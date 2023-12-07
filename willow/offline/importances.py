import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

from ..utils.ad99 import AlexanderDunkerton
from ..utils.datasets import load_datasets, _get_columns_and_index
from ..utils.diagnostics import log
from ..utils.importances import get_shapley_values
from ..utils.plotting import (
    COLORS,
    format_name,
    format_pressure,
    get_letter
)
from ..utils.statistics import standardize

@log
def save_shapley_values(data_dir: str, model_dir: str) -> None:
    """
    Compute and save Shapley values for later plotting.

    Parameters
    ----------
    data_dir : Directory where train and test samples are saved.
    model_dir : Directory where trained model is saved, and where computed
        Shapley values will be saved.

    """

    X_tr, _ = load_datasets(data_dir, 'tr', n_samples=int(1e6), seed=1234)
    X_te, Y = load_datasets(data_dir, 'te', n_samples=int(1e3), seed=5678)

    if 'ad99' in model_dir:
        name_parts = {'wind', 'T'}
        model = AlexanderDunkerton()
        _, col_idx = _get_columns_and_index(name_parts, X_tr.columns)

    else:
        wrapper = joblib.load(os.path.join(model_dir, 'model.pkl'))
        col_idx= wrapper.col_idx
        model = wrapper.model

    features = X_tr.columns[col_idx].tolist()
    pressures = np.array([s.split(' @ ')[-1].split()[0] for s in Y.columns])
    pressures = pressures.astype(float)

    background = X_tr.iloc[:, col_idx].to_numpy()
    X = X_te.iloc[:, col_idx].to_numpy()

    path = os.path.join(model_dir, 'shapley.nc')
    get_shapley_values(
        model, background, X, 
        features=features, 
        pressures=pressures
    ).to_netcdf(path)

def plot_feature_importances(
    model_dirs: list[str],
    output_path: str,
    gini: bool=False,
    levels: list[float]=[200, 100, 10],
    suffix: str=''
) -> None:
    """
    Plot Shapley values and potentially Gini importances.

    Parameters
    ----------
    model_dirs : Directories where Shapley values are saved.
    output_path : Path where image will be saved.
    gini : Whether to also plot Gini importances.
    levels : List of output levels, in hPa, to show importance profiles at.
    suffix : Suffix to specify netCDF file in `model_dir`.

    """

    if len(model_dirs) > 1 and gini:
        raise ValueError('Cannot plot multiple models and Gini importances')

    if suffix:
        suffix = f'-{suffix}'

    importances, features = {}, {}
    for model_dir in model_dirs:
        name = 'shapley' if gini else os.path.basename(model_dir)
        path = os.path.join(model_dir, f'shapley{suffix}.nc')

        with xr.open_dataset(path) as ds:
            importances[name] = _parse_shapley_values(ds)
            features[name] = ds['feature'].values.tolist()
            pressures = ds['pressure'].values

        if gini:
            wrapper = joblib.load(os.path.join(model_dir, 'model.pkl'))
            model = wrapper.model

            forests = (MultioutputBoostedForest, MultioutputRandomForest)
            if not isinstance(model,forests):
                raise TypeError('Only forest models support Gini importance')

            importances['gini'] = model.feature_importances_
            features['gini'] = features['shapley']

    n_subplots = len(levels)
    fig, axes = plt.subplots(ncols=n_subplots)
    fig.set_size_inches(3 * n_subplots, 6)

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]

    keys = [s for s in importances.keys() if 'ad99' not in s]
    cmap = dict(zip(keys, COLORS))
    cmap['ad99-wind-T'] = 'k'

    xmaxes = {name : -np.inf for name in importances}
    make_twin = lambda ax, name: ax.twiny() if name == 'gini' else ax
    groups = [{name : make_twin(ax, name) for name in cmap} for ax in axes]

    for i, (level, group) in enumerate(zip(levels, groups)):
        j = np.argmin(abs(pressures - level))
        for name, data in importances.items():
            # profile = get_profile(data[:, j], features[name])
            profile = data[:40, j]
            xmaxes[name] = max(xmaxes[name], profile.max())

            ax, color = group[name], cmap[name]
            kind = name.capitalize() if name == 'gini' else 'SHAP'
            ax.set_xlabel(f'{kind} importance')

            if kind.lower() in name:
                side = dict(shapley='bottom', gini='top')[name]
                ax.spines[side].set_color(color)

                if name == 'gini': 
                    ax.spines['bottom'].set_visible(False)
                    ax.grid(False)

                ax.tick_params(axis='x', colors=color)
                ax.xaxis.label.set_color(color)

            label = None if gini else format_name(name, True)
            ax.plot(profile, y, color=color, alpha=1, label=label, zorder=1)

        if 'gini' in group: ax = group['shapley']
        ax.barh([y[j]], [10], height=1, color='lightgray', zorder=-2)

        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])
        ax.set_ylim(y[-1], y[0])

        ax.set_ylabel('input pressure (hPa)')
        ax.set_title(f'({get_letter(i)}) {labels[j]} hPa')
        
        ax.set_axisbelow(True)
        if len(model_dirs) > 1 and i == 0:
            ax.legend()

    if not gini: xmaxes = {name : max(xmaxes.values()) for name in xmaxes}
    for name, xmax in xmaxes.items():
        ticks = np.linspace(0, 1.2 * xmax, 5)
        labels = np.round(ticks, 2)

        for group in groups:
            group[name].set_xlim(0, ticks[-1])
            group[name].set_xticks(ticks)
            group[name].set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(output_path)

def plot_scalar_importances(
    model_dirs: list[str],
    output_path: str
) -> None:
    """
    Plot the importance of surface pressure and latitude.

    Parameters
    ----------
    model_dirs : Directories where model SHAP values are saved.
    output_path : Path where image will be saved.

    """

    fig, ax = plt.subplots()
    fig.set_size_inches(1.15 * 3, 1.15 * 6)
    
    colors = ['k'] + [c for _, c in zip(model_dirs[1:], COLORS)]
    for model_dir, color in zip(model_dirs, colors):
        path = os.path.join(model_dir, 'shapley.nc')

        ls = 'dashed' if 'lat_scale' in path else 'solid'
        if 'lat_scale' in path:
            color = 'tab:red'

        with xr.open_dataset(path) as ds:
            pressures = ds.pressure.values
            y = -np.arange(len(pressures))
            labels = [format_pressure(p) for p in pressures]

            name = format_name(model_dir.split('/')[-1], True)
            data = abs(ds['importances']).isel(feature=-1).mean('sample').values
            ax.plot(data, y, color=color, ls=ls, label=name)
 
    ax.set_xlim(0, 0.45)
    ax.set_ylim(y[-1], y[0])

    ax.set_xticks([0, 0.15, 0.3, 0.45])
    ax.set_yticks(y[::3])
    ax.set_yticklabels(labels[::3])

    ax.set_xlabel('latitude SHAP value')
    ax.set_ylabel('prediction level (hPa)')

    # ax.grid(False)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)

def plot_shapley_errors(
    model_dirs: list[str],
    output_path: str,
    levels: list[float]=[200, 100, 10],
    suffix: str=''
) -> None:
    """
    Plot average utility of features to reducing emulator error.

    Parameters
    ----------
    model_dirs : Directories where model SHAP values are saved.
    output_path : Path where image will be saved.
    levels : List of output levels, in hPa, to show analyses at.
    suffix : Suffix to specify netCDF file in `model_dir`.

    """

    if suffix:
        suffix = f'-{suffix}'

    ref_dir, *model_dirs = model_dirs
    # path = os.path.join(ref_dir, f'shapley{suffix}.nc')
    path = os.path.join(ref_dir, 'shapley.nc')
    model = AlexanderDunkerton()

    with xr.open_dataset(path) as ds:
        pressures = ds['pressure'].values
        samples = ds['X'].values

        # data, *_ = standardize(model.predict(samples))
        # targets = xr.DataArray(data, (ds['sample'], ds['pressure']))
        targets = None

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]

    importances, outputs = {}, {}
    for model_dir in model_dirs:
        path = os.path.join(model_dir, f'shapley{suffix}.nc')
        name = os.path.basename(model_dir)

        with xr.open_dataset(path) as ds:
            # assert np.allclose(samples, ds['X'].values)
            importances[name] = ds['importances']
            outputs[name] = ds['predictions']

            if targets is None:
                data, *_ = standardize(model.predict(ds['X'].values))
                targets = xr.DataArray(data, (ds['sample'], ds['pressure']))

    n_columns = len(levels) + 1
    fig, axes = plt.subplots(ncols=n_columns)
    fig.set_size_inches(3 * n_columns, 6)

    for name, data in importances.items():
        output = outputs[name]
        errors = abs(output - targets).mean('sample')
        references = abs(output - targets - data).mean('sample')

        color = 'tab:red' if 'mubofo' in name else 'forestgreen'
        for level, ax in zip(levels, axes):
            k = np.argmin(abs(pressures - level))
            profile = references[k] - errors[k]
            ax.plot(profile[:40], y, color=color)

    for j, (level, ax) in enumerate(zip(levels, axes)):
        k = np.argmin(abs(pressures - level))
        ax.barh(
            [y[k]], [10],
            left=-5,
            height=1,
            color='lightgray',
            zorder=-2
        )

        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])

        ax.set_xlim(-0.1, 2.5)
        ax.set_ylim(y[-1], y[0])

        ax.set_xlabel(r'$\overline{s_{\mathsf{wind}}^\ast}$')
        ax.set_ylabel('input pressure (hPa)')
        ax.set_title(f'({get_letter(j)}) {labels[k]} hPa')

        if j == 0:
            ax.plot([], [], color='tab:red', label='boosted forest')
            ax.plot([], [], color='forestgreen', label='neural network')
            ax.legend()

    window = 3
    y = -np.arange(-window, window + 1)

    ax = axes[-1]
    ax.barh([0], [8], height=0.3, color='lightgray', left=-4, zorder=-2)

    for name, data in importances.items():
        composite = np.zeros(2 * window + 1)
        for k in range(window, 40 - window):
            if pressures[k] > 115:
                break

            output = outputs[name]
            errors = abs(output - targets).mean('sample')
            references = abs(output - targets - data).mean('sample')

            profile = (references[k] - errors[k]).values[:40]
            start, end = k - window, k + window + 1
            composite += profile[start:end]

        color = 'tab:red' if 'mubofo' in name else 'forestgreen'
        ax.plot(composite / (k - window), y, color=color)

    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-window, window)

    ax.set_xlabel(r'$\overline{s_{\mathsf{wind}}^\ast}$')
    ax.set_ylabel('levels above prediction level')
    ax.set_title(f'({get_letter(j + 1)}) composite')

    plt.tight_layout()
    plt.savefig(output_path)

def _parse_shapley_values(ds: xr.Dataset, rms: bool=False) -> np.ndarray:
    """
    Combine Shapley values for individual samples in a consistent way.

    Parameters
    ----------
    ds : Dataset containing Shapley importances.
    rms : Whether to use RMS averaging instead of absolute values.

    Returns
    -------
    importances : Array of importances combined over all samples in `ds`.

    """

    if rms:
        return np.sqrt((ds['importances'] ** 2).mean('sample').values)

    return abs(ds['importances']).mean('sample').values