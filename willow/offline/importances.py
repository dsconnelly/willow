import os

from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

from ..utils.ad99 import AlexanderDunkerton
from ..utils.aliases import Model
from ..utils.datasets import load_datasets, _get_columns_and_index
from ..utils.diagnostics import log
from ..utils.importances import get_profile, get_shapley_values
from ..utils.plotting import COLORS, format_name, format_pressure, get_letter

@log
def plot_feature_importances(
    data_dir: str,
    model_dirs: list[str],
    output_path: str,
    gini: bool=False,
    levels: list[float]=[200, 100, 10]
) -> None:
    """
    Plot Shapley values and potentially Gini importances.

    Parameters
    ----------
    data_dir : Directory where test samples are saved.
    model_dirs : Directories where trained models are saved. Can include a
        directory name containing `'ad99'`, in which case the Shapley values of
        the parameterization itself are computed.
    output_path : Path where image will be saved.
    gini : Whether to also plot Gini importances.
    levels : List of output levels, in hPa, to show importance profiles at.

    """

    if len(model_dirs) > 1 and gini:
        raise ValueError('Cannot plot multiple models and Gini importances')

    X_tr, _ = load_datasets(data_dir, 'tr', n_samples=int(1e6))
    X_te, Y = load_datasets(data_dir, 'te', n_samples=int(1e4))
   
    pressures = np.array([s.split(' @ ')[-1].split()[0] for s in Y.columns])
    pressures = pressures.astype(float)
    
    col_idx: np.ndarray
    model: AlexanderDunkerton | Model
    importances, features = {}, {}

    for model_dir in model_dirs:
        if 'ad99' in model_dir:
            name_parts = {'wind', 'T'}
            model = AlexanderDunkerton()
            _, col_idx = _get_columns_and_index(name_parts, X_tr.columns)

        else:
            wrapper = joblib.load(os.path.join(model_dir, 'model.pkl'))
            col_idx = wrapper.col_idx
            model = wrapper.model

        name = 'shapley' if gini else os.path.basename(model_dir)
        features[name] = X_tr.columns[col_idx].tolist()

        path = os.path.join(model_dir, 'shapley.nc')
        if os.path.exists(path):
            ds = xr.open_dataset(path)

        else:
            background = X_tr.iloc[:, col_idx].to_numpy()
            X = X_te.iloc[:, col_idx].to_numpy()

            ds = get_shapley_values(
                model, background, X, 
                features=features[name], 
                pressures=pressures
            )
            ds.to_netcdf(path)

        importances[name] = abs(ds['importances']).mean('sample').values
        if gini:
            forests = (MultioutputBoostedForest, MultioutputRandomForest)
            if not isinstance(model, forests):
                raise TypeError('Only forest models support Gini importances')

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
            profile = get_profile(data[:, j], features[name])
            xmaxes[name] = max(xmaxes[name], profile.max())

            ax, color = group[name], cmap[name]
            kind = name.capitalize() if name == 'gini' else 'SHAP'
            ax.set_xlabel(f'{kind} importance')

            if kind.lower() in name:
                side = dict(shapley='bottom', gini='top')[name]
                ax.spines[side].set_color(color)
                if kind == 'gini': ax.spines['bottom'].set_visible(False)

                ax.tick_params(axis='x', colors=color)
                ax.xaxis.label.set_color(color)

            label = None if gini else format_name(name, True)
            ax.scatter(profile, y, color=color, label=label)
            ax.plot(profile, y, color=color, alpha=0.3, zorder=1)

        if 'gini' in group: ax = group['shapley']
        ax.barh([y[j]], [10], height=1, color='lightgray', zorder=-2)

        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])
        ax.set_ylim(y[-1], y[0])

        ax.set_ylabel('input pressure (hPa)')
        ax.set_title(f'({get_letter(i)}) {labels[j]} hPa')

        if len(model_dirs) > 1 and i == 0:
            ax.legend()

    if not gini: xmaxes = {name : max(xmaxes.values()) for name in xmaxes}
    for name, xmax in xmaxes.items():
        ticks = np.round(np.linspace(0, 1.2 * xmax, 5), 2)

        for group in groups:
            group[name].set_xlim(0, ticks[-1])
            group[name].set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path)

def plot_shapley_analytics(
    ref_dir: str,
    model_dirs: list[str],
    output_path: str
) -> None:
    """
    Plot disentangled shape and ampltitude analyses of Shapley profiles.

    Parameters
    ----------
    ref_dir : Directory where ground truth Shapley values are saved.
    model_dirs : Directories where model Shapley values are saved.
    output_path : Path where image will be saved.

    """

    corrs: defaultdict[str, np.ndarray]
    sums: defaultdict[str, np.ndarray]

    references = np.zeros((40, 40))
    corrs = defaultdict(lambda: np.zeros(40))
    sums = defaultdict(lambda: np.zeros(40))

    path = os.path.join(ref_dir, 'shapley.nc')
    with xr.open_dataset(path) as ds:
        features = list(ds.feature.values)
        pressures = list(ds.pressure.values)
        importances = abs(ds['importances']).mean('sample').values

        for j in range(40):
            references[j] = get_profile(importances[:, j], features)

    for model_dir in model_dirs:
        name = os.path.basename(model_dir)
        path = os.path.join(model_dir, 'shapley.nc')

        with xr.open_dataset(path) as ds:
            importances = abs(ds['importances']).mean('sample').values

        for j in range(39):
            profile = get_profile(importances[:, j], features)
            corrs[name][j] = np.corrcoef(profile, references[j])[0, 1]
            sums[name][j] = profile.sum() / references[j].sum()

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(6, 6)

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]

    for (name, corr), color in zip(corrs.items(), COLORS):
        axes[0].plot(corr, y, color=color, label=format_name(name))
        axes[1].plot(sums[name], y, color=color)

    for ax in axes:
        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])
        ax.set_ylim(y[-1], y[0])
    
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('SHAP correlation')
    axes[0].set_ylabel('prediction level (hPa)')
    axes[0].legend(loc='lower center')

    axes[1].set_xlim(0.5, 1.5)
    axes[1].set_xlabel('SHAP ratio')

    plt.tight_layout()
    plt.savefig(output_path)
