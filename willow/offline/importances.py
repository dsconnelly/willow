import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

from ..utils.ad99 import AlexanderDunkerton
from ..utils.aliases import Model
from ..utils.datasets import load_datasets, _get_columns_and_index
from ..utils.diagnostics import log
from ..utils.importances import get_profiles, get_shapley_values
from ..utils.plotting import COLORS, format_name, format_pressure

# @log
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

    X, Y = load_datasets(data_dir, 'te', n_samples=int(1e3))
    pressures = np.array([s.split(' @ ')[-1].split()[0] for s in Y.columns])
    pressures = pressures.astype(float)

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]

    col_idx: np.ndarray
    model: AlexanderDunkerton | Model

    importances, features = {}, {}
    for model_dir in model_dirs:
        if 'ad99' in model_dir:
            name_parts = {'wind', 'T'}
            model = AlexanderDunkerton()
            _, col_idx = _get_columns_and_index(name_parts, X.columns)

        else:
            wrapper = joblib.load(os.path.join(model_dir, 'model.pkl'))
            col_idx = wrapper.col_idx
            model = wrapper.model

        name = 'shapley' if gini else os.path.basename(model_dir)
        features[name] = X.columns[col_idx].tolist()

        path = os.path.join(model_dir, 'shapley.nc')
        if os.path.exists(path):
            ds = xr.open_dataset(path)

        else:
            samples = X.iloc[:, col_idx].to_numpy()
            ds = get_shapley_values(model, samples, features[name], pressures)
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

    cmap = dict(zip(importances.keys(), COLORS))
    for name in cmap:
        if 'ad99' in name:
            cmap[name] = 'k'

    xmaxes = {name : -np.inf for name in importances}
    make_twin = lambda ax, name: ax.twiny() if name == 'gini' else ax
    groups = [{name : make_twin(ax, name) for name in cmap} for ax in axes]

    for i, (level, group) in enumerate(zip(levels, groups)):
        j = np.argmin(abs(pressures - level))
        for name, data in importances.items():
            profiles = list(get_profiles(data[:, j], features[name]).values())
            profile = np.stack(profiles).sum(axis=0)
            xmaxes[name] = max(xmaxes[name], profile.max())

            ax, color = group[name], cmap[name]
            kind = name if name == 'gini' else 'shapley'
            ax.set_xlabel(f'{kind.capitalize()} importance')

            if name == kind:
                side = dict(shapley='bottom', gini='top')[kind]
                ax.spines[side].set_color(color)
                if kind == 'gini': ax.spines['bottom'].set_visible(False)

                ax.tick_params(axis='x', colors=color)
                ax.xaxis.label.set_color(color)

            label = None if gini else format_name(name, True)
            ax.scatter(profile, y, color=color, label=label)
            ax.plot(profile, y, color=color, alpha=0.3, zorder=-1)

        ax.barh([y[j]], [10], height=1, color='lightgray', zorder=-2)

        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])
        ax.set_ylim(y[-1], y[0])

        ax.set_ylabel('input pressure (hPa)')
        ax.set_title(f'{labels[j]} hPa')

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

def plot_cois(
    model_dirs: list[str],
    output_path: str,
    kind: str='shapley'
) -> None:
    """
    Plot center of importance by level for several models.

    Parameters
    ----------
    model_dirs : Directories where trained models are saved. If `kind` is
        `'shapley'`, each directory should also contain a `shapley.nc` file.
    output_path : Path where image will be saved.
    kind : Kind of feature importance to use in calculating centers of
        importance; should be either `'shapley'` or `'gini'`.

    """