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
from ..utils.plotting import COLORS, format_pressure

@log
def plot_feature_importances(
    data_dir: str,
    model_dir: str,
    output_path: str,
    kind: str='shapley',
    levels: list[float]=[200, 100, 10]
) -> None:
    """
    Plot Shapley values or Gini importances.

    Parameters
    ----------
    data_dir : Directory where test samples are saved.
    model_dir : Directory where trained model is saved. Alternatively, can be
        a directory name containing `'ad99'`, in which case the Shapley values
        of the parameterization itself are computed.
    output_path : Path where image will be saved.
    kind : Kind of feature importance to plot; either `'shapley'`, `'gini'`, or
        `'both'`, where `'both'` plots both Gini and Shapley scores.
    levels : List of output levels, in hPa, to show importance profiles at.

    """

    col_idx: np.ndarray
    model: AlexanderDunkerton | Model
    X, Y = load_datasets(data_dir, 'te', n_samples=int(1e3))
    
    if 'ad99' in model_dir:
        name_parts = {'wind', 'T'}
        model = AlexanderDunkerton()
        _, col_idx = _get_columns_and_index(name_parts, X.columns)

    else:
        wrapper = joblib.load(os.path.join(model_dir, 'model.pkl'))
        col_idx = wrapper.col_idx
        model = wrapper.model

    features = X.columns[col_idx].tolist()
    pressures = np.array([s.split(' @ ')[-1].split()[0] for s in Y.columns])
    pressures = pressures.astype(float)

    importances = {}
    if kind in ('gini', 'both'):
        forests = (MultioutputBoostedForest, MultioutputRandomForest)
        if not isinstance(model, forests):
            raise TypeError('Only forest models support Gini importances')

        importances['gini'] = model.feature_importances_

    if kind in ('shapley', 'both'):
        path = os.path.join(model_dir, 'shapley.nc')

        if os.path.exists(path):
            ds = xr.open_dataset(path)

        else:
            X = X.iloc[:, col_idx]
            ds = get_shapley_values(model, X.to_numpy(), features, pressures)
            ds.to_netcdf(path)

        importances['shapley'] = abs(ds['importances']).mean('sample').values

    n_subplots = len(levels)
    fig, axes = plt.subplots(ncols=n_subplots)
    fig.set_size_inches(3 * n_subplots, 6)

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]
    cmap = dict(zip(['gini', 'shapley'], COLORS))

    xmax = -np.inf
    for i, (level, ax) in enumerate(zip(levels, axes)):
        j = np.argmin(abs(pressures - level))
        ax.barh([y[j]], [1], color='lightgray', height=1, zorder=-2)

        for kind, data in importances.items():
            profiles = list(get_profiles(data[:, j], features).values())
            profile = np.stack(profiles).sum(axis=0)
            profile = profile / profile.sum()
            
            color = cmap[kind]
            ax.plot(profile, y, color=color, alpha=0.3, zorder=-1)
            ax.scatter(profile, y, color=color, label=kind.capitalize())
            xmax = max(xmax, profile.max())

        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])
        ax.set_ylim(y[-1], y[0])

        ax.set_xlabel(f'normalized importance')
        ax.set_ylabel('input pressure (hPa)')
        
        ax.set_title(f'{labels[j]} hPa')
        if i == 0 and len(importances) > 1:
            ax.legend()

    xmax = 1.2 * xmax
    ticks = np.round(np.linspace(0, xmax, 5), 2)

    for ax in axes:
        ax.set_xlim(0, ticks[-1])
        ax.set_xticks(ticks)

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