import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

from ..utils.aliases import Model
from ..utils.datasets import load_datasets
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
    model_dir : Directory where trained model is saved.
    output_path : Path where image will be saved.
    kind : Kind of feature importance to plot; either `'shapley'` or `'gini'`.
    levels : List of output levels, in hPa, to show importance profiles at.

    """

    X, Y = load_datasets(data_dir, 'te', n_samples=int(1e3))
    wrapper = joblib.load(os.path.join(model_dir, 'model.pkl'))
    model: Model = wrapper.model

    features = X.columns[wrapper.col_idx]
    pressures = np.array([s.split(' @ ')[-1].split()[0] for s in Y.columns])
    pressures = pressures.astype(float)

    if kind == 'shapley':
        path = os.path.join(model_dir, 'shapley.nc')

        if os.path.exists(path):
            ds = xr.open_dataset(path)

        else:
            X = X.iloc[:, wrapper.col_idx]
            ds = get_shapley_values(model, X.to_numpy(), features, pressures)
            ds.to_netcdf(path)

        importances = abs(ds['importances']).mean('sample').values

    elif kind == 'gini':
        forests = (MultioutputBoostedForest, MultioutputRandomForest)
        if not isinstance(model, forests):
            raise TypeError('Only forest models support Gini importances')

        importances = model.feature_importances_
    
    n_subplots = len(levels)
    fig, axes = plt.subplots(ncols=n_subplots)
    fig.set_size_inches(3 * n_subplots, 6)

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]
    cmap = dict(zip(['wind', 'T', 'N', 'shear'], COLORS))

    xmax = -np.inf
    for i, (level, ax) in enumerate(zip(levels, axes)):
        j = np.argmin(abs(pressures - level))
        profiles = get_profiles(importances[:, j], features)

        left = np.zeros(y.shape)
        for name, widths in profiles.items():
            color = cmap[name]
            ax.barh([0], [0], color=color, label=name)

            ax.barh(
                y, widths,
                height=1,
                color=color,
                edgecolor='k',
                left=left,
                alpha=0.3
            )[j].set_alpha(1)

            left = left + widths

        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])
        ax.set_ylim(y[-1] - 0.5, y[0] + 0.5)

        ax.set_xlabel(f'{kind.capitalize()} importance')
        ax.set_ylabel('pressure (hPa)')
        xmax = max(xmax, left.max())

        ax.set_title(f'@ {labels[j]} hPa')
        if i == 0:
            ax.legend()

    xmax = 1.2 * xmax
    ticks = np.round(np.linspace(0, xmax, 5), 3)

    for ax in axes:
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path)