import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.diagnostics import logs, times
from ..utils.plotting import format_pressure
from ..utils.importance import (
    compute_gini_scores,
    compute_shapley_scores,
    get_level_data
)

@logs
@times
def plot_feature_importances(model_dir, output_dir, kind, data_dir=None):
    """
    Plot model feature importances by vertical level.

    Parameters
    ----------
    model_dir : str
        The directory where the trained model is saved.
    output_dir : str
        The directory where plots should be saved.
    kind : str
        The kind of feature importance to use. Must be in {'shapley', 'gini'},
        and 'gini' can only be used for models built on scikit-learn trees.
    data_dir : str
        The directory where the input data is saved, if Shapley values are to be
        recomputed. If None, model_dir should contain an array shapley.npy
        containing precomputed Shapley values.

    """

    model_name = os.path.basename(model_dir)
    if data_dir is not None:
        X, Y = load_datasets(data_dir, 'tr', 5000)
        samples, _ = prepare_datasets(X, Y, model_name, as_array=False)

        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path).model

        if kind == 'shapley':
            ds = compute_shapely_scores(samples, model)
            scores = abs(ds['scores']).mean('sample')
            ds.to_netcdf(os.path.join(model_dir, 'shapley.nc'))

        elif kind == 'gini':
            scores = compute_gini_scores(samples, model)

    else:
        with xr.open_dataset(os.path.join(model_dir, 'shapley.nc')) as ds:
            scores = abs(ds['scores']).mean('sample')

    globals()[f'_plot_{kind}_scores'](model_name, scores, output_dir)
            
_colormap = {
    'wind' : 'royalblue',
    'T' : 'tab:red',
    'Nsq' : 'seagreen'
}

def _plot_gini_scores(model_name, scores, output_dir):
    pressures = scores.index
    y = np.arange(len(pressures))
    data = scores.to_dict('list')
    
    fig, ax = plt.subplots()
    fig.set_size_inches(3, 6)

    _ = _plot_by_level(ax, pressures, data, True)
    ax.set_xlim(0, 0.1)

    plt.tight_layout()

    fname = f'{model_name}-gini-by-level.png'
    plt.savefig(os.path.join(output_dir, fname))

def _plot_shapley_scores(model_name, scores, output_dir):
    levels = [200, 25, 1]
    fig, axes = plt.subplots(ncols=len(levels))
    fig.set_size_inches(3 * len(levels), 6)

    pressures = [format_pressure(p) for p in scores['pressure'].values]
    for j, (level, ax) in enumerate(zip(levels, axes)):
        k, data = get_level_data(scores, level)
        imgs = _plot_by_level(ax, pressures, data, verbose=(j == 0))

        ax.set_xlim(0, 5)
        ax.set_title(f'predictions @ {pressures[k]} hPa')

        for img in imgs:
            for i, bar in enumerate(img):
                if i != k:
                    bar.set_alpha(0.3)

    plt.tight_layout()

    fname = f'{model_name}-shapley-by-level.png'
    plt.savefig(os.path.join(output_dir, fname))

def _plot_by_level(ax, pressures, data, verbose):
    y = np.arange(len(pressures))
    left = np.zeros(len(pressures))
    ax = _setup_level_axis(ax, pressures, y, verbose)

    imgs = []
    for name, widths in data.items():
        color = _colormap[name]
        ax.barh([0], [0], color=color, label=name)

        imgs.append(ax.barh(
            -y, widths,
            height=1,
            color=color,
            edgecolor='k',
            left=left
        ))

        left = left + widths

    if verbose:
        ax.legend()

    return imgs

def _setup_level_axis(ax, pressures, y, set_ylabel):
    ax.set_ylim(-y[-1] - 0.5, -y[0] + 0.5)

    ax.set_yticks(-y[::3])
    ax.set_yticklabels(pressures[::3])

    ax.set_xlabel('importance')
    if set_ylabel:
        ax.set_ylabel('input data pressure (hPa)')
    
    return ax