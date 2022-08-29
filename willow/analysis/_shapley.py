import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.diagnostics import logs, times
from ..utils.plotting import format_pressure
from ..utils.shapley import compute_shapely_values, get_level_data

@logs
@times
def plot_shapley_values(model_dir, output_dir, data_dir=None):
    """
    Plot model Shapley values by pressure and QBO phase.

    Parameters
    ----------
    model_dir : str
        The directory where the trained model is saved.
    output_dir : str
        The directory where plots should be saved.
    data_dir : str
        The directory where the input data is saved, if Shapley values are to be
        recomputed. If None, model_dir should contain an array shapley.npy
        containing precomputed Shapley values.

    """

    model_name = os.path.basename(model_dir)
    shap_path = os.path.join(model_dir, 'shapley.nc')

    if data_dir is not None:
        X, Y = load_datasets(data_dir, 'tr', 5000)
        samples, _ = prepare_datasets(X, Y, model_name, as_array=False)

        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path).model

        ds = compute_shapely_values(samples, model)
        ds.to_netcdf(shap_path)
        scores = ds['scores']
        
    else:
        with xr.open_dataset(shap_path) as ds:
            scores = ds['scores']

    level_path = os.path.join(output_dir, f'{model_name}-by-level.png')
    _plot_by_level(scores, level_path)

_colormap = {
    'wind' : 'royalblue',
    'T' : 'tab:red',
    'Nsq' : 'seagreen'
}

def _plot_by_level(scores, output_path):
    levels = [200, 25, 1]
    fig, axes = plt.subplots(ncols=len(levels))
    fig.set_size_inches(3 * len(levels), 6)

    pressures = [format_pressure(p) for p in scores['pressure'].values]
    y = np.arange(len(pressures))

    scores = abs(scores).mean('sample')
    for j, (level, ax) in enumerate(zip(levels, axes)):
        k, data = get_level_data(scores, level)
        p = pressures[k]

        ax = _setup_level_axis(ax, pressures, y, set_ylabel=(j == 0))
        ax.set_title(f'predictions @ {p} hPa')

        left = np.zeros(len(pressures))
        for name, widths in data.items():
            color = _colormap[name]
            ax.barh([0], [0], color=color, label=name)

            ax.barh(
                -y, widths,
                height=1,
                color=color,
                edgecolor='k',
                alpha=0.3,
                left=left
            )[k].set_alpha(1)

            left = left + widths

        if j == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.clf()
        
def _setup_level_axis(ax, pressures, y, set_ylabel=True):
    ax.set_xlim(0, 5)
    ax.set_ylim(-y[-1] - 0.5, -y[0] + 0.5)

    ax.set_yticks(-y[::3])
    ax.set_yticklabels(pressures[::3])

    ax.set_xlabel('importance')
    if set_ylabel:
        ax.set_ylabel('input data pressure (hPa)')
    
    return ax