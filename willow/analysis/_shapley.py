import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.diagnostics import logs, times
from ..utils.plotting import format_pressure
from ..utils.shapley import combine_by_level, compute_shapely_values

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

    import logging
    logging.info = print

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

def _plot_by_level(scores, output_path):
    levels = [200, 25, 1]
    fig, axes = plt.subplots(ncols=len(levels))
    fig.set_size_inches(3 * len(levels), 6)

    pressures = [format_pressure(p) for p in scores['pressure'].values]
    y = np.arange(len(pressures))

    data = combine_by_level(abs(scores).mean('sample'))
    for j, (level, ax) in enumerate(zip(levels, axes)):
        k = abs(scores['pressure'].values - level).argmin()
        p = str(pressures[k])
        widths = data[p] / (1.02 * data[p].max())

        ax = _setup_level_axis(ax, pressures, y, set_ylabel=(j == 0))
        ax.barh(
            -y, widths,
            height=1,
            color='darkgray',
            edgecolor='k',
            alpha=0.3
        )[k].set_alpha(1)

        ax.set_title(f'predictions @ {p} hPa')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.clf()
        
def _setup_level_axis(ax, pressures, y, set_ylabel=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(-y[-1] - 0.5, -y[0] + 0.5)

    ax.set_yticks(-y[::3])
    ax.set_yticklabels(pressures[::3])

    ax.set_xlabel('importance')
    if set_ylabel:
        ax.set_ylabel('input data pressure (hPa)')
    
    return ax