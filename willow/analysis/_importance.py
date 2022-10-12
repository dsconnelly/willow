import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.ad99 import AlexanderDunkerton
from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.diagnostics import logs, times
from ..utils.plotting import colors, format_name, format_pressure
from ..utils.importance import get_level_data, get_shapley_scores

@logs
@times
def save_shapley_scores(model_dir, data_dir):
    """
    Compute and save Shapley values for plotting later.

    Parameters
    ----------
    model_dir : str
        Directory where the trained model is saved, and where the Shapley values
        will be saved. If model_dir starts with 'ad99', the Shapley values of
        AlexanderDunkerton.predict will be saved instead.
    data_dir : str
        Directory where input data is saved.

    """

    model_name = os.path.basename(model_dir)
    X, Y = load_datasets(data_dir, 'tr', 5000)
    samples, Y = prepare_datasets(X, Y, model_name, as_array=False)

    if model_name.startswith('ad99'):
        model = AlexanderDunkerton()
        Y = Y.to_numpy()

    else:
        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path).model
        Y = None

    ds = get_shapley_scores(samples, model, Y)
    ds.to_netcdf(os.path.join(model_dir, 'shapley.nc'))

def plot_lmis(model_dirs, output_path):
    """
    Plot the Shapley levels of maximum importance for one or more models.

    Parameters
    ----------
    model_dirs : list of str
        Directories for models whose LMIs will be plotted. Each should contain
        a shapley.nc file as created by save_shapley_scores.
    output_path : str
        Where to save the image.

    """

    data = {}
    for model_dir in model_dirs:
        name = format_name(os.path.basename(model_dir))
        with xr.open_dataset(os.path.join(model_dir, 'shapley.nc')) as ds:
            data[name] = abs(ds['scores']).mean('sample')

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    use_colors = colors
    if 'ad99' in model_dirs[0]:
        use_colors = ['k'] + use_colors

    k_lo, k_hi = 11, 37
    for (name, scores), color in zip(data.items(), use_colors):
        levels = scores.pressure.values[::-1]
        xs = np.arange(len(levels))
        ys = np.zeros(xs.shape)

        for k, level in enumerate(levels[1:], start=1):
            _, profiles = get_level_data(scores, level)
            weights = sum(profile for _, profile in profiles.items())

            weights = weights[::-1]
            if 'AD99' in name:
                weights[(k + 2):] = 0

            weights = (weights / weights.sum())
            ys[k] = np.average(xs, weights=weights)

        slope, _ = np.polyfit(xs[k_lo:k_hi], ys[k_lo:k_hi], deg=1)
        ax.scatter(
            xs[1:], ys[1:],
            color=color,
            zorder=2,
            clip_on=False,
            label=f'{name} (slope = {slope:.2f})'
        )

    gray = (0.839, 0.839, 0.839)
    ax.plot(xs[k_lo:k_hi], xs[k_lo:k_hi] + 1, color=gray, zorder=1)

    kwargs = {'lw' : 0, 'fc' : 'none', 'ec' : gray, 'hatch' : 'xx'}
    ax.fill_between([0, k_lo - 0.5], 0, 39, **kwargs)
    ax.fill_between([k_hi - 0.5, 39], 0, 39, **kwargs)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xticks(xs[::3])
    ax.set_yticks(xs[::3])

    levels = [format_pressure(p) for p in levels]
    ax.set_xticklabels(levels[::3], rotation=45)
    ax.set_yticklabels(levels[::3])

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(xs.min(), xs.max())

    ax.set_xlabel('prediction level (hPa)')
    ax.set_ylabel('level of maximum importance (hPa)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, transparent=False)

def plot_shapley_scores(model_dir, output_path, levels=[200, 25, 1]):
    """
    Plot Shapley values for predictions at several levels.

    Parameters
    ----------
    model_dir : str
        Directory containing a shapley.nc file, as created by 
        save_shapley_scores, for the model in question.
    output_path : str
        Where to save the image.
    levels : list of float
        Pressure levels (in hPa) of predictions to plot.

    """

    with xr.open_dataset(os.path.join(model_dir, 'shapley.nc')) as ds:
        scores = abs(ds['scores']).mean('sample')
        pressures = [format_pressure(p) for p in scores['pressure'].values]

    fig, axes = plt.subplots(ncols=len(levels))
    fig.set_size_inches(3 * len(levels), 6)

    xmax = -np.inf
    y = np.arange(len(pressures))

    for j, (level, ax) in enumerate(zip(levels, axes)):
        ax = _setup_level_axis(ax, pressures, y, set_ylabel=(j == 0))
        k, data = get_level_data(scores, level)

        left = np.zeros(y.shape)
        for name, widths in data.items():
            color = _colormap[name]
            ax.barh([0], [0], color=color, label=name)

            if 'ad99' in model_dir:
                widths[:(k - 1)] = 0

            ax.barh(
                -y, widths,
                height=1,
                color=color,
                edgecolor='k',
                left=left,
                alpha=0.3
            )[k].set_alpha(1)

            left = left + widths

        if j == 0:
            ax.legend()

        xmax = max(xmax, left.max())
        ax.set_title(f'predictions @ {pressures[k]} hPa')

    xmax = 1.2 * xmax
    xticks = np.linspace(0, xmax, 6)

    for ax in axes:
        ax.set_xlim(0, xmax)
        ax.set_xticks(xticks)

    plt.tight_layout()
    plt.savefig(output_path, transparent=False)
            
_colormap = dict(zip(['wind', 'T', 'Nsq', 'shear'], colors))

def _setup_level_axis(ax, pressures, y, set_ylabel):
    ax.set_ylim(-y[-1] - 0.5, -y[0] + 0.5)

    ax.set_yticks(-y[::3])
    ax.set_yticklabels(pressures[::3])

    ax.set_xlabel('importance')
    if set_ylabel:
        ax.set_ylabel('input data pressure (hPa)')
    
    return ax