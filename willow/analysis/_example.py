import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.plotting import colors, format_name

_exp = '$\mathregular{^{-1}}$'

def plot_example_profiles(data_dir, model_dirs, output_path):
    """
    Plot a wind profile and the corresponding gravity wave drags.

    Parameters
    ----------
    data_dir : str
        Directory where the datasets saved.
    model_dirs : list of str
        Directories containing trained models whose predictions will be plotted.
    output_path : str
        Path to save the image.

    """

    seconds_per_day = 60 * 60 * 24
    X, Y = load_datasets(data_dir, 'tr', 100)
    Y = Y * seconds_per_day

    jdx, pressures = [], []
    for j, s in enumerate(X.columns):
        if 'wind' in s:
            jdx.append(j)
            pressures.append(s.split(' @ ')[-1].split()[0])

    idx = abs(Y).max(axis=1) < 10
    X, Y = X.drop('time', axis=1)[idx].to_numpy(), Y[idx].to_numpy()

    drags = {'AD99' : Y[0]}
    for model_dir in model_dirs:
        name = format_name(os.path.basename(model_dir))
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        drags[name] = seconds_per_day * model.predict_online(X)[0]

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(8, 6)

    ys = np.arange(len(pressures))
    _plot_wind(axes[0], X[0, jdx], ys, pressures)
    _plot_drags(axes[1], drags, ys, pressures)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, transparent=True)

def _plot_wind(ax, wind, ys, pressures):
    ax.plot(wind, -ys, color='k', marker='o')

    xmin = 10 * (wind.min() // 10)
    xmax = 10 * (wind.max() // 10 + 1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ys[-1], -ys[0])

    ax.set_yticks(-ys[::3])
    ax.set_yticklabels(pressures[::3])

    ax.set_xlabel(f'wind (m s{_exp})')
    ax.set_ylabel('pressure (hPa)')

def _plot_drags(ax, drags, ys, pressures):
    for (name, drag), color in zip(drags.items(), ['k'] + colors):
        ax.plot(drag, -ys, color=color, marker='o', label=name)

    ticks = np.array([-10, -1, -0.1, 0, 0.1, 1, 10])
    ax.set_xscale('symlog', linthresh=1e-1)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-ys[-1], -ys[0])

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    ax.set_yticks(-ys[::3])
    ax.set_yticklabels(pressures[::3])

    ax.set_xlabel(f'GW drag (m s{_exp} day{_exp})')
    ax.set_ylabel('pressure (hPa)')
    ax.legend(loc='lower left')