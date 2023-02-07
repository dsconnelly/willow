import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

from ..utils.ad99 import AlexanderDunkerton
from ..utils.datasets import load_datasets, _get_columns_and_index
from ..utils.mima import R_dry
from ..utils.plotting import (
    COLORS,
    format_latitude,
    format_name,
    format_pressure
)
from ..utils.wrappers import MiMAModel

def plot_example_profiles(
    data_dir: str,
    model_dirs: list[str],
    output_path: str
) -> None:
    """
    Plot examples input profiles and parameterized drags.

    Parameters
    ----------
    data_dir : Directory where samples are saved.
    model_dirs : Directories where trained models are saved.
    output_path : Path where image will be saved.

    """

    X, Y = load_datasets(data_dir, 'tr', n_samples=int(1e5), component='u')
    profile = X.loc[abs(X['latitude']) < 5].mean().to_numpy()

    ad99 = AlexanderDunkerton()
    _, col_idx = _get_columns_and_index({'wind', 'T'}, X.columns)
    drags = {'ad99' : ad99.predict(profile[None, col_idx])}

    cmap = {'ad99' : 'k'}
    for (model_dir, color) in zip(model_dirs, COLORS):
        path = os.path.join(model_dir, 'model.pkl')
        model_name = os.path.basename(model_dir)

        model: MiMAModel = joblib.load(path)
        drags[model_name] = model.predict_online(profile[None])
        cmap[model_name] = color

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(8, 6)

    pressures = [float(s.split(' @ ')[-1].split()[0]) for s in Y.columns]
    labels = list(map(format_pressure, pressures))
    y = -np.arange(len(pressures))

    twin = axes[0].twiny()
    axes[0].plot(profile[:40], y, color='royalblue')
    twin.plot(profile[79:119], y, color='darkorange')

    seconds_per_day = 60 * 60 * 24
    for name, drag in drags.items():
        color = cmap[name]
        data = seconds_per_day * drag.flatten()
        label = format_name(name, True)

        alpha = 0.85 if name == 'ad99' else 0.6
        zorder = 2 if name == 'ad99' else 1

        axes[1].plot(
            data, y,
            color=color,
            label=label,
            alpha=alpha,
            zorder=zorder
        )

    iterator = zip([axes[0], twin], ['royalblue', 'darkorange'])
    for i, (ax, color) in enumerate(iterator):
        ax.spines[['bottom', 'top'][i]].set_color(color)
        ax.spines[['bottom', 'top'][1 - i]].set_visible(False)

        ax.tick_params(axis='x', colors=color)
        ax.xaxis.label.set_color(color)

    axes[0].set_xlim(-30, 30)
    axes[1].set_xlim(-10, 10)
    twin.set_xlim(180, 300)

    axes[1].set_xscale('symlog', linthresh=1e-1)
    ticks = np.array([-10, -1, -0.1, 0, 0.1, 1, 10])

    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f'{x:2g}' for x in ticks])

    for ax in axes:
        ax.set_ylim(y[-1], y[0])
        ax.set_yticks(y[::3])
        ax.set_yticklabels(labels[::3])

        ax.grid(color='lightgray')
        ax.set_axisbelow(True)

    axes[0].set_xlabel('$u$ (m s$^{-1}$)')
    axes[0].set_ylabel('pressure (hPa)')
    twin.set_xlabel('$T$ (K)')

    axes[1].set_xlabel('GW drag (m s$^{-1}$ day$^{-1}$)')
    axes[1].legend(loc='lower center').set_zorder(3)
    
    plt.tight_layout()
    plt.savefig(output_path)

def plot_example_sources(
    output_path: str,
    lats: list[float]=[2, 60],
    wind: float=20
) -> None:
    """
    Plot example AD99 source spectra.

    Parameters
    ----------
    output_path : Where the plot will be saved.
    lats : Latitudes to show.
    wind : Source level wind.

    """

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4.5)

    ad99 = AlexanderDunkerton()
    cs = ad99._get_phase_speeds()

    rho = 31500 / (R_dry * 245)
    for lat, color in zip(lats, COLORS):
        _, Bs = ad99._get_amp_and_Bs(lat, cs, wind)
        label = 'tropics' if abs(lat) < 15 else 'extratropics'

        ax.scatter(cs, rho * Bs, s=4, color=color)
        ax.scatter([], [], s=20, color=color, label=label)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-0.2, 0.2)

    ax.grid(color='lightgray')
    ax.set_axisbelow(True)

    ax.set_xlabel('phase speed (m s$^{-1}$)')
    ax.set_ylabel('momumentum flux (Pa)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)

