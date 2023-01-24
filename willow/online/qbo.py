import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from ..utils.plotting import format_name, format_pressure
from ..utils.qbo import get_qbo_statistics, load_qbo

_CBAR_SCALE = 0.1
_SUBPLOTS_PER_COL = 3

_CMAP = 'RdBu_r'
_VMAX = 75

def plot_qbos(case_dirs: list[str], output_path: str) -> None:
    """
    Plot quasi-biennial oscillations from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : Directories of the MiMA runs with QBOs to be plotted.
    output_path : Path where the plot should be saved.

    """

    n_subplots = len(case_dirs)
    n_rows = min(n_subplots, _SUBPLOTS_PER_COL)
    n_cols = (n_subplots - 1) // _SUBPLOTS_PER_COL + 1

    fig = plt.figure(constrained_layout=True)
    height_ratios = [1] * n_rows + [_CBAR_SCALE]
    gs = GridSpec(n_rows + 1, n_cols, fig, height_ratios=height_ratios)
    fig.set_size_inches(9 * n_cols, 3 * (n_rows + _CBAR_SCALE))

    axes: list[Axes] = []
    for j in range(n_cols):
        for i in range(n_rows):
            if len(axes) < n_subplots:
                axes.append(fig.add_subplot(gs[i, j]))

    levels = np.linspace(-_VMAX, _VMAX, 13)
    for i, (case_dir, ax) in enumerate(zip(case_dirs, axes)):
        u = load_qbo(case_dir, n_years=24)
        period, period_err, amp, amp_err = get_qbo_statistics(u)
        time = u['time'].values

        years = np.array([x.days for x in time - time[0]]) / 360
        pressures = [format_pressure(p) for p in u.pfull.values]
        ys = -np.arange(len(pressures))

        img = ax.contourf(
            years, ys, u.T,
            vmin=-_VMAX,
            vmax=_VMAX,
            cmap=_CMAP,
            levels=levels
        )

        n_ticks = round(years.max() / 4) + 1
        ticks = np.linspace(0, years.max(), n_ticks)
        labels = np.round(ticks).astype(int)

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('year')

        ax.set_yticks(ys[::3])
        ax.set_yticklabels(pressures[::3])
        ax.set_ylabel('pressure (hPa)')

        text = f'{period:.1f} $\pm$ {period_err:.1f} month period\n'
        text += f'{amp:.1f} $\pm$ {amp_err:.1f} m s$^{{-1}}$ amplitude'
        bbox = dict(alpha=0.8, boxstyle='round', edgecolor='k', facecolor='w')
        
        ax.text(
            0.85, 0.17, text,
            ha='center',
            va='center',
            transform=ax.transAxes,
            bbox=bbox
        )

        model_name = os.path.basename(case_dir)
        ax.set_title(f'({chr(97 + i)}) {format_name(model_name)}')

    cax = fig.add_subplot(gs[-1, :])
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label('zonal mean tropical $u$ (m s$^{-1}$)')
    cbar.set_ticks(levels[::2])

    plt.savefig(output_path)