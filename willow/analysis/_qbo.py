import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.qbo import load_qbo, qbo_statistics
from ..utils.plotting import (
    format_name,
    format_pressure,
    get_bounds_and_cmap,
    get_units
)

def plot_qbos(case_dirs, output_path):
    """
    Plot quasi-biennial oscillations from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : list of str
        Directories of the MiMA runs with QBOs to be plotted.
    output_path : str
        Path where the plot should be saved.

    """

    plots_per_col = 3

    n_qbos = len(case_dirs)
    n_rows = min(n_qbos, plots_per_col)
    n_cols = (n_qbos + plots_per_col - 1) // plots_per_col

    cbar_scale = 0.1
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(9 * n_cols, 3 * (n_rows + cbar_scale))

    height_ratios = [1] * n_rows + [cbar_scale]
    gs = gridspec.GridSpec(
        ncols=n_cols, nrows=(n_rows + 1),
        height_ratios=height_ratios,
        figure=fig
    )

    axes, cax = [], fig.add_subplot(gs[-1, :])
    for j in range(n_cols):
        for i in range(n_rows):
            if len(axes) < n_qbos:
                axes.append(fig.add_subplot(gs[i, j]))

    data = {os.path.basename(s) : load_qbo(s, n_years=32) for s in case_dirs}
    # vmin, vmax, cmap = get_bounds_and_cmap('u', data)
    vmin, vmax, cmap = -75, 75, 'RdBu_r'
    levels = np.linspace(vmin, vmax, 13)

    for (name, u), ax in zip(data.items(), axes):
        years = (u.time - u.time[0]) / 360
        ys = np.arange(len(u.pfull))
        ps = [format_pressure(p) for p in u.pfull.values]

        img = ax.contourf(
            years, -ys, u.T, 
            vmin=vmin, 
            vmax=vmax,
            cmap=cmap,
            levels=levels
        )

        ax.set_yticks(-ys[::3])
        ax.set_yticklabels(ps[::3])

        ax.set_xticks(np.linspace(0, years.max(), 9))
        ax.set_xticklabels(np.linspace(0, 32, 9).astype(int))

        ax.set_xlabel('year')
        ax.set_ylabel('p (hPa)')

        period, period_err, amp, amp_err = qbo_statistics(u)
        info = (
            f'{period:.1f} $\pm$ {period_err:.1f} month period\n'
            f'{amp:.1f} $\pm$ {amp_err:.1f} m s$^{{-1}}$ amplitude'
        )

        #ax.set_title(format_name(name))
        ax.set_title(name)
        ax.text(
            0.85, 0.17, info,
            ha='center',
            va='center',
            transform=ax.transAxes,
            bbox=dict(
                edgecolor='k',
                facecolor='w',
                boxstyle='round',
                alpha=0.8
            )
        )

    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label(f'zonal mean tropical $u$ ({get_units("u")})')
    cbar.set_ticks(levels[::2])

    plt.savefig(output_path, bbox_inches='tight')
        
