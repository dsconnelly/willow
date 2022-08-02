import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.qbo import load_qbo, qbo_statistics
from ..utils.plotting import format_pressure

def plot_qbos(case_dirs, output_path):
    """
    Plot quasi-biennial oscillations from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : list of str
        The directories of the MiMA runs with QBOs to be plotted.
    output_path : str
        Path where the plot should be saved.

    """

    n_subplots = len(case_dirs)
    height_ratios = [1] * n_subplots + [0.1]

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(9, 3 * n_subplots + 0.3)
    gs = gridspec.GridSpec(
        ncols=1, nrows=(n_subplots + 1),
        height_ratios=height_ratios,
        figure=fig
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_subplots)]
    cax = fig.add_subplot(gs[n_subplots, 0])

    data = {os.path.basename(s) : load_qbo(s, n_years=12) for s in case_dirs}
    vmax = max([abs(u).max() for _, u in data.items()])

    for (name, u), ax in zip(data.items(), axes):
        years = u.time / 360
        ys = np.arange(len(u.pfull))
        ps = [format_pressure(p) for p in u.pfull.values]

        img = ax.contourf(
            years, -ys, u.T, 
            vmin=(-vmax), 
            vmax=vmax,
            cmap='RdBu_r',
            levels=15
        )

        ax.set_yticks(-ys[::3])
        ax.set_yticklabels(ps[::3])

        ax.set_xlabel('year')
        ax.set_ylabel('p (hPa)')

        period, _, amp, _ = qbo_statistics(u)
        info = f'{period:.2f} month period, {amp:.2f} m s$^{{-1}}$ amplitude'
        ax.set_title(f'{name} \u2014 {info}')

    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label('zonal mean tropical $u$ (m s$^{-1}$)')

    plt.savefig(output_path, dpi=400)
        
