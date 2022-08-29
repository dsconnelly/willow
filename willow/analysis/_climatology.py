import os

import cmocean
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.plotting import format_latitude, format_pressure

def plot_climatology(field, case_dirs, output_path, n_years=12):
    """
    Plot zonal mean climatologies for a specified variable.

    Parameters
    ----------
    field : str
        The name of the variable to plot. Should be one of {'u', 'v', 'T'}.
    case_dirs : list of str
        The directories containing MiMA runs to plot.
    output_path : str
        The path where the plot should be saved.
    n_years : int
        The number of years to include in the climatology, counted backwards
        from the end of the run.

    """

    data = {}
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)

        years = sorted([s for s in os.listdir(case_dir) if s.isdigit()])
        fnames = [os.path.join(case_dir, y, 'atmos_4xdaily.nc') for y in years]

        with xr.open_mfdataset(fnames[-n_years:], decode_times=False) as ds:
            data[name] = ds[field.lower() + '_gwf'].mean(('time', 'lon')).load()

    n_cases = len(case_dirs)
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(8 * (n_cases + 0.1), 6)

    width_ratios = [1] * n_cases + [0.1]
    gs = gridspec.GridSpec(
        nrows=1, ncols=(n_cases + 1),
        width_ratios=width_ratios,
        figure=fig
    )

    axes = [fig.add_subplot(gs[0, j]) for j in range(n_cases)]
    cax = fig.add_subplot(gs[0, -1])

    if field in ('u', 'v'):
        vmax = max([abs(a).max() for _, a in data.items()])
        vmin = -vmax
        cmap = 'RdBu_r'
    
    else:
        vmin = min([a.min() for _, a in data.items()])
        vmax = max([a.max() for _, a in data.items()])
        cmap = cmocean.cm.thermal

    xticks = np.linspace(-90, 90, 7)
    xlabels = [format_latitude(lat) for lat in xticks]

    for (name, a), ax in zip(data.items(), axes):
        lats = np.linspace(-90, 90, len(a['lat']))
        ys = np.arange(len(a['pfull']))
        ps = [format_pressure(p) for p in a['pfull'].values]

        img = ax.contourf(
            lats, -ys, a,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            levels=15
        )

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        ax.set_yticks(-ys[::3])
        ax.set_yticklabels(ps[::3])

        ax.set_xlabel('latitude')
        ax.set_ylabel('pressure (hPa)')
        ax.set_title(name)

    units = {
        'u' : 'm s$^{-1}$',
        'v' : 'm s$^{-1}$',
        'T' : 'K'
    }[field]

    cbar = plt.colorbar(img, cax=cax, orientation='vertical')
    cbar.set_label(f'zonal mean ${field}$ ({units})')

    plt.savefig(output_path, dpi=400)
