import os

from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.mima import get_fnames, get_mima_name
from ..utils.plotting import (
    format_latitude,
    format_pressure,
    get_bounds_and_cmap,
    get_units
)

def plot_climatologies(fields, case_dirs, output_dir):
    """
    Plot zonal mean climatologies for a specified variable.

    Parameters
    ----------
    fields : list of str
        The variables to plot, out of {'u', 'v', 'T', 'N', 'gwd_u', 'gwd_v'}. 
    case_dirs : list of str
        The directories containing MiMA runs to plot.
    output_dir : str
        The directory where output images should be saved.

    """

    data = defaultdict(dict)
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        fnames = get_fnames(case_dir, n_years=2)

        with xr.open_mfdataset(fnames, decode_times=False) as ds:
            ds = ds.mean(('time', 'lon'))
            for field in fields:
                data[field][name] = ds[get_mima_name(field)].load()

    n_cases = len(case_dirs)
    for field in fields:
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(8 * (n_cases + 0.1), 6)

        width_ratios = [1] * n_cases + [0.1]
        gs = gridspec.GridSpec(
            nrows=1, ncols=(n_cases + 1),
            width_ratios=width_ratios,
            figure=fig
        )

        *axes, cax = [fig.add_subplot(gs[0, j]) for j in range(n_cases + 1)]
        vmin, vmax, cmap = get_bounds_and_cmap(field, data[field])
        
        xticks = np.linspace(-90, 90, 7)
        xlabels = [format_latitude(lat) for lat in xticks]

        for (name, a), ax in zip(data[field].items(), axes):
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

        units = get_units(field)
        cbar = plt.colorbar(img, cax=cax, orientation='vertical')
        cbar.set_label(f'zonal mean ${field}$ ({units})')

        plt.savefig(os.path.join(output_dir, f'{field}.png'), dpi=400)
        plt.clf()