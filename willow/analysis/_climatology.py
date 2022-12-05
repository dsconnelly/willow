import os

from collections import defaultdict

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.util import add_cyclic_point

from ..utils.mima import get_fnames, get_mima_name
from ..utils.plotting import (
    format_latitude,
    format_name,
    format_pressure,
    get_bounds_and_cmap,
    get_units
)

def plot_climatologies(fields, case_dirs, kind, output_dir):
    """
    Plot climatological means of outputs from MiMA runs.

    Parameters
    ----------
    fields : list of str
        The variables to plot, out of {'u', 'v', 'T', 'N', 'gwd_u', 'gwd_v'}.
    case_dirs : list of str
        The directories containing MiMA runs to plot.
    kind : str
        The kind of mean to take, out of {'tropical', 'vertical', 'zonal'}.
    output_dir : str
        The directory where images should be saved.

    """

    sels = {'lat' : slice(-5, 5)} if kind == 'tropical' else {}
    means = {
        'tropical' : ('lat', 'lon'),
        'vertical' : ('time', 'pfull'),
        'zonal' : ('time', 'lon')
    }[kind]

    datas = defaultdict(dict)
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        fnames = get_fnames(case_dir)[-15:]

        with xr.open_mfdataset(fnames, decode_times=False) as ds:
            ds = ds.sel(**sels).mean(means)
            for field in fields:
                datas[field][name] = ds[get_mima_name(field)].load()

    ax_kwargs = {}
    if kind == 'vertical':
        ax_kwargs['projection'] = ccrs.PlateCarree()

    plot_func = globals()[f'_plot_{kind}_mean']

    n_cases = len(case_dirs)
    for field, data in datas.items():
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(8 * n_cases, 6.6)

        gs = gridspec.GridSpec(
            nrows=2, ncols=n_cases,
            height_ratios=[1, 0.1],
            figure=fig
        )

        axes = [fig.add_subplot(gs[0, j], **ax_kwargs) for j in range(n_cases)]
        cax = fig.add_subplot(gs[1, :])

        vmin, vmax, cmap = get_bounds_and_cmap(field, data)
        for ax, (name, a) in zip(axes, data.items()):
            img = plot_func(ax, a, vmin, vmax, cmap)
            ax.set_title(format_name(name))

        units = get_units(field)
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        cbar.set_label(f'{kind} mean ${field}$ ({units})')

        fname = f'{field}-{kind}-mean.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=400)
        plt.clf()

def _plot_tropical_mean(ax, a, vmin, vmax, cmap):
    years = a['time'] / 360
    ys = np.arange(len(a['pfull']))
    ps = [format_pressure(p) for p in a['pfull'].values]

    img = ax.contourf(
        years, -ys, a.T,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        levels=15
    )

    xmin = np.ceil(years.min())
    xmax = np.ceil(years.max())
    ax.set_xticks(np.arange(xmin, xmax))

    ax.set_yticks(-ys[::3])
    ax.set_yticklabels(ps[::3])

    ax.set_xlabel('time (years)')
    ax.set_ylabel('pressure (hPa)')

    return img

def _plot_vertical_mean(ax, a, vmin, vmax, cmap):
    lats = np.linspace(-90, 90, len(a['lat']))
    a, lons = add_cyclic_point(a.values, coord=a['lon'].values)

    img = ax.contourf(
        lons, lats, a,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        levels=15
    )

    ax.coastlines()

    return img

def _plot_zonal_mean(ax, a, vmin, vmax, cmap):
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

    lats = np.linspace(-90, 90, 7)
    ax.set_xticks(lats)
    ax.set_xticklabels([format_latitude(lat) for lat in lats])

    ax.set_yticks(-ys[::3])
    ax.set_yticklabels(ps[::3])

    ax.set_xlabel('latitude')
    ax.set_ylabel('pressure (hPa)')

    return img
