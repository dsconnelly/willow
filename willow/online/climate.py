import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind as ttest

from ..utils.mima import open_mima_output
from ..utils.plotting import (
    COLORS,
    format_latitude, 
    format_name,
    format_pressure,
    get_letter
)

def plot_biases(
    case_dirs: list[str],
    output_path: str
) -> None:
    """
    Plot zonal mean sections from several online runs.

    Parameters
    ----------
    case_dirs : Directories where MiMA was run.
    output_path : Path where image should be saved

    """

    ref_dir, *case_dirs = case_dirs
    n_subplots = len(case_dirs)
    height_ratios = [1] * n_subplots + [0.05]

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5 * 2, 4 * sum(height_ratios))

    gs = GridSpec(
        nrows=(n_subplots + 1), ncols=2,
        height_ratios=height_ratios,
        figure=fig
    )

    axes = np.empty((n_subplots, 2), dtype=Axes)
    for j in range(2):
        for i in range(n_subplots):
            axes[i, j] = fig.add_subplot(gs[i, j])
    
    umax, tmax = (3, 1)
    tasks = [
        ('u_gwf', '$u$ (m s$^{-1}$)', Normalize(-umax, umax), 'PuOr_r'),
        ('t_gwf', '$T$ (K)', Normalize(-tmax, tmax), 'RdBu_r')
    ]

    lats = np.linspace(-90, 90, 64)
    ticks = np.linspace(-90, 90, 7)

    references = {}
    with open_mima_output(os.path.join(ref_dir, 'zonal_mean.nc'), 56) as ds:
        for field, *_ in tasks:
            references[field] = ds[field].isel(lon=0).values

        pressures = ds['pfull'].values

    y = -np.arange(len(pressures))
    labels = [format_pressure(p) for p in pressures]

    for i, case_dir in enumerate(case_dirs):
        path = os.path.join(case_dir, 'zonal_mean.nc')
        with open_mima_output(path, 56) as ds:
            for j, (info, ax) in enumerate(zip(tasks, axes[i])):
                field, _, norm, cmap = info

                data = ds[field].isel(lon=0).values                
                _, p_values = ttest(references[field], data, equal_var=False)
                significant = (p_values < 0.05).astype(int)

                bias = (data - references[field]).mean(axis=0)
                ax.contourf(lats, y, bias, levels=15, norm=norm, cmap=cmap)
                ax.contourf(
                    lats, y, significant, 
                    levels=[-0.5, 0.5, 1.5],
                    hatches=[None, '...'],
                    colors='none'
                )

                ax.set_xticks(ticks)
                ax.set_xticklabels([format_latitude(lat) for lat in ticks])
                ax.set_xlabel('latitude')

                ax.set_yticks(y[::3])
                ax.set_yticklabels(labels[::3])
                ax.set_ylabel('pressure (hPa)')

                letter = get_letter(i + 3 * j)
                ax.set_title(f'({letter}) {format_name(case_dir, True)}')
                ax.grid(False)

    caxes = [fig.add_subplot(gs[-1, j]) for j in range(2)]
    for (_, label, norm, cmap), cax in zip(tasks, caxes):
        img = ScalarMappable(norm, cmap)
        cbar = plt.colorbar(img, cax, orientation='horizontal')
        cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 5))
        cbar.set_label(f'bias in zonal mean {label}', size=12)
    
    plt.savefig(output_path)

def plot_distribution_shift(
    case_dirs: list[str],
    output_path: str,
    levels: list[float]=[10, 100, 200]
) -> None:
    """
    Plot profiles with error bars from different MiMA runs.

    Parameters
    ----------
    case_dirs : Directories where MiMA was run.
    output_path : Path where image should be saved.
    levels : List of levels to plot at.

    """

    n_levels = len(levels)
    fig, axes = plt.subplots(nrows=n_levels, ncols=2)
    fig.set_size_inches(10, (5 / 2) * n_levels)

    for case_dir, color in zip(case_dirs, COLORS):
        path = os.path.join(case_dir, 'covariance.nc')
        ppm = case_dir.split('/')[-2]
        name = f'{ppm} ppm CO$_2$'

        with open_mima_output(path) as ds:
            u = ds['u_gwf']

        for j, lat in enumerate(u.lat.values):
            for level, ax in zip(levels, axes[:, j]):
                a = u.isel(lat=j).sel(pfull=level, method='nearest')
                p = format_pressure(a.pfull.item())

                pdf, edges = np.histogram(
                    a.values.flatten(),
                    bins=30,
                    density=True
                )
                
                x = (edges[:-1] + edges[1:]) / 2
                ax.plot(x, pdf, color=color, label=name)
                ax.set_title(f'{format_latitude(lat)}, {p} hPa')
        
    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(-80, 80)
        ax.set_ylim(0, 0.05)

        if i > 2 * n_levels - 3:
            ax.set_xlabel('$u$ (m s$^{-1}$)')

        if not i % 2:
            ax.set_ylabel('density')

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)