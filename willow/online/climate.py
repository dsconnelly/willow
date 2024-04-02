import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.integrate import cumulative_trapezoid as integrate
from scipy.stats import ttest_ind as ttest
from sklearn.decomposition import PCA

from ..utils.mima import get_paths, open_mima_output
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

def plot_tropical_analysis(
    case_dirs: list[str],
    output_path: str,
    recalculate: bool=True,
    ref_dirs: list[str]=None
) -> None:
    """
    Make various latitude/height cross sections for the paper revisions.

    Parameters
    ----------
    case_dirs : Directories where MiMA was run.
    output_path : Where to save the image.
    recalculate : Whether to recalculate the data fields.
    ref_dirs : Directories to show difference in w_star

    """

    if recalculate:
        for case_dir in case_dirs:    
            paths = get_paths(case_dir)
            with open_mima_output(paths, 16) as ds:
            # with open_mima_output(f'{case_dir}/zonal_mean.nc') as ds:
                ds = ds.sel(lat=slice(-20, 20))
                ds['lat'] = np.deg2rad(ds['lat'])
                ds['pfull'] = 100 * ds['pfull']

                w_star = 1000 * _get_w_star(ds)
                gwfu = 86400 * ds['gwfu_cgwd'].mean('lon')
                u = ds['u_gwf'].mean('lon')

                xr.Dataset({
                    'w_star' : w_star,
                    'gwfu' : gwfu,
                    'u' : u
                }).to_netcdf(f'{case_dir}/long-means/revisions.nc')

    colors = ['k'] + [c for c, case_dir in zip(COLORS, case_dirs[1:])]
    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(12, 6)

    tcolor = 'goldenrod'
    taxes = [ax.twiny() for ax in axes]
    
    for i, (case_dir, color) in enumerate(zip(case_dirs, colors)):
        with xr.open_dataset(f'{case_dir}/revisions.nc') as ds:
            ds = ds.sel(pfull=slice(None, 30000), lat=slice(-5, 5))
            u_qbo = ds['u'].sel(pfull=1000, method='nearest').mean('lat').values

            pressures = ds['pfull'].values / 100
            labels = [format_pressure(p) for p in pressures]
            y = -np.arange(len(pressures))

            idx_west = u_qbo > 10
            idx_east = u_qbo < -5

            for idx, ax, tax in zip([idx_west, idx_east], axes[1:], taxes[1:]):
                if i == 0:
                    u = ds['u'].isel(time=idx).mean(('time', 'lat'))
                    tax.plot(u, y, color=tcolor, ls='dashed')

                    tax.set_zorder(5)
                    ax.set_zorder(10)
                    ax.patch.set_visible(False)

                gwf = ds['gwfu'].isel(time=idx).mean(('time', 'lat'))
                ax.plot(gwf, y, color=color)

            if ref_dirs is not None:
                ref_dir = ref_dirs[i]
                with xr.open_dataset(f'{ref_dir}/revisions.nc') as rds:
                    rds = rds.sel(pfull=slice(None, 30000), lat=slice(-5, 5))
                    ref_w = rds['w_star'].mean(('time', 'lat'))

            label = format_name(case_dir, True)
            label = label[:label.index(' (')]

            # axes[0].plot([], [], color=color, label=label)
            w_star = ds['w_star'].mean(('time', 'lat'))
            if ref_dirs is not None:
                w_star = w_star - ref_w

            axes[0].plot(w_star, y, color=color, label=label)

    for ax in axes[1:]:
        ax.set_xlim(-3, 3)
        
    axes[1].set_xlabel('QBOW drag (m s$^{-1}$ day$^{-1}$)')
    axes[2].set_xlabel('QBOE drag (m s$^{-1}$ day$^{-1}$)')

    for i, tax in enumerate(taxes):
        tax.set_xlim(-30, 30)
        tax.set_xlabel('$\\bar{u}$ (m s$^{-1}$)')
        tax.grid(False)

        tax.xaxis.label.set_color('w' if i == 0 else tcolor)
        tax.spines['top'].set_edgecolor('k' if i == 0 else tcolor)
        tax.tick_params(axis='x', colors='w' if i == 0 else tcolor)

    # axes[0].set_xlim(0, 3)
    if ref_dirs is None:
        label = '$\\bar{w}^\\ast$ (mm s$^{-1}$)'
    else:
        label = '$\\Delta\\bar{w}^\\ast$ (mm s$^{-1}$)'

    axes[0].set_xlabel(label)

    for i, ax in enumerate(axes):
        ax.set_yticks(y[::3])
        ax.set_ylim(y[-1], y[0])
        ax.set_yticklabels(labels[::3])
        ax.set_ylabel('pressure (hPa)')

        # pad = 30 if i == 0 else 2
        ax.set_title(f'({get_letter(i)})')

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_path)

def _get_w_star(ds: xr.Dataset) -> xr.DataArray:
    a_earth = 6.37e6
    p_ref = 100000

    v_bar, v_prime = _decompose(ds['v_gwf'])
    theta = ds['t_gwf'] * ((p_ref / ds['pfull']) ** (2 / 7))
    theta_bar, theta_prime = _decompose(theta)

    strat = theta_bar.differentiate('pfull')
    flux = (v_prime * theta_prime).mean('lon')
    v_star = v_bar - (flux / strat).differentiate('pfull')

    cos_lat = np.cos(ds['lat'])
    R = a_earth * cos_lat

    rhs = -(v_star * cos_lat).differentiate('lat') / R
    omega_star = integrate(rhs, rhs['pfull'], axis=1, initial=0)
    omega_star = omega_star * xr.ones_like(v_star)

    rho = (ds['pfull'] / 287.05 / ds['t_gwf']).mean('lon')
    return -omega_star / rho / 9.8

def _decompose(data: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    data_bar = data.mean('lon')
    data_prime = data - data_bar

    return data_bar, data_prime

def plot_pca_shift(
    case_dirs: list[str],
    output_path: str,
    field: str,
    n_samples: int=int(1e6)
) -> None:
    """
    Plot shift in zonal wind or temperature with PCA.

    Parameters
    ----------
    case_dirs : Directories where MiMA was run.
    output_path : Where to save the image.
    field : Either u or t, depending on what should be plotted.
    n_samples : How many samples to plot.

    """

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    for i, (case_dir, color) in enumerate(zip(case_dirs, COLORS)):
        with open_mima_output(get_paths(case_dir), 15) as ds:
            ds = ds.sel(lat=slice(-5, 5))
            
            vname = f'{field.lower()}_gwf'
            u = ds[vname].values.transpose(0, 2, 3, 1).reshape(-1, 40)
            idx = np.random.permutation(u.shape[0])
            u = u[idx[:n_samples]]

        if i == 0: pca = PCA(n_components=2).fit(u)
        x, y = pca.transform(u).T

        label = format_name(case_dir, True)
        label = label[label.index('(') + 1:-1]
        ax.scatter(x, y, s=5, color=color, alpha=0.15, linewidth=0, zorder=10)
        ax.scatter([], [], color=color, label=label)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    amax = {'u' : 150, 'T' : 50}[field]
    ax.set_xlim(-amax, amax)
    ax.set_ylim(-amax, amax)

    ax.set_title(f'tropical ${field}$ samples')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
            
        
