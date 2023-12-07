import os

from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from ..utils.plotting import (
    COLORS,
    format_name,
    format_pressure,
    get_letter,
    get_rows_and_columns
)
from ..utils.qbo import get_qbo_statistics, load_qbo

_CBAR_SCALE = 0.15
_CMAP = 'RdBu_r'
_VMAX = 75

def plot_qbos(case_dirs: list[str], output_path: str,) -> None:
    """
    Plot quasi-biennial oscillations from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : Directories of the MiMA runs with QBOs to be plotted.
    output_path : Path where the plot should be saved.

    """

    n_subplots = len(case_dirs)
    # n_rows, n_cols = get_rows_and_columns(n_subplots)
    n_rows, n_cols = n_subplots // 3, 3

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5 * n_cols, 2.5 * (n_rows + _CBAR_SCALE))
    height_ratios = [1] * n_rows + [_CBAR_SCALE]
    gs = GridSpec(n_rows + 1, n_cols, fig, height_ratios=height_ratios)
    
    axes: list[Axes] = []
    for j in range(n_cols):
        for i in range(n_rows):    
            if len(axes) < n_subplots:
                axes.append(fig.add_subplot(gs[i, j]))

    levels = np.linspace(-_VMAX, _VMAX, 13)
    for i, (case_dir, ax) in enumerate(zip(case_dirs, axes)):
        u = load_qbo(case_dir)
        time = u['time'].values
        
        years = np.array([x.days for x in time - time[0]]) / 360
        start = (years >= years[-1] - 36).argmax()
        u = u.isel(time=slice(start, None))
        years = years[start:]

        pressures = [format_pressure(p) for p in u.pfull.values]
        ys = -np.arange(len(pressures))

        img = ax.contourf(
            years, ys, u.T,
            vmin=-_VMAX,
            vmax=_VMAX,
            cmap=_CMAP,
            levels=levels,
            extend='both'
        )

        n_ticks = round((years.max() - years.min()) / 4) + 1
        ticks = np.linspace(years.min(), years.max(), n_ticks)
        labels = np.round(ticks).astype(int)

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('year')

        ax.set_yticks(ys[::6])
        ax.set_yticklabels(pressures[::6])
        ax.set_ylabel('pressure (hPa)')

        ax.set_title(f'({get_letter(i)}) {format_name(case_dir)}')
        ax.grid(False)

    cax = fig.add_subplot(gs[-1, :])
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label('zonal mean tropical $u$ (m s$^{-1}$)')
    cbar.set_ticks(levels[::2])

    plt.savefig(output_path)

def plot_qbo_statistics(case_dirs: list[str], output_path: str) -> None:
    """
    Plot quasi-biennial oscillation statistics from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : Directories of the MiMA runs with QBO statistics to be plotted.
    output_path : Path where the plot should be saved.

    """

    periods = defaultdict(list)
    period_errs = defaultdict(list)

    amps = defaultdict(list)
    amp_errs = defaultdict(list)

    for case_dir in case_dirs:
        u = load_qbo(case_dir)
        period, period_err, amp, amp_err = get_qbo_statistics(u)
        case_name = case_dir.split('/')[-1]

        periods[case_name].append(period)
        period_errs[case_name].append(period_err)

        amps[case_name].append(amp)
        amp_errs[case_name].append(amp_err)

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(2 * 6, 4.5)

    keys = [scheme for scheme in periods.keys() if 'ad99' not in scheme]
    cmap = dict(zip(keys, COLORS))

    ppms = [390, 800, 1200]
    ticks = np.arange(len(ppms))

    n_models = len(periods)
    width = 1 / (n_models + 1)

    for i, name in enumerate(periods.keys()):
        color = cmap.get(name, 'gray')
        label = format_name(name, simple=True)
        xs = (i - 1) * width - (width / 2 if n_models % 2 == 0 else 0)

        axes[0].bar(
            ticks + xs,
            periods[name],
            yerr=period_errs[name],
            capsize=8,
            width=width,
            color=color,
            edgecolor='k',
            ecolor='k',
            label=label
        )

        axes[1].bar(
            ticks + xs,
            amps[name],
            yerr=amp_errs[name],
            capsize=8,
            width=width,
            color=color,
            edgecolor='k',
            ecolor='k',
        )

    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ppms)

        ax.set_xlim(-0.5, len(ppms) - 0.5)
        ax.set_xlabel('CO$_2$ concentration (ppm)')

        ax.grid(False)
        ax.grid(True, axis='y')
        ax.set_axisbelow(True)

    axes[0].set_ylabel('period (months)')
    axes[1].set_ylabel('amplitude (m s$^{-1}$)')

    axes[0].set_title('(a) QBO period')
    axes[1].set_title('(b) QBO amplitude')
    axes[0].legend()

    # axes[0].set_ylim(20, 60)
    axes[0].set_ylim(20, 45)
    axes[1].set_ylim(10, 25)

    plt.tight_layout()
    plt.savefig(output_path)