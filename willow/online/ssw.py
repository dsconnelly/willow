import os

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import (
    COLORS,
    format_name,
    get_letter
)
from ..utils.ssw import load_ssw, get_ssw_frequency

def plot_ssw_frequencies(case_dirs: list[str], output_path: str) -> None:
    """
    Plot SSW frequencies with intelligent grouping.

    Parameters
    ----------
    case_dirs : Directories of the MiMA runs with SSWs to be plotted.
    output_path : Path where the plot should be saved.

    """

    lines = defaultdict(list)
    errors = defaultdict(list)

    for case_dir in case_dirs:
        u = load_ssw(case_dir)
        freq, error = get_ssw_frequency(u)

        case_name = case_dir.split('/')[-1]
        lines[case_name].append(freq)
        errors[case_name].append(error)

    fig, ax = plt.subplots()
    fig.set_size_inches(1.2 * 6, 1.2 * 4.5)

    keys = [scheme for scheme in lines.keys() if 'ad99' not in scheme]
    cmap = dict(zip(keys, COLORS))

    ppms = [390, 800, 1200]
    ticks = np.arange(len(ppms))

    for i, (scheme, freqs) in enumerate(lines.items()):
        color = cmap.get(scheme, 'gray')
        label = format_name(scheme, simple=True)

        ax.bar(
            ticks + 0.2 * (i - 1) - 0.1,
            freqs, yerr=errors[scheme],
            capsize=8,
            width=0.2,
            color=color,
            edgecolor='k',
            ecolor='k',
            label=label
        )

    ax.set_xticks(ticks)
    ax.set_xticklabels(ppms)

    ax.set_xlim(-0.5, len(ppms) - 0.5)
    ax.set_ylim(0, 14)

    ax.set_xlabel('CO$_2$ concentration (ppm)')
    ax.set_ylabel('SSW frequency (decade$^{-1}$)')
    ax.legend()

    ax.grid(False)
    ax.grid(True, axis='y', zorder=-10)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path)

def plot_ssws(case_dirs: list[str], output_path: str) -> None:
    """
    Plot sudden stratospheric warmings from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : Directories of the MiMA runs with SSWs to be plotted.
    output_path : Path where the plot should be saved.

    """

    n_subplots = len(case_dirs)
    # n_rows, n_cols = get_rows_and_columns(n_subplots)
    n_rows, n_cols = 4, n_subplots // 4

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    fig.set_size_inches(7 * n_cols, 7 / 3 * n_rows)
    axes = axes.T.flatten()

    for i, (case_dir, ax) in enumerate(zip(case_dirs, axes)):
        u, idx = load_ssw(case_dir)
        year, day = u['time'].dt.year, u['time'].dt.dayofyear
        time = (((year - year[0]) * 360 + day - 1) / 360).values

        ax.plot(time, u.values.flatten(), color='k')
        ax.scatter(time[idx], u[idx], marker='x', s=100, color='tab:red')

        months = u['time'].dt.month
        mask = (3 < months) & (months < 11)
        ymin, ymax = -40, 80

        ax.fill_between(
            time, ymin, ymax, 
            where=~mask,
            color='gainsboro',
            zorder=-1
        )

        xmin, xmax = time.min(), time.max()
        n_ticks = round(xmax - xmin) // 4 + 1
        ticks = np.linspace(xmin, xmax, n_ticks)

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks.astype(int))

        ax.set_xlim(time.min(), time.max())
        ax.set_ylim(ymin, ymax)
        ax.grid(zorder=0)

        ax.set_xlabel('year')
        ax.set_ylabel('$u$ (m s$^{-1}$)')

        letter = get_letter(i)
        name = format_name(case_dir)
        rate = 10 * len(idx) / (time.max() - time.min())
        ax.set_title(f'({letter}) {name} \u2014 {rate:.2f} SSW per decade')

    plt.tight_layout()
    plt.savefig(output_path)

