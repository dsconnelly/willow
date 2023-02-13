import os

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import format_name, get_letter, get_rows_and_columns
from ..utils.ssw import load_ssw

def plot_ssws(case_dirs: list[str], output_path: str) -> None:
    """
    Plot sudden stratospheric warmings from one or more MiMA runs.

    Parameters
    ----------
    case_dirs : Directories of the MiMA runs with SSWs to be plotted.
    output_path : Path where the plot should be saved.

    """

    n_subplots = len(case_dirs)
    n_rows, n_cols = get_rows_and_columns(n_subplots)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    fig.set_size_inches(9 * n_cols, 3 * n_rows)
    axes = axes.flatten()

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
        name = format_name(os.path.basename(case_dir))
        rate = 10 * len(idx) / (time.max() - time.min())
        ax.set_title(f'({letter}) {name} \u2014 {rate:.2f} SSW per decade')

    plt.tight_layout()
    plt.savefig(output_path)

