import os

import matplotlib.pyplot as plt
import numpy as np

def plot_online_profiling(case_dirs, output_path):
    """
    Plot online runtime of gravity wave drag parameterizations.

    Parameters
    ----------
    case_dirs : list of str
        The directories containing MiMA runs to analyze.
    output_path : str
        The path where the plot should be saved.

    """
    
    n_cases = len(case_dirs)
    percents = np.zeros((2, n_cases))
    dragtimes = np.zeros((2, n_cases))
    yeartimes = np.zeros((2, n_cases))
    
    colors = []
    for j, case_dir in enumerate(case_dirs):
        log_path = os.path.join(case_dir, 'slurm.out')
        case_percents, case_dragtimes, case_yeartimes = _read_log(log_path)

        percents[0, j] = case_percents.mean()
        percents[1, j] = case_percents.std()

        dragtimes[0, j] = case_dragtimes.mean()
        dragtimes[1, j] = case_dragtimes.std()

        yeartimes[0, j] = case_yeartimes.mean()
        yeartimes[1, j] = case_yeartimes.std()

        kind = os.path.basename(case_dir).split('-')[0]
        colors.append(_colormap.get(kind, 'whitesmoke'))

    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(12, 6)

    _plot(axes[0], percents, colors, 25, '% of runtime in GW drag routine')
    _plot(axes[1], dragtimes, colors, 8, 'minutes in GW drag routine', False)
    _plot(axes[2], yeartimes, colors, 35, 'total minutes per year', False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

_colormap = {
    'mubofo' : 'royalblue',
    'xgboost' : 'seagreen',
    'random' : 'tab:red',
    'WaveNet' : 'mediumorchid'
}

def _plot(ax, data, colors, xmax, xlabel, legend=True):
    n_cases = data.shape[1]
    y = -np.arange(n_cases) - 0.5

    ax.barh(
        y, data[0],
        xerr=data[1],
        height=1,
        color=colors,
        edgecolor='k'
    )

    ax.set_xlim(0, xmax)
    ax.set_ylim(-n_cases, 0)
    ax.set_yticks([])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('parameterization')

    if legend:
        ax.barh([0], [0], color='whitesmoke', edgecolor='k', label='AD99')
        for kind, color in _colormap.items():
            ax.barh([0], [0], color=color, edgecolor='k', label=kind)

        ax.legend()

def _read_log(path):
    keep = lambda c: c in ' .0123456789'
    def parse(line):
        return [float(s) for s in ''.join(filter(keep, line)).split()]

    percents, dragtimes, yeartimes = [], [], []
    with open(path) as f:
        for line in f:
            if 'Total runtime' in line:
                yeartimes.append(parse(line)[2] / 60)
            elif 'Damping: GW drag' in line:
                numbers = parse(line)
                percents.append(100 * numbers[4])
                dragtimes.append(numbers[2] / 60)

    return np.array(percents), np.array(dragtimes), np.array(yeartimes)