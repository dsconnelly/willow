import matplotlib.pyplot as plt

from func2cli import FunctionParser

from .offline import plot_R2_scores
from .online import initialize_coupled_run, plot_qbos
from .preprocessing import save_datasets
from .training import train_emulator

if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['savefig.bbox'] = 'tight'

    FunctionParser([
        save_datasets,
        train_emulator,
        plot_R2_scores,
        initialize_coupled_run,
        plot_qbos
    ]).run()