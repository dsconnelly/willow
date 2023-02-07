import matplotlib.pyplot as plt

from func2cli import FunctionParser

from .offline import (
    plot_example_profiles,
    plot_example_sources,
    plot_feature_importances,
    plot_importance_correlations,
    plot_R2_scores
)
from .online import initialize_coupled_run, plot_qbos, plot_ssws
from .preprocessing import save_datasets
from .training import train_emulator

if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['savefig.bbox'] = 'tight'

    FunctionParser([
        save_datasets,
        train_emulator,
        plot_example_profiles,
        plot_example_sources,
        plot_R2_scores,
        plot_feature_importances,
        plot_importance_correlations,
        initialize_coupled_run,
        plot_qbos,
        plot_ssws
    ]).run()