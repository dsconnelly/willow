import matplotlib.pyplot as plt

from func2cli import FunctionParser

from .offline import (
    plot_example_profiles,
    plot_example_sources,
    plot_feature_importances,
    plot_emulator_drift,
    plot_R2_scores,
    plot_shapley_errors,
    plot_scalar_importances,
    save_shapley_values
)
from .online import (
    initialize_coupled_run,
    plot_biases,
    plot_tropical_analysis,
    plot_pca_shift,
    plot_qbos,
    plot_qbo_statistics,
    plot_ssw_frequencies
)
from .preprocessing import save_datasets
from .training import train_emulator

if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['savefig.bbox'] = 'tight'

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'lightgray'

    plt.rcParams['font.size'] = 12

    FunctionParser([
        save_datasets,
        train_emulator,
        plot_example_profiles,
        plot_example_sources,
        plot_emulator_drift,
        plot_R2_scores,
        save_shapley_values,
        plot_feature_importances,
        plot_scalar_importances,
        plot_shapley_errors,
        initialize_coupled_run,
        plot_biases,
        plot_tropical_analysis,
        plot_pca_shift,
        plot_qbos,
        plot_qbo_statistics,
        plot_ssw_frequencies,
    ]).run()
