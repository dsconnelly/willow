import argparse

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from func2cli import FunctionParser

from .analysis import *
from .coupling import setup_mima
from .preprocessing import make_datasets
from .training import train_emulator

if __name__ == '__main__':
    for font in fm.findSystemFonts('data/fonts'):
        fm.fontManager.addfont(font)

    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['savefig.transparent'] = False

    FunctionParser([
        make_datasets, 
        train_emulator,
        setup_mima,
        plot_example_profiles,
        plot_offline_scores,
        plot_tropical_drag,
        save_shapley_scores,
        plot_shapley_scores,
        plot_shapley_profiles,
        plot_lmis,
        plot_oracle,
        plot_online_profiling,
        plot_climatologies,
        plot_distributions,
        plot_qbos,
        plot_ssws
    ]).run()