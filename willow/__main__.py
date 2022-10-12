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

    plt.rcParams['font.family'] = 'TeX Gyre Adventor'
    plt.rcParams['font.size'] = 12

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'TeX Gyre Adventor'

    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['savefig.transparent'] = False

    FunctionParser([
        make_datasets, 
        train_emulator,
        setup_mima,
        plot_example_profiles,
        plot_offline_scores,
        save_shapley_scores,
        plot_shapley_scores,
        plot_shapley_profiles,
        plot_lmis,
        plot_online_profiling,
        plot_climatologies,
        plot_distributions,
        plot_qbos,
        plot_ssws
    ]).run()