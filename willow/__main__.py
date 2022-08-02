import argparse

from func2cli import FunctionParser

from .analysis import plot_offline_scores
from .coupling import setup_mima
from .preprocessing import make_datasets
from .training import train_emulator

if __name__ == '__main__':
    FunctionParser([
        make_datasets, 
        train_emulator,
        setup_mima,
        plot_offline_scores
    ]).run()