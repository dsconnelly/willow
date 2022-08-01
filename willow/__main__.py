import argparse

from func2cli import FunctionParser

from .analysis import plot_offline_scores
from .data import preprocess
from .online import setup_mima
from .training import train_forest, train_network

if __name__ == '__main__':
    FunctionParser([
        preprocess, 
        train_forest, 
        train_network,
        setup_mima,
        plot_offline_scores
    ]).run()