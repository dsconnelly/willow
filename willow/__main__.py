import argparse

from func2cli import FunctionParser

from .analysis import plot_offline
from .data import preprocess
from .training import train_forest, train_network

if __name__ == '__main__':
    FunctionParser([
        preprocess, 
        train_forest, 
        train_network,
        plot_offline
    ]).run()