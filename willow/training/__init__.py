import logging
import os

import joblib
import numpy as np

from .forests import train_forest
from .networks import train_network

from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.diagnostics import log, profile
from ..utils.statistics import standardize
from ..utils.wrappers import MiMAModel

__all__ = ['train_emulator']

@log
@profile
def train_emulator(data_dir: str, model_dir: str) -> None:
    """
    Train a forest or neural network emulator.

    Parameters
    ----------
    data_dir : Directory where training and test datasets are saved.
    model_dir : Directory where the trained model will be saved. The prefix of
        separated by a hyphen, is used to determine the kind of model. If the
        prefix is one of `'mubofo'` or `'random'`, then the appropriate kind of
        forest will be trained; otherwise, a neural network will be trained and
        the prefix should be the name of a class defined in `networks.py`.

    """

    model_name = os.path.basename(model_dir)
    kind, *_ = model_name.split('-')

    data = load_datasets(data_dir, 'tr')
    X, Y, col_idx = prepare_datasets(*data, model_name)
    Y_scaled, means, stds = standardize(Y)

    logging.info(f'Loaded {X.shape[0]} training samples.')
    train_func = train_forest if kind in ['mubofo', 'random'] else train_network

    np.save('X-p.npy', X)
    np.save('Y-p.npy', Y_scaled)

    with np.errstate(invalid='raise'):
        model = train_func(X, Y_scaled, model_name)

    wrapper = MiMAModel(model_name, model, means, stds, col_idx)
    joblib.dump(wrapper, os.path.join(model_dir, 'model.pkl'))
    