import os

from ._forests import train_scikit_forest, train_xgboost_forest
from ._networks import train_network
from ..utils.diagnostics import logs, times

__all__ = ['train_emulator']

@logs
@times
def train_emulator(data_dir, model_dir):
    """
    Train a forest or neural network emulator.

    Parameters
    ----------
    data_dir : str
        Directory where training and test datasets are saved.
    model_dir : str
        Directory where trained model will be saved. The hyphen-separated prefix
        of the directory name will be used to determine the kind of model. If
        the prefix is one of 'mubofo', 'random', or 'xgboost', then the
        appropriate kind of forest will be trained; otherwise, a neural network
        will be trained and the prefix should be the name of a class defined in
        _architectures.py.

    """

    prefix = os.path.basename(model_dir).split('-')[0]
    if prefix in ['mubofo', 'random']:
        train_scikit_forest(data_dir, model_dir)
    elif prefix == 'xgboost':
        train_xgboost_forest(data_dir, model_dir)
    else:
        train_network(data_dir, model_dir)