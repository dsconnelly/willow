import numpy as np

from .datasets import prepare_datasets
from .statistics import R_squared

def get_scores_by_lev(X, Y, model):
    """
    Calculate R_squared scores by vertical level.

    Parameters
    ----------
    X, Y : pd.DataFrame
        DataFrames of input and output data, as returned by load_datasets.
    model : various
        The model to evaluate.

    """

    X, Y = prepare_datasets(X, Y, model.name)
    
    return R_squared(Y, model.predict(X))
    
