import os

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from .aliases import Dataset

def load_datasets(
    data_dir: str,
    suffix: str,
    n_samples: Optional[int]=None,
    component: str='both',
    phase: Optional[str]=None,
    seed: Optional[int]=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features and targets as `DataFrame`s, with optional subsampling.

    Parameters
    ----------
    data_dir : Directory containing the pickled `DataFrame`s.
    suffix : Suffix to add on to the file names. Should be either `'tr'` or
        `'te'`, depending on whether training or test data should be loaded.
    n_samples : How many rows of each `DataFrame` should be randomly sampled to
        return. If `None`, each `DataFrame` will be returned in full.
    component : If `'both'`, the `DataFrame` is sampled without regard to
        whether zonal or meridional drag is being predicted. If `'u'` or `'v'`,
        only samples from the respective direction are taken.
    phase : If 'west' or 'east', only samples from the corresponding QBO phase
        are returned. If None, all samples are possible.
    seed : Integer to use as seed for subsampling, for reproducibility.

    Returns
    -------
    X, Y : `DataFrame`s with input and output variables. `X` contains all
        possible input features (wind, shear, temperature, buoyancy frequency,
        surface pressure, and latitude). `Y` contains the corresponding gravity
        wave drag profiles for each sample.

    """

    X: pd.DataFrame = pd.read_pickle(os.path.join(data_dir, f'X-{suffix}.pkl'))
    Y: pd.DataFrame = pd.read_pickle(os.path.join(data_dir, f'Y-{suffix}.pkl'))

    if phase is not None:
        u_qbo = X['wind @ 11 hPa'].values

        if phase == 'west':
            idx = u_qbo > 10
        elif phase == 'east':
            idx = u_qbo < -5

        X = X.iloc[idx]
        Y = Y.iloc[idx]

    if component != 'both':
        m = len(X) // 2

        if component == 'u':
            X, Y = X.iloc[:m], Y.iloc[:m]

        elif component == 'v':
            X, Y = X.iloc[m:], Y.iloc[m:]

        else:
            raise ValueError(f'Unknown component {component}')

    if n_samples is not None:
        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, len(X))
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X, Y = X.iloc[idx], Y.iloc[idx]

    return X, Y

def prepare_datasets(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract relevant input features for a given model.

    Parameters
    ----------
    X, Y : `DataFrame`s as returned by `load_datasets`.
    model_name : Name of the model being trained. It should be a list of input
        feature names (potentially among other things), separated by hyphens,
        which can be `'wind'`, `'shear'`, `'T'`, and `'N'`. Location variables
        (surface pressure and latitude) will be included unless `'noloc'` is in
        the list.

    Returns
    -------
    X, Y : Extracted input and output data as `ndarray`s.

    """

    name_parts = set(model_name.split('-'))
    keep_names, idx = _get_columns_and_index(name_parts, X.columns)
    X = X[keep_names]
    
    return X.to_numpy(), Y.to_numpy(), idx

def _get_columns_and_index(
    name_parts: set[str],
    columns: Iterable[str]
) -> tuple[list[str], np.ndarray]:
    """
    Get the column names and indices needed for a given model.

    Parameters
    ----------
    name_parts : Set of strings that were hyphen-separated in the model name.
    columns : List of available feature names.

    Returns
    -------
    keep_names : List of column names to keep for the given model.
    idx : Indices of the columns in `keep_names`.

    """

    allowed = {'wind', 'shear', 'T', 'N'} & name_parts
    if 'noloc' not in name_parts:
        allowed = allowed | {'pressure', 'latitude'}

    keep_names, idx = [], []
    for i, column in enumerate(columns):
        if any([name in column for name in allowed]):
            keep_names.append(column)
            idx.append(i)

    return keep_names, np.array(idx)
