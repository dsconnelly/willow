import os

import numpy as np
import pandas as pd
import torch
    
def load_datasets(data_dir, suffix, n_samples=None):
    """
    Load pandas DataFrames from disk, with optional subsampling.

    Parameters
    ----------
    data_dir : str
        The directory containing the pickled DataFrames.
    suffix : str
        The suffix to be added on to the file names. Should be 'tr' or 'te'.
    n_samples : int
        How many rows of each DataFrame should be randomly sampled to return. If
        None, each DataFrame will be returned in full.

    Returns
    -------
    X, Y : pd.DataFrame
        DataFrames with input and output variables. X contains all possible
        input variables (wind, shear, temperature, buoyancy frequency, surface
        pressure, and latitude). Y contains the corresponding gravity wave drag
        profiles for each sample.

    """

    X = pd.read_pickle(os.path.join(data_dir, f'X-{suffix}.pkl'))
    Y = pd.read_pickle(os.path.join(data_dir, f'Y-{suffix}.pkl'))

    if n_samples is not None:
        idx = np.random.choice(len(X), size=n_samples, replace=False)
        X, Y = X.iloc[idx], Y.iloc[idx]
    
    return X, Y
    
def prepare_datasets(X, Y, model_name, as_array=True, return_col_idx=False):
    """
    Extract the relevant input variables for a given model.

    Parameters
    ----------
    X, Y : pd.DataFrame
        DataFrames of input and output data, as returned by load_datasets.
    model_name : str
        The name of the model being trained. It should be a hyphen-separated
        list of (potentially among other things) input variable names, which can
        be 'wind', 'shear', 'T', and 'Nsq'. Location variables (surface pressure
        and latitude) will be included, unless 'noloc' is in the list.
    as_array : bool
        Whether the outputs should be cast from a pd.DataFrame to an array class
        (either a np.array or a torch.Tensor, depending on model_name).
    return_col_idx : bool
        Whether to return the indices corresponding to the returned columns of
        X. These indices are useful for coupling models to MiMA, where the model
        needs to extract input variables from an unlabeled array.

    Returns
    -------
    X, Y : np.ndarry or torch.Tensor
        The extracted input and output data. If The first hyphen-separated part
        of model_name does not specify AD99 or a kind of forest, the model is
        assumed to be a torch model, and the outputs are cast as torch.Tensors.

    """

    name_parts = model_name.split('-')
    keep, idx = _filter_columns(set(name_parts), X.columns)
    X = X[keep]
    
    if as_array:          
        X, Y = X.to_numpy(), Y.to_numpy()
        if name_parts[0] not in ['ad99', 'mubofo', 'random', 'xgboost']:
            X, Y = torch.tensor(X), torch.tensor(Y)
        
    if return_col_idx:
        return X, Y, idx
        
    return X, Y

def _filter_columns(name_parts, columns):
    allowed = {'wind', 'shear', 'T', 'Nsq'} & name_parts
    if 'noloc' not in name_parts:
        allowed = allowed | {'pressure', 'latitude'}

    keep, idx = [], []
    for i, column in enumerate(columns):
        if any([name in column for name in allowed]):
            keep.append(column)
            idx.append(i)

    return keep, np.array(idx)
