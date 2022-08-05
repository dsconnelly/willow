import os

import numpy as np
import pandas as pd
import torch
    
def load_datasets(data_dir, suffix, n_samples=None):
    X = pd.read_pickle(os.path.join(data_dir, f'X-{suffix}.pkl'))
    Y = pd.read_pickle(os.path.join(data_dir, f'Y-{suffix}.pkl'))

    if n_samples is not None:
        idx = np.random.choice(len(X), size=n_samples, replace=False)
        X, Y = X.iloc[idx], Y.iloc[idx]
    
    return X, Y
    
def prepare_datasets(X, Y, model_name, return_col_idx=False):
    name_parts = model_name.split('-')
    keep, idx = _filter_columns(name_parts, X.columns)                
    X, Y = X[keep].to_numpy(), Y.to_numpy()

    if name_parts[0] not in ['mubofo', 'random', 'xgboost']:
        X, Y = torch.tensor(X), torch.tensor(Y)
        
    if return_col_idx:
        return X, Y, idx
        
    return X, Y

def _filter_columns(name_parts, columns):
    allowed = {'wind', 'shear', 'T', 'Nsq'} & set(name_parts)
    allowed = allowed | {'pressure', 'latitude'}

    keep, idx = [], []
    for i, column in enumerate(columns):
        if any([name in column for name in allowed]):
            keep.append(column)
            idx.append(i)

    return keep, np.array(idx)
