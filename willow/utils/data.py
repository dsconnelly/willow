import os

import numpy as np
import pandas as pd
import torch, torch.nn as nn

class StandardWrapper:
    def __init__(self, name, model, means, stds, col_idx):
        self.name = name
        self.model = model        
        
        if self._is_torch():
            means = means.numpy()
            stds = stds.numpy()
        
        self.means = means
        self.stds = stds
        
        self.col_idx = col_idx
        
    def predict(self, X):
        with torch.no_grad():
            out = self._apply(X)
            
        return self.means + self.stds * out
    
    def predict_online(self, X):
        return self.predict(X[:, self.col_idx])
    
    def _apply(self, X):
        if self._is_torch():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
                
            return self.model(X).numpy()
        
        return self.model.predict(X)
    
    def _is_torch(self):
        return isinstance(self.model, nn.Module)
    
def load_data(data_dir, suffix):
    X = pd.read_pickle(os.path.join(data_dir, f'X-{suffix}.pkl'))
    Y = pd.read_pickle(os.path.join(data_dir, f'Y-{suffix}.pkl'))
    
    return X, Y
    
def prepare_data(X, Y, model_name, return_col_idx=False):
    name_parts = model_name.split('-')
    allowed = {'wind', 'shear', 'T', 'Nsq'}
    keep = allowed.intersection(set(name_parts))
    
    cols, idx = [], []
    for i, s in enumerate(X.columns):
        if any([name in s for name in keep]):
            cols.append(s)
            idx.append(i)
            
    cols = cols + ['surface pressure', 'latitude']
    idx = np.array(idx)
    
    X, Y = X[cols].to_numpy(), Y.to_numpy()
    if name_parts[0] not in ['boosted', 'random']:
        X, Y = torch.tensor(X), torch.tensor(Y)
        
    if return_col_idx:
        return X, Y, idx
        
    return X, Y
    
def standardize(A, means=None, stds=None, return_stats=False):
    if means is None:
        means = A.mean(axis=0)
    if stds is None:
        stds = A.std(axis=0)
        
    if isinstance(A, torch.Tensor):
        out = torch.zeros_like(A)
    elif isinstance(A, np.ndarray):
        out = np.zeros_like(A)
        
    mask = stds != 0
    out[:, mask] = (A[:, mask] - means[mask]) / stds[mask]
    
    if return_stats:
        return out, means, stds
    
    return out