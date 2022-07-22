import time

import numpy as np
import torch, torch.nn as nn

class ScalingWrapper:
    def __init__(self, model, means, stds):
        self.model = getattr(model, 'predict', model)
        self.is_torch = isinstance(model, nn.Module)
        
        if self.is_torch:
            means = means.numpy()
            stds = stds.numpy()
        
        self.means = means
        self.stds = stds
        
    def predict(self, X):
        if self.is_torch:
            X = torch.tensor(X)
            
        with torch.no_grad():
            out = self.model(X)
            
        if self.is_torch:
            out = out.numpy()
            
        return self.means + self.stds * out

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

def timer(func):
    def timed_func(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        runtime = time.time() - start
        
        print(f'{func.__name__} took {runtime:.2f} seconds.')
        
        return output
    
    timed_func.__name__ = func.__name__
    timed_func.__doc__ = func.__doc__
    
    return timed_func