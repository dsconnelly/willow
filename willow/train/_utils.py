import functools
import time

import numpy as np
import torch, torch.nn as nn

class ScalingWrapper:
    def __init__(self, model, means, stds):
        self.model = model        
        if self._is_torch():
            means = means.numpy()
            stds = stds.numpy()
        
        self.means = means
        self.stds = stds
        
    def predict(self, X):
        with torch.no_grad():
            out = self._apply(X)
            
        return self.means + self.stds * out    
    
    def _apply(self, X):
        if self._is_torch():
            return self.model(torch.tensor(X)).numpy()
        
        return self.model.predict(X)
    
    def _is_torch(self):
        return isinstance(self.model, nn.Module)

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
    @functools.wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        runtime = time.time() - start
        
        print(f'{func.__name__} took {runtime:.2f} seconds.')
        
        return output
    
    return timed_func