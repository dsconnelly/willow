import functools
import logging
import os
import sys
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

def logs(func):
    @functools.wraps(func)
    def func_with_logging(*args, **kwargs):
        model_dir = kwargs['model_dir']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        logging.basicConfig(
            filename=os.path.join(model_dir, 'log.out'),
            filemode='w',
            format='%(message)s',
            level=logging.INFO
        )
        
        def handle(exc_type, exc_value, traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, traceback)
                return
            
            logging.error(
                f'{func.__name__} had an uncaught exception:',
                exc_info=(exc_type, exc_value, traceback)
            )
            
        sys.excepthook = handle
        
        return func(*args, **kwargs)
    
    return func_with_logging

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

def times(func):
    @functools.wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        runtime = time.time() - start
        
        logging.info(f'{func.__name__} took {runtime:.2f} seconds.')
        
        return output
    
    return timed_func
