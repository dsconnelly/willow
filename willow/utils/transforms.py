import numpy as np
import torch, torch.nn as nn

class StandardizedModel:
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
        return np.asfortranarray(self.predict(X[:, self.col_idx]))
    
    def _apply(self, X):
        if self._is_torch():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
                
            return self.model(X).double().numpy()
        
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