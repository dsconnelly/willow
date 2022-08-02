import numpy as np
import torch

def R_squared(Y, output):
    means = Y.mean(axis=0)
    ss_res = ((Y - output) ** 2).sum(axis=0)
    ss_tot = ((Y - means) ** 2).sum(axis=0)
    
    mask = ss_tot != 0
    output = np.zeros(Y.shape[1])
    output[mask] = 1 - (ss_res[mask] / ss_tot[mask])
    
    return output

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