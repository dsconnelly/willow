import numpy as np
import torch

def R_squared(Y, output):
    """
    Compute the R-squared score for multiple output channels.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_outputs)
        The true values for each sample and output.
    output : np.ndarray of shape (n_samples, n_outputs)
        The predicted values for each sample and target.

    Returns
    -------
    scores : np.ndarray of shape (n_outputs,)
        The R-squared score for each output. For outputs where the total sum of
        squares is zero (because every sample in Y has the same value for that
        output) scores will contain np.nan.

    """

    means = Y.mean(axis=0)
    ss_res = ((Y - output) ** 2).sum(axis=0)
    ss_tot = ((Y - means) ** 2).sum(axis=0)
    
    mask = ss_tot != 0
    scores = np.nan * np.zeros(Y.shape[1])
    scores[mask] = 1 - (ss_res[mask] / ss_tot[mask])
    
    return scores

def standardize(A, means=None, stds=None, return_stats=False):
    """
    Standardize an array to zero mean and unit variance.

    Parameters
    ----------
    A : np.ndarray or torch.Tensor of shape (n_samples, n_features)
        The array to standardize.
    means : np.ndarray or torch.Tensor of shape (n_features,)
        The mean of each feature. If None, the means of the features in A will
        be calculated and used.
    stds : np.ndarray or torch.Tensor of shape (n_features,)
        The standard deviation of each feature. If None, the standard deviations
        of the features in A will be calculated and used.
    return_stats : bool
        Whether to return the means and standard deviations used in the
        calculation. Most useful when means and stds are passed as None.

    Returns
    -------
    output : np.ndarray or torch.Tensor of shape (n_samples, n_features)
        The standardized array. Has the same shape and type as A.
    means : np.ndarray or torch.Tensor of shape (n_features,)
        The means used to standardize. Only returned if return_stats is True.
    stds : np.ndarray or torch.Tensor of shape (n_features,)
        The standard deviations used to standardize. Only returned if
        return_stats is True. 

    """

    if means is None:
        means = A.mean(axis=0)
    if stds is None:
        stds = A.std(axis=0)
        
    if isinstance(A, torch.Tensor):
        output = torch.zeros_like(A)
    elif isinstance(A, np.ndarray):
        output = np.zeros_like(A)
        
    mask = stds != 0
    output[:, mask] = (A[:, mask] - means[mask]) / stds[mask]
    
    if return_stats:
        return output, means, stds
    
    return output