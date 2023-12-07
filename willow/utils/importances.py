import logging
import warnings

from collections import defaultdict

import numpy as np
import torch
import xarray as xr

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from shap import DeepExplainer, KernelExplainer, TreeExplainer, kmeans

from .ad99 import AlexanderDunkerton
from .aliases import Model
from .statistics import standardize

def get_profile(importances: np.ndarray, features: list[str]) -> np.ndarray:
    """
    Get importances of features with vertical profiles.

    Parameters
    ----------
    importances : Array of feature importances at the level of interest.
    features : List of feature names.

    Returns
    -------
    profile : Array of total importance at each level.
    
    """

    profiles = defaultdict(list)
    for feature, value in zip(features, importances):
        if '@' not in feature:
            continue

        name = feature.split(' @ ')[0]
        profiles[name].append(value)

    def handle(v: list[float]) -> np.ndarray:
        """Convert a list to an array profile, padding with a zero if needed."""

        out = np.array(v)
        if len(out) == 39:
            out = np.append(out, 0)

        return out

    return sum((handle(v) for _, v in profiles.items()), np.zeros(40))

def get_shapley_values(
    model: AlexanderDunkerton | Model,
    background: np.ndarray,
    X: np.ndarray,
    features: list[str],
    pressures: np.ndarray
) -> xr.Dataset:
    """
    Compute Shapley values for various model architectures.

    Parameters
    ----------
    model : Trained model for which to compute the Shapley values.
    background : Array of samples to use as background data.
    X : Array of input samples to compute Shapley values for.
    features : List of input features for the returned `Dataset`.
    pressures : Array of output pressures for the returned `Dataset`.

    Returns
    -------
    xr.Dataset : `Dataset` of Shapley values with coordiantes for sample, input
        feature, and output pressure.

    """

    if isinstance(model, AlexanderDunkerton):
        Y = model.predict(background)
        _, means, stds = standardize(Y)
        f = lambda Z: standardize(model.predict(Z), means, stds)[0]
        predictions = f(X)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', module='sklearn.linear_model')
            
            background = kmeans(background, 100)
            explainer = KernelExplainer(f, background)

            importances = np.stack(explainer.shap_values(X, l1_reg=0))
            expectations = explainer.expected_value

    elif isinstance(model, (MultioutputBoostedForest, MultioutputRandomForest)):
        background = kmeans(background, 100).data
        explainer = KernelExplainer(model.predict, background)

        importances = np.stack(explainer.shap_values(X, nsamples=5000))
        expectations = explainer.expected_value
        predictions = model.predict(X)

        if model.n_outputs_ > 40:
            importances = importances[:40]
            expectations = expectations[:40]
            predictions = predictions[:, :40]
        
    elif isinstance(model, torch.nn.Module):
        background = kmeans(background, 100).data
    
        def f(Z):
            if not isinstance(Z, torch.Tensor):
                Z = torch.tensor(Z)

            with torch.no_grad():
                return model(Z).numpy()

        explainer = KernelExplainer(f, background)
        X_tensor = X

        importances = np.stack(explainer.shap_values(X_tensor))
        expectations = explainer.expected_value

        with torch.no_grad():
            # predictions = model(X_tensor).numpy()
            predictions = f(X_tensor)

    samples = np.arange(X.shape[0]) + 1
    importances = importances.transpose(1, 2, 0)
    coords = dict(sample=samples, feature=features, pressure=pressures)

    data = {
        'importances' : (('sample', 'feature', 'pressure'), importances),
        'expectations' : (('pressure',), expectations),
        'X' : (('sample', 'feature'), X),
        'predictions' : (('sample', 'pressure'), predictions)
    }

    return xr.Dataset(data, coords)