import logging
import warnings

from collections import defaultdict
from typing import Optional

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
        means, stds = Y.mean(axis=0), Y.std(axis=0)
        f = lambda Z: standardize(model.predict(Z), means, stds)[0]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', module='sklearn.linear_model')
            
            background = kmeans(background, 100)
            explainer = KernelExplainer(f, background)
            importances = np.stack(explainer.shap_values(X, l1_reg=0))

    elif isinstance(model, (MultioutputBoostedForest, MultioutputRandomForest)):
        _importances = []
        for i, estimator in enumerate(model.estimators_):
            explainer = TreeExplainer(estimator)
            _importances.append(np.stack(explainer.shap_values(X)))
            logging.info(f'Computed Shapley values for tree {i + 1}.')

        importances = sum(_importances)
        if isinstance(model, MultioutputBoostedForest):
            importances *= model.learning_rate
        elif isinstance(model, MultioutputRandomForest):
            importances /= len(model.estimators_)
        
    elif isinstance(model, torch.nn.Module):
        background = kmeans(background, 100).data
        explainer = DeepExplainer(model, torch.tensor(background))
        importances = np.stack(explainer.shap_values(torch.tensor(X)))

    samples = np.arange(X.shape[0]) + 1
    importances = importances.transpose(1, 2, 0)
    coords = dict(sample=samples, feature=features, pressure=pressures)

    data = {
        'importances' : (('sample', 'feature', 'pressure'), importances),
        'X' : (('sample', 'feature'), X)
    }

    return xr.Dataset(data, coords)