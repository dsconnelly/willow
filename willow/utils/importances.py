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

def get_profiles(
    importances: np.ndarray,
    features: list[str]
) -> dict[str, np.ndarray]:
    """
    Get importances of features with vertical profiles.

    Parameters
    ----------
    importances : Array of feature importances at the level of interest.
    features : List of feature names.

    Returns
    -------
    profiles : Dictionary whose keys are the names of the profiles and whose
        values are arrays of feature importances.
    
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

    return {name : handle(v) for name, v in profiles.items()}

def get_shapley_values(
    model: AlexanderDunkerton | Model,
    X: np.ndarray,
    features: list[str],
    pressures: np.ndarray
) -> xr.Dataset:
    """
    Compute Shapley values for various model architectures.

    Parameters
    ----------
    model : Trained model for which to compute the Shapley values.
    X : Array of input samples containing only those features used
        by `model`.
    features : List of input features for the returned `Dataset`.
    pressures : Array of output pressures for the returned `Dataset`.

    Returns
    -------
    xr.Dataset : `Dataset` of Shapley values with coordiantes for sample, input
        feature, and output pressure.

    """

    if isinstance(model, AlexanderDunkerton):
        Y = model.predict(X)
        means, stds = Y.mean(axis=0), Y.std(axis=0)
        f = lambda Z: standardize(model.predict(Z), means, stds)[0]

        m = round(0.8 * X.shape[0])
        background, X = X[:m], X[m:]
        background = kmeans(background, 100)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', module='sklearn.linear_model')
            
            explainer = KernelExplainer(f, background)
            importances = np.stack(explainer.shap_values(X, l1_reg=0))

    elif isinstance(model, (MultioutputBoostedForest, MultioutputRandomForest)):
        _importances = []
        for i, estimator in enumerate(model.estimators_):
            explainer = TreeExplainer(estimator)
            _importances.append(np.stack(explainer.shap_values(X)))
            logging.info(f'Computed Shapley values for tree {i + 1}.')

        importances = sum(_importances)
        if isinstance(model, MultioutputRandomForest):
            importances /= len(model.estimators_)

    elif isinstance(model, torch.nn.Module):
        m = round(0.8 * X.shape[0])
        background, X = X[:m], X[m:]
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