from collections import defaultdict

import numpy as np
import torch
import xarray as xr

from mubofo import BoostedForestRegressor
from shap import DeepExplainer, TreeExplainer
from sklearn.ensemble import RandomForestRegressor
from torch.nn import Module
from xgboost import Booster

def combine_by_level(scores):
    """
    Combine Shapley values corresponding to input from the same pressure level.

    Parameters
    ----------
    scores : xr.Dataset
        The precomputed Shapley values. Should have pressure and feature
        coordinates, and the feature coordinates should have their corresponding
        pressures preceded by the @ symbol.

    Returns
    -------
    combined : dict
        The dictionary of combined Shapley values, with a key for each pressure
        level found in the feature names in scores (as a string).

    """

    n_pressures = len(scores['pressure'])
    combined = defaultdict(lambda: np.zeros(n_pressures))

    for feature in scores['feature'].values:
        if '@' not in feature:
            continue

        p = feature.split('@')[-1].split()[0]
        combined[p] += scores.sel(feature=feature).values

    return dict(combined)

def compute_shapely_values(samples, model):
    """
    Compute Shapley values for various model architectures.

    Parameters
    ----------
    samples : pandas.DataFrame
        A DataFrame of input samples containing only the features used by the
        provided model.
    model : BoostedForestRegressor or Booster or RandomForestRegressor or Module
        The model for which to compute the Shapley values.

    Returns
    -------
    ds : xr.Dataset
        A Dataset of Shapley values for each sample. Has coordinates for sample,
        input feature, and output pressure.

    """

    features = samples.columns
    X = samples.to_numpy()

    if isinstance(model, (BoostedForestRegressor)):
        tree_scores = []
        for tree in model.estimators_:
            explainer = TreeExplainer(tree)
            tree_scores.append(np.stack(explainer.shap_values(X)))

        scores = sum(tree_scores)

    elif isinstance(model, (Booster, RandomForestRegressor)):
        explainer = TreeExplainer(model)
        scores = np.stack(explainer.shap_values(X))

    elif isinstance(model, Module):
        m = round(0.2 * X.shape[0])
        background, X = X[:m], X[m:]

        explainer = DeepExplainer(model, torch.tensor(background))
        scores = np.stack(explainer.shap_values(torch.tensor(X)))

    samples = np.arange(X.shape[0]) + 1
    pressures = [s.split('@')[-1].split()[0] for s in features if '@' in s]
    pressures = np.sort(np.array([float(p) for p in set(pressures)]))    

    scores = scores.transpose(1, 2, 0)
    coords = {'sample' : samples, 'feature' : features, 'pressure' : pressures}
    data = {
        'scores' : (('sample', 'feature', 'pressure'), scores),
        'X' : (('sample', 'feature'), X)
    }

    return xr.Dataset(data, coords)


    