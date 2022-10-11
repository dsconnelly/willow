import warnings

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import xarray as xr

from mubofo import BoostedForestRegressor
from shap import DeepExplainer, KernelExplainer, TreeExplainer, kmeans
from sklearn.ensemble import RandomForestRegressor
from torch.nn import Module
from xgboost import Booster

from .ad99 import AlexanderDunkerton
from .statistics import standardize

def get_level_data(scores, level):
    """
    Get Shapley values for each input profile at a particular level.

    Parameters
    ----------
    scores : xr.Dataset
        The precomputed Shapley values. Should have pressure and feature
        coordinates, and the feature coordinates should have their corresponding
        pressures preceded by the @ symbol.
    level : float
        The output pressure level. The Shapley values for the pressure level in
        the dataset closest to level will be returned.

    Returns
    -------
    k : int
        The index of the closest pressure level to level.
    profiles : dict
        A dictionary whose keys are the input profiles included in the dataset
        (e.g. 'wind', 'T') and whose values are arrays of Shapley values.

    """

    k = abs(scores['pressure'].values - level).argmin()
    scores = scores.isel(pressure=k)

    profiles = defaultdict(list)
    for feature, score in zip(scores['feature'].values, scores.values):
        if '@' not in feature:
            continue

        name = feature.split(' @ ')[0]
        profiles[name].append(score)

    def handle(v):
        out = np.array(v)
        if len(out) == 39:
            out = np.append(out, 0)

        return out

    return k, {name : handle(v) for name, v in profiles.items()}

def get_shapley_scores(samples, model, Y=None):
    """
    Compute Shapley values for various model architectures.

    Parameters
    ----------
    samples : pandas.DataFrame
        A DataFrame of input samples containing only the features used by the
        provided model.
    model : various
        The model for which to compute the Shapley values.
    Y : np.ndarray
        Model outputs to be used for standardization. Only used if model is an
        AlexanderDunkerton instance.

    Returns
    -------
    ds : xr.Dataset
        A Dataset of Shapley values for each sample. Has coordinates for sample,
        input feature, and output pressure.

    """

    features = samples.columns
    X = samples.to_numpy()

    if isinstance(model, AlexanderDunkerton):
        if Y is None:
            raise ValueError('Must pass Y if model is AlexanderDunkerton')

        means = Y.mean(axis=0)
        stds = Y.std(axis=0)
        f = lambda Z: standardize(model.predict(Z), means, stds)

        background, X = X[:4000], X[-500:]
        background = kmeans(background, 50)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', module='sklearn.linear_model')
            
            explainer = KernelExplainer(f, background)
            scores = np.stack(explainer.shap_values(X, l1_reg=0))

    if isinstance(model, BoostedForestRegressor):
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
    