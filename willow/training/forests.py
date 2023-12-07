import logging

from typing import Any

import numpy as np

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

from ..utils.aliases import Forest

_KWARGS: dict[str, Any] = {
    'n_estimators' : 300,
    'max_depth' : 15,
    'max_samples' : 0.15,
    'max_features' : 0.5
}

_EXTRA_KWARGS: dict[str, Any] = {
    'learning_rate' : 0.1,
    'val_size' : 0.2,
    'threshold' : 0.001,
    'max_patience' : 25,
    'logging' : True
}

def train_forest(X: np.ndarray, Y: np.ndarray, model_name: str) -> Forest:
    """
    Train a boosted or random forest.

    Parameters
    ----------
    X : Array of input features.
    Y : Array of (possibly standardized) targets.
    model_name : Name of the model. The first hyphen-separated substring should
        be either `'mubofo'` or `'random'`. Hyperparameter overrides may be
        included as hyphen-separated substrings where the value is separated
        from the hyperparameter name by an underscore.

    Returns
    -------
    model : Trained boosted or random forest.

    """

    kwargs = _KWARGS.copy()
    kind, *_ = model_name.split('-')
    model_class: type[Forest]

    if kind == 'mubofo':
        model_class = MultioutputBoostedForest
        kwargs.update(_EXTRA_KWARGS)

    elif kind == 'random':
        model_class = MultioutputRandomForest

    logging.info(f'Training a {model_class.__name__}.')
    _override_parameters(model_name, kwargs)

    return model_class(**kwargs).fit(X, Y)

def _override_parameters(model_name: str, kwargs: dict[str, Any]) -> None:
    """
    Parse a model name and perform any hyperparameter overrides.

    Parameters
    ----------
    model_name : Name of the model, with hyperparameter overrides specified as
        described in the docstring for `train_forest`.
    kwargs : Dictionary of parameters with values to override.

    """

    for override in [s for s in model_name.split('-') if '_' in s]:
        *name_parts, value = override.split('_')
        name = '_'.join(name_parts)

        if name in kwargs:
            caster = type(kwargs[name])
            kwargs[name] = caster(value)
            logging.info(f'Setting {name} to {value}.')
