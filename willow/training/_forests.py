import logging
import os

import joblib
import xgboost as xgb

from mubofo import BoostedForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.statistics import standardize
from ..utils.wrappers import MiMAModel

_N_TREES = 300
_MAX_DEPTH = 15
_LEARNING_RATE = 0.08
_MAX_SAMPLES = 0.3
_MAX_FEATURES = 0.5
_PATIENCE = 20

def train_scikit_forest(data_dir, model_dir):    
    model_name = os.path.basename(model_dir)
    kind = model_name.split('-')[0]
    
    X, Y = load_datasets(data_dir, 'tr')
    X, Y, col_idx = prepare_datasets(X, Y, model_name, return_col_idx=True)
    Y_scaled, means, stds = standardize(Y, return_stats=True)
    
    kwargs = {
        'n_estimators' : _N_TREES,
        'max_depth' : _MAX_DEPTH,
        'max_samples' : _MAX_SAMPLES,
        'max_features' : _MAX_FEATURES
    }
    
    if kind == 'random':
        model_class = RandomForestRegressor
        
    elif kind == 'mubofo':
        model_class = BoostedForestRegressor
        
        kwargs['learning_rate'] = _LEARNING_RATE
        kwargs['val_size'] = 0.2
        kwargs['patience'] = _PATIENCE
        kwargs['threshold'] = 0.01
        kwargs['verbose'] = True
        
    else:
        raise ValueError(f'Unknown forest type: {kind}')
        
    logging.info(f'Loaded {X.shape[0]} samples.')
    logging.info(f'Training a {model_class.__name__}.')
    
    if 'pca' in model_name:
        logging.info('Using PCA in training pipeline.')
        
        estimator_class = model_class
        def model_class(**kwargs):
            return make_pipeline(
                StandardScaler(),
                PCA(n_components=0.9),
                estimator_class(**kwargs)
            )
        
    model = model_class(**kwargs).fit(X, Y_scaled)
    model = MiMAModel(model_name, model, means, stds, col_idx)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    
def train_xgboost_forest(data_dir, model_dir):
    model_name = os.path.basename(model_dir)

    X, Y = load_datasets(data_dir, 'tr')
    X, Y, col_idx = prepare_datasets(X, Y, model_name, return_col_idx=True)
    X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.2)

    Y_tr_scaled, means, stds = standardize(Y_tr, return_stats=True)
    Y_va_scaled = standardize(Y_va, means, stds)

    (n_tr, _), (n_va, _) = X_tr.shape, X_va.shape
    logging.info(f'Loaded {n_tr + n_va} samples, using {n_tr} for training.')
    logging.info(f'Training an xgboost model.')

    params = {
        #'max_depth' : _MAX_DEPTH,
        'max_depth' : 5,
        'eta' : _LEARNING_RATE,
        'subsample' : _MAX_SAMPLES,
        'colsample_bynode' : _MAX_FEATURES,
    }

    data_tr = xgb.DMatrix(X_tr, label=Y_tr_scaled)
    data_va = xgb.DMatrix(X_va, label=Y_va_scaled)
    
    model = xgb.train(
        params, data_tr,
        num_boost_round=_N_TREES,
        evals=[(data_va, 'val')],
        #early_stopping_rounds=_PATIENCE,
        early_stopping_rounds=5
    )
    
    model = MiMAModel(model_name, model, means, stds, col_idx)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))