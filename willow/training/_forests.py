import logging
import os

import joblib

from mubofo import BoostedForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from ..utils.datasets import load_data, prepare_data
from ..utils.transforms import StandardizedModel, standardize

def train_forest(data_dir, model_dir):    
    model_name = os.path.basename(model_dir)
    kind = model_name.split('-')[0]
    
    X, Y = load_datasets(data_dir, 'tr')
    X, Y, col_idx = prepare_datasets(X, Y, model_name, return_col_idx=True)
    Y_scaled, means, stds = standardize(Y, return_stats=True)
    
    kwargs = {
        'n_estimators' : 300,
        'max_depth' : 15,
        'max_samples' : 0.3,
        'max_features' : 0.5
    }
    
    if kind == 'random':
        model_class = RandomForestRegressor
        
    elif kind == 'boosted':
        model_class = BoostedForestRegressor
        
        kwargs['learning_rate'] = 0.08
        kwargs['val_size'] = 0.2
        kwargs['patience'] = 20
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
                PCA(n_components=0.8),
                estimator_class(**kwargs)
            )
        
    model = model_class(**kwargs).fit(X, Y_scaled)
    model = StandardizedModel(model_name, model, means, stds, col_idx)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    