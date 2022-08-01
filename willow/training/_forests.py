import logging
import os

import joblib

from mubofo import BoostedForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from ..utils.data import StandardWrapper, load_data, prepare_data, standardize
from ..utils.diagnostics import logs, times

@logs
@times
def train_forest(data_dir, model_dir):
    """
    Train a boosted or random forest.

    Parameters
    ----------
    data_dir : str
        Directory where training and test datasets are saved.
    model_dir : str
        Directory where trained model will be saved. Should be prefixed with
        either 'boosted' or 'random', separated by a hyphen, to determine the
        kind of forest to train.

    """
    
    model_name = os.path.basename(model_dir)
    kind = model_name.split('-')[0]
    
    X, Y = load_data(data_dir, 'tr')
    X, Y, col_idx = prepare_data(X, Y, model_name, return_col_idx=True)
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
    model = StandardWrapper(model_name, model, means, stds, col_idx)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    