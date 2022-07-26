import os

import joblib
import numpy as np

from mubofo import BoostedForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from ._utils import ScalingWrapper, standardize, timer

@timer
def train_forest(data_dir, model_dir, kind):
    """
    Train a boosted or random forest.
    
    Parameters
    ----------
    data_dir : str
        Directory where training and test datasets are saved.
    model_dir : str
        Directory where trained model will be saved.
    kind : str
        The kind of forest to train. Must be either 'boosted' or 'random'.
    
    """
    
    X = np.load(os.path.join(data_dir, 'X-tr.npy'))
    Y = np.load(os.path.join(data_dir, 'Y-tr.npy'))
    Y_scaled, means, stds = standardize(Y, return_stats=True)
    
    kwargs = {
        'n_estimators' : 300,
        'max_depth' : 15,
        'max_samples' : 0.2,
        'max_features' : 0.5
    }
    
    if kind == 'random':
        model_class = RandomForestRegressor
        
    elif kind == 'boosted':
        model_class = BoostedForestRegressor
        
        kwargs['verbose'] = True
        kwargs['val_size'] = 0.2
        kwargs['patience'] = 20
        kwargs['threshold'] = 0.01
        
    else:
        raise ValueError(f'Unknown forest type: {kind}')
        
    print(f'Loaded {X.shape[0]} samples.')
    print(f'Training a {model_class.__name__}.')
    
    if False and 'pca' in model_dir:
        print('Using PCA in training pipeline.')
        
        estimator_class = model_class
        def model_class(**kwargs):
            return make_pipeline(
                PCA(n_components=10),
                estimator_class(**kwargs)
            )
        
    model = model_class(**kwargs).fit(X, Y_scaled)
    model = ScalingWrapper(model, means, stds)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    