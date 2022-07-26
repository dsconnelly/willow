import os
import logging
import time

from importlib import import_module

import joblib
import numpy as np
import torch, torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split

from ._utils import ScalingWrapper, logs, standardize, times

@logs
@times
def train_network(data_dir, model_dir, class_name):
    """
    Train a neural network.
    
    Parameters
    ----------
    data_dir : str
        Directory where training and test datasets are saved.
    model_dir : str
        Directory where trained model will be saved.
    class_name : str
        Name of neural network model class.
        
    """
    
    X = torch.tensor(np.load(os.path.join(data_dir, 'X-tr.npy')))
    Y = torch.tensor(np.load(os.path.join(data_dir, 'Y-tr.npy')))
    
    n_samples, n_in = X.shape
    _, n_out = Y.shape
    
    idx = torch.randperm(n_samples)
    X, Y = X[idx], Y[idx]
    
    m = round(0.8 * n_samples)
    X_tr, Y_tr = X[:m], Y[:m]
    X_va, Y_va = X[m:], Y[m:]
    
    Y_tr_scaled, means, stds = standardize(Y_tr, return_stats=True)
    Y_va_scaled = standardize(Y_va, means, stds)
    
    architectures = import_module('._architectures', package='willow.train')
    model = getattr(architectures, class_name)(n_in, n_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f'Loaded {n_samples} samples, using {m} for training.')
    logging.info(f'Training a {class_name} with {n_params} parameters.')
    
    dataset = TensorDataset(X_tr, Y_tr_scaled)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    def loss_func(Y, output):
        return ((Y - output) ** 2).mean()
    
    max_epochs = 40
    max_hours = 16
    
    i, training_start = 1, time.time()
    while True:
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            loss = loss_func(Y_batch, model(X_batch))
            loss.backward()
            optimizer.step()
  
        with torch.no_grad():
            loss = loss_func(Y_va_scaled, model(X_va)).item()
            logging.info(f'Epoch {i}: validation loss is {loss:3f}')
        
        hours = (time.time() - training_start) / 3600
        if hours > max_hours or i == max_epochs:
            logging.info(f'Terminating after {i} epochs.')
            model = ScalingWrapper(model, means, stds)
            joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
            
            return
            
        i = i + 1
    