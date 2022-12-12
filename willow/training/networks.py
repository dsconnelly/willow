import logging
import time

from importlib import import_module

import numpy as np
import torch

from torch.nn import MSELoss, Module
from torch.utils.data import DataLoader, TensorDataset

_MAX_EPOCHS = 60
_MAX_HOURS = 16

def train_network(X: np.ndarray, Y: np.ndarray, model_name: str) -> Module:
    """
    Train a neural network.

    Parameters
    ----------
    X : Array of input features.
    Y : Array of (possibly standardized) targets.
    model_name : Name of the model. The first hyphen-separated substring should
        be the name of a class in `architectures.py`.

    Returns
    -------
    model : Trained neural network.

    """

    n_samples, n_in = X.shape
    _, n_out = Y.shape

    idx = torch.randperm(n_samples)
    X, Y = X[idx], Y[idx]

    m = round(0.8 * n_samples)
    X_tr, Y_tr = torch.tensor(X[:m]), torch.tensor(Y[:m])
    X_va, Y_va = torch.tensor(X[m:]), torch.tensor(Y[m:])

    class_name = model_name.split('-')[0]
    architectures = import_module('.architectures', 'willow.training')
    model: Module = getattr(architectures, class_name)(n_in, n_out)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Using {m} samples for training.')
    logging.info(f'Training a {class_name} with {n_params} parameters.')

    dataset = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = MSELoss()

    epoch, start = 1, time.time()
    while True:
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            loss_func(model(X_batch), Y_batch).backward()
            optimizer.step()

        with torch.no_grad():
            loss = loss_func(model(X_va), Y_va).item()
            logging.info(f'Epoch {epoch}: validation loss is {loss:.3f}.')

        hours = (time.time() - start) / 3600
        if epoch == _MAX_EPOCHS or hours > _MAX_HOURS:
            logging.info(f'Terminating training after {epoch} epochs.')

            return model.eval()

        epoch = epoch + 1
