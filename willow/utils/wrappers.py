import numpy as np
import torch, torch.nn as nn
import xgboost as xgb

class StandardizedModel:
    """Wrapper for models to handle I/O sanitization and unstandardization."""

    def __init__(self, name, model, means, stds):
        """
        Initialize a StandardizedModel.

        Parameters
        ----------
        name : str
            The name of the model.
        model : sklearn.base.BaseEstimator or xgboost.Booster or nn.Module
            The trained model. Can be an object implementing the scikit-learn
            estimator API, an xgboost Booster, or a torch model.
        means, stds : np.ndarray or torch.Tensor
            The means and standard deviations to be used in unstandardization of
            the model output. If model is an nn.Module, means and stds should be
            torch.Tensors; otherwise, they should be np.ndarrays.

        """

        self.name = name
        self.model = model      

        self.is_torch = isinstance(self.model, nn.Module)
        self.is_xgboost = isinstance(self.model, xgb.Booster) 
        
        if self.is_torch:
            means = means.numpy()
            stds = stds.numpy()
    
        self.means = means
        self.stds = stds

    def predict(self, X):
        """
        Use self.model to make a prediction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        output : np.ndarray of shape (n_samples, n_outputs)
            The output of self.model, unstandardized to be in physical space.

        """

        return self.means + self.stds * self._apply(X)

    def _apply(self, X):
        if self.is_torch:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
                
            with torch.no_grad():
                return self.model(X).double().numpy()

        if self.is_xgboost:
            X = xgb.DMatrix(X)

        return self.model.predict(X).astype(np.float64)

class MiMAModel(StandardizedModel):
    """StandardizedModel subclass to handle online predictions."""

    def __init__(self, name, model, means, stds, col_idx):
        """
        Initialize a MiMA model.

        Parameters
        ----------
        name, model, means, stds
            Same as for StandardizedModel.
        col_idx : np.ndarray of shape (161,)
            The indices of the columns in the full data array passed by MiMA
            corresponding to input variables to the trained model. Most likely
            obtained as an output from willow.utils.datasets.prepare_datasets.

        """

        super().__init__(name, model, means, stds)
        self.col_idx = col_idx

    def predict_online(self, X):
        """
        Apply self.model to a full data array passed by MiMA.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, 161)
            The full data array passed by MiMA, containing all possible model
            input variables.
        
        Returns
        -------
        output : np.ndarray of shape (n_samples, n_outputs)
            The result of first taking only the columns in self.col_idx of X
            and then passing the data into self.predict.

        """
        
        return np.asfortranarray(self.predict(X[:, self.col_idx]))