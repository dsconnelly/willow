import numpy as np
import torch, torch.nn as nn
import xgboost as xgb

class StandardizedModel:
    def __init__(self, name, model, means, stds):
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
        return self.means + self.stds * self._apply(X)

    def _apply(self, X):
        if self.is_torch:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
                
            with torch.no_grad():
                return self.model(X).double().numpy()

        if self.is_xgboost:
            X = xgb.DMatrix(X)

        return self.model.predict(X)

class MiMAModel(StandardizedModel):
    def __init__(self, name, model, means, stds, col_idx):
        super().__init__(name, model, means, stds)
        self.col_idx = col_idx

    def predict_online(self, X):
        return np.asfortranarray(self.predict(X[:, self.col_idx]))