import numpy as np
import pandas as pd
import torch

from .aliases import Dataset, Model

class MiMAModel:
    """Model wrapper that handles standardization and column indexing."""

    def __init__(
        self,
        name: str,
        model: Model,
        means: np.ndarray,
        stds: np.ndarray,
        col_idx: np.ndarray,
    ) -> None:
        """
        Initialize a MiMAModel.

        Parameters
        ----------
        name : Name of the model.
        model : Trained regressor. Can be an object implementing the
            scikit-learn estimator API (namely, one with a predict method) or a
            torch model.
        means, stds : Means and standard deviations used to standardize the
            output columns during training, to be used to dimensionalize the
            model output.
        col_idx : Array of integers indexing the columns of the full feature
            array corresponding to those features used by `model`.

        """

        self.name = name
        self.model = model
        self.col_idx = col_idx

        self.means = means
        self.stds = stds
        
    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model and return the still-dimensionless output.

        Parameters
        ----------
        X : Array of input features. Should already have been indexed to contain
            only those columns that the model was trained on.
        
        Returns
        -------
        output : Model predictions. Assuming the training outputs were
            standardized, this array is still dimensionless and needs to be
            unstandardized to be physically meaningful.

        """

        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                return self.model(torch.tensor(X)).double().numpy()

        Y = self.model.predict(X).astype(np.double)
        if Y.shape[1] > 40:
            Y = Y[:, :-1]

        return Y

    def predict(self, X: Dataset) -> np.ndarray:
        """
        Apply the model to the selected features and unstandardize the result.

        Parameters
        ----------
        X : Input features for each sample. Should contain all possible input
            features; the appropriate columns are selected in this function. If
            a `DataFrame`, will be cast to an `ndarray` before prediction.

        Returns
        -------
        output : Model predictions, unstandardized so as to have physical units.

        """

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return self.means + self.stds * self.apply(X[:, self.col_idx])

    def predict_online(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model and return a result that MiMA can use.

        Parameters
        ----------
        X : Full input feature array passed by MiMA with all possible features.

        Returns
        -------
        output : Unstandardized model predictions stored in Fortran order.

        """

        return np.asfortranarray(self.predict(X))