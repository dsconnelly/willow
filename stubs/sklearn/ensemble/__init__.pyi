import numpy as np

class RandomForestRegressor:
    def fit(self, X: np.ndarray, Y: np.ndarray) -> RandomForestRegressor:
        ...
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...