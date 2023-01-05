import numpy as np

from scipy.sparse import csr_matrix

class DecisionTreeRegressor:
    tree_: Tree

    def decision_path(self, X : np.ndarray) -> csr_matrix:
        ...

class Tree:
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
