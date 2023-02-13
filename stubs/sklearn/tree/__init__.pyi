import numpy as np

from scipy.sparse import csr_matrix

from ._tree import Tree

class DecisionTreeRegressor:
    tree_: Tree

    def decision_path(self, X : np.ndarray) -> csr_matrix:
        ...
