import numpy as np

from mubofo import MultioutputDecisionTree

from .aliases import Forest

def prune_forest(forest: Forest, max_depths: np.ndarray) -> Forest:
    for estimator, max_depth in zip(forest.estimators_, max_depths):
        _prune_tree(estimator.tree_, 0, 1, max_depth)

    return forest

def _prune_tree(
    tree: MultioutputDecisionTree,
    node: int,
    depth: int,
    max_depth: int
):
    left = tree.children_left[node]
    right = tree.children_right[node]

    if left == -1 or right == -1:
        return

    if depth > max_depth:
        tree.children_left[node] = -1
        tree.children_right[node] = -1

    _prune_tree(tree, left, depth + 1, max_depth)
    _prune_tree(tree, right, depth + 1, max_depth)
        
if __name__ == '__main__':
    from joblib import load
    forest = load('models/mubofo-wind-T-s/model.pkl').model
    prune_forest(forest.estimators_[0], max_depth=10)
