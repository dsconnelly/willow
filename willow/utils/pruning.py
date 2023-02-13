import numpy as np

from sklearn.tree._tree import Tree

from .aliases import Forest

def get_max_depths(forest: Forest) -> np.ndarray:
    """
    Get the maximum depth of each tree in a forest.

    Parameters
    ----------
    forest : Forest to traverse.

    Returns
    -------
    max_depths : Maximum depth of each tree in `forest`.

    """

    trees = [e.tree_ for e in forest.estimators_]
    depths = [_get_max_depth(tree, 0) for tree in trees]

    return np.array(depths)

def prune_forest(forest: Forest, max_depths: int | np.ndarray) -> Forest:
    """
    Prune each tree in a forest to a given depth.

    Parameters
    ----------
    forest : Forest to prune.
    max_depths : Array of depths to prune each tree to. If an `int`, every tree
        will be pruned to that depth.

    """
    
    trees = [e.tree_ for e in forest.estimators_]
    if isinstance(max_depths, int):
        max_depths = max_depths * np.ones(len(trees))

    for tree, max_depth in zip(trees, max_depths):
        _prune_tree(tree, 0, 1, max_depth)

    return forest

def _get_max_depth(tree: Tree, node: int) -> int:
    """
    Recursively get the maximum depth of a decision tree.

    Parameters
    ----------
    tree : Tree to traverse.
    node : Current node.

    Returns
    -------
    max_depth : Maximum depth from the current node down.

    """

    left = tree.children_left[node]
    right = tree.children_right[node]

    if left == -1:
        return 0

    return max(_get_max_depth(tree, left), _get_max_depth(tree, right)) + 1

def _prune_tree(tree: Tree, node: int, depth: int, max_depth: int) -> None:
    """
    Recursively prune a decision tree to a given maximum depth.

    Parameters
    ----------
    tree : Tree to be pruned.
    node : Current node.
    depth : Current depth.
    max_depth : Depth past which tree will be pruned.

    """

    left = tree.children_left[node]
    right = tree.children_right[node]

    if left == -1 or depth > max_depth:
        tree.children_left[node] = -1
        tree.children_right[node] = -1

        return

    _prune_tree(tree, left, depth + 1, max_depth)
    _prune_tree(tree, right, depth + 1, max_depth)
