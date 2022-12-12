from typing import Optional, cast

import numpy as np

def standardize(
    A: np.ndarray,
    means: Optional[np.ndarray]=None,
    stds: Optional[np.ndarray]=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize an array to zero mean and unit variance in each column.

    Parameters
    ----------
    A : Array to standardize.
    means : Means to use during standardization. If `None`, the means of the
        columns of `A` will be calculated and used.
    stds : Standard deviations to use during standardization. If `None`, the
        standard deviations of the columns in `A` will be calculated and used.

    Returns
    -------
    output : Standardized array of same shape and dtype as `A`.
    means : Column means used for standardization.
    stds : Column standard deviations used for standardization.

    """

    if means is None:
        means = cast(np.ndarray, A.mean(axis=0))
    if stds is None:
        stds = cast(np.ndarray, A.std(axis=0))

    mask = stds != 0
    output = np.zeros_like(A)
    output[:, mask] = (A[:, mask] - means[mask]) / stds[mask]

    return output, means, stds