from typing import Any, Literal, Optional, Union, cast, overload

import numpy as np

from sklearn.metrics import r2_score as _r2_score

_R2_KWARGS: dict[str, Any] = {
    'multioutput' : 'raw_values',
    'force_finite' : True
}

@overload
def R2_score(
    Y: np.ndarray,
    output: np.ndarray,
    reduce: Literal[True]=...
) -> float:
    ...

@overload
def R2_score(
    Y: np.ndarray,
    output: np.ndarray,
    reduce: Literal[False]
) -> np.ndarray:
    ...

def R2_score(
    Y: np.ndarray,
    output: np.ndarray,
    reduce: bool=True
) -> Union[float, np.ndarray]:
    """
    Calculate the coefficient of determination, with useful defaults.

    Parameters
    ----------
    Y : Array of targets.
    output : Array of predicted values.
    reduce : Whether to reduce the scores for each targets to a single score.
        If `True`, for output channels that are constant in `Y`, the score will
        be treated as zero unless it is exactly correct and constant in `output`
        as well, in which case it will be treated as one.

    Returns
    -------
    score : Coefficient of determination, or array of coefficients if `reduce`
        is `False`.

    """

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = _r2_score(Y, output, **_R2_KWARGS)
        scores[np.isinf(scores)] = np.nan

    if reduce:
        return np.nanmean(scores)

    return scores
    
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

def std_with_error(
    a: np.ndarray,
    confidence: float=0.95,
    n_resamples: int=int(1e4)
) -> tuple[float, float]:
    """
    Calculate standard deviation with boostrapped error estimates.

    Parameters
    ----------
    a : Array to calculate standard deviation of.
    confidence : Width of error bars. For example, `confidence=0.95` returns
        an error bar containing at least 95% of the bootstrapped statistics.
    n_resamples : Number of bootstrapped subsamples to use.

    Returns
    -------
    std : Estimated standard deviation.
    error : One-sided error bar.

    """

    stds = np.zeros(n_resamples)
    for i in range(n_resamples):
        stds[i] = np.std(np.random.choice(a, size=a.shape[0]))

    std = np.std(a)
    m = int(n_resamples * (1 - confidence) / 2)
    left, *_, right = abs(np.sort(stds)[m:(-m)] - std)

    return std, max(left, right)