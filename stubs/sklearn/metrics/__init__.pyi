from typing import Literal, overload

import numpy as np

def r2_score(
    Y: np.ndarray,
    output: np.ndarray,
    multitoutput: str,
    force_finite: bool
) -> np.ndarray:
    ...