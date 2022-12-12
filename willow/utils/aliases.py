from typing import TypeAlias, Union

import numpy as np
import pandas as pd
import torch

from mubofo import BoostedForestRegressor
from sklearn.ensemble import RandomForestRegressor

Dataset: TypeAlias = Union[np.ndarray, pd.DataFrame]
Forest: TypeAlias = Union[BoostedForestRegressor, RandomForestRegressor]
Model: TypeAlias = Union[Forest, torch.nn.Module]