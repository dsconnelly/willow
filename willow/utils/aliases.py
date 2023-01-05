from typing import TypeAlias, Union

import numpy as np
import pandas as pd
import torch

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

Dataset: TypeAlias = Union[np.ndarray, pd.DataFrame]
Forest: TypeAlias = Union[MultioutputBoostedForest, MultioutputRandomForest]
Model: TypeAlias = Union[Forest, torch.nn.Module]