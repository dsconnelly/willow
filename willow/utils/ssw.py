import os
import warnings

from typing import Optional

import numpy as np
import xarray as xr

from .mima import open_mima_output

def load_ssw(case_dir: str) -> tuple[xr.DataArray, np.ndarray]:
    """
    Load the SSW wind and count the number of warmings from a MiMA run.

    Parameters
    ----------
    case_dir : Directory where MiMA was run. Must contain a file named `ssw.nc`.

    Returns
    -------
    u : DataArray of SSW (10 hPa at 60N) daily mean winds.
    idx : Integer indices of starts of warming events.

    """

    path = os.path.join(case_dir, 'ssw.nc')
    with open_mima_output(path, n_years=24) as ds:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)

            u = ds['u_gwf'].isel(lat=0, lon=0)
            u = u.sel(pfull=10, method='nearest')
            u = u.resample(time='1D').mean(('time')).load()

    idx: list[int] = []
    count: Optional[int] = None

    for i, (time, wind) in enumerate(zip(u['time'].values, u.values)):
        if count is None:
            if 3 < time.month < 11:
                continue

            if wind < 0:
                idx.append(i)
                count = 0

        elif 4 < time.month < 11:
            if count < 10:
                idx.pop()

            count = None
            continue

        elif wind > 0:
            count = count + 1
            if count == 20:
                count = None

        else:
            count = 0

    return u, np.array(idx)

    