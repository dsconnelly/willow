import os
import warnings

from typing import Optional

import numpy as np
import xarray as xr

from .mima import open_mima_output
from .statistics import stat_with_error

_N_YEARS = 56

def load_ssw(case_dir: str) -> xr.DataArray:
    """
    Load the SSW wind and count the number of warmings from a MiMA run.

    Parameters
    ----------
    case_dir : Directory where MiMA was run. Must contain a file named `ssw.nc`.

    Returns
    -------
    u : DataArray of SSW (10 hPa at 60N) daily mean winds.

    """

    path = os.path.join(case_dir, 'zonal_mean.nc')
    with open_mima_output(path, n_years=_N_YEARS) as ds:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)

            u = ds['u_gwf'].sel(lat=60, method='nearest').isel(lon=0)
            u = u.sel(pfull=10, method='nearest')
            u = u.resample(time='1D').mean(('time')).load()

    return u

def get_ssw_frequency(u: xr.DataArray) -> tuple[float, float]:
    """
    Get the SSW frequency along with the bootstrapped uncertainty.

    Parameters
    ----------
    u : DataArray of SSW (10 hPa at 60N) daily mean winds.

    Returns
    -------
    freq : SSW frequency per decade.
    error : Bootstrapped uncertainty.

    """

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

    winters = np.zeros(_N_YEARS + 1)
    start = u.time[0].item().year

    for i in idx:
        time = u.time[i].item()
        year = time.year - start

        if time.month > 10:
            year = year + 1

        winters[year] += 1

    def stat_func(a):
        return 10 * a.sum() / len(a)

    return stat_with_error(winters, stat_func=stat_func)
