import os

from typing import Optional

import dask
import numpy as np
import xarray as xr

from cftime import num2date

_CALENDAR = '360_day'
_UNITS = 'days since 0001-01-01 00:00:00'

R_dry = 287.04
c_p = 7 * R_dry / 2
grav = 9.8

def get_paths(case_dir: str) -> list[str]:
    """
    Get the paths to files containing MiMA output.

    Parameters
    ----------
    case_dir : Directory where MiMA was run. Should contain subdirectories with
        names corresponding to the years of the run, each of which contains
        exactly one file called `'atmos_4xdaily.nc'`.

    Returns
    -------
    fnames : Sorted list of paths to netCDF files containing MiMA output.

    """

    years = sorted([s for s in os.listdir(case_dir) if s.isdigit()])
    paths = [os.path.join(case_dir, y, 'atmos_4xdaily.nc') for y in years]

    return paths

def open_mima_output(
    src: str | list[str],
    n_years: Optional[int]=None
) -> xr.Dataset:
    """
    Read MiMA output files and decode the time coordinate.

    Parameters
    ----------
    src : Path or list of paths to netCDF files to read.
    n_years : Number of years of data to return. If greater than the length of
        the run minus three years, only that part of the run will be returned.

    Returns
    -------
    ds : `Dataset` with decoded time coordinate.

    """

    ds = xr.open_mfdataset(src, decode_times=False)
    ds['time'] = num2date(ds['time'].values, _UNITS, _CALENDAR)

    time = ds['time'].values
    years = np.array([t.year for t in time])

    n_available = round((time[-1] - time[0]).days / 360) - 3
    n_years = min(n_years, n_available) if n_years else n_available
    start = years[-1] - n_years

    with dask.config.set(**{'array.slicing.split_large_chunks' : False}):
        return ds.isel(time=(years >= start))
