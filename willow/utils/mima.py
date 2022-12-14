import os

import xarray as xr

from cftime import num2date

_CALENDAR = '360_day'
_UNITS = 'days since 0001-01-01 00:00:00'

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

def open_mima_output(src: str | list[str]) -> xr.Dataset:
    """
    Read MiMA output files and decode the time coordinate.

    Parameters
    ----------
    src : Path or list of paths to netCDF files to read.

    Returns
    -------
    ds : `Dataset` with decoded time coordinate.

    """

    ds = xr.open_mfdataset(src, decode_times=False)
    ds['time'] = num2date(ds['time'].values, _UNITS, _CALENDAR)

    return ds