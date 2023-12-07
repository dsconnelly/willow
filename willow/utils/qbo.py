import os
import warnings

from typing import Optional

import numpy as np
import scipy.signal as signal
import xarray as xr

from ..utils.mima import open_mima_output
from ..utils.statistics import stat_with_error

def get_qbo_statistics(u: xr.DataArray) -> tuple[float, float, float, float]:
    """
    Compute the period and amplitude of a QBO signal.

    Parameters
    ----------
    u : QBO winds as returned by `load_qbo`.

    Returns
    -------
    period, period_err : Period and error estimate, in months, as estimated by
        `_get_qbo_period` using the default method.
    amp, amp_err : Amplitude and error estimate, in meters per second, as
        estimated by `_get_qbo_amplitude`.

    """

    u = _apply_butterworth(u)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        period, period_err = _get_qbo_period(u)
        amp, amp_err = _get_qbo_amplitude(u)

    return period, period_err, amp, amp_err

def load_qbo(case_dir: str) -> xr.DataArray:
    """
    Load the QBO wind from a MiMA run.

    Parameters
    ----------
    case_dir : Directory where MiMA was run. Must contain a file named `qbo.nc`.

    Returns
    -------
    u : DataArray of QBO (tropical zonal mean) winds.

    """

    path = os.path.join(case_dir, 'zonal_mean.nc')
    with open_mima_output(path, n_years=56) as ds:
        u = ds['u_gwf'].sel(
            pfull=slice(None, 115),
            lat=slice(-5, 5)
        ).mean(('lat', 'lon'))

    return u.load()

def _apply_butterworth(u: xr.DataArray) -> xr.DataArray:
    """
    Apply a Butterworth filter to smooth the QBO signal.

    Parameters
    ----------
    u : Unfiltered QBO winds.

    Returns
    -------
    u : Filtered QBO winds.

    """

    sos = signal.butter(9, 1 / 120, output='sos', fs=1)
    func = lambda a: signal.sosfilt(sos, a)

    return xr.apply_ufunc(
        func, u,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True
    ).transpose('time', 'pfull')

def _get_qbo_amplitude(u: xr.DataArray, level: float=10) -> tuple[float, float]:
    """
    Calculate the QBO amplitude at a given pressure level.

    Parameters
    ----------
    u : (Possibly filtered) QBO winds.
    level : Pressure level to compute the amplitude at, in hPa.

    Returns
    -------
    amp : QBO amplitude, calculated as the standard deviation of the winds at
        the specified pressure level.
    error : Error in amplitude estimate, calculated by bootstrap resampling.

    """

    return stat_with_error(u.sel(pfull=level, method='nearest').values)

def _get_qbo_period(
    u: xr.DataArray,
    level: float=10,
    method: str='fourier'
) -> tuple[float, float]:
    """
    Calculate the QBO period at a given pressure level.

    Parameters
    ----------
    u : (Possibly filtered) QBO winds.
    level : Pressure level to compute the period at, in hPa.
    method : Either `'crossings'` or `'fourier'`, depending on whether the
        period should be estimated using gaps between zero crossings or the
        period of the dominant Fourier mode.

    Returns
    -------
    period : QBO period, calculated according to `method`.
    error : Error in period estimate. If `method == 'crossings'`, the error is
        the standard deviation in the periods of each cycle, whereas if
        `method == 'fourier'`, the error is the half-width of the spectrum.

    """

    u = u.sel(pfull=level, method='nearest')

    if method == 'crossings':
        crossings = (u.values[:-1] < 0) & (u.values[1:] > 0)
        days = np.diff(u['time'].values[:-1][crossings])
        months = np.array([x.days for x in days]) / 30

        return months.mean(), months.std()

    elif method == 'fourier':
        u_hat = np.fft.fft(u, n=int(2.5e6))
        freqs = np.fft.fftfreq(u_hat.shape[0], d=1)
        idx = (freqs > 0) & (freqs >= 1 / len(u))

        powers = abs(u_hat[idx])
        periods = 1 / freqs[idx] / 30

        k = powers.argmax()
        period = periods[k]
        half_max = powers[k] / 2

        starts, = np.where((powers[:-1] < half_max) & (powers[1:] > half_max))
        ends, = np.where((powers[:-1] > half_max) & (powers[1:] < half_max))
        starts, ends = starts[starts < k], ends[ends > k]

        left = np.nan if not starts.size else periods[starts[-1]]
        right = np.nan if not ends.size else periods[ends[0]]
        error = np.nanmax([left - period, period - right])

        return period, error

    raise ValueError(f'Unknown method \'{method}\' for QBO period')
