import os

import numpy as np
import scipy.signal as signal
import xarray as xr

from .mima import get_fnames

def load_qbo(case_dir, n_years=24):
    """
    Load the QBO signal from a MiMA run with yearly output subdirectories.

    Parameters
    ----------
    case_dir : str
        The directory where MiMA was run. Should contain subdirectories named
        with two-digit integers corresponding to each year of ouput, each of
        which contains an 'atmos_4xdaily.nc' file.
    n_years : int
        The number of years of data to return. Years will be counted back from
        the end of the MiMA run.

    Returns
    -------
    u : xr.DataArray
        A DataArray containing the daily tropical-mean zonal-mean zonal wind at
        pressure levels less than 115 hPa for years indicated.

    """

    path = os.path.join(case_dir, 'qbo.nc')
    if os.path.exists(path):
        with xr.open_dataset(path, decode_times=False) as ds:
            u = ds['u_gwf'].sel(pfull=slice(None, 115))
            u = u.isel(time=slice(-(360 * n_years), None))
            u = u.assign_coords(time=u['time'].astype(int))

    else:
        fnames = get_fnames(case_dir)
        with xr.open_mfdataset(fnames[-n_years:], decode_times=False) as ds:
            u = ds['u_gwf'].sel(pfull=slice(None, 115)).sel(lat=slice(-5, 5))
            u = u.groupby(u['time'].astype(int)).mean(('time', 'lat', 'lon'))

    return u.load()

def qbo_statistics(u):
    """
    Compute the period and amplitude of a QBO signal.

    Parameters
    ----------
    u : xr.DataArray
        The QBO signal, as returned by load_qbo.

    Returns
    -------
    period : float
        The QBO period in months. Calculated by taking a Fourier transform and
        returning the period with maximum spectral power at 27 hPa.
    period_err : float
        The error in the period calculation in months. Calculated as the
        half-width of the spectral peak used to select the period.
    amp : float
        The QBO amplitude in meters per second. Calculated as the standard
        deviation of the signal at 20 hPa.
    amp_err : float
        The error in the amplitude calculation in meters per second. Calculated
        by bootstrapping subsamples, computing their standard deviation, and
        returning the half-width of the 95% confidence interval.

    """

    u = _apply_butterworth(u)
    period, period_err = _qbo_period(u)
    amp, amp_err = _qbo_amplitude(u)

    return period, period_err, amp, amp_err

def _apply_butterworth(u):
    sos = signal.butter(9, 1 / 120, output='sos', fs=1)
    func = lambda a: signal.sosfilt(sos, a)

    return xr.apply_ufunc(
        func, u,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True
    ).transpose('time', 'pfull')

def _qbo_amplitude(u, level=20):
    return _std_with_error(u.sel(pfull=level, method='nearest').values)

def _qbo_period(u, level=27):
    u = u.sel(pfull=level, method='nearest').values
    u_hat = np.fft.fft(u, n=int(2.5e6))

    freqs = np.fft.fftfreq(u_hat.shape[0])
    idx = (freqs > 0) & (freqs >= 1 / len(u))

    powers = abs(u_hat[idx])
    periods = (1 / freqs[idx]) / 30

    k = powers.argmax()
    period = periods[k]
    half_max = powers[k] / 2

    starts, = np.where((powers[:-1] < half_max) & (powers[1:] > half_max))
    ends, = np.where((powers[:-1] > half_max) & (powers[1:] < half_max))
    start, end = starts[starts < k][-1], ends[ends > k][0]

    left, right = periods[start], periods[end]
    error = max(left - period, period - right)

    return period, error

def _std_with_error(a, confidence=0.95, n_resamples=int(1e4)):
    stds = np.zeros(n_resamples)
    for i in range(n_resamples):
        stds[i] = np.std(np.random.choice(a, size=a.shape[0]))

    std = np.std(a)
    m = int(((1 - confidence) / 2) * n_resamples)
    left, *_, right = abs(np.sort(stds)[m:(-m)] - std)
    error = max(left, right)

    return std, error