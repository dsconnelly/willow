import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cftime import date2num, num2date

from ..utils.mima import get_fnames

def plot_ssws(case_dirs, output_path):
    """
    Make a bar graph of the sudden stratospheric warming occurrences in MiMA.

    Parameters
    ----------
    case_dirs : list of str
        Directories of the MiMA runs with SSWs to count and plot.
    output_path : str
        Path where the plot should be saved.

    """

    units = 'days since 0001-01-01 00:00:00'
    calendar = '360_day'

    data = {}
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        path = os.path.join(case_dir, 'ssw.nc')

        n_years = 12
        if os.path.exists(path):
            with xr.open_dataset(path, decode_times=False) as ds:
                u = ds['u_gwf'].isel(time=slice(-(360 * n_years), None))

        else:
            fnames = get_fnames(case_dir, n_years=n_years)
            with xr.open_mfdataset(fnames, decode_times=False) as ds:
                u = ds['u_gwf'].sel(lat=60, method='nearest').mean('lon')
                
        u = u.sel(pfull=10, method='nearest')
        u['time'] = num2date(u['time'].values, units, calendar)
        u = u.resample(time='1D').mean(('time')).load()

        idx = _find_ssws(u)
        data[name] = (idx, u)

    fig, axes = plt.subplots(nrows=len(case_dirs), squeeze=False)
    fig.set_size_inches(9, 3 * len(case_dirs))

    for (name, (idx, u)), ax in zip(data.items(), axes[:, 0]):
        years = date2num(u['time'], units, calendar) / 360
        months = u['time'].dt.month
        mask = (3 < months) & (months < 11)

        ax.plot(years, u.values.flatten(), color='k', zorder=2)
        ax.scatter(
            years[idx], u[idx], 
            marker='x', s=100, 
            color='tab:red', 
            zorder=3
        )

        ymin, ymax = -40, 80
        ax.fill_between(
            years, ymin, ymax, 
            where=~mask,
            color='gainsboro',
            zorder=1
        )

        ax.set_xlim(years.min(), years.max())
        ax.set_ylim(ymin, ymax)
        ax.grid(True, zorder=0)

        ax.set_xlabel('year')
        ax.set_ylabel('$u$ (m s$^{-1}$)')

        n_decades = (years.max() - years.min()) / 10
        rate = len(idx) / n_decades
        ax.set_title(f'{name} \u2014 {rate:.2f} SSW decade$^{{-1}}$')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

def _find_ssws(u):
    idx, count = [], None
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

    return np.array(idx)