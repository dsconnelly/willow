import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.mima import get_fnames, get_mima_name
from ..utils.plotting import colors, format_latitude, format_pressure, get_units

def plot_distributions(case_dirs, field, latitude, pressure, output_path):
    """
    Make a plot comparing distributions of a MiMA output variable.

    Parameters
    ----------
    case_dirs : list of str
        Directories containing MiMA runs to comapre.
    field : str
        Name of the field to show.
    latitude : float
        Latitude at which the comparison should be made.
    pressure : float
        Pressure (in hPa) at which the comparison should be made.
    output_path : str
        Path where the plot should be saved.

    """

    datas = {}
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        fnames = get_fnames(case_dir, n_years=12)

        with xr.open_mfdataset(fnames, decode_times=False) as ds:
            ds = ds.sel(lat=latitude, pfull=pressure, method='nearest')
            datas[name] = ds[get_mima_name(field)].values.flatten()

    bmin = min([data.min() for _, data in datas.items()])
    bmax = max([data.max() for _, data in datas.items()])
    bins = np.linspace(bmin, bmax, 21)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    for (name, data), color in zip(datas.items(), colors):
        ax.hist(data, bins, color=color, label=name, alpha=0.4)

    ax.set_xlim(bmin, bmax)
    ax.set_xlabel(f'{field} ({get_units(field)})')
    ax.set_ylabel('number of occurrences')
    
    latitude = format_latitude(latitude)
    pressure = format_pressure(pressure)

    ax.set_title(f'distributions at {latitude} and {pressure} hPa')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
