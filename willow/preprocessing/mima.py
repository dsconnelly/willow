import os

import dask
import numpy as np
import pandas as pd
import xarray as xr

from ..utils.mima import get_paths
from ..utils.plotting import format_pressure

def save_datasets(
    case_dir: str,
    output_dir: str,
    n_samples: int=int(1e5)
) -> None:
    """
    Read MiMA output files and save training and test sets.

    Parameters
    ----------
    case_dir : Directory where MiMA was run.
    output_dir : Directory where training and test `DataFrame`s will be saved.
    n_samples : The number of samples to draw. Half the samples will be in the
        zonal direction and half in the meridional. 

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paths = get_paths(case_dir)[-8:]
    pairs = {'tr' : paths[:4], 'te' : paths[4:]}
    keep = lambda s: 'gwf' in s or 'cgwd' in s or s == 'ps'

    profiles = ['wind', 'shear', 'T', 'N']
    scalars = ['surface pressure', 'latitude', 'time']

    for suffix, paths in pairs.items():
        ds: xr.Dataset
        with xr.open_mfdataset(paths, decode_times=False) as ds:
            keep_names = filter(keep, ds.keys())
            ds = ds[list(keep_names)]

            config: dict = {'array.slicing.split_large_chunks' : False}
            with dask.config.set(**config):
                ds = _sample_dataset(ds, n_samples // 2)

            T = ds['t_gwf'].values
            N = ds['bf_cgwd'].values
            zero_cols = N.min(axis=0) == 0
            N[:, zero_cols] = 0

            ps = ds['ps'].values.reshape(-1, 1)
            lat = ds['lat'].values.reshape(-1, 1)
            time = ds['time'].values.reshape(-1, 1)

            Xs, Ys = [], []
            for component in ('u', 'v'):
                wind = ds[f'{component}_gwf'].values
                shear = wind[:, :-1] - wind[:, 1:]
                
                Xs.append(np.hstack((wind, shear, T, N, ps, lat, time)))
                Ys.append(ds[f'gwf{component}_cgwd'].values)
                
            pressures = [format_pressure(p) for p in ds['pfull'].values]
            make_names = lambda s: [f'{s} @ {p} hPa' for p in pressures]

            columns_X = sum(map(make_names, profiles), []) + scalars
            columns_X.remove(f'shear @ {pressures[-1]} hPa')
            columns_Y = make_names('drag')

            X = pd.DataFrame(np.vstack(Xs).astype(np.double), columns=columns_X)
            Y = pd.DataFrame(np.vstack(Ys).astype(np.double), columns=columns_Y)

            X.to_pickle(os.path.join(output_dir, f'X-{suffix}.pkl'))
            Y.to_pickle(os.path.join(output_dir, f'Y-{suffix}.pkl'))

def _sample_dataset(ds: xr.Dataset, n_samples: int) -> xr.Dataset:
    """
    Sample points from a `Dataset` weighted by the cosine of latitude.

    Parameters
    ----------
    ds : Dataset to sample from. Should have a `'lat'` coordinate.
    n_samples : Number of points to sample.

    Returns
    -------
    ds : Sampled dataset.

    """

    ds = ds.stack(sample=('time', 'lat', 'lon'))
    ds = ds.reset_index('sample').transpose('sample', 'pfull')
    
    weights = abs(np.cos(np.radians(ds['lat'].values)))
    weights = weights / weights.sum()

    idx = np.sort(np.random.choice(
        len(weights),
        size=n_samples,
        replace=False,
        p=weights
    ))

    return ds.isel(sample=idx)
