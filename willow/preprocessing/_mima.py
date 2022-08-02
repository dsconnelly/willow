import os

import dask
import numpy as np
import pandas as pd
import xarray as xr

from ..utils.plotting import get_pressures

def make_datasets(case_dir, output_dir, n_samples=int(5e6)):
    """
    Read MiMA output data and save training and test sets.

    Parameters
    ----------
    case_dir : str
        The directory for the MiMA run.
    output_dir : str
        The directory where training and test data will be saved.
    n_samples : int
        The number of samples to draw for each of u and v.

    """
    
    all_fnames = _get_fnames(case_dir)
    suffix_to_fnames = {
        'tr' : all_fnames[-5:-1], 
        'te' : all_fnames[-1]
    }
    
    for suffix, fnames in suffix_to_fnames.items():
        with xr.open_mfdataset(fnames, decode_times=False) as ds:
            ds = ds[[
                'u_gwf', 'v_gwf', 
                't_gwf', 'bf_cgwd', 'ps', 
                'gwfu_cgwd', 'gwfv_cgwd'
            ]]
            
            config = {'array.slicing.split_large_chunks': False}
            with dask.config.set(**config):
                ds = _sample_dataset(ds, n_samples)
            
            Xs, Ys = [], []
            for component in ['u', 'v']:
                wind = ds[f'{component}_gwf'].values
                shear = wind[:, :-1] - wind[:, 1:]
                
                Xs.append(np.hstack((
                    wind, shear,
                    ds['t_gwf'].values,
                    ds['bf_cgwd'].values,
                    ds['ps'].values.reshape(-1, 1),
                    ds['lat'].values.reshape(-1, 1)
                )))
                
                Ys.append(ds[f'gwf{component}_cgwd'].values)
                
            pressures = get_pressures()
            profile_names = lambda s: [f'{s} @ {p} hPa' for p in pressures]
            
            columns_X = (
                profile_names('wind') +
                profile_names('shear')[:-1] +
                profile_names('T') +
                profile_names('Nsq') +
                ['surface pressure', 'latitude']
            )
            
            columns_Y = profile_names('drag')
                
            X, Y = np.vstack(Xs), np.vstack(Ys)
            df_X = pd.DataFrame(X, columns=columns_X)
            df_Y = pd.DataFrame(Y, columns=columns_Y)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            df_X.to_pickle(os.path.join(output_dir, f'X-{suffix}.pkl'))
            df_Y.to_pickle(os.path.join(output_dir, f'Y-{suffix}.pkl'))
                
def _get_fnames(case_dir):
    is_year_dir = lambda f: f.is_dir() and f.name.isnumeric()
    year_dirs = [f.path for f in os.scandir(case_dir) if is_year_dir(f)]
    fnames = [f'{year_dir}/atmos_4xdaily.nc' for year_dir in year_dirs]
    
    return sorted(fnames)

def _sample_dataset(ds, n_samples):
    ds = ds.stack(sample=('time', 'lat', 'lon'))
    ds = ds.reset_index('sample')
    ds = ds.transpose('sample', 'pfull')

    weights = abs(np.cos(np.pi * ds['lat'].values / 180))
    weights = weights / weights.sum()

    idx = np.sort(np.random.choice(
        len(weights), 
        size=n_samples, 
        p=weights,
        replace=False
    ))

    return ds.isel(sample=idx)

