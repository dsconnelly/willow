import os

import dask
import numpy as np
import xarray as xr

def preprocess(case_dir, output_dir, n_samples):
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
                'bf_cgwd', 'ps', 
                'gwfu_cgwd', 'gwfv_cgwd'
            ]]
            
            config = {'array.slicing.split_large_chunks': False}
            with dask.config.set(**config):
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

                ds = ds.isel(sample=idx)
            
            Xs, Ys = [], []
            for component in ['u', 'v']:
                Xs.append(np.hstack((
                    ds[f'{component}_gwf'].values,
                    ds['bf_cgwd'].values,
                    ds['ps'].values.reshape(-1, 1),
                    ds['lat'].values.reshape(-1, 1)
                )))
                
                Ys.append(ds[f'gwf{component}_cgwd'].values)
                
            X, Y = np.vstack(Xs), np.vstack(Ys)
            np.save(f'{output_dir}/X-{suffix}.npy', X)
            np.save(f'{output_dir}/Y-{suffix}.npy', Y)
             
def _get_fnames(case_dir):
    is_year_dir = lambda f: f.is_dir() and f.name.isnumeric()
    year_dirs = [f.path for f in os.scandir(case_dir) if is_year_dir(f)]
    fnames = [f'{year_dir}/atmos_4xdaily.nc' for year_dir in year_dirs]
    
    return sorted(fnames)