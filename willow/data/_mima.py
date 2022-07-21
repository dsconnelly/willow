import os

def process(case_dir, n_samples=int(1e7)):
    suffixes = ['tr', 'te']
    spans = [(0, 4 * 1440), (4 * 1440, 5 * 1440)]

    is_year_dir = f.is_dir() and f.name.isnumeric()
    year_dirs = [f.path for f in os.scandir(case_dir) if is_year_dir(f)]
    fnames = [f'{year_dir}/atmos_4xdaily.nc' for year_dir in year_dirs]
