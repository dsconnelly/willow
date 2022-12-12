import os

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