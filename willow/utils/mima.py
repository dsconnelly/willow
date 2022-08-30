import os

def get_fnames(case_dir, n_years=12):
    """
    Get the filenames corresponding to the last chunk of a MiMA run.

    Parameters
    ----------
    case_dir : str
        The directory containing the MiMA run.
    n_years : int
        The number of years to retain, counting from the end of the run.

    Returns
    -------
    fnames : list of str
        A sorted list of names of the netCDF files for the requested years.

    """

    years = sorted([s for s in os.listdir(case_dir) if s.isdigit()])
    fnames = [os.path.join(case_dir, y, 'atmos_4xdaily.nc') for y in years]

    return fnames[-n_years:]

def get_mima_name(field):
    """
    Get the name MiMA uses for a physical field.

    Parameters
    ----------
    field : str
        The name of the field.

    Returns
    -------
    name : str
        The name MiMA uses for the field.
        
    """

    if field in ('u', 'v', 'T'):
        return field.lower() + '_gwf'

    if field == 'N':
        return 'bf_cgwd'

    if field.startswith('gwd'):
        return f'gwf{field[-1]}_cgwd'
