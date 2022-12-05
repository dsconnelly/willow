import cmocean.cm as cm

colors = [
    'tab:red',
    'royalblue',
    'forestgreen',
    (0.341, 0.024, 0.549),
    'darkorange',
    'hotpink',
    'springgreen'
]

def format_latitude(lat):
    """
    Format a latitude for plot tick labels.

    Parameters
    ----------
    lat : float
        The latitude to format. Should be between -90 and 90.

    Returns
    -------
    lat : str
        The formatted latitude. Contains the absolute value of the input
        latitude followed by a degree sign and, if the latitude is not zero, one
        of 'N' and 'S' chosen according to its sign.

    """

    suffix = ''
    if lat < 0:
        suffix = 'S'
    elif lat > 0:
        suffix = 'N'
        
    return f'$\mathregular{{ {int(abs(lat))}^o {suffix} }}$'

def format_name(name, simple=False):
    """
    Parse the name of a MiMA run for displaying in plots.

    Parameters
    ----------
    name : str
        The name of the run.
    simple : bool
        If True, the formatted name will just identify the model architecture.

    Returns
    -------
    name : str
        The formatted name, with trailing suffixes and perturbations removed.

    """

    if 'ad99' in name:
        return 'AD99'

    kind, *name_parts = name.split('-')
    kind = {
        'mubofo' : 'boosted forest',
        'xgboost' : 'XGBoost',
        'random' : 'random forest',
        'WaveNet' : 'neural network'
    }[kind]

    if simple:
        return kind

    if 'noloc' not in name_parts:
        name_parts.extend(['$\\vartheta$', '$p_\mathrm{s}$'])

    to_drop = ['noloc', 'control', '4xco2', 'o3hole', 'full']
    is_valid = lambda s: s not in to_drop and (len(s) > 1 or s in 'NT')

    name_parts = list(filter(is_valid, name_parts[1:]))
    for i, part in enumerate(name_parts):
        if len(part) == 1:
            name_parts[i] = f'${part}$'

    return kind + ' (' + ', '.join(name_parts) + ')'

def format_pressure(p):
    """
    Format a pressure for plot tick labels.

    Parameters
    ----------
    p : float
        The pressure to format. Should be in hPa.
    
    Returns
    -------
    p : float or int
        The formatted pressure. If the pressure is less than 10 hPa, the output
        is the pressure rounded to two significant figures; otherwise, the
        output is the pressure rounded to the nearest integer.

    """

    if p < 1:
        return f'{p:.2f}'
    elif p < 10:
        return f'{p:.1f}'
    
    return int(round(p, 0))

def get_bounds_and_cmap(field, data):
    """
    Get the right bounds and colormap for displaying physical fields.

    Parameters
    ----------
    field : str
        The name of the field.
    data : dict
        Dictionary with whose values are array-like and contain the data for
        each of the plots to be made.

    Returns
    -------
    vmin, vmax : float
        Minimum and maximum values for the plot.
    cmap : str or matplotlib.colors.Colormap
        The colormap to use.

    """

    if field in ('u', 'v', 'gwf_u', 'gwf_v'):
        vmax = max([abs(a).max() for _, a in data.items()])

        return -vmax, vmax, 'RdBu_r'

    cmap = {'T' : cm.thermal, 'N' : cm.tempo}[field]
    vmin = min([a.min() for _, a in data.items()])
    vmax = max([a.max() for _, a in data.items()])

    return vmin, vmax, cmap

def get_units(field):
    """
    Get the formatted units for a given field.

    Parameters
    ----------
    field : str
        The name of the field.

    Returns
    -------
    units : str
        The formatted units.

    """

    if field in ('u', 'v'):
        return 'm s$^{-1}$'

    if field == 'T':
        return 'K'

    if field == 'N':
        return 's$^{-1}$'

    if field.startswith('gwf'):
        return 'm s$^{-2}$'
