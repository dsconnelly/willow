from itertools import cycle

import numpy as np

COLORS = cycle([
    'tab:red',
    'royalblue',
    'forestgreen'
])

def format_latitude(lat: float) -> str:
    """
    Format a latitude value for reasonable display.

    Parameters
    ----------
    lat : Latitude to format. Should be between -90 and 90.

    Returns
    -------
    lat : Format latitude. Contains the absolute value of the input latitude
        followed by a degree sign and, if the latitude is not zero, one of `'N'`
        or `'S'` chosen according to its sign.

    """

    value = int(abs(lat))
    suffix = ['', 'N', 'S'][int(np.sign(lat))]

    return f'$\mathregular{{ {value}^o {suffix} }}$'

def format_name(name: str) -> str:
    """
    Format the name of a model or MiMA run for reasonable display.

    Parameters
    ----------
    name : Name of the model or MiMA run.

    Returns
    -------
    name : Name formatted for display in plots.

    """

    if 'ad99' in name:
        return 'AD99'

    kind, *name_parts = name.split('-')
    kind = {
        'mubofo' : 'boosted forest',
        'random' : 'random forest',
        'WaveNet' : 'neural network'
    }[kind]

    features = [s for s in name_parts if s in ['wind', 'shear', 'N', 'T']]
    if 'noloc' not in name_parts:
        features.extend(['$\\vartheta$', '$p_\mathrm{s}$'])

    for i, feature in enumerate(features):
        if len(feature) == 1:
            features[i] = f'${feature}$'

    return kind + ' (' + ', '.join(features) + ')'

def format_pressure(p: float) -> str:
    """
    Format a pressure value for reasonable display.

    Parameters
    ----------
    p : Pressure to format, in hPa.

    Returns
    -------
    p : Formatted pressure. If `p < 10`, the output is `p` rounded to two
        significant figures; otherwise, `p` is rounded to the nearest integer.

    """

    if p < 1:
        return f'{p:.2f}'
    
    if p < 10:
        return f'{p:.1f}'

    return str(round(p, 0))