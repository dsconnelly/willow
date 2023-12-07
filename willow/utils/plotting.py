from itertools import cycle

import numpy as np

COLORS = cycle([
    'tab:red',
    'royalblue',
    'forestgreen',
    'darkorange',
    'darkviolet',
    'fuchsia',
    'gold',
    'cyan',
    'lawngreen'
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

def format_name(name: str, simple=False) -> str:
    """
    Format the name of a model or MiMA run for reasonable display.

    Parameters
    ----------
    name : Name of the model or MiMA run.
    simple : Whether to just return the kind of model.

    Returns
    -------
    name : Name formatted for display in plots.

    """

    co2 = None
    if '/' in name:
        *_, co2, name = name.split('/')

    kind, *name_parts = name.split('-')
    kind = {
        'ad99' : 'AD99',
        'mubofo' : 'boosted forest',
        'random' : 'random forest',
        'WaveNet' : 'neural network'
    }[kind]

    if 'lat_scale' in name:
        part = [x for x in name_parts if x.startswith('lat_scale')][0]
        *_, s = part.split('_')

        if co2 is not None:
            return f'retrained {kind} ({co2} ppm CO$_2$)'

        return f'retrained\n{kind}'
        # return f'{kind} (s = {s})'

    if co2 is not None:
        return f'{kind} ({co2} ppm CO$_2$)'

    if simple or 'AD99' in kind:
        return kind

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

    return str(int(round(p, 0)))

def get_letter(i: int) -> str:
    """
    Return a letter of the alphabet for use in subplot titles.

    Parameters
    ----------
    i : Index of the subplot, starting at zero.

    Returns
    -------
    letter : `(i + 1)`th letter of the alphabet.

    """

    return chr(97 + i)

def get_rows_and_columns(n_subplots: int) -> tuple[int, int]:
    """
    Choose a pleasing arrangement of subplots.

    Parameters
    ----------
    n_subplots : Number of subplots to create.

    Returns
    -------
    n_rows : Number of rows to use.
    n_cols : Number of columns to use.

    """

    if n_subplots < 4:
        return n_subplots, 1

    if not n_subplots % 2:
        return n_subplots // 2, 2

    if not n_subplots % 3:
        return n_subplots // 3, 3

    return (n_subplots // 2) + 1, 2