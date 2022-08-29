colors = [
    'tab:red',
    'royalblue',
    'forestgreen',
    'darkviolet',
    'darkorange'
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
        
    return f'{int(abs(lat))}$^\circ${suffix}'

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
