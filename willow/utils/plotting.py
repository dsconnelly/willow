colors = [
    'tab:red',
    'royalblue',
    'forestgreen',
    'darkviolet',
    'darkorange'
]

def format_latitude(lat):
    suffix = ''
    if lat < 0:
        suffix = 'S'
    elif lat > 0:
        suffix = 'N'
        
    return f'{int(abs(lat))}{suffix}'

def get_pressures():
    return [_format_pressure(p) for p in _pressures]

def _format_pressure(p):
    if p < 1:
        return f'{p:.2f}'
    elif p < 10:
        return f'{p:.1f}'
    
    return int(round(p, 0))

_pressures = [
    0.1781917, 0.5561894, 0.723815, 0.9395257, 1.216319, 1.570452, 2.022183, 
    2.596651, 3.324925, 4.245241, 5.404436, 6.859625, 8.680108, 10.94954, 
    13.76836, 17.25646, 21.55608, 26.83488, 33.28912, 41.14669, 50.67014, 
    62.15909, 75.95193, 92.42645, 111.9987, 135.1196, 162.2685, 193.9424, 
    230.6404, 272.8413, 320.9724, 375.3678, 436.214, 503.4754, 576.7974, 
    655.3696, 737.7247, 821.3965, 902.1687, 970.5498
]
