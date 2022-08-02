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

def format_pressure(p):
    if p < 1:
        return f'{p:.2f}'
    elif p < 10:
        return f'{p:.1f}'
    
    return int(round(p, 0))
