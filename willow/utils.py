import builtins

def parse_func(func):
    lines = filter(lambda s: s.strip(), func.__doc__.split('\n'))
    lines = [line[4:] for line in lines]
    help_str = lines[0]

    params = []
    for line in lines[3:]:
        if not line.startswith('    '):
            name, type_name = line.split(' : ')
            params.append({
                'name' : name,
                'metavar' : name.replace('_', '-'),
                'type' : _get_type(type_name),
                'help' : []
            })

        else:
            params[-1]['help'].append(line[4:].strip())

    for param in params:
        param['help'] = ' '.join(param['help'])

    return func.__name__.replace('_', '-'), help_str, params

def _get_type(type_name):
    try:
        return getattr(builtins, type_name)
    except AttributeError:
        return globals()[type_name]
