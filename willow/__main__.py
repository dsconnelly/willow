import argparse

from .data import preprocess
from .train import train_forest
from .utils import parse_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser('willow')
    adder = parser.add_subparsers()
    
    for func in [preprocess, train_forest]:
        command, help_str, params = parse_func(func)
        subparser = adder.add_parser(command, help=help_str)
        subparser.set_defaults(func=func)

        for param in params:
            name = param.pop('name')
            subparser.add_argument(name, **param)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop('func')
    func(**kwargs)
