import argparse

from willow.data import process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--process-dir',
        type=str,
        help='process MiMA output in a directory'
    )

    parser.parse_args()
    if parser.process_dir:
        print('Going to process data now.')
