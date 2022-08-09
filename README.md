`willow` is the repository where I manage code written for the [DataWave](https://datawaveproject.github.io/) project as part of my doctoral research at the [Center for Atmosphere Ocean Science](https://caos.cims.nyu.edu/dynamic/) at New York University. I chose the name `willow` because the work contained in this repository concerns tree-based data-driven gravity wave parameterizations.

At present, `willow` provides straightforward and well-documented functions for training emulators, running them online in [MiMA](https://github.com/mjucker/MiMA), and conducting various analyses of offline and online performance. Most functionality has documentation immediately accessible from the command line, as demonstrated below.
```console
$ python -m willow train-emulator -h
usage: __main__.py train-emulator [-h] data-dir model-dir

Train a forest or neural network emulator.

positional arguments:
  data-dir    Directory where training and test datasets are saved.
  model-dir   Directory where trained model will be saved. The hyphen-separated prefix of the directory name will be used to determine the kind of model. If the prefix is one of 'mubofo', 'random', or
              'xgboost', then the appropriate kind of forest will be trained; otherwise, a neural network will be trained and the prefix should be the name of a class defined in -architectures.py.

optional arguments:
  -h, --help  show this help message and exit
```

