`willow` is the repository where I manage code written for the [DataWave](https://datawaveproject.github.io/) project as part of my doctoral research at the [Center for Atmosphere Ocean Science](https://caos.cims.nyu.edu/dynamic/) at New York University. I chose the name `willow` because the work contained in this repository concerns tree-based data-driven gravity wave parameterizations. 

At present, `willow` provides straightforward and well-documented functions for training emulators, running them online in [MiMA](https://github.com/mjucker/MiMA), and conducting various analyses of offline and online performance. Usage help can be displayed from the command line.
```console
$ python -m willow -h
usage: __main__.py [-h] {make-datasets,train-emulator,setup-mima,plot-offline-scores,plot-shapley-values,plot-qbos} ...

positional arguments:
  {make-datasets,train-emulator,setup-mima,plot-offline-scores,plot-shapley-values,plot-qbos}
    make-datasets       Read MiMA output data and save training and test sets.
    train-emulator      Train a forest or neural network emulator.
    setup-mima          Set up a MiMA run with an emulator for online testing.
    plot-offline-scores
                        Plot training and test R-squared scores by level and latitude.
    plot-shapley-values
                        Plot model Shapley values by pressure and QBO phase.
    plot-qbos           Plot quasi-biennial oscillations from one or more MiMA runs.

optional arguments:
  -h, --help            show this help message and exit
```
Each option also has its own help message.
```console
$ python -m willow train-emulator -h
usage: __main__.py train-emulator [-h] data-dir model-dir

Train a forest or neural network emulator.

positional arguments:
  data-dir    Directory where training and test datasets are saved.
  model-dir   Directory where trained model will be saved. The hyphen-separated prefix of the directory name will be used to determine the kind of model. If
              the prefix is one of 'mubofo', 'random', or 'xgboost', then the appropriate kind of forest will be trained; otherwise, a neural network will be
              trained and the prefix should be the name of a class defined in -architectures.py.

optional arguments:
  -h, --help  show this help message and exit
```
