`willow` is the repository where I manage code written for the [DataWave](https://datawaveproject.github.io/) project as part of my doctoral research at the [Center for Atmosphere Ocean Science](https://caos.cims.nyu.edu/dynamic/) at New York University. I chose the name `willow` because the work contained in this repository concerns tree-based data-driven gravity wave parameterizations. 

At present, `willow` provides straightforward and well-documented functions for training emulators, running them online in [MiMA](https://github.com/mjucker/MiMA), and conducting various analyses of offline and online performance. Usage help can be displayed from the command line.
```console
$ python -m willow -h
usage: __main__.py [-h]
                   {save-datasets,train-emulator,plot-example-profiles,plot-example-sources,plot-R2-scores,save-shapley-values,plot-feature-importances,plot-scalar-importances,plot-shapley-errors,initialize-coupled-run,plot-biases,plot-qbos,plot-qbo-statistics,plot-ssw-frequencies}
                   ...

positional arguments:
  {save-datasets,train-emulator,plot-example-profiles,plot-example-sources,plot-R2-scores,save-shapley-values,plot-feature-importances,plot-scalar-importances,plot-shapley-errors,initialize-coupled-run,plot-biases,plot-qbos,plot-qbo-statistics,plot-ssw-frequencies}
    save-datasets       Read MiMA output files and save training and test sets.
    train-emulator      Train a forest or neural network emulator.
    plot-example-profiles
                        Plot examples input profiles and parameterized drags.
    plot-example-sources
                        Plot example AD99 source spectra.
    plot-R2-scores      Plot training and test R2 scores by level and latitude.
    save-shapley-values
                        Compute and save Shapley values for later plotting.
    plot-feature-importances
                        Plot Shapley values and potentially Gini importances.
    plot-scalar-importances
                        Plot the importance of surface pressure and latitude.
    plot-shapley-errors
                        Plot average utility of features to reducing emulator error.
    initialize-coupled-run
                        Set up a MiMA run coupled with a data-driven emulator for online testing.
    plot-biases         Plot zonal mean sections from several online runs.
    plot-qbos           Plot quasi-biennial oscillations from one or more MiMA runs.
    plot-qbo-statistics
                        Plot quasi-biennial oscillation statistics from one or more MiMA runs.
    plot-ssw-frequencies
                        Plot SSW frequencies with intelligent grouping.

options:
  -h, --help            show this help message and exit
```
Each option also has its own help message.
```console
$ python -m willow train-emulator -h
usage: __main__.py train-emulator [-h] data-dir model-dir

Train a forest or neural network emulator.

positional arguments:
  data-dir    Directory where training and test datasets are saved.
  model-dir   Directory where the trained model will be saved. The prefix of separated by a hyphen, is used to determine the kind of model. If the prefix is one of `'mubofo'` or `'random'`, then the appropriate kind of forest will be trained; otherwise, a neural network will be trained and the prefix should be the name of a class defined in `architectures.py`.

options:
  -h, --help  show this help message and exit
```
