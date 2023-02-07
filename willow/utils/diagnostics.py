import functools
import logging
import os
import sys
import time

from typing import Callable

def log(func: Callable) -> Callable:
    """
    Decorate certain functions to enable automatic logging.

    Parameters
    ----------
    func : Function to set up logging for. Should take 'model_dir' as a
        positional argument. The log will be written to a file
        `f'{func.__name__.replace("_", "-")}.log'` in model_dir.

    Returns
    -------
    func_with_logging : Function with the same signature as func that configures
        logging as described above before calling func.

    """

    @functools.wraps(func)
    def func_with_logging(*args, **kwargs):
        try:
            model_dir = kwargs['model_dir']
        except KeyError:
            model_dir = kwargs['model_dirs'][0]

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        fname = func.__name__.replace('_', '-') + '-log.out'
        path = os.path.join(model_dir, fname)

        logging.basicConfig(
            filename=path,
            filemode='w',
            format='%(message)s',
            level=logging.INFO
        )

        def handle(exc_type, exc_value, traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, traceback)
                return

            message = f'{func.__name__} had an uncaught exception:'
            info = (exc_type, exc_value, traceback)
            logging.error(message, exc_info=info)

        sys.excepthook = handle

        return func(*args, **kwargs)

    return func_with_logging

def profile(func: Callable) -> Callable:
    """
    Decorate a function to log its runtime.

    Parameters
    ----------
    func : Function to be timed.
    
    Returns
    -------
    func_with_profiling : Function with the same signature as func that prints
        the runtime before returning the return value of func.

    """

    @functools.wraps(func)
    def func_with_profiling(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        runtime = time.time() - start

        logging.info(f'{func.__name__} took {runtime:.2f} seconds.')

        return output

    return func_with_profiling