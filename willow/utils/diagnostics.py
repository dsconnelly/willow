import functools
import logging
import os
import sys
import time

def logs(func):
    """
    Decorate functions with certain arguments to enable automatic logging.

    Parameters
    ----------
    func : callable
        The function to set up logging for. Should take 'model_dir' as a
        positional argument.

    Returns
    -------
    func_with_logging : callable
        A function that, before calling func, reads the 'model_dir' positional
        argument and creates a log file in that directory, to which standard
        output and Python tracebacks will be written.

    """

    @functools.wraps(func)
    def func_with_logging(*args, **kwargs):
        model_dir = kwargs['model_dir']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        fname = func.__name__.replace('_', '-') + '-log.out'
        logging.basicConfig(
            filename=os.path.join(model_dir, fname),
            filemode='w',
            format='%(message)s',
            level=logging.INFO
        )
        
        def handle(exc_type, exc_value, traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, traceback)
                return
            
            logging.error(
                f'{func.__name__} had an uncaught exception:',
                exc_info=(exc_type, exc_value, traceback)
            )
            
        sys.excepthook = handle
        
        return func(*args, **kwargs)
    
    return func_with_logging

def times(func):
    """
    Decorate functions to log their runtimes.

    Parameters
    ----------
    func : callable
        The function to be timed.

    Returns
    -------
    timed_func : callable
        A function that checks the time before and after func is called and logs
        a message with the runtime before returning the output of func.
        
    """

    @functools.wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        runtime = time.time() - start
        
        logging.info(f'{func.__name__} took {runtime:.2f} seconds.')
        
        return output
    
    return timed_func
