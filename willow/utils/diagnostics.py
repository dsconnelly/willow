import functools
import logging
import os
import sys
import time

def logs(func):
    @functools.wraps(func)
    def func_with_logging(*args, **kwargs):
        model_dir = kwargs['model_dir']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        logging.basicConfig(
            filename=os.path.join(model_dir, 'log.out'),
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
    @functools.wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        runtime = time.time() - start
        
        logging.info(f'{func.__name__} took {runtime:.2f} seconds.')
        
        return output
    
    return timed_func
