import logging

import os
import errno
from functools import wraps
from hashlib import md5

import numpy as np

CACHE_DIR = "./_cache/"

# ensure cache directory exists
try:
    os.makedirs(CACHE_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


def cache_numpy_result(read_cache, write_cache=True, force_update=False, hash_method='readable'):
    """
    Cache any function that has hashable (or string representable) arguments and returns a numpy object
    :param read_cache: Enable caching of function with this decorator
    :param write_cache: Create a cached result if none exists and read_cache is enabled
    :param force_update: Force a cache update (ignores previously cached results)
    :param hash_method: Either 'hash', 'readable', or a callable object to return a hash-like representation of the
            functions args and kwargs. Default: 'readable'
    :return: The function result, cached if use_cache is enabled
    """

    valid_hash_funcs = {
        'hash': lambda args, kwargs: md5(str(
            (args, frozenset(kwargs.items()))
        ).encode()).hexdigest(),
        'readable': lambda args, kwargs: '_'.join(
            tuple(str(x) for x in args) +
            tuple('_'.join(str(y) for y in kw) for kw in kwargs.items())
        )
    }
    try:
        hash_func = valid_hash_funcs[hash_method]
    except KeyError:
        msg = "hash_method argument value must be one of {}".format(', '.join(valid_hash_funcs.keys()))
        raise ValueError(msg) from None

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not read_cache:
                # Don't cache anything
                return func(*args, **kwargs)
            # create cache file path
            hashkey = '{}_{}.npy'.format(func.__name__, hash_func(args, kwargs))
            cachepath = os.path.join(CACHE_DIR, hashkey)

            logging.debug("Caching key {}".format(hashkey))
            if force_update:
                logging.debug("Forcing update")
                result = func(*args, **kwargs)
                np.save(cachepath, result)
            else:
                try:
                    result = np.load(cachepath)
                    logging.debug("Found in cache")
                except (IOError, FileNotFoundError) as e:
                    logging.debug("Not found in cache")
                    result = func(*args, **kwargs)
                    if write_cache:
                        # logging.debug("Writing to cache")
                        np.save(cachepath, result)
            return result
        return wrapper

    return decorator
