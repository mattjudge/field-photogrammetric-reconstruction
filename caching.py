import logging

import os
import errno
from functools import wraps
from hashlib import md5
from zipfile import BadZipFile

import numpy as np

CACHE_DIR = "./_cache/"

# ensure cache directory exists
try:
    os.makedirs(CACHE_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


def _save_numpy(path, values, compress):
    if type(values) is not tuple:
        values = (values,)
    if not all(isinstance(v, np.ndarray) for v in values):
        raise TypeError("Results returned by cached function must all be of type np.ndarray") from None
    if compress:
        return np.savez_compressed(path, *values)
    else:
        return np.savez(path, *values)


def _load_numpy(path):
    npzfile = np.load(path)
    files = npzfile.files
    if len(files) == 1:
        return npzfile[files[0]]
    else:
        # multiple values
        return tuple(npzfile[f] for f in files)


def _func_hash_md5(func, args, kwargs):
    return md5(str(
        (func.__name__, args, frozenset(kwargs.items()))
    ).encode()).hexdigest()


def _func_hash_readable(func, args, kwargs):
    return '{fnm}_{args}_{kwargs}_h{hash}'.format(
        fnm=func.__name__,
        args='_'.join(str(x) for x in args),
        kwargs='_'.join('_'.join(str(y) for y in kw) for kw in kwargs.items()),
        hash=_func_hash_md5(func, args, kwargs)[:6]
    )


def cache_numpy_result(enable_cache, write_cache=True, force_update=False, compress=True, hash_method='hash'):
    """
    Cache any function that has hashable (or string representable) arguments and returns a numpy object
    :param enable_cache: Enable caching of function with this decorator
    :param write_cache: Create a cached result if none exists and enable_cache is enabled
    :param force_update: Force a cache update (ignores previously cached results)
    :param compress: True if the cache should be compressed
    :param hash_method: Either 'hash', or 'readable'. Default: 'hash'
    :return: The function result, cached if use_cache is enabled
    """

    valid_hash_funcs = {
        'hash': _func_hash_md5,
        'readable': _func_hash_readable
    }
    try:
        hash_func = valid_hash_funcs[hash_method]
    except KeyError:
        msg = "hash_method argument value must be one of {}".format(', '.join(valid_hash_funcs.keys()))
        raise ValueError(msg) from None

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enable_cache:
                # Don't cache anything
                return func(*args, **kwargs)
            # create cache file path
            hash_key = '{}.npz'.format(hash_func(func, args, kwargs))
            cache_path = os.path.join(CACHE_DIR, hash_key)

            def run_func_update_cache():
                res = func(*args, **kwargs)
                if force_update or write_cache:
                    # logging.debug("Writing to cache")
                    _save_numpy(cache_path, res, compress)
                return res

            if force_update:
                logging.debug("Cache: Forcing update on {}".format(hash_key))
                result = run_func_update_cache()
            else:
                try:
                    result = _load_numpy(cache_path)
                    logging.debug("Cache: Found {}".format(hash_key))
                except (IOError, FileNotFoundError):
                    logging.debug("Cache: Not found {}".format(hash_key))
                    result = run_func_update_cache()
                except BadZipFile:
                    logging.warning("Cache: Corrupted file, ignoring {}".format(hash_key))
                    result = run_func_update_cache()
            return result
        return wrapper

    return decorator


if __name__ == '__main__':
    pass
    # logging.basicConfig(level=logging.DEBUG)
    #
    # @cache_numpy_result(True, hash_method='readable')
    # def test(a, b):
    #     return np.arange(a), np.arange(b)
    #
    # print(test(4, 5))
    # print(test(5, 4))
    # print(test(4, 5))
