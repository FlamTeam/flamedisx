""""xenon resource cache system

Copy-paste from https://github.com/XENONnT/straxen/blob/master/straxen/common.py
"""
from base64 import b32encode
import gzip
from hashlib import sha1
import json
import os
import os.path as osp
import pickle
import urllib.request

import numpy as np
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


# In-memory resource cache
_resource_cache = dict()

# Formats for which the original file is text, not binary
_text_formats = ['text', 'csv', 'json']


@export
def get_resource(x, fmt=None):
    """Return contents of file or URL x
    :param fmt: Format to parse contents into. If not specified, will use the
    file extension (minus dot) as the format

    Do NOT mutate the result you get. Make a copy if you're not sure.
    If you mutate resources it will corrupt the cache, cause terrible bugs in
    unrelated code, tears unnumbered ye shall shed, not even the echo of
    your lamentations shall pass over the mountains, etc.
    """
    if fmt is None:
        fmt = os.path.splitext(x)[1]
        if not fmt:
            raise ValueError(f"Please specify format for {x}")
        fmt = fmt[1:]  # Removes dot

    if x in _resource_cache:
        # Retrieve from in-memory cache
        return _resource_cache[x]

    if '://' in x:
        # Web resource; look first in on-disk cache
        # to prevent repeated downloads.
        cache_fn = deterministic_hash(x)
        cache_folders = ['./resource_cache',
                         '/tmp/straxen_resource_cache',
                         '/dali/lgrandi/strax/resource_cache']
        for cache_folder in cache_folders:
            try:
                os.makedirs(cache_folder, exist_ok=True)
            except (PermissionError, OSError):
                continue
            cf = osp.join(cache_folder, cache_fn)
            if osp.exists(cf):
                result = get_resource(cf, fmt=fmt)
                break
        else:
            print(f'Did not find {cache_fn} in cache, downloading {x}')
            result = urllib.request.urlopen(x).read()
            is_binary = fmt not in _text_formats
            if not is_binary:
                result = result.decode()

            # Store in as many caches as possible
            m = 'wb' if is_binary else 'w'
            available_cf = None
            for cache_folder in cache_folders:
                if not osp.exists(cache_folder):
                    continue
                cf = osp.join(cache_folder, cache_fn)
                try:
                    with open(cf, mode=m) as f:
                        f.write(result)
                except Exception:
                    pass
                else:
                    available_cf = cf
            if available_cf is None:
                raise RuntimeError(
                    f"Could not store {x} in on-disk cache,"
                    "none of the cache directories are writeable??")

            # Retrieve result from file-cache
            # (so we only need one format-parsing logic)
            result = get_resource(available_cf, fmt=fmt)

    else:
        # File resource
        if fmt in ['npy', 'npy_pickle']:
            result = np.load(x, allow_pickle=fmt == 'npy_pickle')
            if isinstance(result, np.lib.npyio.NpzFile):
                # Slurp the arrays in the file, so the result can be copied,
                # then close the file so its descriptors does not leak.
                result_slurped = {k: v[:] for k, v in result.items()}
                result.close()
                result = result_slurped
        elif fmt == 'pkl':
            with open(x, 'rb') as f:
                result = pickle.load(f)
        elif fmt == 'pkl.gz':
            with gzip.open(x, 'rb') as f:
                result = pickle.load(f)
        elif fmt == 'json.gz':
            with gzip.open(x, 'rb') as f:
                result = json.load(f)
        elif fmt == 'json':
            with open(x, mode='r') as f:
                result = json.load(f)
        elif fmt == 'binary':
            with open(x, mode='rb') as f:
                result = f.read()
        elif fmt == 'text':
            with open(x, mode='r') as f:
                result = f.read()
        elif fmt == 'csv':
            result = pd.read_csv(x)
        else:
            raise ValueError(f"Unsupported format {fmt}!")

    # Store in in-memory cache
    _resource_cache[x] = result

    return result


@export
def hashablize(obj):
    """Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    """
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, '__iter__'):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


@export
def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing
    a container hierarchy
    """
    digest = sha1(json.dumps(hashablize(thing)).encode('ascii')).digest()
    return b32encode(digest)[:length].decode('ascii').lower()
