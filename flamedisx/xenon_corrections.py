from base64 import b32encode
import logging
import gzip
from hashlib import sha1
import json
import os
import os.path as osp
import re
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

import flamedisx as fd
export, __all__ = fd.exporter()


@export
class InterpolateAndExtrapolate(object):
    """Linearly interpolate- and extrapolate using inverse-distance
    weighted averaging between nearby points.
    """

    def __init__(self, points, values, neighbours_to_use=None):
        """
        :param points: array (n_points, n_dims) of coordinates
        :param values: array (n_points) of values
        :param neighbours_to_use: Number of neighbouring points to use for
        averaging. Default is 2 * dimensions of points.
        """
        self.kdtree = cKDTree(points)
        self.values = values
        if neighbours_to_use is None:
            neighbours_to_use = points.shape[1] * 2
        self.neighbours_to_use = neighbours_to_use

    def __call__(self, points):
        distances, indices = self.kdtree.query(points, self.neighbours_to_use)
        # If one of the coordinates is NaN, the neighbour-query fails.
        # If we don't filter these out, it would result in an IndexError
        # as the kdtree returns an invalid index if it can't find neighbours.
        result = np.ones(len(points)) * float('nan')
        valid = (distances < float('inf')).max(axis=-1)
        result[valid] = np.average(
            self.values[indices[valid]],
            weights=1/np.clip(distances[valid], 1e-6, float('inf')),
            axis=-1)
        return result


@export
class InterpolatingMap(object):
    """Correction map that computes values using inverse-weighted distance
    interpolation.

    The map must be specified as a json translating to a dictionary like this:
        'coordinate_system' :   [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...],
        'map' :                 [value1, value2, value3, value4, ...]
        'another_map' :         idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, etc',
        'timestamp':            unix epoch seconds timestamp

    with the straightforward generalization to 1d and 3d.

    The default map name is 'map', I'd recommend you use that.

    For a 0d placeholder map, use
        'points': [],
        'map': 42,
        etc

    """
    data_field_names = ['timestamp', 'description', 'coordinate_system',
                        'name', 'irregular']

    def __init__(self, data):
        self.log = logging.getLogger('InterpolatingMap')

        if isinstance(data, bytes):
            data = gzip.decompress(data).decode()
        self.data = json.loads(data)

        self.coordinate_system = cs = self.data['coordinate_system']
        if not len(cs):
            self.dimensions = 0
        else:
            self.dimensions = len(cs[0])
        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys()
                                 if k not in self.data_field_names])
        self.log.debug('Map name: %s' % self.data['name'])
        self.log.debug('Map description:\n    ' +
                       re.sub(r'\n', r'\n    ', self.data['description']))
        self.log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            map_data = np.array(self.data[map_name])
            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments
                # and always return a single value
                def itp_fun(positions):
                    return map_data * np.ones_like(positions)
            else:
                itp_fun = InterpolateAndExtrapolate(points=np.array(cs),
                                                    values=np.array(map_data))

            self.interpolators[map_name] = itp_fun

    def __call__(self, positions, map_name='map'):
        """Returns the value of the map at the position given by coordinates
        :param positions: array (n_dim) or (n_points, n_dim) of positions
        :param map_name: Name of the map to use. Default is 'map'.
        """
        return self.interpolators[map_name](positions)


cache_dict = dict()


@export
def get_resource(x, fmt='text'):
    """Return contents of file or URL x
    :param binary: Resource is binary. Return bytes instead of a string.
    """
    is_binary = fmt != 'text'

    # Try to retrieve from in-memory cache
    if x in cache_dict:
        return cache_dict[x]

    if '://' in x:
        # Web resource; look first in on-disk cache
        # to prevent repeated downloads.
        cache_fn = deterministic_hash(x)
        cache_folders = ['./resource_cache',
                         '/tmp/straxen_resource_cache',
                         '/project2/lgrandi/strax_resource_cache',
                         '/dali/lgrandi/strax/resource_cache']
        for cache_folder in cache_folders:
            try:
                os.makedirs(cache_folder, exist_ok=True)
            except (PermissionError, OSError):
                continue
            cf = osp.join(cache_folder, cache_fn)
            if osp.exists(cf):
                return get_resource(cf, fmt=fmt)

        # Not found in any cache; download
        result = urllib.request.urlopen(x).read()
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
                f"Could not load {x},"
                "none of the on-disk caches are writeable??")

        # Retrieve result from file-cache
        # (so we only need one format-parsing logic)
        return get_resource(available_cf, fmt=fmt)

    # File resource
    if fmt == 'npy':
        result = np.load(x)
    elif fmt == 'binary':
        with open(x, mode='rb') as f:
            result = f.read()
    elif fmt == 'text':
        with open(x, mode='r') as f:
            result = f.read()

    # Store in in-memory cache
    cache_dict[x] = result

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


@export
def pax_file(x):
    """Return URL to file hosted in the pax repository master branch"""
    return 'https://raw.githubusercontent.com/XENON1T/pax/master/pax/data/' + x
