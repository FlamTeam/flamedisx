import subprocess

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
# Remove once tf.repeat is available in the tf api
from tensorflow.python.ops.ragged.ragged_util import repeat  # yes, it IS used!
lgamma = tf.math.lgamma

o = tf.newaxis
FLOAT_TYPE = tf.float32
INT_TYPE = tf.int32

# Maximum p_electron and
# Minimum p_electron probability fluctuation
# Lower than this, numerical instabilities will occur in the
# beta-binom pmf.
MAX_MEAN_P = 0.95
MIN_FLUCTUATION_P = 0.005


def exporter():
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    def decorator(obj):
        all_.append(obj.__name__)
        return obj
    return decorator, all_


export, __all__ = exporter()
__all__ += ['float_type', 'exporter', 'repeat',
            'MIN_FLUCTUATION_P', 'MAX_MEAN_P']


@export
def float_type():
    return FLOAT_TYPE


@export
def int_type():
    return INT_TYPE


@export
def lookup_axis1(x, indices, fill_value=0):
    """Return values of x at indices along axis 1,
       returning fill_value for out-of-range indices.
    """
    # Save shape of x and flatten
    ind_shape = indices.shape
    a, b = x.shape
    x = tf.reshape(x, [-1])

    legal_index = indices < b

    # Convert indices to legal indices in flat array
    indices = tf.clip_by_value(indices, 0., b - 1.)
    indices = indices + b * tf.range(a, dtype=float_type())[:, o, o]
    indices = tf.reshape(indices, shape=(-1,))
    indices = tf.dtypes.cast(indices, dtype=int_type())

    # Do indexing
    result = tf.reshape(tf.gather(x,
                                  indices),
                        shape=ind_shape)

    # Replace illegal indices with fill_value, cast to float explicitly
    return tf.cast(tf.where(legal_index,
                            result,
                            tf.zeros_like(result) + fill_value),
                   dtype=float_type())


@export
def tf_to_np(x):
    """Convert (list/tuple of) tensors x to numpy"""
    if isinstance(x, (list, tuple)):
        return tuple([tf_to_np(y) for y in x])
    if isinstance(x, np.ndarray):
        return x
    return x.numpy()


@export
def np_to_tf(x):
    """Convert (list/tuple of) arrays x to tensorflow"""
    if isinstance(x, (list, tuple)):
        return tuple([np_to_tf(y) for y in x])
    if isinstance(x, tf.Tensor):
        return x
    return tf.convert_to_tensor(x, dtype=float_type())

@export
def cart_to_pol(x, y):
    return (x**2 + y**2)**0.5, np.arctan2(y, x)

@export
def pol_to_cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

@export
def tf_log10(x):
    return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))


@export
def safe_p(ps):
    """Clip probabilities to be in [1e-5, 1 - 1e-5]
    NaNs are replaced by 1e-5.
    """
    ps = tf.where(tf.math.is_nan(ps),
                  tf.zeros_like(ps, dtype=float_type()),
                  tf.cast(ps, dtype=float_type()))
    ps = tf.clip_by_value(ps, 1e-5, 1 - 1e-5)
    return ps


@export
def beta_params(mean, sigma):
    """Convert (p_mean, p_sigma) to (alpha, beta) params of beta distribution
    """
    # From Wikipedia:
    # variance = 1/(4 * (2 * beta + 1)) = 1/(8 * beta + 4)
    # mean = 1/(1+beta/alpha)
    # =>
    # beta = (1/variance - 4) / 8
    # alpha
    b = (1. / (8. * sigma ** 2) - 0.5)
    a = b * mean / (1. - mean)
    return a, b


@export
def beta_binom_pmf(x, n, p_mean, p_sigma):
    """Return probability mass function of beta-binomial distribution.

    That is, give the probability of obtaining x successes in n trials,
    if the success probability p is drawn from a beta distribution
    with mean p_mean and standard deviation p_sigma.
    """
    # Avoid numerical instabilities
    # TODO: is there a better way?
    p_mean = tf.clip_by_value(p_mean, 0., MAX_MEAN_P)
    p_sigma = tf.clip_by_value(p_sigma, MIN_FLUCTUATION_P, 1.)

    a, b = beta_params(p_mean, p_sigma)
    res = tf.exp(
        lgamma(n + 1.) + lgamma(x + a) + lgamma(n - x + b)
        + lgamma(a + b)
        - (lgamma(x + 1.) + lgamma(n - x + 1.)
           + lgamma(a) + lgamma(b) + lgamma(n + a + b)))
    return tf.where(tf.math.is_finite(res),
                    res,
                    tf.zeros_like(res, dtype=float_type()))


@export
def is_numpy_number(x):
    try:
        return (np.issubdtype(x.dtype, np.integer)
                or np.issubdtype(x.dtype, np.floating))
    except (AttributeError, TypeError):
        return False


@export
def symmetrize_matrix(x):
    upper = tf.linalg.band_part(x, 0, -1)
    diag = tf.linalg.band_part(x, 0, 0)
    return (upper - diag) + tf.transpose(upper)


@export
def j2000_to_event_time(dates):
    """Convert a numpy array of j2000 timestamps to event_times
    which are ns unix timestamps. This is the reverse of wimprates.j2000
    """
    zero = pd.to_datetime('2000-01-01T12:00')
    nanoseconds_per_day = 1e9 * 3600 * 24
    return nanoseconds_per_day * dates + zero.value


@export
def index_lookup_dict(names):
    return dict(zip(
        names,
        [tf.constant(i, dtype=int_type())
         for i in range(len(names))]))


@export
def values_to_constants(kwargs):
    for k, v in kwargs.items():
        if isinstance(v, (float, int)) or is_numpy_number(v):
            kwargs[k] = tf.constant(v, dtype=float_type())
    return kwargs


@export
def wilks_crit(confidence_level):
    """Return critical value from Wilks' theorem for upper limits"""
    return stats.norm.ppf(confidence_level) ** 2


@export
def run_command(command):
    """Run command and show its output in STDOUT"""
    # Is there no easier way??
    with subprocess.Popen(
            command.split(),
            bufsize=1,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT) as p:
        for line in iter(p.stdout.readline, ''):
            print(line.rstrip())
