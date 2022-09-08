from pathlib import Path
import subprocess

import inspect
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

lgamma = tf.math.lgamma
o = tf.newaxis
FLOAT_TYPE = tf.float32
INT_TYPE = tf.int32

# Extreme mean and standard deviations give numerical errors
# in the beta-binomial.
MAX_MEAN_P = 0.95            # issue #36
MIN_FLUCTUATION_P = 0.005    # issue #36
MIN_MEAN_P = 0.011           # issue #83. Adjust if changing MIN_FLUCTUATION_P!
# The MAX_FLUCTUATION_P depends on the mean, see issue #83.


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
__all__.extend([
    'float_type', 'exporter',
    'MIN_FLUCTUATION_P', 'MAX_MEAN_P'])


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
    ind_shape = tf.shape(indices)
    a = tf.shape(x)[0]
    b = tf.shape(x)[1]
    x = tf.reshape(x, [-1])

    legal_index = tf.cast(indices, dtype=int_type()) < b

    # Convert indices to legal indices in flat array
    indices = tf.clip_by_value(indices, 0., tf.cast(b, dtype=float_type())-1.)
    indices = indices + tf.cast(b, dtype=float_type()) * \
        tf.range(a, dtype=float_type())[:, o, o]
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
    if isinstance(x, pd.Series):
        x = x.values
    elif isinstance(x, pd.DataFrame):
        raise ValueError("Cannot convert pd.DataFrame's to tensors!")
    elif isinstance(x, (list, tuple)):
        return tuple([np_to_tf(y) for y in x])
    elif isinstance(x, tf.Tensor):
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
def beta_params(mean, sigma, force_valid=True):
    """Convert (p_mean, p_sigma) to (alpha, beta) params of beta distribution

    :param force_valid: If true, adjust values to give valid, stable, and
      unimodal beta distributions. See issues #36 and #83
    """
    m, v = mean, sigma ** 2
    m, v = np_to_tf(m), np_to_tf(v)  # We're called from numpy sometimes

    if force_valid:
        m = tf.clip_by_value(m, MIN_MEAN_P, MAX_MEAN_P)
        max_v = tf.maximum(
            (m-1)**2 * m / (2-m),
            m**2 * (1-m)/(1+m))
        v = tf.clip_by_value(v, MIN_FLUCTUATION_P**2, max_v)

    a = m * (m/v - m**2/v - 1.)
    b = a * (1/m - 1)
    return a, b


@export
def beta_binom_pmf(x, n, p_mean, p_sigma):
    """Return probability mass function of beta-binomial distribution.

    That is, give the probability of obtaining x successes in n trials,
    if the success probability p is drawn from a beta distribution
    with mean p_mean and standard deviation p_sigma.
    """
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
def index_lookup_dict(names, column_widths=None):
    """Return dictionary mapping names to successive tensor indices
     (tf.constant integers.)

    :param column_widths: dictionary mapping names to column width.
        For columns with width > 1, the result contains a tensor slice.
    """
    names = list(names)
    if column_widths is None:
        column_widths = dict()

    result = dict()
    i = 0
    while names:
        name = names.pop(0)
        width = column_widths.get(name, 1)
        if width == 1:
            result[name] = tf.constant(i, dtype=int_type())
        else:
            result[name] = slice(tf.constant(i, dtype=int_type()),
                                 tf.constant(i + width, dtype=int_type()))
        i += width

    return result


@export
def values_to_constants(kwargs):
    """Return dictionary with python/numpy values replaced by tf.constant"""
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
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT) as p:
        for line in iter(p.stdout.readline, ''):
            print(line.rstrip())


@export
def load_config(config_files=None):
    """Return dictionary of configuration options from (a) python file(s)"""
    if config_files is None:
        return {}
    if isinstance(config_files, str):
        # Support one or multiple files
        config_files = (config_files,)

    config = dict()
    for filename in config_files:
        if not filename.endswith('.py'):
            # This is (hopefully) a named config shipped with flamedisx
            filename = Path(__file__).parent / 'configs' / (filename + '.py')

        # Adapted from https://stackoverflow.com/a/37611448
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
        captured_locals = dict()
        exec(code, globals(), captured_locals)
        config.update({
            k: v for k, v in captured_locals.items()
            if not k.startswith('_')})

    return config


# Taken from straxen to filter arguments for interpolators
@export
def filter_kwargs(func, kwargs):
    """Filter out keyword arguments that
        are not in the call signature of func
        and return filtered kwargs dictionary
    """
    params = inspect.signature(func).parameters
    if any([str(p).startswith('**') for p in params.values()]):
        # if func accepts wildcard kwargs, return all
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
