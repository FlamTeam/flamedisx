import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# Remove once tf.repeat is available in the tf api
from tensorflow.python.ops.ragged.ragged_util import repeat  # yes, it IS used!

o = tf.newaxis
FLOAT_TYPE = tf.float32


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
__all__ += ['float_type', 'exporter', 'repeat']


@export
def float_type():
    return FLOAT_TYPE


@export
def lookup_axis1(x, indices, fill_value=0):
    """Return values of x at indices along axis 1,
       returning fill_value for out-of-range indices.
    """
    # Save shape of x and flatten
    a, b = x.shape
    x = tf.reshape(x, [-1])

    legal_index = indices < b

    # Convert indices to legal indices in flat array
    indices = tf.dtypes.cast(indices, dtype=tf.int32)
    indices = tf.clip_by_value(indices, 0, b - 1)
    indices = indices + b * tf.range(a)[:, o, o]

    # Do indexing
    result = tf.reshape(tf.gather(x,
                                  tf.reshape(indices, shape=(-1,))),
                        shape=indices.shape)

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

    Implemented using Dirichlet Multinomial distribution which is
    identically the Beta-Binomial distribution when len(beta_pars) == 2
    """
    # TODO: check if the number of successes wasn't reversed in the original
    # code. Should we have [x, n-x] or [n-x, x]?

    beta_pars = tf.stack(beta_params(p_mean, p_sigma), axis=-1)

    # DirichletMultinomial only gives correct output on float64 tensors!
    # Cast inputs to float64 explicitly!
    beta_pars = tf.cast(beta_pars, dtype=tf.float64)
    x = tf.cast(x, dtype=tf.float64)
    n = tf.cast(n, dtype=tf.float64)

    counts = tf.stack([x, n-x], axis=-1)
    res = tfp.distributions.DirichletMultinomial(
        n,
        beta_pars,
        # validate_args=True,
        # allow_nan_stats=False
        ).prob(counts)
    res = tf.cast(res, dtype=float_type())
    return tf.where(tf.math.is_finite(res),
                    res,
                    tf.zeros_like(res, dtype=float_type()))