import numpy as np
import tensorflow as tf

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
__all__ += ['float_type', 'exporter']


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
