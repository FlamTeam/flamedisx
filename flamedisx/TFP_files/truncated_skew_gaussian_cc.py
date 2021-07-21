# Copyright 2021 Robert James
# ============================================================================
"""The Skew Gaussian distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

import functools

import scipy.special

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import skew_gaussian
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'TruncatedSkewGaussianCC',
]


class TruncatedSkewGaussianCC(distribution.Distribution):
  """The Skew Gaussian distribution with `loc`, `scale` and `skewness` parameters.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z * (1 + erf(alpha * (x - mu) / (sigma * sqrt(2))))
  Z = (2 pi sigma**2)**0.5
  ```

  where `loc = mu` is the mean, `scale = sigma` is the std. deviation, `skewness = alpha` is the skewness, and, `Z`
  is the normalization constant.

  """

  def __init__(self,
               loc,
               scale,
               skewness,
               limit,
               validate_args=False,
               allow_nan_stats=True,
               name='TruncatedSkewGaussianCC'):
    """Construct Skew Gaussian distributions with mean, stddev and skewness `loc`, `scale` and `skewness`.

    The parameters must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
      skewness: Floating point tensor; the skewness of the distribution(s).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc`, `scale` or `skewness` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, skewness, limit], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, dtype=dtype, name='skewness')
      self._limit = tensor_util.convert_nonref_to_tensor(
          limit, dtype=dtype, name='limit')
      super(TruncatedSkewGaussianCC, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale', 'skewness', 'limit'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 4)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0, skewness=0, limit=0)

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
    return self._scale

  @property
  def skewness(self):
    """Distribution parameter for skewness."""
    return self._skewness

  @property
  def limit(self):
    """"""
    return self._limit

  def _batch_shape_tensor(self, loc=None, scale=None, skewness=None, limit=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(self.loc if loc is None else loc),
        prefer_static.shape(self.scale if scale is None else scale),
        prefer_static.shape(self.skewness if skewness is None else skewness),
        prefer_static.shape(self.limit if limit is None else limit))

  def _batch_shape(self):
    return functools.reduce(tf.broadcast_static_shape, (
        self.loc.shape, self.scale.shape, self.skewness.shape, self.limit.shape))

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    limit = tf.convert_to_tensor(self.limit)
    bounded_log_prob = tf.where((x > limit),
                                dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
                                tf.math.log(skew_gaussian.SkewGaussian(loc=self.loc,scale=scale,skewness=skewness).cdf(x+0.5) \
                                - skew_gaussian.SkewGaussian(loc=self.loc,scale=scale,skewness=skewness).cdf(x-0.5)))
    bounded_log_prob = tf.where(tf.math.is_nan(bounded_log_prob),
                                dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
                                bounded_log_prob)
    dumping_log_prob = tf.where((x == limit),
                                tf.math.log(1 - skew_gaussian.SkewGaussian(loc=self.loc,scale=scale,skewness=skewness).cdf(x-0.5)),
                                bounded_log_prob)
    return dumping_log_prob

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `loc`, `scale`, `skewness` and `limit` must have compatible shapes; '
            'loc.shape={}, scale.shape={}, skewness.shape={}, limit.shape={}.'.format(
                self.loc.shape, self.scale.shape, self.skewness.shape, self.limit.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access both arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))

    return assertions
