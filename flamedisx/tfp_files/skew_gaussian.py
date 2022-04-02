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

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util

import flamedisx as fd

export, __all__ = fd.exporter()


@export
class SkewGaussian(distribution.Distribution):
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
               owens_t_terms=2,
               validate_args=False,
               allow_nan_stats=True,
               name='SkewGaussian'):
    """Construct Skew Gaussian distributions with mean, stddev and skewness `loc`, `scale` and `skewness`.

    The parameters must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
      skewness: Floating point tensor; the skewness of the distribution(s).
      owens_t_terms: Number of terms to use in the expansion of Owen's T function
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
      dtype = dtype_util.common_dtype([loc, scale, skewness], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, dtype=dtype, name='skewness')
      self.owens_t_terms = owens_t_terms
      super(SkewGaussian, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale', 'skewness'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0, skewness=0)

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

  def _batch_shape_tensor(self, loc=None, scale=None, skewness=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(self.loc if loc is None else loc),
        prefer_static.shape(self.scale if scale is None else scale),
        prefer_static.shape(self.skewness if skewness is None else skewness))

  def _batch_shape(self):
    return functools.reduce(tf.broadcast_static_shape, (
        self.loc.shape, self.scale.shape, self.skewness.shape))

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    log_value = 1 + tf.math.erf(skewness/(np.sqrt(2)*scale) * (x - self.loc))
    log_value = tf.where(log_value <= 0, 1e-10, log_value)
    log_unnormalized = -0.5 * tf.math.squared_difference(
        x / scale, self.loc / scale) + tf.math.log(log_value)
    log_normalization = tf.constant(
        0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(scale)
    return log_unnormalized - log_normalization

  @staticmethod
  def owensT1(h, a, terms):
    hs = -0.5 * h * h
    exp_hs = tf.math.exp(hs)

    ci = -1 + exp_hs
    val = tf.math.atan(a) * tf.ones_like(hs)

    for i in range(terms):
        val += ci * tf.math.pow(a,2*tf.cast(i,'float32')+1) / (2*tf.cast(i,'float32')+1)
        ci = -ci + tf.math.pow(hs,tf.cast(i+1,'float32')) / tf.exp(tf.math.lgamma(tf.cast(i+2,'float32'))) * exp_hs

    val = val / (2 * np.pi)

    return val

  def _cdf(self, x):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    h = tf.cast((x - self.loc)/scale,'float32')
    a = tf.cast(skewness,'float32')

    owens_t_eval = 0.5 * normal.Normal(loc=0.,scale=1.).cdf(h) + 0.5 * normal.Normal(loc=0.,scale=1.).cdf(a*h) - normal.Normal(loc=0.,scale=1.).cdf(h) * normal.Normal(loc=0.,scale=1.).cdf(a*h)

    return 0.5 * (1. + tf.math.erf(1./(np.sqrt(2.)*scale) * (x - self.loc))) - \
    tf.cast(tf.where(a > tf.ones_like(a), 2. * (owens_t_eval - self.owensT1(a*h,1./a,self.owens_t_terms)), 2. * self.owensT1(h,a,self.owens_t_terms)),'float32')

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `loc`, `scale` and `skewness` must have compatible shapes; '
            'loc.shape={}, scale.shape={}, skewness.shape={}.'.format(
                self.loc.shape, self.scale.shape, self.skewness.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access both arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))

    return assertions
