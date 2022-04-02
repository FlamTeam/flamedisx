# Copyright 2021 Robert James
# ============================================================================
"""The discretised Truncated Skew Gaussian distribution, with continuity correction, class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

import functools

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util

import flamedisx as fd

export, __all__ = fd.exporter()


@export
class TruncatedSkewGaussianCC(distribution.Distribution):
  """

  """

  def __init__(self,
               loc,
               scale,
               skewness,
               limit,
               owens_t_terms=2,
               validate_args=False,
               allow_nan_stats=True,
               name='TruncatedSkewGaussianCC'):
    """Construct Truncated Skew Gaussian distributions with mean, stddev and skewness `loc`, `scale` and `skewness`.
    Distribition is truncated at `limit`.
    Designed to be used for discrete random variables, with an in-built continuity correction.

    The parameters must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
      skewness: Floating point tensor; the skewness of the distribution(s).
      limit: Floating point tensor; the point above which all probability
        mass is zero-ed out and re-dumped into the the probability mass of
        limit.
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
      TypeError: if `loc`, `scale`, `skewness` or `limit` have different `dtype`.
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
      self.owens_t_terms = owens_t_terms
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
    """Distribution parameter for limit"""
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
    skew_gauss = fd.tfp_files.SkewGaussian(loc=self.loc,scale=scale,skewness=skewness,owens_t_terms=self.owens_t_terms)

    cdf_upper = skew_gauss.cdf(x+0.5)
    cdf_lower = skew_gauss.cdf(x-0.5)

    minus_inf = dtype_util.as_numpy_dtype(x.dtype)(-np.inf)

    bounded_log_prob = tf.where((x > limit),
                                minus_inf,
                                tf.math.log(cdf_upper - cdf_lower))
    bounded_log_prob = tf.where(tf.math.is_nan(bounded_log_prob),
                                minus_inf,
                                bounded_log_prob)
    dumping_log_prob = tf.where((x == limit),
                                tf.math.log(1 - cdf_lower),
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
