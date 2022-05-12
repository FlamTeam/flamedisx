"""
Routines for estimating the total expected events
and its variation with parameters.
"""

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd

export, __all__ = fd.exporter()


@export
class MuEstimator:

    n_trials = int(1e5)  # Number of trials per mu simulation
    progress = True      # Whether to show progress bar during building

    def __init__(
            self,
            source: fd.Source,
            n_trials=None,
            progress=None,
            **param_specs):
        if n_trials is not None:
            self.n_trials = n_trials
        if progress is not None:
            self.progress = progress

        # Extract bounds and options (like n_anchors) from param_specs
        self.bounds = dict()
        self.options = dict()
        for pname, spec in param_specs.items():
            assert len(spec) in (2, 3)
            # Explicit type casting here, to avoid confusing tensorflow device
            # placement messages if users mix up ints and floats
            self.bounds[pname] = float(spec[0]), float(spec[1])
            if len(spec) == 3:
                if isinstance(spec[2], dict):
                    opts = spec[2]
                else:
                    # Backwards compatibility: the third element used to be
                    # number of anchors
                    opts = dict(n_anchors=spec[2])
                self.options[pname] = opts

        # Remove parameters the source does not take
        # Consistent with Source.__init__, don't complain / discard silently.
        # MuEstimator.__call__, however, expects to be called with filtered params
        param_specs = {k: v for k, v in param_specs.items() if k in source.defaults}

        # Build the necessary interpolators
        self.build(source)

    def build(self, source: fd.Source):
        raise NotImplementedError

    def __call__(self, **params):
        raise NotImplementedError


@export
class CrossInterpolator(MuEstimator):

    def build(self, source: fd.Source):
        # Estimate mu under the current defaults
        self.base_mu = tf.constant(
            source.estimate_mu(n_trials=self.n_trials),
            dtype=fd.float_type())

        # Estimate mu variation along each direction
        self.mus = dict()   # parameter -> tensor of mus along anchors
        _iter = self.bounds.items()
        if self.progress:
            _iter = tqdm(_iter, desc="Estimating mus")
        for pname, (start, stop) in _iter:
            n_anchors = int(self.options.get(pname, {}).get('n_anchors', 2))
            self.mus[pname] = tf.convert_to_tensor(
                 [source.estimate_mu(**{pname: x}, n_trials=self.n_trials)
                  for x in np.linspace(start, stop, n_anchors)],
                 dtype=fd.float_type())

    def __call__(self, **kwargs):
        mu = self.base_mu
        for pname, v in kwargs.items():
            mu *= tfp.math.interp_regular_1d_grid(
                x=v,
                x_ref_min=self.bounds[pname][0],
                x_ref_max=self.bounds[pname][1],
                y_ref=self.mus[pname]) / self.base_mu
        return mu


@export
class SimulateEachCall(MuEstimator):
    """Estimate the mu with a new simulation __every single call__!

    Use for debugging / if you know what you are getting into...
    """

    def build(self, source: fd.Source):
        self.source = source

    def __call__(self, **kwargs):
        return self.source.estimate_mu(**kwargs)
