"""
Routines for estimating the total expected events
and its variation with parameters.
"""
from functools import partial

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
    options: dict
    bounds: dict
    param_options: dict  # dict param -> dict of options per parameter

    def __init__(
            self,
            source: fd.Source,
            n_trials=None,
            progress=None,
            options=None,
            **param_specs):
        if n_trials is not None:
            self.n_trials = n_trials
        if progress is not None:
            self.progress = progress
        if options is None:
            options = dict()
        self.options = options

        # Extract bounds and options (like n_anchors) from param_specs
        self.bounds = dict()
        self.param_options = dict()
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
                self.param_options[pname] = opts

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
class CrossInterpolatedMu(MuEstimator):
    """Build piecewise-linear estimates of the relative change in mu along
    each single parameter, then multiply the relative changes.
    """

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
            n_anchors = int(self.param_options.get(pname, {}).get('n_anchors', 2))
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
class SimulateEachCallMu(MuEstimator):
    """Estimate the expected number of events with a new simulation
    __every single call__.

    Use for debugging / if you know what you are getting into...
    """

    def build(self, source: fd.Source):
        self.source = source

    def __call__(self, **kwargs):
        return self.source.estimate_mu(**kwargs)


@export
class ConstantMu(MuEstimator):
    """Assume the expected number of events does not depend
    on the fitted parameters
    """
    def build(self, source: fd.Source):
        self.mu = source.estimate_mu(n_trials=self.n_trials)

    def __call__(self, **kwargs):
        return self.mu


@export
class CombinedMu(MuEstimator):
    """Combine the results of different mu estimators for different subsets
    of parameters. The final mu is the product of all relative changes.

    Use e.g. as

        est = fd.CombinedMu.from_estimators(
            {
                ('elife', 'g1'): fd.GridInterpolatedMu,
                ('g2', 'single_electron_size'): fd.GridInterpolatedMu,
            },
            default=fd.CrossInterpolatedMu)
        fd.LogLikelihood(..., mu_estimators=est)

    This will setup two (2d) grid interpolators, and use cross interpolation
    for the remaining parameters.
    """
    @classmethod
    def from_estimators(cls, spec_dict, default=CrossInterpolatedMu):
        return partial(cls, options=dict(spec_dict=spec_dict, default=default))

    def build(self, source: fd.Source):
        # Check the sub_estimators option is specified correctly
        # Note we copy self.options['spec_dict'], since we mutate it later
        assert self.options.get('spec_dict'), "Use CombinedMu.from_estimators"
        spec_dict = {**self.options['spec_dict']}
        params_specified = sum([spec['params'] for spec in spec_dict])
        if len(set(params_specified)) != len(params_specified):
            raise ValueError("CombinedMu specification duplicates parameters")

        # Add a specification for the default estimator
        missing_params = set(self.bounds.keys()) - set(params_specified)
        if missing_params:
            spec_dict[tuple(missing_params)] = self.options['default']

        # Build sub-estimators (for different parameter sets)
        self.estimators = []
        for spec, est in spec_dict.items():
            if isinstance(est, fd.MuEstimator):
                # Already-initialized estimator, e.g. loaded from pickle
                self.estimators.append(est)
                continue
            # Have to build this sub-estimator -- collect its options
            if isinstance(est, dict):
                # We got a dictionary specifying estimator class and options
                est_class = est['class']
                est_options = est.get('options', dict())
                n_trials = est.get('n_trials', self.n_trials)
                progress = est.get('progress', self.progress)
            elif issubclass(est, fd.MuEstimator):
                # We just got a class name -- use default options
                est_class = est
                est_options = est.get('options', dict())
                n_trials = self.n_trials
                progress = self.progress
            else:
                raise ValueError(f"Can't build mu estimator for {spec},"
                                 f" {est} is not a mu estimator?")

            # Create pname -> (min, max, options) dict for this estimator
            param_specs = {
                pname: self.bounds[pname] + (self.param_options[pname],)
                for pname in spec['params']}
            # Finally, build the estimator
            self.estimators.append(est_class)(
                source=self.source,
                n_trials=n_trials,
                progress=progress,
                options=est_options,
                **param_specs
            )

        # Get base mus: mus estimated at the default values by the different
        # estimators
        self.base_mus = [e() for e in self.estimators]
        # Compute the mean. TODO: weight appropriately if n_trials varies?
        self.mean_base_mu = float(np.mean(self.base_mus))

    def __call__(self, **kwargs):
        # Predicted mus by each estimator
        pred_mus = [
            est(**{k: v for k, v in kwargs.items() if k in spec['params']})
            for est, spec in self.options['sub_estimator_specs'].items()]
        # Relative increase over base mu
        pred_increase = [
            pred_mu / base_mu
            for pred_mu, base_mu in zip(pred_mus, self.base_mus)
        ]
        # Multiply the mean base mu by all the relative increases
        # to get the combined mu estimate
        return self.mean_base_mu * tf.math.reduce_prod(pred_increase)


@export
class GridInterpolatedMu:
    """Linearly interpolate the estimated mu on an n-dimensional grid"""

    def build(self, source: fd.Source):
        raise NotImplementedError
