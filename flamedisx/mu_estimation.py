"""
Routines for estimating the total expected events
and its variation with parameters.
"""
import itertools
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
        kwargs = {param_name: kwargs[param_name] for param_name in self.bounds}

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

    def __call__(self, **params):
        return self.source.estimate_mu(**params)


@export
class ConstantMu(MuEstimator):
    """Assume the expected number of events does not depend
    on the fitted parameters
    """
    def __init__(self, *args, input_mu=None, **kwargs):
        if input_mu is not None:
            self.mu = input_mu
        else:
            self.mu = None

        super().__init__(*args, **kwargs)

    def build(self, source: fd.Source):
        if self.mu is None:
            self.mu = source.estimate_mu(n_trials=self.n_trials)

    def __call__(self, **params):
        result = self.mu
        # Add zero terms so gradient evaluates to 0, not None
        for pname, value in params.items():
            result += 0 * value
        return result


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
        assert 'spec_dict' in self.options, "Use CombinedMu.from_estimators"
        spec_dict = {
            tuple(params) if isinstance(params, (list, tuple)) else (params,): est
            for params, est in self.options['spec_dict'].items()
        }

        params_specified = sum([params for params in spec_dict.keys()], tuple())
        if len(set(params_specified)) != len(params_specified):
            raise ValueError("CombinedMu specification duplicates parameters")

        # Add a specification for the default estimator
        missing_params = set(self.bounds.keys()) - set(params_specified)
        if missing_params:
            spec_dict[tuple(missing_params)] = self.options['default']

        # Build sub-estimators (for different parameter sets)
        self.estimators = dict()
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
            elif is_mu_estimator_class(est):
                # We just got a class; don't pass any options
                est_class = est
                est_options = dict()
                n_trials = self.n_trials
                progress = self.progress
            else:
                raise ValueError(f"Can't build mu estimator for {spec},"
                                 f" {est} is not a mu estimator?")

            # Create pname -> (min, max, options) dict for this estimator
            param_specs = {
                pname: self.bounds[pname] + (self.param_options.get(pname, dict()),)
                for pname in spec}
            # Finally, build the estimator
            self.estimators[spec] = est_class(
                source=source,
                n_trials=n_trials,
                progress=progress,
                options=est_options,
                **param_specs
            )

        # Get base mus: mus estimated at the default values by the different
        # estimators
        self.base_mus = [e(**source.defaults) for e in self.estimators.values()]
        # Compute the mean. TODO: weight appropriately if n_trials varies?
        self.mean_base_mu = float(np.mean(self.base_mus))

    def __call__(self, **kwargs):
        # Predicted mus by each estimator
        pred_mus = [
            est(**{k: v for k, v in kwargs.items() if k in params})
            for params, est in self.estimators.items()]
        # Relative increase over base mu
        pred_increase = [
            pred_mu / base_mu
            for pred_mu, base_mu in zip(pred_mus, self.base_mus)
        ]
        # Multiply the mean base mu by all the relative increases
        # to get the combined mu estimate
        return self.mean_base_mu * tf.math.reduce_prod(pred_increase)


@export
class GridInterpolatedMu(MuEstimator):
    """Linearly interpolate the estimated mu on an n-dimensional grid"""

    def __init__(self, *args, **kwargs):
        if ('n_trials' not in kwargs) or (kwargs['n_trials'] is None):
            kwargs['n_trials'] = int(1e6)

        super().__init__(*args, **kwargs)

    def build(self, source: fd.Source):
        param_lowers = []
        param_uppers = []
        grid_shape = ()
        grid_dict = dict()
        for pname, (start, stop) in self.bounds.items():
            param_lowers.append(start)
            param_uppers.append(stop)
            n_anchors = int(self.param_options.get(pname, {}).get('n_anchors', 3))
            grid_shape += (n_anchors,)
            grid_dict[pname] = np.linspace(start, stop, n_anchors)

        self.param_lowers = fd.np_to_tf(np.asarray(param_lowers))
        self.param_uppers = fd.np_to_tf(np.asarray(param_uppers))

        # Convert dict of anchors to a list of grid points
        # (like sklearn.ParameterGrid)
        keys, values = grid_dict.keys(), grid_dict.values()
        param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if self.progress:
            param_grid = tqdm(param_grid, desc="Estimating mus")

        mu_grid = [
            source.estimate_mu(**params, n_trials=self.n_trials)
            for params in param_grid
        ]
        self.mu_grid = fd.np_to_tf(np.asarray(mu_grid).reshape(grid_shape))

    def __call__(self, **kwargs):
        # Match kwargs order to grid param order
        # (LogLikelihood.mu already filtered params)
        kwargs = {param_name: kwargs[param_name] for param_name in self.bounds}

        return tfp.math.batch_interp_regular_nd_grid(
            [list(kwargs.values())],
            x_ref_min=self.param_lowers,
            x_ref_max=self.param_uppers,
            y_ref=self.mu_grid,
            axis=-len(self.bounds))[0]


@export
def is_mu_estimator_class(x):
    if isinstance(x, partial):
        return is_mu_estimator_class(x.func)
    if type(x) is not type:
        # x is no class (issubclass would crash, annoyingly)
        return False
    return issubclass(x, fd.MuEstimator)
