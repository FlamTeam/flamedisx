import flamedisx as fd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import typing as ty
import scipy
from iminuit import Minuit


export, __all__ = fd.exporter()
__all__ += ['LOWER_RATE_MULTIPLIER_BOUND']

# Setting this to 0 does work, but makes the inference rather slow
# (at least for scipy); probably there is a relative xtol computation,
# which fails when x -> 0.
LOWER_RATE_MULTIPLIER_BOUND = 1e-9

o = tf.newaxis


##
# Objective creation
##

class ObjectiveResult(ty.NamedTuple):
    fun: ty.Union[np.ndarray, tf.Tensor]
    grad: ty.Union[np.ndarray, tf.Tensor]


class Objective:
    """Construct the function that is minimized by the optimizer.
    That function should take one argument x_guess that is a list of values
    each belonging to the parameters being varied.

    :param lf: LogLikelihood object implementing the likelihood to minimize
    :param arg_names: List of parameter names whose values are varied by the
    minimizer
    :param fix: Dict of parameter names and value which are kept fixed (and
    whose gradients are omitted)
    :param numpy_in_out: Converts inputs to tensors and outputs to numpy arrays
    if True, default False
    :param autograph: Use tf.function inside likelihood, default True
    :param nan_val: Value to use if likelihood evaluates to NaN
    :param memoize: Whether to cache values during minimization.
    """
    _cache: dict
    numpy_in_out = False
    memoize = False

    def __init__(self,
                 lf: fd.LogLikelihood,
                 arg_names: ty.List[str],
                 fix: ty.Dict[str, ty.Union[float, tf.constant]],
                 autograph=True,
                 nan_val=float('inf')):
        self.lf = lf
        self.arg_names = arg_names
        self.fix = fix
        self.autograph = autograph
        self.nan_val = nan_val
        self._cache = dict()

    def __call__(self, x):
        """Return (objective, gradient)"""
        if self.memoize:
            assert self.numpy_in_out, "Fix memoization for tensorflow..."
            memkey = tuple(x)
            if memkey in self._cache:
                return self._cache[memkey]

        if self.numpy_in_out:
            x = fd.np_to_tf(x)

        assert len(self.arg_names) == len(x)
        # Build parameter dict, pair arg_names with x_guess values and add
        # fixed pars
        params = {**{self.arg_names[i]: x[i]
                     for i in range(len(x))},
                  **self.fix}

        ll, grad = self._inner_fun_and_grad(params)

        # Check NaNs
        if tf.math.is_nan(ll):
            tf.print(f"Objective at {x} is Nan!")
            ll = tf.constant(self.nan_val, dtype=tf.float32)

        if self.numpy_in_out:
            ll = ll.numpy()
            grad = grad.numpy()
        result = ObjectiveResult(fun=ll, grad=grad)
        if self.memoize:
            self._cache[memkey] = result
        return result

    def _inner_fun_and_grad(self, params):
        # Get -2lnL and its gradient
        return self.lf.minus_ll(
            **params,
            autograph=self.autograph,
            omit_grads=tuple(self.fix.keys()))

    def fun_and_grad(self, x):
        r = self(x)
        return r.fun, r.grad

    def fun(self, x):
        """Return only objective"""
        return self(x).fun

    def grad(self, x):
        """Return only gradient"""
        return self(x).grad

    def absolute_to_relative_tol(self, x_guess, llr_tolerance):
        return abs(llr_tolerance / self.fun(x_guess))

    def prepare_bounds(self, bounds):
        """Return dictionary of {parameter: (left, right)} bounds
        given None or another such dictionary, setting defaults
        for rate multipliers"""
        if bounds is None:
            bounds = dict()
        for p in self.arg_names:
            if p.endswith('_rate_multiplier'):
                bounds.setdefault(p, (LOWER_RATE_MULTIPLIER_BOUND, None))
        return bounds


class ScipyObjective(Objective):
    numpy_in_out = True
    memoize = True

    def minimize(self, x_guess, get_lowlevel_result=False, use_hessian=False,
                 llr_tolerance=None, bounds=None, **kwargs):
        # TODO implement optimizer methods the use hessian
        kwargs.setdefault('method', 'TNC')

        bounds = self.prepare_bounds(bounds)
        kwargs['bounds'] = [bounds.get(x, (None, None))
                            for x in self.arg_names]

        # Note the default 'tol' option is interpreted as xtol for TNC.
        # ftol is cryptically described as "precision goal"... but from the code
        # https://github.com/scipy/scipy/blob/81d2318e3a9ab172c05645e5d663979f7c594472/scipy/optimize/tnc/tnc.c#L844
        # it appears this is the absolute relative change in f to trigger
        # convergence. (not 100% sure, might be relative...)
        kwargs.setdefault('options', dict())
        if llr_tolerance is not None:
            kwargs['options'].setdefault('ftol', llr_tolerance)
        res = scipy.optimize.minimize(self.fun, x_guess, jac=self.grad,
                                      **kwargs)
        if get_lowlevel_result:
            return res
        return {**dict(zip(self.arg_names, res.x)), **self.fix}


class TensorFlowObjective(Objective):
    def minimize(self, x_guess, get_lowlevel_result=False, bounds=None,
                 use_hessian=True, llr_tolerance=None, **kwargs):
        x_guess = fd.np_to_tf(x_guess)

        if use_hessian:
            guess = {**dict(zip(self.arg_names, x_guess)), **self.fix}
            # This optimizer can use the hessian information
            # Compute the inverse hessian at the guess
            inv_hess = self.lf.inverse_hessian(guess,
                                        omit_grads=tuple(self.fix.keys()))
            # Explicitly symmetrize the matrix
            inv_hess = fd.symmetrize_matrix(inv_hess)
        else:
            inv_hess = None

        # Unfortunately we can only set the relative tolerance for the
        # objective; we'd like to set the absolute one.
        # Use the guess log likelihood to normalize;
        if llr_tolerance is not None:
            kwargs.setdefault(
                'f_relative_tolerance',
                self.absolute_to_relative_tol(x_guess, llr_tolerance))

        res = tfp.optimizer.bfgs_minimize(
            self.fun_and_grad,
            initial_position=x_guess,
            initial_inverse_hessian_estimate=inv_hess,
            **kwargs)
        if get_lowlevel_result:
            return res
        if res.failed:
            raise ValueError(f"Optimizer failure! Result: {res}")

        res = res.position
        res = {k: res[i].numpy() for i, k in enumerate(self.arg_names)}
        return {**res, **self.fix}


class MinuitObjective(Objective):
    numpy_in_out = True
    memoize = True

    def minimize(self, x_guess, bounds=None, use_hessian=False,
                 get_lowlevel_result=False,
                 llr_tolerance=None, **kwargs):

        kwargs.setdefault('error',
                          np.maximum(x_guess * 0.1,
                                     1e-3 * np.ones_like(x_guess)))

        bounds = self.prepare_bounds(bounds)
        for param_name, b in bounds.items():
            kwargs.setdefault('limit_' + param_name, b)

        fit = Minuit.from_array_func(self.fun, x_guess, grad=self.grad,
                                     errordef=0.5,
                                     name=self.arg_names,
                                     **kwargs)

        if llr_tolerance is not None:
            # From https://iminuit.readthedocs.io/en/latest/reference.html
            # and https://root.cern.ch/download/minuit.pdf
            # this value is multiplied by 0.001 * 0.5, and then gives the
            # estimated vertical distance to the minimum needed to stop
            # Note the first reference gives 0.0001 instead of 0.001!
            # TODO make issue?
            fit.tol = llr_tolerance/(0.001 * 0.5)

        fit.migrad()

        if get_lowlevel_result:
            return fit
        return fit.fitarg


OBJECTIVES = dict(tfp=TensorFlowObjective,
                  minuit=MinuitObjective,
                  scipy=ScipyObjective)


##
# Bestfit functions
##

@export
def get_bestfit_objective(optimizer):
    if optimizer not in OBJECTIVES:
        raise ValueError(f"Optimizer {optimizer} not supported")
    return OBJECTIVES[optimizer]


##
# Interval estimation
##

class IntervalObjective(Objective):

    # Add constant offset to objective, so objective is not 0 at the minimum
    # and relative tolerances mean something.
    _offset = 1

    def __init__(self, m2ll_best, critical_quantile, target_parameter,
                 *args, t_ppf=None, t_ppf_grad=None, **kwargs):
        super().__init__(*args, **kwargs)
        if t_ppf:
            assert self.t_ppf_grad is not None
            self.t_ppf = t_ppf
            self.t_ppf_grad = t_ppf_grad

        self.critical_quantile = critical_quantile
        self.m2ll_best = m2ll_best
        self.target_parameter = target_parameter

        self.numpy_in_out = True

    def t_ppf(self, target_param_value):
        """Return critical value given parameter value and critical
        quantile.
        Asymptotic case using Wilk's theorem, does not depend
        on the value of the target parameter."""
        return wilks_crit(self.critical_quantile)

    def t_ppf_grad(self, target_param_value):
        """Return derivative of t_ppf wrt target_param_value"""
        return 0

    def _inner_fun_and_grad(self, params):
        fun, grad = super()._inner_fun_and_grad(params)
        ll_ratio = fun - self.m2ll_best
        x = params[self.target_parameter]
        crit = self.t_ppf(x)
        diff = ll_ratio - crit

        return (self._offset + diff**2,
                2 * diff * (grad - self.t_ppf_grad(x)))

    def absolute_to_relative_tol(self, x_guess, llr_tolerance):
        # We know the objective is self.offset at the minimum,
        # so the relative to absolute tolerance conversion is easy:
        return llr_tolerance / self._offset


class TensorFlowIntervalObjective(IntervalObjective, TensorFlowObjective):
    """IntervalObjective using TensorFlow optimizer"""


class MinuitIntervalObjective(IntervalObjective, MinuitObjective):
    """IntervalObjective using Minuit optimizer"""


class ScipyIntervalObjective(IntervalObjective, ScipyObjective):
    """IntervalObjective using Scipy optimizer"""


INTERVALOBJECTIVES = dict(tfp=TensorFlowIntervalObjective,
                          minuit=MinuitIntervalObjective,
                          scipy=ScipyIntervalObjective)


def get_interval_objective(optimizer):
    if optimizer not in INTERVALOBJECTIVES:
        raise ValueError(f"Optimizer {optimizer} not supported")
    return INTERVALOBJECTIVES[optimizer]


@export
def one_parameter_interval(
        lf: fd.LogLikelihood,
        parameter: str,
        bound: ty.Tuple[ty.Union[float, None], ty.Union[float, None]],
        guess: ty.Dict[str, float],
        m2ll_best: float,
        critical_quantile: float,
        optimizer: str,
        fix: ty.Dict[str, float] = None,
        llr_tolerance=None,
        t_ppf=None,
        t_ppf_grad=None,
        **kwargs):
    """Compute upper/lower/central interval on parameter at confidence level"""

    if fix is None:
        fix = dict()
    arg_names = [k for k in lf.param_names if k not in fix]
    x_guess = np.array([guess[k] for k in arg_names])

    optimizer = get_interval_objective(optimizer)

    if llr_tolerance is not None:
        # The objective is squared:
        llr_tolerance = llr_tolerance ** 2

    # Construct t-stat objective + grad
    obj = optimizer(lf=lf, arg_names=arg_names, fix=fix, m2ll_best=m2ll_best,
                    target_parameter=parameter, t_ppf=t_ppf,
                    t_ppf_grad=t_ppf_grad, critical_quantile=critical_quantile)
    res = obj.minimize(x_guess, use_hessian=False, get_lowlevel_result=False,
                       bounds={parameter: bound},
                       llr_tolerance=llr_tolerance, **kwargs)

    return res[parameter]


@export
def wilks_crit(confidence_level):
    """Return critical value from Wilks' theorem for upper limits"""
    return scipy.stats.norm.ppf(confidence_level) ** 2