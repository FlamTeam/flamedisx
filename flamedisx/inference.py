import flamedisx as fd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import typing as ty
from scipy.optimize import minimize
from iminuit import Minuit


export, __all__ = fd.exporter()
o = tf.newaxis


##
# Bestfit functions
##

@export
def bestfit_tf(lf: fd.LogLikelihood,
               arg_names: ty.List[str],
               x_guess: np.array,
               fix: ty.Dict[str, float],
               use_hessian=True,
               llr_tolerance=0.1,
               get_lowlevel_result=False,
               **kwargs):
    """Minimize the -2lnL using TFP optimization"""
    f = Objective(lf, arg_names=arg_names, fix=fix)

    x_guess = fd.np_to_tf(x_guess)

    if use_hessian:
        guess = {**dict(zip(arg_names, x_guess)), **fix}
        # This optimizer can use the hessian information
        # Compute the inverse hessian at the guess
        inv_hess = lf.inverse_hessian(guess, omit_grads=tuple(fix.keys()))
        # Explicitly symmetrize the matrix
        inv_hess = fd.symmetrize_matrix(inv_hess)
    else:
        inv_hess = None

    # Unfortunately we can only set the relative tolerance for the
    # objective; we'd like to set the absolute one.
    # Use the guess log likelihood to normalize;
    if llr_tolerance is not None:
        kwargs.setdefault('f_relative_tolerance',
                          llr_tolerance/f.fun(x_guess).numpy())

    res = tfp.optimizer.bfgs_minimize(f, x_guess,
                                      initial_inverse_hessian_estimate=inv_hess,
                                      **kwargs)
    if get_lowlevel_result:
        return res
    if res.failed:
        raise ValueError(f"Optimizer failure! Result: {res}")

    res = res.position
    res = {k: res[i].numpy() for i, k in enumerate(arg_names)}
    return {**res, **fix}

@export
def bestfit_minuit(lf: fd.LogLikelihood,
                   arg_names: ty.List[str],
                   x_guess: ty.Union[ty.List[float], np.array],
                   fix: ty.Dict[str, float],
                   use_hessian=True,
                   return_errors=False,
                   autograph=True,
                   get_lowlevel_result=False,
                   **kwargs):
    """Minimize the -2lnL using Minuit optimization"""
    # TODO: memoize kan ook wel voor minuit toch...
    f = Objective(lf, arg_names, fix, numpy_in_out=True)

    for i in range(len(x_guess)):
        # Set initial step sizes of 0.1 * guess
        kwargs.setdefault('error_' + arg_names[i], x_guess[i] * 0.1)

    fit = Minuit.from_array_func(f.fun, x_guess, grad=f.grad,
                                 errordef=0.5, name=arg_names, **kwargs)

    fit.migrad()
    fit_result = dict(fit.values)
    if use_hessian:
        fit.hesse()

    if get_lowlevel_result:
        return fit

    fit_errors = dict()
    for (k, v) in fit.errors.items():
        fit_errors[k + '_error'] = v

    fit_result = {**fit_result, **fix}
    if return_errors:
        return fit_result, fit_errors
    return fit_result


@export
def bestfit_scipy(lf: fd.LogLikelihood,
                  arg_names: ty.List[str],
                  x_guess: np.array,
                  fix: ty.Dict[str, float],
                  get_lowlevel_result=False,
                  method='TNC', tol=5e-3, **kwargs):
    """Minimize the -2lnL using SciPy optimization"""
    f = Objective(lf, arg_names, fix, memoize=True, numpy_in_out=True)
    bounds = [(1e-9, None) if n.endswith('_rate_multiplier') else (None, None)
              for n in arg_names]           # TODO: can user give bounds?
    res = minimize(f.fun, x_guess,
                   jac=f.grad,
                   method=method,
                   bounds=bounds,
                   tol=tol,
                   **kwargs)
    if get_lowlevel_result:
        return res

    return {**dict(zip(arg_names, res.x)), **fix}


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
    :param memoize: Whether to cache values during minimization. Useful
    in combination with separate_func_grad=True...
    """
    _cache: dict

    def __init__(self,
                 lf: fd.LogLikelihood,
                 arg_names: ty.List[str],
                 fix: ty.Dict[str, ty.Union[float, tf.constant]],
                 numpy_in_out=False,
                 autograph=True,
                 nan_val=float('inf'),
                 memoize=False):
        self.lf = lf
        self.arg_names = arg_names
        self.fix = fix
        self.numpy_in_out = numpy_in_out
        self.autograph = autograph
        self.nan_val = nan_val
        self.memoize = memoize
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

        # Get -2lnL and its gradient
        ll, grad = self.lf.minus_ll(
            **params,
            autograph=self.autograph,
            omit_grads=tuple(self.fix.keys()))

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

    def fun_and_grad(self, x):
        r = self(x)
        return r.fun, r.grad

    def fun(self, x):
        """Return only objective"""
        return self(x).fun

    def grad(self, x):
        """Return only gradient"""
        return self(x).grad

##
# Interval estimation
##

@export
def one_parameter_interval(lf: fd.LogLikelihood, parameter: str,
                           bound: ty.Tuple[float, float],
                           guess: ty.Dict[str, float],
                           t_ppf: ty.Callable[[float, float], float],
                           ll_best: float,
                           critical_quantile: float):
    """Compute upper/lower/central interval on parameter at confidence level"""
    # TODO: try with other minimizers

    #TODO add possible fixed parameters
    arg_names = lf.param_names
    x_guess = np.array([guess[k] for k in arg_names])

    # Construct t-stat objective + grad, get regular objective first
    f = Objective(lf=lf, arg_names=arg_names, fix=dict(),
                  numpy_in_out=True, memoize=True)
    # Wrap in new func minimizing:
    # (2 (lnL max - lnL s) - critical_value) ** 2
    def t_fun(x):
        return (f(x).fun - ll_best - t_ppf(x, critical_quantile)) ** 2

    def t_grad(x):
        r = f(x)
        # t_fun is g=f(x)**2 - const, so dg/dx = 2*f(x)*df/dx
        return 2 * r.fun * r.grad

    bounds = [(1e-9, None) if n.endswith('_rate_multiplier') else (None, None)
              for n in arg_names]
    bounds[arg_names.index(parameter)] = bound
    res = minimize(t_fun, x_guess,
                   jac=t_grad,
                   method='TNC',
                   bounds=bounds,
                   tol=1e-5)

    return res.x[arg_names.index(parameter)]
