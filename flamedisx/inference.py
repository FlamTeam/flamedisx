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
    objective = make_objective(lf, arg_names=arg_names, fix=fix)

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
                          llr_tolerance/objective(x_guess)[0].numpy())

    res = tfp.optimizer.bfgs_minimize(objective, x_guess,
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
    fun, jac = make_objective(lf,
                              arg_names=arg_names,
                              fix=fix,
                              separate_func_grad=True,
                              numpy_in_out=True)

    for i in range(len(x_guess)):
        # Set initial step sizes of 0.1 * guess
        kwargs.setdefault('error_' + arg_names[i], x_guess[i] * 0.1)

    fit = Minuit.from_array_func(fun, x_guess, grad=jac,
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
    precomp = dict()

    fun, jac = make_objective(lf,
                              arg_names,
                              fix,
                              separate_func_grad=True,
                              memoize_dict=precomp,
                              numpy_in_out=True)
    bounds = [(1e-9, None) if n.endswith('_rate_multiplier') else (None, None)
              for n in arg_names]
    res = minimize(fun, x_guess,
                   jac=jac,
                   method=method,
                   bounds=bounds,
                   tol=tol,
                   **kwargs)
    if get_lowlevel_result:
        return res

    return {**dict(zip(arg_names, res.x)), **fix}

def make_objective(lf: fd.LogLikelihood,
                   arg_names: ty.List[str],
                   fix: ty.Dict[str, ty.Union[float, tf.constant]],
                   separate_func_grad=False,
                   numpy_in_out=False, autograph=True,
                   nan_val=float('inf'), memoize_dict=None):
    """Construct the function that is minimized by the optimizer.
    That function should take one argument x_guess that is a list of values
    each belonging to the parameters being varied.

    :param lf: LogLikelihood object implementing the likelihood to minimize
    :param arg_names: List of parameter names whose values are varied by the
    minimizer
    :param fix: Dict of parameter names and value which are kept fixed (and
    whose gradients are omitted)
    :param separate_func_grad: If True returns two objective functions for
    the likelihood and gradient respectively, otherwise return one objective
    function, default False.
    :param numpy_in_out: Converts inputs to tensors and outputs to numpy arrays
    if True, default False
    :param autograph: Use tf.function inside likelihood, default True
    :param nan_val: Value to use if likelihood evaluates to NaN
    :param memoize_dict: Dict which stores calls to objective. Useful
    in combination with separate_func_grad=True

    Returns function that evaluates the likelihood at x_guess and returns the
    likelihood and gradient (or two functions for likelihood and gradient
    separately if separate_func_grad=True)
    """
    if isinstance(memoize_dict, dict):
        memoize=True
    else:
        memoize=False

    def objective(x_guess):
        if memoize:
            memkey = tuple(x_guess)
            if memkey in memoize_dict:
                return memoize_dict[memkey]
        if numpy_in_out:
            x_guess = fd.np_to_tf(x_guess)

        assert len(arg_names) == len(x_guess)
        # Build parameter dict, pair arg_names with x_guess values and add
        # fixed pars
        params = {**{arg_names[i]: x_guess[i] for i in range(len(x_guess))},
                  **fix}

        # Get -2lnL and its gradient
        ll, grad = lf.minus_ll(**params,
                               autograph=autograph,
                               omit_grads=tuple(fix.keys()))
        # Check NaNs
        if tf.math.is_nan(ll):
            tf.print(f"Objective at {x_guess} is Nan!")
            ll = tf.constant(nan_val, dtype=tf.float32)

        if numpy_in_out:
            if memoize:
                memoize_dict[memkey] = (ll.numpy(), grad.numpy())
            return ll.numpy(), grad.numpy()
        if memoize:
            memoize_dict[memkey] = (ll, grad)
        return ll, grad

    if not separate_func_grad:
        return objective

    def fun(x_guess):
        return objective(x_guess)[0]

    def jac(x_guess):
        return objective(x_guess)[1]

    return fun, jac

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

    #TODO add possible fixed parameters
    arg_names = lf.param_names
    x_guess = np.array([guess[k] for k in arg_names])

    # Cache repeated calls to objective
    precomp = dict()
    # Construct t-stat objective + grad, get regular objective first
    objective = make_objective(lf, arg_names, fix=dict(),
                               numpy_in_out=True, memoize_dict=precomp)
    # Wrap in new func minimizing:
    # (2 (lnL max - lnL s) - critical_value) ** 2
    def t_fun(x):
        return (objective(x)[0] - ll_best - t_ppf(x, critical_quantile)) ** 2

    def t_grad(x):
        # original function and gradient f(x), df/dx
        f, grad = objective(x)
        # t_fun is g=f(x)**2 - const, so dg/dx = 2*f(x)*df/dx
        return 2 * f * grad

    bounds = [(1e-9, None) if n.endswith('_rate_multiplier') else (None, None)
              for n in arg_names]
    bounds[arg_names.index(parameter)] = bound
    res = minimize(t_fun, x_guess,
                   jac=t_grad,
                   method='TNC',
                   bounds=bounds,
                   tol=1e-5)

    return res.x[arg_names.index(parameter)]
