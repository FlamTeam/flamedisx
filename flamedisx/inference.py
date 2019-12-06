import flamedisx as fd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import typing as ty
from scipy import optimize as scipy_optimize
from iminuit import Minuit


export, __all__ = fd.exporter()
__all__ += ['LOWER_RATE_MULTIPLIER_BOUND',
            'SUPPORTED_OPTIMIZERS',
            'SUPPORTED_INTERVAL_OPTIMIZERS']

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

    :param lf: LogLikelihood object implementing the likelihood to minimize
    :param guess: {param: value} guess for the result
    :param fix: {param: value} for parameters to fix, or None (to fit all).
    Gradients for fixed parameters are (of course) omitted.
    :param bounds: {param: (left, right)} bounds, if any (otherwise None)
    :param llr_tolerance: Allowed distance between true and found solution
    in -2 log likelihood.
    :param nan_val: Value to pass to optimizer if likelihood evaluates to NaN
    :param get_lowlevel_result: Return low-level result from optimizer directly
    :param use_hessian: If supported, use Hessian to improve error estimate
    :param return_errors: If supported, return error estimates on parameters
    """
    numpy_in_out = False   # Minimizer works on numpy arrays, not tensors
    memoize = False        # Cache values during minimization.
    require_complete_guess = True   # Require a guess for all fitted parameters
    arg_names: ty.List = None

    _cache: dict

    def __init__(self, *,
                 lf: fd.LogLikelihood,
                 guess: ty.Dict[str, float] = None,
                 fix: ty.Dict[str, ty.Union[float, tf.constant]] = None,
                 bounds: dict = None,
                 llr_tolerance=0.05,
                 nan_val=float('inf'),
                 get_lowlevel_result=False,
                 get_history=False,
                 use_hessian=False,
                 return_errors=False,
                 optimizer_kwargs: dict = None):
        if guess is None:
            guess = dict()
        if fix is None:
            fix = dict()
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()

        self.lf = lf
        self.guess = {**guess, **fix}
        self.fix = fix
        # Bounds need processing, see deeper in init
        self.llr_tolerance = llr_tolerance
        self.nan_val = nan_val
        self.get_lowlevel_result = get_lowlevel_result
        self.return_history = get_history
        self.use_hessian = use_hessian
        self.return_errors = return_errors
        self.optimizer_kwargs = optimizer_kwargs

        # The if is only here to support MockInference with static arg_names
        if self.arg_names is None:
            self.arg_names = [
                k for k in self.lf.param_names if k not in self.fix]
        self._cache = dict()
        if self.return_history:
            self._history = []

        if self.require_complete_guess:
            for k in self.arg_names:
                if k not in self.guess:
                    raise ValueError("Incomplete guess: {k} missing")

        if bounds is None:
            bounds = dict()
        for p in self.arg_names:
            if p.endswith('_rate_multiplier'):
                bounds.setdefault(p, (LOWER_RATE_MULTIPLIER_BOUND, None))
        self.bounds = bounds

        self.absolute_ftol = self.llr_tolerance

    def _dict_to_array(self, x: dict) -> np.array:
        """Convert from {parameter: value} dictionary to numpy array"""
        return np.array([x[k] for k in self.arg_names])

    def _array_to_dict(self, x: ty.Union[np.ndarray, tf.Tensor]) -> dict:
        """Convert from array/tensor to {parameter: value} dictionary"""
        assert isinstance(x, (np.ndarray, tf.Tensor))
        assert len(x) == len(self.arg_names)
        return {k: x[i]
                for i, k in enumerate(self.arg_names)}

    def __call__(self, x):
        """Return (objective, gradient)"""
        memkey = None
        if self.memoize:
            assert self.numpy_in_out, "Fix memoization for tensorflow..."
            memkey = tuple(x)
            if memkey in self._cache:
                return self._cache[memkey]

        params = {**self._array_to_dict(x), **self.fix}
        ll, grad = self._inner_fun_and_grad(params)

        # Check NaNs
        if tf.math.is_nan(ll):
            tf.print(f"Objective at {x} is Nan!")
            ll = tf.constant(self.nan_val, dtype=tf.float32)

        if self.numpy_in_out:
            ll = ll.numpy()
            grad = grad.numpy()

        if self.return_history:
            self._history.append(dict(params=params, ll=ll, grad=grad))

        result = ObjectiveResult(fun=ll, grad=grad)
        if self.memoize:
            self._cache[memkey] = result
        return result

    def _inner_fun_and_grad(self, params):
        # Get -2lnL and its gradient
        return self.lf.minus_ll(
            **params,
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

    def relative_ftol_guess(self):
        return abs(self.absolute_ftol / self._inner_fun_and_grad(self.guess)[0])

    def _lowlevel_shortcut(self, res):
        if self.get_lowlevel_result:
            return True, res
        if self.return_history:
            return True, self._history
        return False, res


class ScipyObjective(Objective):
    numpy_in_out = True
    memoize = True

    def minimize(self):
        if self.return_errors:
            raise NotImplementedError(
                "Scipy minimizer does not yet support return errors")

        # TODO implement optimizer methods to use the Hessian,
        # see https://github.com/FlamTeam/flamedisx/pull/60#discussion_r354832569

        kwargs: ty.Dict[str, ty.Any] = self.optimizer_kwargs
        kwargs.setdefault('method', 'TNC')
        kwargs['bounds'] = [self.bounds.get(x, (None, None))
                            for x in self.arg_names]

        # Note the default 'tol' option is interpreted as xtol for TNC.
        # ftol is cryptically described as "precision goal"... but from the code
        # https://github.com/scipy/scipy/blob/81d2318e3a9ab172c05645e5d663979f7c594472/scipy/optimize/tnc/tnc.c#L844
        # it appears this is the absolute relative change in f to trigger
        # convergence. (not 100% sure, might be relative...)
        kwargs.setdefault('options', dict())
        if self.absolute_ftol is not None:
            kwargs['options'].setdefault('ftol', self.absolute_ftol)

        res = scipy_optimize.minimize(
            fun=self.fun,
            x0=self._dict_to_array(self.guess),
            jac=self.grad,
            **kwargs)

        ret, res = self._lowlevel_shortcut(res)
        if ret:
            return res
        return {**dict(zip(self.arg_names, res.x)), **self.fix}


class TensorFlowObjective(Objective):

    def minimize(self):
        if self.return_errors:
            raise NotImplementedError(
                "Tensorflow minimizer does not yet support return errors")

        if self.use_hessian:
            # This optimizer can use the hessian information
            # Compute the inverse hessian at the guess
            inv_hess = self.lf.inverse_hessian(
                self.guess,
                omit_grads=tuple(self.fix.keys()))
            # Explicitly symmetrize the matrix
            inv_hess = fd.symmetrize_matrix(inv_hess)
        else:
            inv_hess = None

        kwargs = self.optimizer_kwargs

        # Unfortunately we can only set the relative tolerance for the
        # objective; we'd like to set the absolute one.
        # Use the guess log likelihood to normalize;
        if self.absolute_ftol is not None:
            kwargs.setdefault(
                'f_relative_tolerance',
                self.relative_ftol_guess())

        x_guess = fd.np_to_tf(self._dict_to_array(self.guess))

        res = tfp.optimizer.bfgs_minimize(
            self.fun_and_grad,
            initial_position=x_guess,
            initial_inverse_hessian_estimate=inv_hess,
            **kwargs)
        ret, res = self._lowlevel_shortcut(res)
        if ret:
            return res
        if res.failed:
            raise ValueError(f"Optimizer failure! Result: {res}")

        res = res.position
        res = {k: res[i].numpy() for i, k in enumerate(self.arg_names)}
        return {**res, **self.fix}


class MinuitObjective(Objective):
    numpy_in_out = True
    memoize = True

    def minimize(self):
        kwargs = self.optimizer_kwargs
        x_guess = self._dict_to_array(self.guess)
        kwargs.setdefault('error',
                          np.maximum(x_guess * 0.1,
                                     1e-3 * np.ones_like(x_guess)))

        for param_name, b in self.bounds.items():
            kwargs.setdefault('limit_' + param_name, b)

        fit = Minuit.from_array_func(self.fun, x_guess, grad=self.grad,
                                     errordef=0.5,
                                     name=self.arg_names,
                                     **kwargs)

        if self.absolute_ftol is not None:
            # From https://iminuit.readthedocs.io/en/latest/reference.html
            # and https://root.cern.ch/download/minuit.pdf
            # this value is multiplied by 0.001 * 0.5, and then gives the
            # estimated vertical distance to the minimum needed to stop
            # Note the first reference gives 0.0001 instead of 0.001!
            # See https://github.com/scikit-hep/iminuit/issues/353
            fit.tol = self.absolute_ftol/(0.001 * 0.5)

        fit.migrad()
        if self.return_errors == 'hesse':
            fit.hesse()
        elif self.return_errors == 'minos':
            fit.minos()

        ret, res = self._lowlevel_shortcut(fit)
        if ret:
            return res
        return fit.fitarg


SUPPORTED_OPTIMIZERS = dict(tfp=TensorFlowObjective,
                            minuit=MinuitObjective,
                            scipy=ScipyObjective)


##
# Interval estimation
##

class IntervalObjective(Objective):

    # We can guesstimate from the bestfit and Hessian
    require_complete_guess = False

    # Add constant offset to objective, so objective is not 0 at the minimum
    # and relative tolerances mean something.
    _offset = 1

    def __init__(self, *,
                 target_parameter,
                 bestfit,
                 direction: int,
                 critical_quantile,
                 tol_multiplier=1.,
                 sigma_guess=None,
                 t_ppf=None,
                 t_ppf_grad=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.target_parameter = target_parameter
        self.bestfit = bestfit
        self.direction = direction
        self.critical_quantile = critical_quantile
        self.tol_multiplier = tol_multiplier

        if sigma_guess is None:
            # Estimate one sigma interval using parabolic approx.
            sigma_guess = fd.cov_to_std(
                self.lf.inverse_hessian(bestfit)
            )[0][self.lf.param_names.index(self.target_parameter)]
        self.sigma_guess = sigma_guess

        if t_ppf:
            assert self.t_ppf_grad is not None
            self.t_ppf = t_ppf
            self.t_ppf_grad = t_ppf_grad

        # TODO: add reference to computation for this
        self.tilt = 4 * self.llr_tolerance * self.critical_quantile

        # Store bestfit target, maximum likelihood and slope
        self.bestfit_tp = self.bestfit[self.target_parameter]
        self.m2ll_best, _grad_at_bestfit = \
            super()._inner_fun_and_grad(bestfit)
        self.bestfit_tp_slope = _grad_at_bestfit[
            self.lf.param_names.index(self.target_parameter)]

        # Incomplete guess support
        if self.target_parameter not in self.guess:
            # Estimate crossing point from Wilks' theorem
            dy = fd.wilks_crit(self.critical_quantile)

            # Guess the distance to the crossing point
            # based on the Hessian
            dx_1 = (2 * dy) ** 0.5 * abs(self.sigma_guess)

            # ... or the slope (for boundary solutions)
            dx_2 = abs(dy / self.bestfit_tp_slope)

            # Take the smaller of the two and add it on the correct side
            # TODO: Is the best one always smallest? Don't know...
            dx = min(dx_1, dx_2)
            tp_guess = max(
                fd.LOWER_RATE_MULTIPLIER_BOUND,
                self.bestfit_tp + self.direction * dx)

            if self.target_parameter.endswith('rate_multiplier'):
                tp_guess = max(tp_guess, fd.LOWER_RATE_MULTIPLIER_BOUND)
        else:
            tp_guess = self.guess[self.target_parameter]

        # Check guess is in bounds
        lb, rb = self.bounds.get(self.target_parameter, (None, None))
        if lb is not None:
            assert lb <= tp_guess, f"Guess {tp_guess} below lower bound {lb}"
        if rb is not None:
            assert tp_guess <= rb, f"Guess {tp_guess} above upper bound {rb}"

        self.guess = {**bestfit,
                      **{self.target_parameter: tp_guess},
                      **self.guess}

        # Objective involves square likelihood, so square tolerance too:
        self.absolute_ftol = self.tol_multiplier * self.llr_tolerance**2

    def t_ppf(self, target_param_value):
        """Return critical value given parameter value and critical
        quantile.
        Asymptotic case using Wilk's theorem, does not depend
        on the value of the target parameter."""
        return fd.wilks_crit(self.critical_quantile)

    def t_ppf_grad(self, target_param_value):
        """Return derivative of t_ppf wrt target_param_value"""
        return 0

    def _inner_fun_and_grad(self, params):
        x = params[self.target_parameter]
        x_norm = (x - self.bestfit_tp) / self.sigma_guess

        fun, grad = super()._inner_fun_and_grad(params)
        diff = (fun - self.m2ll_best) - self.t_ppf(x)
        fun = diff ** 2
        grad = 2 * diff * (grad - self.t_ppf_grad(x))

        # Add 'tilt' to push the minimum to extreme values of the parameter of
        # interest. Without this, we would find any solution on the ellipsoid
        # where our likelihood equals the target amplitude.
        fun += - self.direction * self.tilt * x_norm
        extra_grad = - self.direction * self.tilt / self.sigma_guess
        grad += tf.where(
            tf.equal(self.arg_names, self.target_parameter),
            extra_grad,
            tf.constant(0., dtype=fd.float_type()))

        return fun + self._offset, grad

    def absolute_to_relative_tol(self, llr_tolerance):
        # We know the objective is self.offset at the minimum,
        # so the relative to absolute tolerance conversion is easy:
        return llr_tolerance / self._offset


class TensorFlowIntervalObjective(IntervalObjective, TensorFlowObjective):
    """IntervalObjective using TensorFlow optimizer"""


class MinuitIntervalObjective(IntervalObjective, MinuitObjective):
    """IntervalObjective using Minuit optimizer"""


class ScipyIntervalObjective(IntervalObjective, ScipyObjective):
    """IntervalObjective using Scipy optimizer"""


SUPPORTED_INTERVAL_OPTIMIZERS = dict(tfp=TensorFlowIntervalObjective,
                                     minuit=MinuitIntervalObjective,
                                     scipy=ScipyIntervalObjective)
