import flamedisx as fd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import typing as ty

export, __all__ = fd.exporter()
o = tf.newaxis


@export
class LogLikelihood:
    param_defaults: ty.Dict[str, float]
    data: pd.DataFrame
    sources: ty.Dict[str, fd.Source]
    mu_iterpolators: ty.Dict[str, ty.Callable]

    def __init__(
            self,
            sources: ty.Dict[str, fd.Source.__class__],
            data: pd.DataFrame,
            free_rates: ty.Union[None, str, ty.Tuple[str]] = None,
            batch_size=10,
            max_sigma=3,
            n_trials=int(1e5),
            **common_param_specs):

        param_defaults = dict()

        if free_rates is None:
            free_rates = tuple()
        if isinstance(free_rates, str):
            free_rates = (free_rates,)
        for sn in free_rates:
            if sn not in sources:
                raise ValueError(f"Can't free rate of unknown source {sn}")
            param_defaults[sn + '_rate_multiplier'] = 1.

        # Create sources. Have to copy data, it's modified by set_data
        self.sources = {
            sname: sclass(data.copy(),
                          max_sigma=max_sigma,
                          fit_params=list(common_param_specs.keys()),
                          batch_size=batch_size)
            for sname, sclass in sources.items()}
        del sources  # so we don't use it by accident
        del data  # use data from sources (which is now annotated)

        for pname in common_param_specs:
            # Check defaults for common parameters are consistent between
            # sources
            defs = [s.defaults[pname] for s in self.sources.values()]
            if len(set([x.numpy() for x in defs])) > 1:
                raise ValueError(
                    f"Inconsistent defaults {defs} for common parameters")
            param_defaults[pname] = defs[0]

        # Set n_batches, batch_size and data from any source
        for s in self.sources.values():
            self.n_batches = s.n_batches
            self.batch_size = s.batch_size
            self.n_padding = s.n_padding
            self.data = s.data
            break

        self.param_defaults = {k: tf.constant(v, dtype=fd.float_type())
                               for k, v in param_defaults.items()}
        self.param_names = list(param_defaults.keys())

        self.mu_itps = {
            sname: s.mu_interpolator(n_trials=n_trials,
                                     data=s.data,
                                     **common_param_specs)
            for sname, s in self.sources.items()}
        # Not used, but useful for mu smoothness diagnosis
        self.param_specs = common_param_specs

    def __call__(self, **kwargs):
        assert 'second_order' not in kwargs, 'Roep gewoon log_likelihood aan'
        return self.log_likelihood(second_order=False, **kwargs)[0].numpy()

    def log_likelihood(self, autograph=True, second_order=False, **kwargs):
        if second_order:
            # Compute the likelihood, jacobian and hessian
            # Use only non-tf.function version, in principle works with
            # but this leads to very long tracing times and we only need
            # hessian once
            f = self._log_likelihood_grad2
        else:
            # Computes the likelihood and jacobian
            f = self._log_likelihood_tf if autograph else self._log_likelihood

        params = self.prepare_params(kwargs)
        n_params = len(self.param_defaults)
        lls = tf.constant(0., dtype=fd.float_type())
        llgrads = tf.zeros(n_params, dtype=fd.float_type())
        llgrads2 = tf.zeros((n_params, n_params), dtype=fd.float_type())

        for i_batch in tf.range(self.n_batches, dtype=fd.int_type()):
            v = f(i_batch, autograph, **params)
            lls += v[0]
            llgrads += v[1]
            if second_order:
                llgrads2 += v[2]

        if second_order:
            return lls, llgrads, llgrads2
        return lls, llgrads

    def minus_ll(self, *, autograph=True, **kwargs):
        ll, grad = self.log_likelihood(autograph=autograph, **kwargs)
        return -2 * ll, -2 * grad

    def prepare_params(self, kwargs):
        for k in kwargs:
            if k not in self.param_defaults:
                raise ValueError(f"Unknown parameter {k}")
        # tf.function doesn't support {**x, **y} dict merging
        # return {**self.param_defaults, **kwargs}
        z = self.param_defaults.copy()
        for k, v in kwargs.items():
            if isinstance(v, (float, int)) or fd.is_numpy_number(v):
               kwargs[k] = tf.constant(v, dtype=fd.float_type())
        z.update(kwargs)
        return z

    def _get_rate_mult(self, sname, kwargs):
        rmname = sname + '_rate_multiplier'
        if rmname in self.param_names:
            return kwargs[rmname]
        return tf.constant(1., dtype=fd.float_type())

    def _source_kwargnames(self, source_name):
        """Return parameter names that apply to source"""
        return [pname for pname in self.param_names
                if not pname.endswith('_rate_multiplier')
                and pname in self.sources[source_name].defaults]

    def _filter_source_kwargs(self, kwargs, source_name):
        """Return {param: value} dictionary with keyword arguments
        for source, with values extracted from kwargs"""
        return {pname: kwargs[pname]
                for pname in self._source_kwargnames(source_name)}

    def _param_i(self, pname):
        """Return index of parameter pname"""
        return self.param_names.index(pname)

    def mu(self, **kwargs):
        mu = tf.constant(0., dtype=fd.float_type())
        for sname, s in self.sources.items():
            mu += (self._get_rate_mult(sname, kwargs)
                   * self.mu_itps[sname](**self._filter_source_kwargs(kwargs, sname)))
        return mu

    def _log_likelihood(self, i_batch, autograph, **params):
        par_list = list(params.values())
        with tf.GradientTape() as t:
            t.watch(par_list)
            ll = self._log_likelihood_inner(i_batch, params, autograph)
        grad = t.gradient(ll, par_list)
        return ll, tf.stack(grad)

    @tf.function
    def _log_likelihood_tf(self, i_batch, autograph, **params):
        print("Tracing _log_likelihood")
        return self._log_likelihood(i_batch, autograph, **params)

    def _log_likelihood_grad2(self, i_batch, autograph, **params):
        par_list = list(params.values())
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(par_list)
            with tf.GradientTape() as t:
                t.watch(par_list)
                ll = self._log_likelihood_inner(i_batch, params, autograph)
            grads = t.gradient(ll, par_list)
        hessian = [t2.gradient(grad, par_list) for grad in grads]
        del t2
        return ll, tf.stack(grads), tf.stack(hessian)

    @tf.function
    def _log_likelihood_grad2_tf(self, i_batch, autograph, **params):
        print("Tracing _log_likelihood_grad2_tf")
        return self._log_likelihood_grad2(i_batch, autograph, **params)

    def _log_likelihood_inner(self, i_batch, params, autograph):
        # Does for loop over sources, not batches
        # Sum over sources is first in likelihood

        # Compute differential rates and their gradients from all sources
        # drs = list[n_sources] of [n_events] tensors
        drs = tf.zeros((self.batch_size,), dtype=fd.float_type())
        for sname, s in self.sources.items():
            rate_mult = self._get_rate_mult(sname, params)
            dr = s.differential_rate(s.data_tensor[i_batch],
                                     autograph=autograph,
                                     **self._filter_source_kwargs(params, sname))
            drs += dr * rate_mult

        # Sum over events and remove padding
        n = tf.where(tf.equal(i_batch, tf.constant(self.n_batches - 1, dtype=fd.int_type())),
                     self.batch_size,
                     self.batch_size - self.n_padding)
        ll = tf.reduce_sum(tf.math.log(drs[:n]))

        # Add mu once (to the first batch)
        ll += tf.where(tf.equal(i_batch, tf.constant(0, dtype=fd.int_type())),
                       -self.mu(**params),
                       0.)
        return ll

    def guess(self):
        """Return array of parameter guesses"""
        return np.array(list(self.param_defaults.values()))

    def params_to_dict(self, values):
        """Return parameter {name: value} dictionary"""
        return {k: v for k, v in zip(self.param_names, tf.unstack(values))}

    def bestfit(self, guess=None,
                optimizer=tfp.optimizer.lbfgs_minimize,
                llr_tolerance=0.1,
                get_lowlevel_result=False, **kwargs):
        """Return best-fit parameter tensor

        :param guess: Guess parameters: array or tensor of same length
        as param_names.
        If omitted, will use guess from source defaults.
        :param llr_tolerance: stop minimizer if change in -2 log likelihood
        becomes less than this (roughly: using guess to convert to
        relative tolerance threshold)
        """
        _guess = self.guess()
        if isinstance(guess, dict):
            # Modify guess with user-specified vars
            for k, v in guess.items():
                _guess[self._param_i(k)] = v
        elif isinstance(guess, (np.ndarray, tf.Tensor)):
            _guess = fd.tf_to_np(guess)
        _guess = fd.np_to_tf(_guess)
        del guess

        # Unfortunately we can only set the relative tolerance for the
        # objective; we'd like to set the absolute one.
        # Use the guess log likelihood to normalize;
        if llr_tolerance is not None:
            kwargs.setdefault('f_relative_tolerance',
                              llr_tolerance/self.minus_ll(**self.params_to_dict(_guess))[0])

        # Minimize multipliers to the guess, rather than the guess itself
        # This is a basic kind of standardization that helps make the gradient
        # vector reasonable.
        x_norm = tf.ones(len(_guess), dtype=fd.float_type())

        # Set guess for objective function
        self._guess = _guess

        res = optimizer(self.objective, x_norm, **kwargs)
        if get_lowlevel_result:
            return res
        if res.failed:
            raise ValueError(f"Optimizer failure! Result: {res}")
        return res.position * _guess

    @tf.function
    def objective(self, x_norm):
        print("Tracing objective")
        x = x_norm * self._guess
        ll, grad = self.minus_ll(**self.params_to_dict(x))
        if tf.math.is_nan(ll):
            tf.print(f"Objective at {x_norm} is Nan!")
            ll *= float('inf')
            grad *= float('nan')
        return ll, grad * self._guess

    def inverse_hessian(self, params):
        """Return inverse hessian (square tensor)
        of -2 log_likelihood at params
        """
        # Also Tensorflow has tf.hessians, but:
        # https://github.com/tensorflow/tensorflow/issues/29781

        # In case params is a numpy vector
        params = fd.np_to_tf(params)

        # Get second order derivatives of likelihood at params
        _, _, grad2_ll = self.log_likelihood(**self.params_to_dict(params),
                                             autograph=False,
                                             second_order=True)
        return tf.linalg.inv(-2 * grad2_ll)

    def summary(self, bestfit, inverse_hessian=None, precision=3):
        """Print summary information about best fit"""
        if inverse_hessian is None:
            inverse_hessian = self.inverse_hessian(bestfit)
        inverse_hessian = fd.tf_to_np(inverse_hessian)

        stderr, cov = cov_to_std(inverse_hessian)
        for i, pname in enumerate(self.param_names):
            template = "{pname}: {x:.{precision}g} +- {xerr:.{precision}g}"
            print(template.format(
                pname=pname,
                x=bestfit[i],
                xerr=stderr[i],
                precision=precision))

        df = pd.DataFrame(
            {p1: {p2: cov[i1, i2]
                  for i2, p2 in enumerate(self.param_names)}
             for i1, p1 in enumerate(self.param_names)},
            columns=self.param_names)

        # Get rows in the correct order
        df['index'] = [self.param_names.index(x)
                       for x in df.index.values]
        df = df.sort_values(by='index')
        del df['index']

        print("Correlation matrix:")
        pd.set_option('precision', 3)
        print(df)
        pd.reset_option('precision')


@export
def cov_to_std(cov):
    """Return (std errors, correlation coefficent matrix)
    given covariance matrix cov
    """
    std_errs = np.diag(cov) ** 0.5
    corr = cov * np.outer(1 / std_errs, 1 / std_errs)
    return std_errs, corr
