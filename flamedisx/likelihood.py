import flamedisx as fd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import typing as ty
from scipy import stats


export, __all__ = fd.exporter()
o = tf.newaxis

DEFAULT_DSETNAME = 'the_dataset'

@export
class LogLikelihood:
    param_defaults: ty.Dict[str, float]

    mu_iterpolators: ty.Dict[str, ty.Callable]

    # Source name -> Source instance
    sources: ty.Dict[str, fd.Source]

    # Source name -> dataset name
    d_for_s = ty.Dict[str, str]

    # Datasetname -> value
    batch_size: ty.Dict[str, int]
    n_batches: ty.Dict[str, int]
    n_padding: ty.Dict[str, int]

    dsetnames: ty.List

    def __init__(
            self,
            sources: ty.Union[
                ty.Dict[str, fd.Source.__class__],
                ty.Dict[str, ty.Dict[str, fd.Source.__class__]]],
            data: ty.Union[
                None,
                pd.DataFrame,
                ty.Dict[str, pd.DataFrame]] = None,
            free_rates: ty.Union[None, str, ty.Tuple[str]] = None,
            batch_size=10,
            max_sigma=3,
            n_trials=int(1e5),
            log_constraint=None,
            **common_param_specs):
        """

        :param sources: Dictionary {datasetname : {sourcename: class,, ...}, ...}
        or just {sourcename: class} in case you have one dataset
        Every source name must be unique.
        :param data: Dictionary {datasetname: pd.DataFrame}
        or just pd.DataFrame if you have one dataset or None if you
        set data later.
        :param free_rates: names of sources whose rates are floating
        :param batch_size:
        :param max_sigma:
        :param n_trials:
        :param **common_param_specs:  param_name = (min, max, anchors), ...
        """
        param_defaults = dict()

        if isinstance(data, pd.DataFrame) or data is None:
            # Only one dataset
            data = {DEFAULT_DSETNAME: data}
        if not isinstance(list(sources.values())[0], dict):
            sources: ty.Dict[str, ty.Dict[str, fd.Source.__class__]] = \
                {DEFAULT_DSETNAME: sources}
        assert data.keys() == sources.keys(), "Inconsistent dataset names"
        self.dsetnames = list(data.keys())

        # Flatten sources and fill data for source
        self.sources: ty.Dict[str, fd.Source.__class__] = dict()
        self.d_for_s = dict()
        for dsetname, ss in sources.items():
            for sname, s in ss.items():
                self.d_for_s[sname] = dsetname
                if sname in self.sources:
                    raise ValueError(f"Duplicate source name {sname}")
                self.sources[sname] = s
        del sources  # so we don't use it by accident

        if free_rates is None:
            free_rates = tuple()
        if isinstance(free_rates, str):
            free_rates = (free_rates,)
        for sn in free_rates:
            if sn not in self.sources:
                raise ValueError(f"Can't free rate of unknown source {sn}")
            param_defaults[sn + '_rate_multiplier'] = 1.

        # Determine default parameters for each source
        defaults_in_sources = {
            sname : sclass.find_defaults()[2]
            for sname, sclass in self.sources.items()}

        # Create sources. Have to copy data, it's modified by Source.set_data
        self.sources = {
            sname: sclass(data=(None
                                if data[self.d_for_s[sname]] is None
                                else data[self.d_for_s[sname]].copy()),
                          max_sigma=max_sigma,
                          fit_params=list(k for k in common_param_specs.keys()
                                          if k in defaults_in_sources[sname].keys()),
                          batch_size=batch_size)
            for sname, sclass in self.sources.items()}
        del data  # use data from sources (which is now annotated)

        for pname in common_param_specs:
            # Check defaults for common parameters are consistent between
            # sources
            defs = [s.defaults[pname]
                    for s in self.sources.values()
                    if pname in s.defaults]
            if len(set([x.numpy() for x in defs])) > 1:
                raise ValueError(
                    f"Inconsistent defaults {defs} for common parameters")
            param_defaults[pname] = defs[0]

        # Store n_batches, batch_size and n_padding for all datasets
        self.n_batches = dict()
        self.batch_size = dict()
        self.n_padding = dict()
        for sname, s in self.sources.items():
            self.n_batches[self.d_for_s[sname]] = s.n_batches
            self.batch_size[self.d_for_s[sname]] = s.batch_size
            self.n_padding[self.d_for_s[sname]] = s.n_padding

        self.param_defaults = {k: tf.constant(v, dtype=fd.float_type())
                               for k, v in param_defaults.items()}
        self.param_names = list(param_defaults.keys())

        self.mu_itps = {
            sname: s.mu_function(n_trials=n_trials,
                                 **{p_name: par for p_name, par in common_param_specs.items()
                                 if p_name in defaults_in_sources[sname].keys()})
            for sname, s in self.sources.items()}
        # Not used, but useful for mu smoothness diagnosis
        self.param_specs = common_param_specs

        # Add the constraint
        if log_constraint is None:
            log_constraint = lambda **kwargs: 0.
        self.log_constraint = log_constraint

    def set_data(self,
                 data: ty.Union[pd.DataFrame, ty.Dict[str, pd.DataFrame]]):
        """set new data for sources in the likelihood.
        Data is passed in the same format as for __init__
        Data can contain any subset of the original data keys to only
        update specific datasets.
        """
        if isinstance(data, pd.DataFrame):
            # Only one dataset
            assert len(self.dsetnames) == 1, \
                "You passed a DataFrame but there are multiple datasets"
            data = {DEFAULT_DSETNAME: data}

        for sname, source in self.sources.items():
            dname = self.d_for_s[sname]
            if dname in data:
                source.set_data(data[dname])
                # Update batches and padding
                self.n_batches[dname] = source.n_batches
                self.batch_size[dname] = source.batch_size
                self.n_padding[dname] = source.n_padding
            elif dname not in self.dsetnames:
                raise ValueError(f"Dataset name {dname} not known")

    def simulate(self, rate_multipliers=None, fix_truth=None, **params):
        """Simulate events from sources, optionally pass custom
        rate_multipliers and fix_truth.
        """
        if rate_multipliers is None:
            rate_multipliers = dict()

        # Collect Source event DFs in ds
        ds = []
        for sname, s in self.sources.items():
            rmname = sname + '_rate_multiplier'
            if rmname in rate_multipliers:
                rm = rate_multipliers[rmname]
            else:
                rm = self._get_rate_mult(sname, params)

            # mean number of events to simulate, rate mult times mu before
            # efficiencies, the simulator deals with the efficiencies
            mu = rm * s.mu_before_efficiencies(
                **self._filter_source_kwargs(params, sname))
            # Simulate this many events from source
            n_to_sim = np.random.poisson(mu)
            if n_to_sim == 0:
                continue
            d = s.simulate(n_to_sim,
                           fix_truth=fix_truth,
                           **self._filter_source_kwargs(params,
                                                        sname))
            # If events were simulated add them to the list
            if len(d) > 0:
                # Keep track of what source simulated which events
                d['source'] = sname
                ds.append(d)

        # Check if each source returned the same set of columns since
        # columns outside the intersection will be filled with NaN values
        if len(ds) > 1:
            col_sets = set([frozenset(df.columns) for df in ds])
            assert len(col_sets) == 1, \
                f"Sources did not all simulate the same columns: {col_sets}"

        # Concatenate results and shuffle them.
        # Adding empty DataFrame ensures pd.concat doesn't fail if
        # n_to_sim is 0 for all sources or all sources return 0 events
        ds = pd.concat([pd.DataFrame()] + ds, sort=False)
        return ds.sample(frac=1).reset_index(drop=True)

    def __call__(self, **kwargs):
        assert 'second_order' not in kwargs, 'Roep gewoon log_likelihood aan'
        return self.log_likelihood(second_order=False, **kwargs)[0].numpy()

    def log_likelihood(self, autograph=True, second_order=False,
                       omit_grads=tuple(), **kwargs):
        if second_order:
            # Compute the likelihood, jacobian and hessian
            f = self._log_likelihood_grad2
        else:
            # Computes the likelihood and jacobian
            f = self._log_likelihood

        params = self.prepare_params(kwargs)
        n_grads = len(self.param_defaults) - len(omit_grads)
        ll = tf.constant(0., dtype=fd.float_type())
        llgrad = tf.zeros(n_grads, dtype=fd.float_type())
        llgrad2 = tf.zeros((n_grads, n_grads), dtype=fd.float_type())

        for dsetname in self.dsetnames:
            for i_batch in tf.range(self.n_batches[dsetname], dtype=fd.int_type()):
                v = f(i_batch, dsetname, autograph, omit_grads=omit_grads, **params)
                ll += v[0]
                llgrad += v[1]
                if second_order:
                    llgrad2 += v[2]

        if second_order:
            return ll, llgrad, llgrad2
        return ll, llgrad

    def minus_ll(self, *, autograph=True, omit_grads=tuple(), **kwargs):
        ll, grad = self.log_likelihood(
            autograph=autograph, omit_grads=omit_grads, **kwargs)
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

    def mu(self, dsetname, **kwargs):
        mu = tf.constant(0., dtype=fd.float_type())
        for sname, s in self.sources.items():
            if self.d_for_s[sname] != dsetname:
                continue
            mu += (self._get_rate_mult(sname, kwargs)
                   * self.mu_itps[sname](**self._filter_source_kwargs(kwargs, sname)))
        return mu

    def _log_likelihood(self, i_batch, dsetname, autograph,
                        omit_grads=tuple(), **params):
        if omit_grads is None:
            omit_grads = []
        grad_par_list = [x for k, x in params.items()
                         if k not in omit_grads]
        with tf.GradientTape() as t:
            t.watch(grad_par_list)
            ll = self._log_likelihood_inner(
                i_batch, params, dsetname, autograph)
        grad = t.gradient(ll, grad_par_list)
        return ll, tf.stack(grad)

    def _log_likelihood_grad2(self, i_batch, dsetname, autograph,
                              omit_grads=tuple(), **params):
        if omit_grads is None:
            omit_grads = []
        grad_par_list = [x for k, x in params.items()
                         if k not in omit_grads]
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(grad_par_list)
            with tf.GradientTape() as t:
                t.watch(grad_par_list)
                ll = self._log_likelihood_inner(i_batch, params,
                                                dsetname, autograph)
            grads = t.gradient(ll, grad_par_list)
        hessian = [t2.gradient(grad, grad_par_list) for grad in grads]
        del t2
        return ll, tf.stack(grads), tf.stack(hessian)

    def _log_likelihood_inner(self, i_batch, params, dsetname, autograph):
        # Does for loop over datasets and sources, not batches
        # Sum over sources is first in likelihood

        # Compute differential rates from all sources
        # drs = list[n_sources] of [n_events] tensors
        drs = tf.zeros((self.batch_size[dsetname],), dtype=fd.float_type())
        for sname, s in self.sources.items():
            if not self.d_for_s[sname] == dsetname:
                continue
            rate_mult = self._get_rate_mult(sname, params)
            dr = s.differential_rate(s.data_tensor[i_batch],
                                     autograph=autograph,
                                     **self._filter_source_kwargs(params, sname))
            drs += dr * rate_mult

        # Sum over events and remove padding
        n = tf.where(tf.equal(i_batch,
                              tf.constant(self.n_batches[dsetname] - 1,
                                          dtype=fd.int_type())),
                     self.batch_size[dsetname] - self.n_padding[dsetname],
                     self.batch_size[dsetname])
        ll = tf.reduce_sum(tf.math.log(drs[:n]))

        # Add mu once (to the first batch)
        # and constraint really only once (to first batch of first dataset)
        ll += tf.where(tf.equal(i_batch, tf.constant(0, dtype=fd.int_type())),
                       -self.mu(dsetname, **params)
                           + (self.log_constraint(**params)
                              if dsetname == self.dsetnames[0] else 0.),
                       0.)
        return ll

    def guess(self):
        """Return dictionary of parameter guesses"""
        return self.param_defaults

    def params_to_dict(self, values):
        """Return parameter {name: value} dictionary"""
        return {k: v for k, v in zip(self.param_names, tf.unstack(values))}

    def bestfit(self,
                guess=None,
                fix=None,
                optimizer='scipy',
                llr_tolerance=0.01,
                get_lowlevel_result=False,
                use_hessian=True,
                return_errors=False,
                autograph=True,
                nan_val=float('inf'),
                **kwargs):
        """Return best-fit parameter dict

        :param guess: Guess parameters: dict {param: guess} of guesses to use.
        :param fix: dict {param: value} of parameters to keep fixed
        during the minimzation.
        :param optimizer: 'tf', 'minuit' or 'scipy'
        :param llr_tolerance: stop minimizer if change in -2 log likelihood
        becomes less than this (roughly: using guess to convert to
        relative tolerance threshold)
        :param get_lowlevel_result: Returns the full optimizer result instead
        of only the best fit parameters. Bool.
        :param use_hessian: Passes the hessian estimated at the guess to the
        optimizer. Bool.
        :param return_errors: If using the minuit minimizer, instead return
        a 2-tuple of (bestfit dict, error dict).
        :param autograph: If true (default), use tensorflow's autograph
        during minimization.
        """
        if fix is None:
            fix = dict()
        if guess is None:
            guess = dict()

        # Names of parameters we want to fit
        arg_names = [k for k in self.param_names if k not in fix]

        if not isinstance(guess, dict):
            raise ValueError("Guess must be a dictionary")
        # These are all the parameters and their default or fixed values
        guess = {**self.guess(), **guess, **fix}
        # Build x_guess list
        x_guess = np.array([guess[k] for k in arg_names])

        Optimizer = fd.get_bestfit_objective(optimizer)
        obj = Optimizer(lf=self, arg_names=arg_names, fix=fix,
                        autograph=autograph, nan_val=nan_val)
        res = obj.minimize(x_guess, get_lowlevel_result=get_lowlevel_result,
                           use_hessian=use_hessian,
                           llr_tolerance=llr_tolerance, **kwargs)

        names = list(arg_names) + list(fix.keys())
        result, errors = (
            {k: v for k, v in res.items() if k in names},
            {k: v for k, v in res.items() if k.startswith('error_')})

        if return_errors:
            return result, errors
        return result

    def one_parameter_interval(self, parameter,
                               guess=None,
                               bestfit=None,
                               fix=None,
                               confidence_level=0.9, kind='upper', t_ppf=None,
                               t_ppf_grad=None, optimizer='scipy',
                               llr_tolerance=0.005):
        """Compute upper/lowel/central interval of parameter at confidence
        level assuming Wilk's theorem if t_ppf=None. Use critical value
        curve t_ppf for non-asymptotic case.

        :param parameter: string, the parameter to set the interval on
        :param bestfit: dictionary of parameter, value pairs, global best-fit.
        :param guess: dictionary of parameter, value pairs, guess
        (only necessary if not providing bestfit)
        :param confidence_level: The confidence level of the method
        :param kind: 'upper', 'lower' or 'central' type of limit/interval
        :param t_ppf: returns critical value as function of parameter
        :param llr_tolerance: see bestfit

        Return limit if kind='upper' or 'lower', returns interval if 'central'
        """
        if bestfit is None:
            assert guess is not None, "Provide either bestfit or guess"
            # Determine global bestfit and global minimum -2lnL
            bestfit = self.bestfit(guess, fix=fix, optimizer=optimizer)
        # TODO, we can avoid this call and have bestfit return ll_best
        ll_best = self.minus_ll(**bestfit)[0]

        # Set (bound, critical_quantile) for the desired kind of limit
        if kind == 'upper':
            limits = [((bestfit[parameter], None),  # UL bound
                       confidence_level)]          # UL critical quantile
        elif kind == 'lower':
            limits = [((None, bestfit[parameter]),  # LL bound
                       1 - confidence_level)]      # LL critical quantile
        elif kind == 'central':
            limits = [((None, bestfit[parameter]),        # LL bound
                       (1 - confidence_level) / 2),      # LL critical quantile
                      ((bestfit[parameter], None),        # UL bound
                       1 - (1 - confidence_level) / 2)]  # UL critical quantile
        else:
            raise ValueError(f"kind must be upper/lower/central but is {kind}")

        res = [fd.one_parameter_interval(self, parameter, bound, guess,
                                         ll_best, critical_quantile=q,
                                         optimizer=optimizer, fix=fix,
                                         t_ppf=t_ppf, t_ppf_grad=t_ppf_grad,
                                         llr_tolerance=llr_tolerance)
               for bound, q in limits]

        if len(res) == 1:
            return res[0]
        return res

    def inverse_hessian(self, params, omit_grads=tuple()):
        """Return inverse hessian (square tensor)
        of -2 log_likelihood at params
        """
        # Also Tensorflow has tf.hessians, but:
        # https://github.com/tensorflow/tensorflow/issues/29781

        # Get second order derivatives of likelihood at params
        _, _, grad2_ll = self.log_likelihood(**params,
                                             autograph=False,
                                             omit_grads=omit_grads,
                                             second_order=True)

        return tf.linalg.inv(-2 * grad2_ll)

    def summary(self, bestfit=None, fix=None, guess=None,
                inverse_hessian=None, precision=3):
        """Print summary information about best fit"""
        if fix is None:
            fix = dict()
        if bestfit is None:
            bestfit = self.bestfit(guess=guess, fix=fix)

        params = {**bestfit, **fix}
        if inverse_hessian is None:
            inverse_hessian = self.inverse_hessian(
                params,
                omit_grads=tuple(fix.keys()))
        inverse_hessian = fd.tf_to_np(inverse_hessian)

        stderr, cov = cov_to_std(inverse_hessian)

        var_par_i = 0
        for i, pname in enumerate(self.param_names):
            if pname in fix:
                print("{pname}: {x:.{precision}g} (fixed)".format(
                    pname=pname, x=fix[pname], precision=precision))
            else:
                template = "{pname}: {x:.{precision}g} +- {xerr:.{precision}g}"
                print(template.format(
                    pname=pname,
                    x=bestfit[pname],
                    xerr=stderr[var_par_i],
                    precision=precision))
                var_par_i += 1

        var_pars = [x for x in self.param_names if x not in fix]
        df = pd.DataFrame(
            {p1: {p2: cov[i1, i2]
                  for i2, p2 in enumerate(var_pars)}
             for i1, p1 in enumerate(var_pars)},
            columns=var_pars)

        # Get rows in the correct order
        df['index'] = [var_pars.index(x)
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
