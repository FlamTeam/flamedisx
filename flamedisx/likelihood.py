from copy import deepcopy
import warnings

import flamedisx as fd
import numpy as np
import pandas as pd
import tensorflow as tf
import typing as ty


export, __all__ = fd.exporter()

o = tf.newaxis
DEFAULT_DSETNAME = 'the_dataset'


@export
class LogLikelihood:
    param_defaults: ty.Dict[str, float]

    dsetnames: ty.List[str]              # Dataset names
    sources: ty.Dict[str, fd.Source]     # Source name -> Source instance

    # Track which source takes which dataset, and converse
    dset_for_source = ty.Dict[str, str]
    sources_in_dset = ty.Dict[str, str]

    # Tensor with batch info
    # First dimension runs over datasets (same indices as dsetnames)
    # Second dimension is (n_batches, batch_size, n_padding)
    batch_info: tf.Tensor

    dsetnames: ty.List

    # Concatenated data tensors for all sources in each dataset
    # dsetname -> Tensor
    data_tensors: ty.Dict[str, tf.Tensor]

    # Track which columns in the data tensor belong to which sources
    # dsetname -> [start, stop] array over sources
    column_indices: ty.Dict[str, np.ndarray]

    def __init__(
            self,
            sources: ty.Union[
                ty.Dict[str, fd.Source.__class__],
                ty.Dict[str, ty.Dict[str, fd.Source.__class__]]],
            data: ty.Union[
                None,
                pd.DataFrame,
                ty.Dict[str, pd.DataFrame]] = None,
            free_rates=None,
            batch_size=10,
            max_sigma=3,
            n_trials=int(1e5),
            log_constraint=None,
            bounds_specified=True,
            **common_param_specs):
        """

        :param sources: Dictionary {datasetname : {sourcename: class,, ...}, ...}
        or just {sourcename: class} in case you have one dataset
        Every source name must be unique.

        :param data: Dictionary {datasetname: pd.DataFrame}
        or just pd.DataFrame if you have one dataset or None if you
        set data later.

        :param free_rates: names of sources whose rates are floating

        :param batch_size: Number of events to use for a computation batch.
            Higher numbers give better performance, especially for GPUs,
            at the cost of more memory

        :param max_sigma: Maximum sigma to use in bounds estimation.
            Higher numbers give better accuracy, at the cost of performance.

        :param n_trials: Number of Monte-Carlo trials for mu estimation.

        :param bounds_specified: If True (default), optimizers will be
            constrained within the specified parameter ranges.

        :param log_constraint: Logarithm of constraint to include in likelihood

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
        self.dset_for_source = dict()
        self.sources_in_dset = dict()
        for dsetname, ss in sources.items():
            self.sources_in_dset[dsetname] = []
            for sname, s in ss.items():
                self.dset_for_source[sname] = dsetname
                self.sources_in_dset[dsetname].append(sname)
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
            sname: sclass.find_defaults()[2]
            for sname, sclass in self.sources.items()}

        # Create sources
        self.sources = {
            sname: sclass(data=None,
                          max_sigma=max_sigma,
                          fit_params=list(k for k in common_param_specs.keys()
                                          if k in defaults_in_sources[sname].keys()),
                          batch_size=batch_size)
            for sname, sclass in self.sources.items()}

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

        self.param_defaults = fd.values_to_constants(param_defaults)
        self.param_names = list(param_defaults.keys())

        if bounds_specified:
            self.default_bounds = {
                p_name: (start, stop)
                for p_name, (start, stop, n) in common_param_specs.items()}
        else:
            self.default_bounds = dict()

        self.mu_itps = {
            sname: s.mu_function(
                n_trials=n_trials,
                **{p_name: par
                   for p_name, par in common_param_specs.items()
                   if p_name in defaults_in_sources[sname].keys()})
            for sname, s in self.sources.items()}
        # Not used, but useful for mu smoothness diagnosis
        self.param_specs = common_param_specs

        # Add the constraint
        if log_constraint is None:
            log_constraint = lambda **kwargs: 0.
        self.log_constraint = log_constraint

        self.set_data(data)

    def set_data(self,
                 data: ty.Union[pd.DataFrame, ty.Dict[str, pd.DataFrame]]):
        """set new data for sources in the likelihood.
        Data is passed in the same format as for __init__
        Data can contain any subset of the original data keys to only
        update specific datasets.
        """
        if isinstance(data, pd.DataFrame):
            assert len(self.dsetnames) == 1, \
                "You passed one DataFrame but there are multiple datasets"
            data = {DEFAULT_DSETNAME: data}

        is_none = [d is None for d in data.values()]
        if any(is_none):
            if not all(is_none):
                warnings.warn("Cannot set only one dataset to None: "
                              "setting all to None instead.",
                              UserWarning)
            for s in self.sources.values():
                s.set_data(None)
                return

        batch_info = np.zeros((len(self.dsetnames), 3), dtype=np.int)
        adjusted_rate_for = {d: False for d in self.dsetnames}

        for sname, source in self.sources.items():
            dname = self.dset_for_source[sname]
            if dname not in data:
                warnings.warn(f"Dataset {dname} not provided in set_data")
                continue

            # Copy ensures annotations don't clobber
            source.set_data(deepcopy(data[dname]))

            # Update batch info
            dset_index = self.dsetnames.index(dname)
            batch_info[dset_index, :] = [
                source.n_batches, source.batch_size, source.n_padding]

            # If the rate is free, update the rate multiplier default
            # (i.e. the fallback guess) once per dataset.
            # That is, our guess will be that the first free source produces
            # enough events to explain the observed event count.
            rmname = sname + '_rate_multiplier'
            if rmname in self.param_names and not adjusted_rate_for[dname]:
                mu_dset = self.mu(dsetname=dname)
                mu_source = self.mu(source=sname)
                mu_others = mu_dset - mu_source
                n_observed = len(data[dname])
                self.param_defaults[rmname] *= (
                        (n_observed - mu_others) / mu_source)
                adjusted_rate_for[dname] = True

        self.batch_info = tf.convert_to_tensor(batch_info, dtype=fd.int_type())

        # Build a big data tensor for each dataset.
        # Each source has an [n_batches, batch_size, n_columns] tensor.
        # Since the number of columns are different, we must concat along
        # axis=2 and track which indices belong to which source.
        self.data_tensors = {
            dsetname: tf.concat(
                [self.sources[sname].data_tensor
                 for sname in self.sources_in_dset[dsetname]],
                axis=2)
            for dsetname in self.dsetnames}

        self.column_indices = dict()
        for dsetname in self.dsetnames:
            # Do not use len(cols_to_cache), some sources have extra columns...
            stop_idx = np.cumsum([self.sources[sname].data_tensor.shape[2]
                                  for sname in self.sources_in_dset[dsetname]])
            self.column_indices[dsetname] = np.transpose([
                np.concatenate([[0], stop_idx[:-1]]),
                stop_idx])

    def simulate(self, fix_truth=None, **params):
        """Simulate events from sources.
        """
        # Collect Source event DFs in ds
        ds = []
        for sname, s in self.sources.items():
            # mean number of events to simulate, rate mult times mu before
            # efficiencies, the simulator deals with the efficiencies
            rm = self._get_rate_mult(sname, params)
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

        # Concatenate results and shuffle them.
        # Adding empty DataFrame ensures pd.concat doesn't fail if
        # n_to_sim is 0 for all sources or all sources return 0 events
        ds = pd.concat([pd.DataFrame()] + ds, sort=False)
        return ds.sample(frac=1).reset_index(drop=True)

    def __call__(self, **kwargs):
        assert 'second_order' not in kwargs, 'Roep gewoon log_likelihood aan'
        return self.log_likelihood(second_order=False, **kwargs)[0]

    def log_likelihood(self, second_order=False,
                       omit_grads=tuple(), **kwargs):
        params = self.prepare_params(kwargs)
        n_grads = len(self.param_defaults) - len(omit_grads)
        ll = 0.
        llgrad = np.zeros(n_grads, dtype=np.float64)
        llgrad2 = np.zeros((n_grads, n_grads), dtype=np.float64)

        for dsetname in self.dsetnames:
            # Getting this from the batch_info tensor is much slower
            n_batches = self.sources[self.sources_in_dset[dsetname][0]].n_batches

            for i_batch in range(n_batches):
                # Iterating over tf.range seems much slower!
                results = self._log_likelihood(
                    tf.constant(i_batch, dtype=fd.int_type()),
                    dsetname=dsetname,
                    data_tensor=self.data_tensors[dsetname][i_batch],
                    batch_info=self.batch_info,
                    omit_grads=omit_grads,
                    second_order=second_order,
                    **params)
                ll += results[0].numpy().astype(np.float64)

                if len(self.param_names):
                    llgrad += results[1].numpy().astype(np.float64)
                    if second_order:
                        llgrad2 += results[2].numpy().astype(np.float64)

        if second_order:
            return ll, llgrad, llgrad2
        return ll, llgrad, None

    def minus2_ll(self, *, omit_grads=tuple(), **kwargs):
        result = self.log_likelihood(omit_grads=omit_grads, **kwargs)
        ll, grad = result[:2]
        hess = -2 * result[2] if result[2] is not None else None
        return -2 * ll, -2 * grad, hess

    def prepare_params(self, kwargs):
        for k in kwargs:
            if k not in self.param_defaults:
                raise ValueError(f"Unknown parameter {k}")
        return {**self.param_defaults, **fd.values_to_constants(kwargs)}

    def _get_rate_mult(self, sname, kwargs):
        rmname = sname + '_rate_multiplier'
        if rmname in self.param_names:
            return kwargs.get(rmname, self.param_defaults[rmname])
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

    def mu(self, dsetname=None, source=None, **kwargs):
        """Return expected number of events
        :param dsetname: ... for just this dataset
        :param source: ... for just this source.
        You must provide either dsetname or source, since it makes no sense to
        add events from multiple dataset
        """
        kwargs = {**self.param_defaults, **kwargs}
        if dsetname is None and source is None:
            raise ValueError("Provide either dsetname or source")
        mu = tf.constant(0., dtype=fd.float_type())
        for sname, s in self.sources.items():
            if dsetname is not None and self.dset_for_source[sname] != dsetname:
                continue
            if source is not None and sname != source:
                continue
            mu += (self._get_rate_mult(sname, kwargs)
                   * self.mu_itps[sname](**self._filter_source_kwargs(kwargs, sname)))
        return mu

    @tf.function
    def _log_likelihood(self,
                        i_batch, dsetname, data_tensor, batch_info,
                        omit_grads=tuple(), second_order=False, **params):
        # Stack the params to create a single node
        # to differentiate with respect to.
        grad_par_stack = tf.stack([
            params[k] for k in self.param_names
            if k not in omit_grads])

        # Retrieve individual params from the stacked node,
        # then add back the params we do not differentiate w.r.t.
        params_unstacked = dict(zip(
            [x for x in self.param_names if x not in omit_grads],
            tf.unstack(grad_par_stack)))
        for k in omit_grads:
            params_unstacked[k] = params[k]

        # Forward computation
        ll = self._log_likelihood_inner(
            i_batch, params_unstacked, dsetname, data_tensor, batch_info)

        # Autodifferentiation. This is why we use tensorflow:
        grad = tf.gradients(ll, grad_par_stack)[0]
        if second_order:
            return ll, grad, tf.hessians(ll, grad_par_stack)[0]
        return ll, grad, None

    def _log_likelihood_inner(self, i_batch, params,
                              dsetname, data_tensor, batch_info):
        """Return log likelihood contribution of one batch in a dataset

        This loops over sources in the dataset and events in the batch,
        but not not over datasets or batches.
        """
        # Retrieve batching info. Cannot use tuple-unpacking, tensorflow
        # doesn't like it when you iterate over tenstors
        dataset_index = self.dsetnames.index(dsetname)
        n_batches = batch_info[dataset_index, 0]
        batch_size = batch_info[dataset_index, 1]
        n_padding = batch_info[dataset_index, 2]

        # Compute differential rates from all sources
        # drs = list[n_sources] of [n_events] tensors
        drs = tf.zeros((batch_size,), dtype=fd.float_type())
        for source_i, sname in enumerate(self.sources_in_dset[dsetname]):
            s = self.sources[sname]
            rate_mult = self._get_rate_mult(sname, params)

            col_start, col_stop = self.column_indices[dsetname][source_i]
            dr = s.differential_rate(
                data_tensor[:, col_start:col_stop],
                # We are already tracing; if we call the traced function here
                # it breaks the Hessian (it will give NaNs)
                autograph=False,
                **self._filter_source_kwargs(params, sname))
            drs += dr * rate_mult

        # Sum over events and remove padding
        n = tf.where(tf.equal(i_batch, n_batches - 1),
                     batch_size - n_padding,
                     batch_size)
        ll = tf.reduce_sum(tf.math.log(drs[:n]))

        # Add mu once (to the first batch)
        # and constraint really only once (to first batch of first dataset)
        ll += tf.where(tf.equal(i_batch, tf.constant(0, dtype=fd.int_type())),
                       -self.mu(dsetname=dsetname, **params)
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
                bounds=None,
                optimizer='scipy',
                get_lowlevel_result=False,
                get_history=False,
                use_hessian=True,
                return_errors=False,
                nan_val=float('inf'),
                optimizer_kwargs=None,
                allow_failure=False):
        """Return best-fit parameter dict

        :param guess: Guess parameters: dict {param: guess} of guesses to use.
        Any omitted parameters will be guessed at LogLikelihood.defaults()
        :param fix: dict {param: value} of parameters to keep fixed
        during the minimzation.
        :param optimizer: 'tf', 'minuit' or 'scipy'
        :param get_lowlevel_result: Returns the full optimizer result instead
        of the best fit parameters. Bool.
        :param get_history: Returns the history of optimizer calls instead
        of the best fit parameters. Bool.
        :param use_hessian: If True, uses flamedisxs' exact Hessian
        in the optimizer. Otherwise, most optimizers estimate it by finite-
        difference calculations.
        :param return_errors: If using the minuit minimizer, instead return
        a 2-tuple of (bestfit dict, error dict).
        In case optimizer is minuit, you can also pass 'hesse' or 'minos' here.
        :param allow_failure: If True, raise a warning instead of an exception
        if there is an optimizer failure.
        """
        if bounds is None:
            bounds = dict()
        if guess is None:
            guess = dict()
        if not isinstance(guess, dict):
            raise ValueError("Must specify bestfit guess as a dictionary")

        # Check the likelihood has a finite value and gradient before starting
        val, grad, hess = self.log_likelihood(**guess,
                                              second_order=use_hessian)
        if not np.isfinite(val):
            raise ValueError("The likelihood is - infinity at your guess, "
                             "please guess better, remove outlier events, or "
                             "add a fallback background model.")
        if not np.all(np.isfinite(grad)):
            raise ValueError("The likelihood is finite at your guess, "
                             "but the gradient is not. Are you starting at a "
                             "cusp?")
        if use_hessian:
            if hess is None:
                raise RuntimeError("Likelihood did't provide Hessian!")
            if not np.all(np.isfinite(hess)):
                raise ValueError("The likelihood and gradient are finite at "
                                 "your guess, but the Hessian is not. "
                                 "Are you starting at an unusual point? "
                                 "You could also try use_hessian=False.")

        opt = fd.SUPPORTED_OPTIMIZERS[optimizer]
        res = opt(
            lf=self,
            guess={**self.guess(), **guess},
            fix=fix,
            bounds={**self.default_bounds, **bounds},
            nan_val=nan_val,
            get_lowlevel_result=get_lowlevel_result,
            get_history=get_history,
            use_hessian=use_hessian,
            return_errors=return_errors,
            optimizer_kwargs=optimizer_kwargs,
            allow_failure=allow_failure,
        ).minimize()
        if get_lowlevel_result or get_history:
            return res

        # TODO: This is to deal with a minuit-specific convention,
        # should either put this to minuit or force others into same mold.
        names = self.param_names
        result, errors = (
            {k: v for k, v in res.items() if k in names},
            {k: v for k, v in res.items() if k.startswith('error_')})
        if return_errors:
            # Filter out errors and return separately
            return result, errors

        return result

    def interval(self, parameter, **kwargs):
        """Return central confidence interval on parameter.
        Options are the same as for limit."""
        kwargs.setdefault('kind', 'central')
        return self.limit(parameter, **kwargs)

    def limit(
            self,
            parameter,
            bestfit=None,
            guess=None,
            fix=None,
            bounds=None,
            confidence_level=0.9,
            kind='upper',
            sigma_guess=None,
            t_ppf=None,
            t_ppf_grad=None,
            t_ppf_hess=None,
            optimizer='scipy',
            get_history=False,
            get_lowlevel_result=False,
            # Set so 90% CL intervals actually report ~90.25% intervals
            # asymptotically due to the tilt... if a pen-and-paper computation
            # Jelle did a long time ago is actually correct
            tilt_overshoot=0.037,
            optimizer_kwargs=None,
            use_hessian=True,
            allow_failure=False,
    ):
        """Return frequentist limit or confidence interval

        :param parameter: string, the parameter to set the interval on
        :param bestfit: {parameter: value} dictionary, global best-fit.
        If omitted, will compute it using bestfit.
        :param guess: {param: value} guess of the result, or None.
        If omitted, nuisance parameters will be guessed equal to bestfit.
        If omitted, guess for target parameters will be based on asymptotic
        parabolic computation.
        :param fix: {param: value} to fix during interval computation.
        Result is only valid if same parameters were fixed for bestfit.
        :param confidence_level: Requried confidence level of the interval
        :param kind: Type of interval, 'upper', 'lower' or 'central'
        :param sigma_guess: Guess for one sigma uncertainty on the target
        parameter. If not provided, will be computed from Hessian.
        :param t_ppf: returns critical value as function of parameter
        Use Wilks' theorem if omitted.
        :param t_ppf_grad: return derivative of t_ppf
        :param t_ppf_hess: return second derivative of t_ppf
        :param tilt_overshoot: Set tilt so the limit's log likelihood will
        overshoot the target value by roughly this much.
        :param optimizer_kwargs: dict of additional arguments for optimizer
        :param allow_failure: If True, raise a warning instead of an exception
        if there is an optimizer failure.
        :param use_hessian: If True, uses flamedisxs' exact Hessian
        in the optimizer. Otherwise, most optimizers estimate it by finite-
        difference calculations.

        Returns a float (for upper or lower limits)
        or a 2-tuple of floats (for a central interval)
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        if bounds is None:
            bounds = dict()

        if bestfit is None:
            # Determine global bestfit
            if optimizer=='nlin':
                # This optimizer is only for interval setting.
                # Use scipy to get best-fit first
                bestfit = self.bestfit(fix=fix, optimizer='scipy')
            else:
                bestfit = self.bestfit(fix=fix, optimizer=optimizer)

        lower_bound = None
        if parameter.endswith('rate_multiplier'):
            lower_bound = fd.LOWER_RATE_MULTIPLIER_BOUND

        # Set (bound, critical_quantile) for the desired kind of limit
        if kind == 'upper':
            requested_limits = [
                dict(bound=(bestfit[parameter], None),
                     crit=confidence_level,
                     direction=1,
                     guess=guess)]
        elif kind == 'lower':
            requested_limits = [
                dict(bound=(lower_bound, bestfit[parameter]),
                     crit=1 - confidence_level,
                     direction=-1,
                     guess=guess)]
        elif kind == 'central':
            if guess is None:
                guess = (None, None)
            elif not isinstance(guess, tuple) or not len(guess) == 2:
                raise ValueError("Guess for central interval must be a 2-tuple")
            requested_limits = [
                dict(bound=(lower_bound, bestfit[parameter]),
                     crit=(1 - confidence_level) / 2,
                     direction=-1,
                     guess=guess[0]),
                dict(bound=(bestfit[parameter], None),
                     direction=+1,
                     crit=1 - (1 - confidence_level) / 2,
                     guess=guess[1])]
        else:
            raise ValueError(f"kind must be upper/lower/central but is {kind}")

        result = []
        for req in requested_limits:
            opt = fd.SUPPORTED_INTERVAL_OPTIMIZERS[optimizer]

            res = opt(
                # To generic objective
                lf=self,
                guess=req['guess'],
                fix=fix,
                bounds={
                    **self.default_bounds,
                    parameter: req['bound'],
                    **bounds},
                # TODO: nan_val
                get_lowlevel_result=get_lowlevel_result,
                get_history=get_history,
                use_hessian=use_hessian,
                optimizer_kwargs=optimizer_kwargs,
                allow_failure=allow_failure,

                # To IntervalObjective
                target_parameter=parameter,
                bestfit=bestfit,
                direction=req['direction'],
                critical_quantile=req['crit'],
                tilt_overshoot=tilt_overshoot,
                sigma_guess=sigma_guess,
                t_ppf=t_ppf,
                t_ppf_grad=t_ppf_grad,
                t_ppf_hess=t_ppf_hess,
            ).minimize()
            if get_lowlevel_result or get_history:
                result.append(res)
            else:
                result.append(res[parameter])

        if len(result) == 1:
            return result[0]
        return result

    def inverse_hessian(self, params, omit_grads=tuple()):
        """Return inverse hessian (square tensor)
        of -2 log_likelihood at params
        """
        # Also Tensorflow has tf.hessians, but:
        # https://github.com/tensorflow/tensorflow/issues/29781

        # Get second order derivatives of likelihood at params
        _, _, grad2_ll = self.log_likelihood(**params,
                                             omit_grads=omit_grads,
                                             second_order=True)

        return np.linalg.inv(-2 * grad2_ll)

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
