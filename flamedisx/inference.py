import flamedisx as fd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import typing as ty

export, __all__ = fd.exporter()


@export
class LogLikelihood:
    param_defaults: ty.Dict[str, float]
    data: pd.DataFrame
    sources: ty.Dict[str, fd.ERSource]
    mu_iterpolators: ty.Dict[str, ty.Callable]

    def __init__(
            self,
            sources: ty.Dict[str, fd.ERSource],
            data: pd.DataFrame,
            # source_params: ty.Union[
            #     None, ty.Dict[str, ty.Dict[str, tuple]]] = None,
            free_rates: ty.Union[None, str, ty.Tuple[str]] = None,
            n_trials=int(1e5),
            **common_params):

        # if source_params is None:
        #     source_params = dict()

        param_defaults = dict()

        if free_rates is None:
            free_rates = tuple()
        if isinstance(free_rates, str):
            free_rates = (free_rates,)
        for sn in free_rates:
            if sn not in sources:
                raise ValueError(f"Can't free rate of unknown source {sn}")
            param_defaults[sn + '_rate_multiplier'] = 1

        for pname in common_params:
            # Check defaults for common parameters are consistent between
            # sources
            defs = [s.defaults[pname] for s in sources.values()]
            if len(set([x.numpy() for x in defs])) > 1:
                raise ValueError(
                    f"Inconsistent defaults {defs} for common parameters")
            param_defaults[pname] = defs[0]

        # Set data. Have to copy it, since data is modified by set_data
        for sname, s in sources.items():
            s.set_data(data.copy())

        self.data = data
        self.sources = sources
        self.param_defaults = param_defaults
        self.param_names = list(param_defaults.keys())
        self.mu_itps = {
            sname: s.mu_interpolator(n_trials=n_trials,
                                     **common_params)
            for sname, s in sources.items()}

        # Not used, but useful for mu smoothness diagnosis
        self.param_specs = common_params

    @tf.function
    def log_likelihood(self, ptensor):
        return self._log_likelihood(ptensor)

    @tf.function
    def minus_ll(self, ptensor):
        return self._minus_ll(ptensor)

    def _log_likelihood(self, ptensor):
        if not len(ptensor) == len(self.param_names):
            raise ValueError(
                f"Likelihood takes {len(self.param_names)} params "
                f"but you gave {len(ptensor)}")

        mu = tf.constant(0., dtype=fd.float_type())
        lls = tf.zeros(len(self.data), dtype=fd.float_type())

        for sname, s in self.sources.items():
            rmname = sname + '_rate_multiplier'
            if rmname in self.param_names:
                rm = ptensor[self._param_i(rmname)]
            else:
                rm = 1.
            source_kwargs = self._source_kwargs(ptensor)

            mu += rm * self.mu_itps[sname](**source_kwargs)
            lls += rm * s.likelihood(**source_kwargs)

        return -mu + tf.reduce_sum(fd.tf_log10(lls))

    def _minus_ll(self, ptensor):
        return -2 * self._log_likelihood(ptensor)

    def guess(self):
        """Return tensor of parameter guesses"""
        return tf.convert_to_tensor(
            list(self.param_defaults.values()),
            fd.float_type())

    def _source_kwargs(self, ptensor):
        """Return {param: value} dictionary with keyword arguments
        for source, with values extracted from ptensor"""
        return {pname: ptensor[self._param_i(pname)]
                for pname in self.param_names
                if not pname.endswith('_rate_multiplier')}

    def _param_i(self, pname):
        """Return index of parameter pname"""
        return self.param_names.index(pname)

    def params_to_dict(self, values):
        """Return parameter {name: value} dictionary"""
        values = fd.tf_to_np(values)
        return {k: v
                for k, v in zip(self.param_names, values)}

    def bestfit(self, guess=None, optimizer = tfp.optimizer.VariationalSGD(batch_size=10,
                                               total_num_examples=len(self.data)),
            #optimizer=tfp.optimizer.lbfgs_minimize,
                llr_tolerance=0.01,
                get_lowlevel_result=False, **kwargs):
        """Return best-fit parameter tensor

        :param guess: Guess parameters: array or tensor of same length
        as param_names.
        If omitted, will use guess from source defaults.
        :param llr_tolerance: stop minimizer if change in -2 log likelihood
        becomes less than this (roughly: using guess to convert to
        relative tolerance threshold)
        """
        if guess is None:
            guess = self.guess()
        guess = fd.np_to_tf(guess)

        # Unfortunately we can only set the relative tolerance for the
        # objective; we'd like to set the absolute one.
        # Use the guess log likelihood to normalize;
        if llr_tolerance is not None:
            kwargs.setdefault('f_relative_tolerance',
                              llr_tolerance/self._minus_ll(guess))

        # Minimize multipliers to the guess, rather than the guess itself
        # This is a basic kind of standardization that helps make the gradient
        # vector reasonable.
        x_norm = tf.ones(len(guess), dtype=fd.float_type())
        @tf.function
        def objective(x_norm):
            y = self._minus_ll(x_norm * guess)
            #with tf.GradientTape() as t:
            #    t.watch(x_norm)
            #    y = self._minus_ll(x_norm * guess)
            #return y, t.gradient(y, x_norm)
            return y, optimizer.get_gradients(y,x_norm)

        res = optimizer.minimize(objective, x_norm, **kwargs)
        if get_lowlevel_result:
            return res
        if res.failed:
            raise ValueError(f"Optimizer failure! Result: {res}")
        return res.position * guess

    def inverse_hessian(self, params, save_ram=True):
        """Return inverse hessian (square numpy matrix)
        of -2 log_likelihood at params
        """
        # TODO: add more memory-efficient computation method

        # I could only get higher-order derivatives to work
        # after splitting the parameter vector in separate variables,
        # and using the un-@tf.function'ed likelihood.
        #
        # Tensorflow has tf.hessians, but:
        # https://github.com/tensorflow/tensorflow/issues/29781

        #xc = [tf.Variable(q)
        #      for q in fd.tf_to_np(params)]

        if save_ram:
            # Slower but more RAM-efficient algorithm
            n = len(self.param_names)
            hessian = np.zeros((n, n))
            for i1 in tqdm(range(n),
                    desc='Computing hessian'):
                for i2 in range(n):
                    if i2 > i1:
                        continue

                    xc = [tf.constant(q)
                            for q in fd.tf_to_np(params) 
                            ]
                    with tf.GradientTape(persistent=True) as t2:
                        t2.watch(xc[i2])
                        with tf.GradientTape() as t:
                            t.watch(xc[i1])
                            ptensor = tf.stack(xc)
                            y = self._minus_ll(ptensor)
                        grad = t.gradient(y, xc[i1])
                        hessian[i1, i2] = t2.gradient(grad, xc[i2]).numpy()
                    del t2
            for i1 in range(n):
                for i2 in range(n):
                    if i2 > i1:
                        hessian[i1, i2] = hessian[i2, i1]
            return np.linalg.inv(hessian)

        # Faster RAM-guzzling algorithm
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape(persistent=True) as t:
                ptensor = tf.stack(xc)
                y = self._minus_ll(ptensor)

            grads = [t.gradient(y, q) for q in xc]

        hessian = np.vstack(
            [np.array([t2.gradient(g, x)
                       for x in xc])
             for g in grads])

        return np.linalg.inv(hessian)
    def hessian(self, params, save_ram=True):
        """Return inverse hessian (square numpy matrix)
        of -2 log_likelihood at params
        """
        # TODO: add more memory-efficient computation method

        # I could only get higher-order derivatives to work
        # after splitting the parameter vector in separate variables,
        # and using the un-@tf.function'ed likelihood.
        #
        # Tensorflow has tf.hessians, but:
        # https://github.com/tensorflow/tensorflow/issues/29781

        #xc = [tf.Variable(q)
        #      for q in fd.tf_to_np(params)]

        if save_ram:
            # Slower but more RAM-efficient algorithm
            n = len(self.param_names)
            hessian = np.zeros((n, n))
            for i1 in tqdm(range(n),
                    desc='Computing hessian'):
                for i2 in range(n):
                    if i2 > i1:
                        continue

                    xc = [tf.constant(q)
                            for q in fd.tf_to_np(params) 
                            ]
                    with tf.GradientTape(persistent=True) as t2:
                        t2.watch(xc[i2])
                        with tf.GradientTape() as t:
                            t.watch(xc[i1])
                            ptensor = tf.stack(xc)
                            y = self._minus_ll(ptensor)
                        grad = t.gradient(y, xc[i1])
                        hessian[i1, i2] = t2.gradient(grad, xc[i2]).numpy()
                    del t2
            for i1 in range(n):
                for i2 in range(n):
                    if i2 > i1:
                        hessian[i1, i2] = hessian[i2, i1]
            return hessian


    def summary(self, bestfit, inverse_hessian=None, precision=3):
        """Print summary information about best fit"""
        if inverse_hessian is None:
            inverse_hessian = self.inverse_hessian(bestfit)
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
