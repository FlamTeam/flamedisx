import flamedisx as fd
import numpy as np
from scipy import stats
import pickle as pkl
from tqdm.auto import tqdm
import typing as ty

import tensorflow as tf

export, __all__ = fd.exporter()

@export
class FrequentistUpperLimitRatesOnly():
    """NOTE: currently single dataset only

    Arguments:
        - xxx: yyy

    """

    def __init__(
            self,
            signal_source_names,
            background_source_names,
            sources: ty.Dict[str, fd.Source.__class__],
            arguments: ty.Dict[str, ty.Dict[str, ty.Union[int, float]]] = None,
            batch_size=100,
            max_sigma=None,
            max_sigma_outer=None,
            n_trials=None,
            rate_gaussian_constraints: ty.Dict[str, ty.Tuple[float, float]] = None,
            defaults=None,
            ntoys=1000,
            skip_reservoir=False):

        if arguments is None:
            arguments = dict()

        for key in sources.keys():
            if key not in arguments.keys():
                arguments[key] = dict()

        if defaults is None:
            defaults = dict()

        if rate_gaussian_constraints is None:
            rate_gaussian_constraints = dict()

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.ntoys = ntoys
        self.batch_size = batch_size

        self.rate_gaussian_constraints = {f'{key}_rate_multiplier': value for key, value in rate_gaussian_constraints.items()}

        self.test_stat_dists = dict()
        self.observed_test_stats = dict()
        self.p_vals = dict()

        self.rm_bounds=dict()
        for source_name in self.background_source_names:
            self.rm_bounds[source_name] = (0., 2.)

        # Create sources
        self.sources = sources
        self.source_objects = {
            sname: sclass(**(arguments.get(sname)),
                          data=None,
                          max_sigma=max_sigma,
                          max_sigma_outer=max_sigma_outer,
                          batch_size=self.batch_size,
                          **defaults)
            for sname, sclass in sources.items()}

        if skip_reservoir:
            self.reservoir = None
        else:
            # Create frozen source reservoir
            self.reservoir = fd.frozen_reservoir.make_event_reservoir(ntoys=ntoys, **self.source_objects)

    def test_statistic_tmu_tilde(self, mu_test, signal_source_name, likelihood):
        fix_dict = {f'{signal_source_name}_rate_multiplier': mu_test}
        guess_dict = {f'{signal_source_name}_rate_multiplier': mu_test}
        guess_dict_nuisance = dict()

        for source_name in self.background_source_names:
            guess_dict[f'{source_name}_rate_multiplier'] = 1.
            guess_dict_nuisance[f'{source_name}_rate_multiplier'] = 1.

        bf_conditional = likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)

        bf_unconditional = likelihood.bestfit(guess=guess_dict, suppress_warnings=True)

        if bf_unconditional[f'{signal_source_name}_rate_multiplier'] < 0.:
            fix_dict[f'{signal_source_name}_rate_multiplier'] = 0.
            bf_unconditional = likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)

        ll_conditional = likelihood(**bf_conditional)
        ll_unconditional = likelihood(**bf_unconditional)

        return -2. * (ll_conditional - ll_unconditional)

    def get_test_stat_dists(self, mus_test=None, output=False, input_path=None):
        if input_path is not None:
            try:
                self.test_stat_dists = pkl.load(open(input_path, 'rb'))
                return
            except Exception:
                print("Could not load TS distributions; re-calculating")

        assert mus_test is not None, 'Must pass in mus to be scanned over'
        assert self.reservoir is not None, 'Must popualte frozen source reservoir'

        self.test_stat_dists = dict()
        for signal_source in self.signal_source_names:
            test_stat_dists = dict()

            # Create likelihood
            sources = dict()
            for background_source in self.background_source_names:
                sources[background_source] = self.sources[background_source]
            sources[signal_source] = self.sources[signal_source]

            likelihood_fast = fd.LogLikelihood(sources={sname: fd.FrozenReservoirSource for sname in sources.keys()},
                                               arguments = {sname: {'source_type': sclass, 'source_name': sname, 'reservoir': self.reservoir}
                                                            for sname, sclass in sources.items()},
                                               progress=False,
                                               batch_size=self.batch_size,
                                               free_rates=tuple([sname for sname in sources.keys()]))

            rm_bounds = self.rm_bounds.copy()
            rm_bounds[signal_source] = (-5., 50.)
            likelihood_fast.set_rate_multiplier_bounds(**rm_bounds)

            def log_constraint(**kwargs):
                log_constraint = sum( -0.5 * ((value - kwargs[f'{key}_constraint'][0]) / kwargs[f'{key}_constraint'][1])**2 for key, value in kwargs.items() if key in self.rate_gaussian_constraints.keys() )
                return log_constraint
            likelihood_fast.set_log_constraint(log_constraint)

            for mu_test in tqdm(mus_test, desc='Scanning over mus'):
                ts_dist = self.toy_test_statistic_dist(mu_test, signal_source, likelihood_fast)
                test_stat_dists[mu_test] = ts_dist

            self.test_stat_dists[signal_source] = test_stat_dists

            if output is True:
                pkl.dump(self.test_stat_dists, open('test_stat_dists.pkl', 'wb'))


    def toy_test_statistic_dist(self, mu_test, signal_source_name, likelihood_fast):
        rm_value_dict = {f'{signal_source_name}_rate_multiplier': mu_test}

        ts_values = []

        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            constraint_extra_args = dict()
            for source_name in self.background_source_names:
                domain = np.linspace(self.rm_bounds[source_name][0], self.rm_bounds[source_name][1], 1000)
                gaussian_constraint = self.rate_gaussian_constraints[f'{source_name}_rate_multiplier']
                constraint = np.exp(-0.5 * ((domain - gaussian_constraint[0]) / gaussian_constraint[1])**2)
                constraint /= np.sum(constraint)

                draw = tf.cast(np.random.choice(domain, 1, p=constraint)[0], fd.float_type())
                rm_value_dict[f'{source_name}_rate_multiplier'] = draw
                constraint_extra_args[f'{source_name}_rate_multiplier_constraint'] = \
                    (draw, tf.cast(self.rate_gaussian_constraints[f'{source_name}_rate_multiplier'][1], fd.float_type()))

            likelihood_fast.set_constraint_extra_args(**constraint_extra_args)

            toy_data = likelihood_fast.simulate(**rm_value_dict)
            likelihood_fast.set_data(toy_data)

            ts_values.append(self.test_statistic_tmu_tilde(mu_test, signal_source_name, likelihood_fast))

        return ts_values

    def get_observed_test_stats(self, mus_test=None, data=None, output=False, input_path=None):
        if input_path is not None:
            try:
                self.observed_test_stats = pkl.load(open(input_path, 'rb'))
                return
            except Exception:
                print("Could not load observed test statistics; re-calculating")

        assert mus_test is not None, 'Must pass in mus to be scanned over'
        assert data is not None, 'Must pass in data'

        self.observed_test_stats = dict()
        for signal_source in self.signal_source_names:
            observed_test_stats = dict()

            # Create likelihood
            sources = dict()
            for background_source in self.background_source_names:
                sources[background_source] = self.sources[background_source]
            sources[signal_source] = self.sources[signal_source]

            likelihood_full = fd.LogLikelihood(sources=sources,
                                               progress=False,
                                               batch_size=self.batch_size,
                                               free_rates=tuple([sname for sname in sources.keys()]))

            rm_bounds = self.rm_bounds.copy()
            rm_bounds[signal_source] = (-5., 50.)
            likelihood_full.set_rate_multiplier_bounds(**rm_bounds)

            def log_constraint(**kwargs):
                log_constraint = sum( -0.5 * ((value - kwargs[f'{key}_constraint'][0]) / kwargs[f'{key}_constraint'][1])**2 for key, value in kwargs.items() if key in self.rate_gaussian_constraints.keys() )
                return log_constraint
            likelihood_fast.set_log_constraint(log_constraint)

            constraint_extra_args = dict()
            for source_name in self.background_source_names:
                constraint_extra_args[f'{source_name}_rate_multiplier_constraint'] = \
                    (tf.cast(self.rate_gaussian_constraints[f'{source_name}_rate_multiplier'][0], fd.float_type()),
                     tf.cast(self.rate_gaussian_constraints[f'{source_name}_rate_multiplier'][1], fd.float_type()))

            likelihood_full.set_constraint_extra_args(**constraint_extra_args)

            likelihood_full.set_data(data)

            for mu_test in tqdm(mus_test, desc='Scanning over mus'):
                observed_test_stats[mu_test] = self.test_statistic_tmu_tilde(mu_test, signal_source, likelihood_full)

            self.observed_test_stats[signal_source] = observed_test_stats

            if output is True:
                pkl.dump(self.observed_test_stats, open('observed_test_stats.pkl', 'wb'))

    def get_p_vals(self):
        self.p_vals = dict()
        for signal_source in self.signal_source_names:
            test_stat_dists = self.test_stat_dists[signal_source]
            observed_test_stats = self.observed_test_stats[signal_source]

            assert len(test_stat_dists) > 0, f'Must generate test statistic distributions first for {signal_source}'
            assert len(observed_test_stats) > 0, f'Must calculate observed test statistics first for {signal_source}'
            assert test_stat_dists.keys() == observed_test_stats.keys(), \
                f'Must get test statistic distributions and observed test statistics for {signal_source} with the same mu values'

            p_vals = dict()
            for mu_test in observed_test_stats.keys():
                p_vals[mu_test] = (100. - stats.percentileofscore(test_stat_dists[mu_test],
                                                                  observed_test_stats[mu_test])) / 100.

            self.p_vals[signal_source] = p_vals

    def get_interval(self, conf_level=0.1, return_p_vals=False):
        self.get_p_vals()

        lower_lim_all = dict()
        upper_lim_all = dict()
        for signal_source in self.signal_source_names:
            these_pvals = self.p_vals[signal_source]

            mus = list(these_pvals.keys())
            pvals = list(these_pvals.values())

            upper_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) < 0.).flatten()
            lower_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) > 0.).flatten()

            if len(lower_lims > 0):
                lower_mu_left = mus[lower_lims[0]]
                lower_mu_right = mus[lower_lims[0] + 1]
                lower_pval_left = pvals[lower_lims[0]]
                lower_pval_right = pvals[lower_lims[0] + 1]

                lower_gradient = (lower_pval_right - lower_pval_left) / (lower_mu_right - lower_mu_left)
                lower_lim = (conf_level - lower_pval_left) / lower_gradient + lower_mu_left
            else:
                lower_lim = None

            assert(len(upper_lims) > 0), 'No upper limit found!'
            upper_mu_left = mus[upper_lims[-1]]
            upper_mu_right = mus[upper_lims[-1] + 1]
            upper_pval_left = pvals[upper_lims[-1]]
            upper_pval_right = pvals[upper_lims[-1] + 1]

            upper_gradient = (upper_pval_right - upper_pval_left) / (upper_mu_right - upper_mu_left)
            upper_lim = (conf_level - upper_pval_left) / upper_gradient + upper_mu_left

            lower_lim_all[signal_source] = lower_lim
            upper_lim_all[signal_source] = upper_lim

        if return_p_vals is True:
            return self.p_vals, lower_lim_all, upper_lim_all
        else:
            return lower_lim_all, upper_lim_all
