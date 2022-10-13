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
            pre_estimated_mus: ty.Dict[str, float] = None,
            max_rm_dict: ty.Dict[str, float] = None,
            batch_size_diff_rate=100,
            batch_size_rates=10000,
            max_sigma=None,
            max_sigma_outer=None,
            n_trials=None,
            rate_gaussian_constraints: ty.Dict[str, ty.Tuple[float, float]] = None,
            rm_bounds: ty.Dict[str, ty.Tuple[float, float]] = None,
            defaults=None,
            ntoys=1000,
            input_reservoir=None):

        if arguments is None:
            arguments = dict()

        if pre_estimated_mus is None:
            self.pre_estimated_mus = dict()
            for key in sources.keys:
                self.pre_estimated_mus[key] = None
        else:
            self.pre_estimated_mus = pre_estimated_mus

        for key in sources.keys():
            if key not in arguments.keys():
                arguments[key] = dict()

        if defaults is None:
            defaults = dict()

        if rate_gaussian_constraints is None:
            rate_gaussian_constraints = dict()

        if rm_bounds is None:
            rm_bounds=dict()

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.ntoys = ntoys
        self.batch_size_diff_rate = batch_size_diff_rate
        self.batch_size_rates = batch_size_rates

        self.rate_gaussian_constraints = {f'{key}_rate_multiplier': value for key, value in rate_gaussian_constraints.items()}
        self.rm_bounds = rm_bounds

        self.test_stat_dists = dict()
        self.unconditional_bfs = dict()
        self.observed_test_stats = dict()
        self.p_vals = dict()

        # Create sources
        self.sources = sources
        self.source_objects = {
            sname: sclass(**(arguments.get(sname)),
                          data=None,
                          max_sigma=max_sigma,
                          max_sigma_outer=max_sigma_outer,
                          batch_size=self.batch_size_diff_rate,
                          **defaults)
            for sname, sclass in sources.items()}

        if input_reservoir is not None:
            # Read in frozen source reservoir
            self.reservoir = pkl.load(open(input_reservoir, 'rb'))
        else:
            # Create frozen source reservoir
            self.reservoir = fd.frozen_reservoir.make_event_reservoir(ntoys=ntoys, max_rm_dict=max_rm_dict, **self.source_objects)


    def test_statistic_tmu_tilde(self, mu_test, signal_source_name, likelihood):
        fix_dict = {f'{signal_source_name}_rate_multiplier': mu_test}
        guess_dict = {f'{signal_source_name}_rate_multiplier': mu_test}
        guess_dict_nuisance = dict()

        for background_source in self.background_source_names:
            if background_source in self.rate_gaussian_constraints:
                guess_dict[f'{background_source}_rate_multiplier'] = self.rate_gaussian_constraints[background_source][0]
                guess_dict_nuisance[f'{background_source}_rate_multiplier'] = self.rate_gaussian_constraints[background_source][0]
            else:
                guess_dict[f'{background_source}_rate_multiplier'] = 1.
                guess_dict_nuisance[f'{background_source}_rate_multiplier'] = 1.

        bf_conditional = likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)

        bf_unconditional = likelihood.bestfit(guess=guess_dict, suppress_warnings=True)

        if bf_unconditional[f'{signal_source_name}_rate_multiplier'] < 0.:
            fix_dict[f'{signal_source_name}_rate_multiplier'] = 0.
            bf_unconditional = likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)

        ll_conditional = likelihood(**bf_conditional)
        ll_unconditional = likelihood(**bf_unconditional)

        return -2. * (ll_conditional - ll_unconditional), bf_unconditional

    def get_test_stat_dists(self, mus_test=None, input_dists=None, dists_output_name=None):
        if input_dists is not None:
            self.test_stat_dists = pkl.load(open(input_dists, 'rb'))
            return

        assert mus_test is not None, 'Must pass in mus to be scanned over'
        assert self.reservoir is not None, 'Must populate frozen source reservoir'

        self.test_stat_dists = dict()
        self.unconditional_bfs = dict()
        for signal_source in self.signal_source_names:
            test_stat_dists = dict()
            unconditional_bfs = dict()

            # Create likelihood
            sources = dict()
            for background_source in self.background_source_names:
                sources[background_source] = self.sources[background_source]
            sources[signal_source] = self.sources[signal_source]

            likelihood_fast = fd.LogLikelihood(sources={sname: fd.FrozenReservoirSource for sname in sources.keys()},
                                               arguments = {sname: {'source_type': sclass, 'source_name': sname, 'reservoir': self.reservoir, 'input_mu': self.pre_estimated_mus[sname]}
                                                            for sname, sclass in sources.items()},
                                               progress=False,
                                               batch_size=self.batch_size_rates,
                                               free_rates=tuple([sname for sname in sources.keys()]))

            rm_bounds = dict()
            if signal_source in self.rm_bounds.keys():
                rm_bounds[signal_source] = self.rm_bounds[signal_source]
            for background_source in self.background_source_names:
                if background_source in self.rm_bounds.keys():
                    rm_bounds[background_source] = self.rm_bounds[background_source]

            likelihood_fast.set_rate_multiplier_bounds(**rm_bounds)

            def log_constraint(**kwargs):
                log_constraint = sum( -0.5 * ((value - kwargs[f'{key}_constraint'][0]) / kwargs[f'{key}_constraint'][1])**2 for key, value in kwargs.items() if key in self.rate_gaussian_constraints.keys() )
                return log_constraint
            likelihood_fast.set_log_constraint(log_constraint)

            these_mus_test = mus_test[signal_source]
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                ts_dist = self.toy_test_statistic_dist(mu_test, signal_source, likelihood_fast)
                test_stat_dists[mu_test] = ts_dist[0]
                unconditional_bfs[mu_test] = ts_dist[1]

            self.test_stat_dists[signal_source] = test_stat_dists
            self.unconditional_bfs[signal_source] = unconditional_bfs

            if dists_output_name is not None:
                pkl.dump(self.test_stat_dists, open(dists_output_name, 'wb'))
                pkl.dump(self.unconditional_bfs, open(dists_output_name[:-4] + '_fits.pkl', 'wb'))


    def toy_test_statistic_dist(self, mu_test, signal_source_name, likelihood_fast):
        rm_value_dict = {f'{signal_source_name}_rate_multiplier': mu_test}

        ts_values = []
        unconditional_bfs = []

        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                assert background_source in self.rm_bounds.keys(), 'Must provide bounds when using a Gaussian constraint'
                domain = np.linspace(self.rm_bounds[background_source][0], self.rm_bounds[background_source][1], 1000)
                gaussian_constraint = self.rate_gaussian_constraints[f'{background_source}_rate_multiplier']
                constraint = np.exp(-0.5 * ((domain - gaussian_constraint[0]) / gaussian_constraint[1])**2)
                constraint /= np.sum(constraint)

                draw = tf.cast(np.random.choice(domain, 1, p=constraint)[0], fd.float_type())
                rm_value_dict[f'{background_source}_rate_multiplier'] = draw
                constraint_extra_args[f'{background_source}_rate_multiplier_constraint'] = \
                    (draw, tf.cast(self.rate_gaussian_constraints[f'{background_source}_rate_multiplier'][1], fd.float_type()))

            likelihood_fast.set_constraint_extra_args(**constraint_extra_args)

            toy_data = likelihood_fast.simulate(**rm_value_dict)
            likelihood_fast.set_data(toy_data)

            ts_result = self.test_statistic_tmu_tilde(mu_test, signal_source_name, likelihood_fast)
            ts_values.append(ts_result[0])
            unconditional_bfs.append(ts_result[1])

        return ts_values, unconditional_bfs

    def get_observed_test_stats(self, mus_test=None, data=None, input_test_stats=None, test_stats_output_name=None):
        if input_test_stats is not None:
            self.observed_test_stats = pkl.load(open(input_test_stats, 'rb'))
            return

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
                                               batch_size=self.batch_size_rates,
                                               free_rates=tuple([sname for sname in sources.keys()]))

            rm_bounds = dict()
            if signal_source in self.rm_bounds.keys():
                rm_bounds[signal_source] = self.rm_bounds[signal_source]
            for background_source in self.background_source_names:
                if background_source in self.rm_bounds.keys():
                    rm_bounds[background_source] = self.rm_bounds[background_source]

            likelihood_full.set_rate_multiplier_bounds(**rm_bounds)

            def log_constraint(**kwargs):
                log_constraint = sum( -0.5 * ((value - kwargs[f'{key}_constraint'][0]) / kwargs[f'{key}_constraint'][1])**2 for key, value in kwargs.items() if key in self.rate_gaussian_constraints.keys() )
                return log_constraint
            likelihood_full.set_log_constraint(log_constraint)

            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                constraint_extra_args[f'{background_source}_rate_multiplier_constraint'] = \
                    (tf.cast(self.rate_gaussian_constraints[f'{background_source}_rate_multiplier'][0], fd.float_type()),
                     tf.cast(self.rate_gaussian_constraints[f'{background_source}_rate_multiplier'][1], fd.float_type()))

            likelihood_full.set_constraint_extra_args(**constraint_extra_args)

            likelihood_full.set_data(data)

            these_mus_test = mus_test[signal_source]
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                observed_test_stats[mu_test] = self.test_statistic_tmu_tilde(mu_test, signal_source, likelihood_full)[0]

            self.observed_test_stats[signal_source] = observed_test_stats

            if test_stats_output_name is not None:
                pkl.dump(self.observed_test_stats, open(test_stats_output_name, 'wb'))

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
