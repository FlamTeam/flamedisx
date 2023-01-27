import flamedisx as fd
import numpy as np
from scipy import stats
import pickle as pkl
from tqdm.auto import tqdm
import typing as ty
from numpy.typing import NDArray
from math import isclose

import tensorflow as tf

export, __all__ = fd.exporter()


@export
class FrequentistIntervalRatesOnlyTemplates():
    """NOTE: currently works for a single dataset only.

    Arguments:
        - signal_source_names: tuple of names for signal sources (e.g. WIMPs of different
            masses) of which we want to get limits/intervals
        - background_source_names: tuple of names for background sources
        - sources: dictionary {sourcename: class} of all signal and background source classes
        - arguments: dictionary {sourcename: {kwarg1: value, ...}, ...}, for
            passing keyword arguments to source constructors
        - batch_size: batch size that will be used for the RM fits
        - rate_gaussian_constraints: dictionary {sourcename: (mu, sigma)} to pass the mean and width for
            any Gaussian constraints on background rate multipliers
        - other_constraints:
        - rm_bounds: dictionary {sourcename: (lower, upper)} to set fit bounds on the rate multipliers
        - ntoys: number of toys that will be run to get test statistic distributions
        - log_constraint_fn:
    """

    def __init__(
            self,
            signal_source_names: ty.Tuple[str],
            background_source_names: ty.Tuple[str],
            sources: ty.Dict[str, fd.Source.__class__],
            arguments: ty.Dict[str, ty.Dict[str, ty.Union[int, float]]] = None,
            batch_size=10000,
            rate_gaussian_constraints: ty.Dict[str, ty.Tuple[float, float]] = None,
            other_constraints: ty.Dict[str, ty.Tuple[NDArray[float], NDArray[float]]] = None,
            rm_bounds: ty.Dict[str, ty.Tuple[float, float]] = None,
            ntoys=1000,
            log_constraint_fn=None):

        for key in sources.keys():
            if key not in arguments.keys():
                arguments[key] = dict()

        if rate_gaussian_constraints is None:
            rate_gaussian_constraints = dict()

        if other_constraints is None:
            other_constraints = dict()

        if rm_bounds is None:
            rm_bounds = dict()
        else:
            for bounds in rm_bounds.values():
                assert bounds[0] >= 0., 'Currently do not support negative rate multipliers'

        if log_constraint_fn is None:
            def log_constraint_fn(**kwargs):
                return 0.
            self.log_constraint_fn = log_constraint_fn
        else:
            self.log_constraint_fn = log_constraint_fn

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.ntoys = ntoys
        self.batch_size = batch_size

        self.rate_gaussian_constraints = {f'{key}_rate_multiplier': value for
                                          key, value in rate_gaussian_constraints.items()}
        self.other_constraints = {f'{key}_rate_multiplier': value for
                                  key, value in other_constraints.items()}
        self.rm_bounds = rm_bounds

        self.test_stat_dists = dict()
        self.observed_test_stats = dict()
        self.p_vals = dict()

        self.sources = sources
        self.arguments = arguments

    def test_statistic_tmu_tilde(self, mu_test, signal_source_name, likelihood, guess_dict):
        """Internal function to evaluate the test statistic of equation 11 in
        https://arxiv.org/abs/1007.1727.
        """
        # To fix the signal RM in the conditional fit
        fix_dict = {f'{signal_source_name}_rate_multiplier': mu_test}

        guess_dict_nuisance = guess_dict.copy()
        guess_dict_nuisance.pop(f'{signal_source_name}_rate_multiplier')

        # Conditional fit
        bf_conditional = likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)
        # Uncnditional fit
        bf_unconditional = likelihood.bestfit(guess=guess_dict, suppress_warnings=True)

        ll_conditional = likelihood(**bf_conditional)
        ll_unconditional = likelihood(**bf_unconditional)

        # Return the test statistic
        return -2. * (ll_conditional - ll_unconditional), bf_unconditional

    def get_test_stat_dists(self, mus_test=None, input_dists=None, dists_output_name=None):
        """Get test statistic distributions.

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate multipliers
                to be tested for each signal source
            - input_dists: path to input test statistic distributions, if we wish to pass this
            - dists_output_name: name of file in which to save test statistic distributions,
                if this is desired
        """
        if input_dists is not None:
            # Read in test statistic distributions
            self.test_stat_dists = pkl.load(open(input_dists, 'rb'))
            return

        assert mus_test is not None, 'Must pass in mus to be scanned over'

        self.test_stat_dists = dict()
        unconditional_best_fits = dict()
        # test_stats_no_signal = dict()
        # unconditional_best_fits_no_signal = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            test_stat_dists = dict()
            unconditional_bfs = dict()
            # ts_no_signal = dict()
            # unconditional_bfs_no_signal = dict()

            sources = dict()
            arguments = dict()
            for background_source in self.background_source_names:
                sources[background_source] = self.sources[background_source]
                arguments[background_source] = self.arguments[background_source]
            sources[signal_source] = self.sources[signal_source]
            arguments[signal_source] = self.arguments[signal_source]

            # Create likelihood of TemplateSources
            likelihood = fd.LogLikelihood(sources=sources,
                                          arguments=arguments,
                                          progress=False,
                                          batch_size=self.batch_size,
                                          free_rates=tuple([sname for sname in sources.keys()]))

            rm_bounds = dict()
            if signal_source in self.rm_bounds.keys():
                rm_bounds[signal_source] = self.rm_bounds[signal_source]
            else:
                rm_bounds[signal_source] = (0., None)
            for background_source in self.background_source_names:
                if background_source in self.rm_bounds.keys():
                    rm_bounds[background_source] = self.rm_bounds[background_source]
                else:
                    rm_bounds[background_source] = (0., None)

            # Pass rate multiplier bounds to likelihood
            likelihood.set_rate_multiplier_bounds(**rm_bounds)

            # Pass constraint function to likelihood
            likelihood.set_log_constraint(self.log_constraint_fn)

            # Save the test statistic values and unconditional best fits for each toy, for each
            # signal RM scanned over, for this signal source
            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                ts_dist = self.toy_test_statistic_dist(mu_test, signal_source, likelihood)
                test_stat_dists[mu_test] = ts_dist[0]
                unconditional_bfs[mu_test] = ts_dist[1]
                # ts_no_signal[mu_test] = ts_dist[2]
                # unconditional_bfs_no_signal[mu_test] = ts_dist[3]

            self.test_stat_dists[signal_source] = test_stat_dists
            unconditional_best_fits[signal_source] = unconditional_bfs
            # test_stats_no_signal[signal_source] = ts_no_signal
            # unconditional_best_fits_no_signal[signal_source] = unconditional_bfs_no_signal

        # Output test statistic distributions and fits
        if dists_output_name is not None:
            pkl.dump(self.test_stat_dists, open(dists_output_name, 'wb'))
            pkl.dump(unconditional_best_fits, open(dists_output_name[:-4] + '_fits.pkl', 'wb'))
            # pkl.dump(test_stats_no_signal, open(dists_output_name[:-4] + '_no_signal.pkl', 'wb'))
            # pkl.dump(test_stats_no_signal, open(dists_output_name[:-4] + '_fits_no_signal.pkl', 'wb'))

    def toy_test_statistic_dist(self, mu_test, signal_source_name, likelihood):
        """Internal function to get a test statistic distribution for a given signal source
        and signal RM.
        """
        rm_value_dict = {f'{signal_source_name}_rate_multiplier': mu_test}

        ts_values = []
        unconditional_bfs = []
        # ts_values_no_signal = []
        # unconditional_bfs_no_signal = []

        # Loop over toys
        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                # Prepare to sample background RMs from constraint functions
                if f'{background_source}_rate_multiplier' in self.rate_gaussian_constraints:
                    assert background_source in self.rm_bounds.keys(), \
                        'Must provide bounds when using a Gaussian constraint'
                    domain = np.linspace(self.rm_bounds[background_source][0], self.rm_bounds[background_source][1], 1000)
                    gaussian_constraint = self.rate_gaussian_constraints[f'{background_source}_rate_multiplier']
                    constraint = np.exp(-0.5 * ((domain - gaussian_constraint[0]) / gaussian_constraint[1])**2)
                    constraint /= np.sum(constraint)

                elif f'{background_source}_rate_multiplier' in self.other_constraints:
                    domain = self.other_constraints[f'{background_source}_rate_multiplier'][0]
                    constraint = self.other_constraints[f'{background_source}_rate_multiplier'][1]
                    assert np.shape(domain) == np.shape(constraint), \
                        'Shapes of contraint domain and values do not match'
                    assert isclose(np.sum(constraint), 1.), \
                        'Constraint values should sum to 1'

                else:
                    assert background_source in self.rm_bounds.keys(), \
                        'Must provide bounds if no constraint provided'
                    domain = np.linspace(self.rm_bounds[background_source][0], self.rm_bounds[background_source][1], 1000)
                    constraint = np.ones_like(domain)
                    constraint /= np.sum(constraint)

                # Sample background RMs from constraint functions. Remove first and last elements of domain to
                # avoid finite precision when casting leading to guesses outside the bounds, if the endpoints
                # are drawn
                domain = domain[1:-1]
                constraint = constraint[1:-1]
                constraint /= np.sum(constraint)
                draw = tf.cast(np.random.choice(domain, 1, p=constraint)[0], fd.float_type())
                rm_value_dict[f'{background_source}_rate_multiplier'] = draw
                # Recall: we want to shift the constraint in the likelihood based on the background RMs we draw
                if f'{background_source}_rate_multiplier' in self.rate_gaussian_constraints:
                    constraint_extra_args[f'{background_source}_rate_multiplier_constraint'] = \
                        (draw, tf.cast(self.rate_gaussian_constraints[f'{background_source}_rate_multiplier'][1],
                                       fd.float_type()))
                elif f'{background_source}_rate_multiplier' in self.other_constraints:
                    constraint_extra_args[f'{background_source}_rate_multiplier_constraint'] = \
                        (draw, 0.)

            # Shift the constraint in the likelihood based on the background RMs we drew
            likelihood.set_constraint_extra_args(**constraint_extra_args)

            # Simulate and set data
            toy_data = likelihood.simulate(**rm_value_dict)

            likelihood.set_data(toy_data)

            ts_result = self.test_statistic_tmu_tilde(mu_test, signal_source_name, likelihood, rm_value_dict)
            ts_values.append(ts_result[0])
            unconditional_bfs.append(ts_result[1])

            # if len(toy_data[toy_data['source'] == signal_source_name]) == 0:
            #     ts_values_no_signal.append(ts_result[0])
            #     unconditional_bfs_no_signal.append(ts_result[1])

        # Return the test statistic and unconditional best fit
        return ts_values, unconditional_bfs #, ts_values_no_signal, unconditional_bfs_no_signal

    def get_observed_test_stats(self, mus_test=None, data=None, input_test_stats=None, test_stats_output_name=None,
                                mu_estimators=None):
        """Get observed test statistics.

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate multipliers
                to be tested for each signal source
            - data: observed dataset
            - input_test_stats: path to input observed test statistics, if we wish to pass this
            - test_stats_output_name: name of file in which to save observed test statistics,
                if this is desired
            - mu_estimators: dictionary {sourcename: fd.ConstantMu(SourceClass, mu)} if we want to use
                the same pre-estimated mus as those used for the test statistic distribitions
        """
        if input_test_stats is not None:
            # Read in observed test statistics
            self.observed_test_stats = pkl.load(open(input_test_stats, 'rb'))
            return

        assert mus_test is not None, 'Must pass in mus to be scanned over'
        assert data is not None, 'Must pass in data'

        self.observed_test_stats = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            observed_test_stats = dict()

            sources = dict()
            for background_source in self.background_source_names:
                sources[background_source] = self.sources[background_source]
            sources[signal_source] = self.sources[signal_source]

            # Create likelihood of regular flamedisx sources
            likelihood_full = fd.LogLikelihood(sources=sources,
                                               progress=False,
                                               batch_size=self.batch_size_diff_rate,
                                               free_rates=tuple([sname for sname in sources.keys()]),
                                               mu_estimators=mu_estimators)

            rm_bounds = dict()
            if signal_source in self.rm_bounds.keys():
                rm_bounds[signal_source] = self.rm_bounds[signal_source]
            else:
                rm_bounds[signal_source] = (0., None)
            for background_source in self.background_source_names:
                if background_source in self.rm_bounds.keys():
                    rm_bounds[background_source] = self.rm_bounds[background_source]
                else:
                    rm_bounds[background_source] = (0., None)

            # Pass rate multiplier bounds to likelihood
            likelihood_full.set_rate_multiplier_bounds(**rm_bounds)

            # Set up Gaussian log constraint for background rate multipliers
            def log_constraint(**kwargs):
                log_constraint = sum(-0.5 * ((value - kwargs[f'{key}_constraint'][0]) /
                                     kwargs[f'{key}_constraint'][1])**2 for key, value in kwargs.items()
                                     if key in self.rate_gaussian_constraints.keys())
                return log_constraint
            likelihood_full.set_log_constraint(log_constraint)

            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                constraint_extra_args[f'{background_source}_rate_multiplier_constraint'] = \
                    (tf.cast(self.rate_gaussian_constraints[f'{background_source}_rate_multiplier'][0],
                             fd.float_type()),
                     tf.cast(self.rate_gaussian_constraints[f'{background_source}_rate_multiplier'][1],
                             fd.float_type()))

            # The constraints are not shifted here
            likelihood_full.set_constraint_extra_args(**constraint_extra_args)

            # Set data
            likelihood_full.set_data(data)

            # Set the guesses
            guess_dict = {f'{signal_source}_rate_multiplier': 0.}
            for background_source in self.background_source_names:
                guess_dict[f'{background_source}_rate_multiplier'] = 1.

            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                observed_test_stats[mu_test] = self.test_statistic_tmu_tilde(mu_test, signal_source, likelihood_full, guess_dict)[0]

            self.observed_test_stats[signal_source] = observed_test_stats

            # Output observed test statistics
            if test_stats_output_name is not None:
                pkl.dump(self.observed_test_stats, open(test_stats_output_name, 'wb'))

    def get_p_vals(self):
        """Internal function to get p-value curves.
        """
        self.p_vals = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get test statistic distribitions and observed test statistics
            test_stat_dists = self.test_stat_dists[signal_source]
            observed_test_stats = self.observed_test_stats[signal_source]

            assert len(test_stat_dists) > 0, f'Must generate test statistic distributions first for {signal_source}'
            assert len(observed_test_stats) > 0, f'Must calculate observed test statistics first for {signal_source}'
            assert test_stat_dists.keys() == observed_test_stats.keys(), \
                f'Must get test statistic distributions and observed test statistics for {signal_source} with ' \
                'the same mu values'

            p_vals = dict()
            # Loop over signal rate multipliers
            for mu_test in observed_test_stats.keys():
                # Compute the p-value from the observed test statistic and the distribition
                p_vals[mu_test] = (100. - stats.percentileofscore(test_stat_dists[mu_test],
                                                                  observed_test_stats[mu_test],
                                                                  kind='weak')) / 100.

            # Record p-value curve
            self.p_vals[signal_source] = p_vals

    def get_interval(self, conf_level=0.1, return_p_vals=False):
        """Get either upper limit, or possibly upper and lower limits.
        Before using this get_test_stat_dists() and get_observed_test_stats() must
        have bene called.

        Arguments:
            - conf_level: confidence level to be used for the limit/interval
            - return_p_vals: whether or not to output the p-value curves
        """
        # Get the p-value curves
        self.get_p_vals()

        lower_lim_all = dict()
        upper_lim_all = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            these_pvals = self.p_vals[signal_source]

            mus = list(these_pvals.keys())
            pvals = list(these_pvals.values())

            # Find points where the p-value curve cross the critical value, decreasing
            upper_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) < 0.).flatten()
            # Find points where the p-value curve cross the critical value, increasing
            lower_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) > 0.).flatten()

            if len(lower_lims > 0):
                # Take the lowest increasing crossing point, and interpolate to get an upper limit
                lower_mu_left = mus[lower_lims[0]]
                lower_mu_right = mus[lower_lims[0] + 1]
                lower_pval_left = pvals[lower_lims[0]]
                lower_pval_right = pvals[lower_lims[0] + 1]

                lower_gradient = (lower_pval_right - lower_pval_left) / (lower_mu_right - lower_mu_left)
                lower_lim = (conf_level - lower_pval_left) / lower_gradient + lower_mu_left
            else:
                # We have no lower limit
                lower_lim = None

            assert len(upper_lims) > 0, 'No upper limit found!'
            # Take the highest decreasing crossing point, and interpolate to get an upper limit
            upper_mu_left = mus[upper_lims[-1]]
            upper_mu_right = mus[upper_lims[-1] + 1]
            upper_pval_left = pvals[upper_lims[-1]]
            upper_pval_right = pvals[upper_lims[-1] + 1]

            upper_gradient = (upper_pval_right - upper_pval_left) / (upper_mu_right - upper_mu_left)
            upper_lim = (conf_level - upper_pval_left) / upper_gradient + upper_mu_left

            lower_lim_all[signal_source] = lower_lim
            upper_lim_all[signal_source] = upper_lim

        if return_p_vals is True:
            # Return p-value curves, and intervals/limits
            return self.p_vals, lower_lim_all, upper_lim_all
        else:
            # Return intervals/limits
            return lower_lim_all, upper_lim_all
