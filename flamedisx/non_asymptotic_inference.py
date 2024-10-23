import flamedisx as fd
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
import typing as ty

from copy import deepcopy

import tensorflow as tf

export, __all__ = fd.exporter()


@export
class TestStatistic():
    """Class to evaluate a test statistic based on a conidtional and unconditional
    maximum likelihood fit. Override the evaluate() method in derived classes.

    Arguments:
        - likelihood: fd.LogLikelihood instance with data already set
    """
    def __init__(self, likelihood):
        self.likelihood = likelihood

    def __call__(self, mu_test, signal_source_name, guess_dict,
                 asymptotic=False):
        # To fix the signal RM in the conditional fit
        fix_dict = {f'{signal_source_name}_rate_multiplier': tf.cast(mu_test, fd.float_type())}

        guess_dict_nuisance = guess_dict.copy()
        guess_dict_nuisance.pop(f'{signal_source_name}_rate_multiplier')

        # Conditional fit
        bf_conditional = self.likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)
        # Uncnditional fit
        bf_unconditional = self.likelihood.bestfit(guess=guess_dict, suppress_warnings=True)

        # Return the test statistic, unconditional fit and conditional fit
        if not asymptotic:
            return self.evaluate(bf_unconditional, bf_conditional), bf_unconditional, bf_conditional
        else:
            return self.evaluate_asymptotic_pval(bf_unconditional, bf_conditional,
                                                 mu_test), bf_unconditional, bf_conditional


@export
class TestStatisticTMuTilde(TestStatistic):
    """Evaluate the test statistic of equation 11 in https://arxiv.org/abs/1007.1727.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, bf_unconditional, bf_conditional):
        ll_conditional = self.likelihood(**bf_conditional)
        ll_unconditional = self.likelihood(**bf_unconditional)

        ts = -2. * (ll_conditional - ll_unconditional)
        if ts < 0.:
            return 0.
        else:
            return ts

    def evaluate_asymptotic_pval(self, bf_unconditional, bf_conditional, mu_test):
        ll_conditional = self.likelihood(**bf_conditional)
        ll_unconditional = self.likelihood(**bf_unconditional)

        ts = -2. * (ll_conditional - ll_unconditional)

        cov = 2. * self.likelihood.inverse_hessian(bf_unconditional)
        sigma_mu = np.sqrt(cov[0][0])

        if ts < (mu_test**2 / sigma_mu**2):
            F = 2. * stats.norm.cdf(np.sqrt(ts)) - 1.
        else:
            F = stats.norm.cdf(np.sqrt(ts)) + stats.norm.cdf((ts + (mu_test**2 / sigma_mu**2)) / (2. * mu_test / sigma_mu)) - 1.

        pval = 1. - F
        return pval


@export
class TestStatisticDistributions():
    """ Class to store test statistic distribution values (pass in as a list),
    as well as (optionally) conditional and unconditional fit dictionaries for
    each toy, for a range of values of the parameter of interest being tested ('mu').
    """
    def __init__(self):
        self.ts_dists = dict()
        self.unconditional_best_fits = dict()
        self.conditional_best_fits = dict()

    def add_ts_dist(self, mu_test, ts_values):
        self.ts_dists[mu_test] = np.array(ts_values)

    def add_unconditional_best_fit(self, mu_test, fit_values):
        self.unconditional_best_fits[mu_test] = fit_values

    def add_conditional_best_fit(self, mu_test, fit_values):
        self.conditional_best_fits[mu_test] = fit_values

    def get_p_vals(self, observed_test_stats, inverse=False):
        """Evaluate the p-value for a set of observed test statistics using the
        stored test statistic distributions. Pass inverse=True to evaluate the integral
        from -infinity to t_obs instead of from t_obs to +infinity.
        """
        p_vals = dict()
        assert self.ts_dists.keys() == observed_test_stats.test_stats.keys(), \
            'POI values for observed test statistics and test statistic distributions ' \
            'do not match'
        for mu_test in observed_test_stats.test_stats.keys():
            if not inverse:
                p_vals[mu_test] = (100. - stats.percentileofscore(self.ts_dists[mu_test],
                                                                  observed_test_stats.test_stats[mu_test],
                                                                  kind='weak')) / 100.
            else:
                p_vals[mu_test] = stats.percentileofscore(self.ts_dists[mu_test],
                                                          observed_test_stats.test_stats[mu_test],
                                                          kind='weak') / 100.
        return p_vals

    def get_crit_vals(self, conf_level):
        """Get the critical value from the test statistic distribution for a test of
        some confidence level.
        """
        crit_vals = ObservedTestStatistics()
        for mu_test, ts_dist in self.ts_dists.items():
            crit_vals.add_test_stat(mu_test, np.quantile(ts_dist, 1. - conf_level))
        return crit_vals


@export
class ObservedTestStatistics():
    """ Class to observed test statistic values as well as (optionally) conditional and
    unconditional fit dictionaries, for a range of values of the parameter of interest
    being tested ('mu').
    """
    def __init__(self):
        self.test_stats = dict()
        self.unconditional_best_fits = dict()
        self.conditional_best_fits = dict()

    def add_test_stat(self, mu_test, observed_ts):
        self.test_stats[mu_test] = observed_ts

    def add_unconditional_best_fit(self, mu_test, fit_values):
        self.unconditional_best_fits[mu_test] = fit_values

    def add_conditional_best_fit(self, mu_test, fit_values):
        self.conditional_best_fits[mu_test] = fit_values


@export
class TSEvaluation():
    """NOTE: currently works for a single dataset only.
    Class for evaluating both observed test statistics and test statistic distributions.

    Arguments:
        - test_statistic: class type of the test statistic to be used
        - signal_source_names: tuple of names for signal sources (e.g. WIMPs of different
            masses)
        - background_source_names: tuple of names for background sources
        - expected_background_counts: dictionary of expected counts for background sources
        - gaussian_constraint_widths: dictionary giving the constraint width for all sources
            using Gaussian constraints for their rate nuisance parameters
        - sample_other_constraints: dictionary of functions to sample constraint means
            in the toys for any sources using non-Gaussian constraints for their rate nuisance
            parameters. Argument to the function will be either the prior expected counts,
            or the number of counts at the conditional MLE, depending on the mode
        - likelihood: BLAH
        - ntoys: number of toys that will be run to get test statistic distributions
    """
    def __init__(
            self,
            test_statistic: TestStatistic.__class__,
            signal_source_names: ty.Tuple[str],
            background_source_names: ty.Tuple[str],
            expected_background_counts: ty.Dict[str, float] = None,
            gaussian_constraint_widths: ty.Dict[str, float] = None,
            sample_other_constraints: ty.Dict[str, ty.Callable] = None,
            likelihood=None,
            ntoys=1000):

        if gaussian_constraint_widths is None:
            gaussian_constraint_widths = dict()

        if sample_other_constraints is None:
            sample_other_constraints = dict()

        self.ntoys = ntoys

        self.likelihood = likelihood
        self.test_statistic = test_statistic

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.expected_background_counts = expected_background_counts
        self.gaussian_constraint_widths = gaussian_constraint_widths
        self.sample_other_constraints = sample_other_constraints

    def run_routine(self, mus_test=None, save_fits=False,
                    observed_data=None,
                    observed_test_stats=None,
                    generate_B_toys=False,
                    simulate_dict_B=None, toy_data_B=None, constraint_extra_args_B=None,
                    toy_batch=0,
                    asymptotic=False, asymptotic_sensitivity=False):
        """If observed_data is passed, evaluate observed test statistics. Otherwise,
        obtain test statistic distributions (for both S+B and B-only).

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate
                multipliers to be tested for each signal source
            - save_fits: if True, unconditional and conditional fits will be saved along with
                the test statistic value
            - observed_data: pass this to evaluate the observed test statistics
            - observed_test_stats: if obtaining test statistic distributions, and this is
                passed, and the conditional best fits were saved for the observed data, the
                background counts for the toys will be fixed to the observed conditional best fits,
                and the constraint centers which are randomised for each toy will be centered
                around the conditional best fits. Otherwise, the prior expected counts will be used
                in place
            - generate_B_toys: if true, the routine run will be a generation of background-only
                datasets
            - simulate_dict_B: first return argument of the result of calling this function with
                generate_B_toys=True)
            - toy_data_B: second return argument of the result of calling this function with
                generate_B_toys=True)
            - toy_data_B: third return argument of the result of calling this function with
                generate_B_toys=True)
            - toy_batch: if parallelising toys, this should correspond to the parallel batch index
                (starting at 0) being run, to ensure the correct background-only toys are accessed
        """
        if observed_test_stats is not None:
            self.observed_test_stats = observed_test_stats
        else:
            self.observed_test_stats = None

        if toy_data_B is not None:
            assert simulate_dict_B is not None, \
                'Must pass all of simulate_dict_B, toy_data_B and \
                    constraint_extra_args_B'
            assert constraint_extra_args_B is not None, \
                'Must pass all of simulate_dict_B, toy_data_B and \
                    constraint_extra_args_B'
            self.simulate_dict_B = simulate_dict_B
            self.toy_data_B = toy_data_B
            self.constraint_extra_args_B = constraint_extra_args_B
            self.toy_batch = toy_batch

        observed_test_stats_collection = dict()
        test_stat_dists_SB_collection = dict()
        test_stat_dists_SB_disco_collection = dict()
        test_stat_dists_B_collection = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            observed_test_stats = ObservedTestStatistics()
            test_stat_dists_SB = TestStatisticDistributions()
            test_stat_dists_SB_disco = TestStatisticDistributions()
            test_stat_dists_B = TestStatisticDistributions()

            # Get likelihood
            likelihood = deepcopy(self.likelihood)

            assert hasattr(likelihood, 'likelihoods'), 'Logic only currently works for combined likelihood'
            for ll in likelihood.likelihoods.values():
                sources_remove = []
                params_remove = []
                for sname in ll.sources:
                    if (sname != signal_source) and (sname not in self.background_source_names):
                        sources_remove.append(sname)
                        params_remove.append(f'{sname}_rate_multiplier')
            likelihood.rebuild(sources_remove=sources_remove,
                               params_remove=params_remove)

            # Where we want to generate B-only toys
            if generate_B_toys:
                toy_data_B_all = []
                constraint_extra_args_B_all = []
                for i in tqdm(range(self.ntoys), desc='Background-only toys'):
                    simulate_dict_B, toy_data_B, constraint_extra_args_B = \
                        self.sample_data_constraints(0., signal_source, likelihood)
                    toy_data_B_all.append(toy_data_B)
                    constraint_extra_args_B_all.append(constraint_extra_args_B)
                simulate_dict_B.pop(f'{signal_source}_rate_multiplier')
                return simulate_dict_B, toy_data_B_all, constraint_extra_args_B_all

            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                # Case where we want observed test statistics
                if observed_data is not None:
                    self.get_observed_test_stat(observed_test_stats, observed_data,
                                                mu_test, signal_source, likelihood, save_fits=save_fits,
                                                asymptotic=asymptotic)
                # Case where we want test statistic distributions
                else:
                    self.toy_test_statistic_dist(test_stat_dists_SB, test_stat_dists_B,
                                                 test_stat_dists_SB_disco,
                                                 mu_test, signal_source, likelihood,
                                                 save_fits=save_fits,
                                                 asymptotic_sensitivity=asymptotic_sensitivity)

            if observed_data is not None:
                observed_test_stats_collection[signal_source] = observed_test_stats
            else:
                test_stat_dists_SB_collection[signal_source] = test_stat_dists_SB
                test_stat_dists_SB_disco_collection[signal_source] = test_stat_dists_SB_disco
                test_stat_dists_B_collection[signal_source] = test_stat_dists_B

        if asymptotic_sensitivity:
            return test_stat_dists_B_collection

        if observed_data is not None:
            return observed_test_stats_collection
        else:
            return test_stat_dists_SB_collection, test_stat_dists_SB_disco_collection, \
                test_stat_dists_B_collection

    def sample_data_constraints(self, mu_test, signal_source_name, likelihood):
        """Internal function to sample the toy data and constraint central values
        following a frequentist procedure. Method taken depends on whether conditional
        best fits were passed.
        """
        simulate_dict = dict()
        constraint_extra_args = dict()
        for background_source in self.background_source_names:
            # Case where we use the conditional best fits as constraint centers and simulated values
            if self.observed_test_stats is not None:
                try:
                    conditional_bfs_observed = self.observed_test_stats[signal_source_name].conditional_best_fits
                    expected_background_counts = \
                        conditional_bfs_observed[mu_test][f'{background_source}_rate_multiplier']
                except Exception:
                    raise RuntimeError("Could not find observed conditional best fits")
            # Case where we use the prior expected counts as constraint centers and simualted values
            else:
                expected_background_counts = self.expected_background_counts[background_source]

            # Sample constraint centers
            if background_source in self.gaussian_constraint_widths:
                draw = stats.norm.rvs(loc=expected_background_counts,
                                      scale=self.gaussian_constraint_widths[background_source])
                constraint_extra_args[f'{background_source}_expected_counts'] = tf.cast(draw, fd.float_type())

            elif background_source in self.sample_other_constraints:
                draw = self.sample_other_constraints[background_source](expected_background_counts)
                constraint_extra_args[f'{background_source}_expected_counts'] = tf.cast(draw, fd.float_type())

            simulate_dict[f'{background_source}_rate_multiplier'] = tf.cast(expected_background_counts, fd.float_type())
            simulate_dict[f'{signal_source_name}_rate_multiplier'] = tf.cast(mu_test, fd.float_type())

        if self.observed_test_stats is not None:
            conditional_bfs_observed = self.observed_test_stats[signal_source_name].conditional_best_fits[mu_test]
            non_rate_params_added = []
            for pname, fitval in conditional_bfs_observed.items():
                if (pname not in simulate_dict) and (pname in likelihood.param_defaults):
                    simulate_dict[pname] = fitval
                    non_rate_params_added.append(pname)

        toy_data = likelihood.simulate(**simulate_dict)

        if self.observed_test_stats is not None:
            for pname in non_rate_params_added:
                simulate_dict.pop(pname)

        return simulate_dict, toy_data, constraint_extra_args

    def toy_test_statistic_dist(self, test_stat_dists_SB, test_stat_dists_B,
                                test_stat_dists_SB_disco,
                                mu_test, signal_source_name, likelihood,
                                save_fits=False, asymptotic_sensitivity=False):
        """Internal function to get test statistic distribution.
        """
        ts_values_SB = []
        ts_values_SB_disco = []
        ts_values_B = []
        if save_fits:
            unconditional_bfs_SB = []
            conditional_bfs_SB = []
            unconditional_bfs_B = []
            conditional_bfs_B = []

        # Loop over toys
        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            if not asymptotic_sensitivity:
                simulate_dict_SB, toy_data_SB, constraint_extra_args_SB = \
                    self.sample_data_constraints(mu_test, signal_source_name, likelihood)

                # S+B toys

                # Shift the constraint in the likelihood based on the background RMs we drew
                likelihood.set_constraint_extra_args(**constraint_extra_args_SB)
                # Set data
                if hasattr(likelihood, 'likelihoods'):
                    for component, data in toy_data_SB.items():
                        likelihood.set_data(data, component)
                else:
                    likelihood.set_data(toy_data_SB)
                # Create test statistic
                test_statistic_SB = self.test_statistic(likelihood)
                # Guesses for fit
                guess_dict_SB = simulate_dict_SB.copy()
                for key, value in guess_dict_SB.items():
                    if value < 0.1:
                        guess_dict_SB[key] = 0.1
                # Evaluate test statistics
                ts_result_SB = test_statistic_SB(mu_test, signal_source_name, guess_dict_SB)
                ts_result_SB_disco = test_statistic_SB(0., signal_source_name, guess_dict_SB)
                # Save test statistics, and possibly fits
                ts_values_SB.append(ts_result_SB[0])
                ts_values_SB_disco.append(ts_result_SB_disco[0])
                if save_fits:
                    unconditional_bfs_SB.append(ts_result_SB[1])
                    conditional_bfs_SB.append(ts_result_SB[2])

            # B-only toys

            try:
                # Guesses for fit
                guess_dict_B = self.simulate_dict_B.copy()
                guess_dict_B[f'{signal_source_name}_rate_multiplier'] = 0.
                for key, value in guess_dict_B.items():
                    if value < 0.1:
                        guess_dict_B[key] = 0.1
                toy_data_B = self.toy_data_B[toy+(self.toy_batch*self.ntoys)]
                constraint_extra_args_B = self.constraint_extra_args_B[toy]
            except Exception:
                raise RuntimeError("Could not find background-only datasets")

            # Shift the constraint in the likelihood based on the background RMs we drew
            likelihood.set_constraint_extra_args(**constraint_extra_args_B)
            # Set data
            if hasattr(likelihood, 'likelihoods'):
                for component, data in toy_data_B.items():
                    likelihood.set_data(data, component)
            else:
                likelihood.set_data(toy_data_B)
            # Create test statistic
            test_statistic_B = self.test_statistic(likelihood)
            # Evaluate test statistic
            ts_result_B = test_statistic_B(mu_test, signal_source_name, guess_dict_B,
                                           asymptotic=asymptotic_sensitivity)
            # Save test statistic, and possibly fits
            ts_values_B.append(ts_result_B[0])
            if save_fits:
                unconditional_bfs_B.append(ts_result_SB[1])
                conditional_bfs_B.append(ts_result_SB[2])

        if asymptotic_sensitivity:
            test_stat_dists_B.add_ts_dist(mu_test, ts_values_B)
            if save_fits:
                test_stat_dists_B.add_unconditional_best_fit(mu_test, unconditional_bfs_B)
                test_stat_dists_B.add_conditional_best_fit(mu_test, conditional_bfs_B)
            return

        # Add to the test statistic distributions
        test_stat_dists_SB.add_ts_dist(mu_test, ts_values_SB)
        test_stat_dists_SB_disco.add_ts_dist(mu_test, ts_values_SB_disco)
        test_stat_dists_B.add_ts_dist(mu_test, ts_values_B)

        # Possibly save the fits
        if save_fits:
            test_stat_dists_SB.add_unconditional_best_fit(mu_test, unconditional_bfs_SB)
            test_stat_dists_SB.add_conditional_best_fit(mu_test, conditional_bfs_SB)
            test_stat_dists_B.add_unconditional_best_fit(mu_test, unconditional_bfs_B)
            test_stat_dists_B.add_conditional_best_fit(mu_test, conditional_bfs_B)

    def get_observed_test_stat(self, observed_test_stats, observed_data,
                               mu_test, signal_source_name, likelihood, save_fits=False,
                               asymptotic=False):
        """Internal function to evaluate observed test statistic.
        """
        # The constraints are centered on the expected values
        constraint_extra_args = dict()
        for background_source in self.background_source_names:
            constraint_extra_args[f'{background_source}_expected_counts'] = \
                self.expected_background_counts[background_source]

        likelihood.set_constraint_extra_args(**constraint_extra_args)

        # Set data
        if hasattr(likelihood, 'likelihoods'):
            for component, data in observed_data.items():
                likelihood.set_data(data, component)
        else:
            likelihood.set_data(observed_data)

        # Create test statistic
        test_statistic = self.test_statistic(likelihood)
        # Guesses for fit
        guess_dict = {f'{signal_source_name}_rate_multiplier': tf.cast(0.1, fd.float_type())}
        for background_source in self.background_source_names:
            guess_dict[f'{background_source}_rate_multiplier'] = tf.cast(self.expected_background_counts[background_source], fd.float_type())
        for key, value in guess_dict.items():
            if value < 0.1:
                guess_dict[key] = tf.cast(0.1, fd.float_type())
        # Evaluate test statistic
        ts_result = test_statistic(mu_test, signal_source_name, guess_dict,
                                   asymptotic=asymptotic)

        # Add to the test statistic collection
        observed_test_stats.add_test_stat(mu_test, ts_result[0])

        # Possibly save the fits
        if save_fits:
            observed_test_stats.add_unconditional_best_fit(mu_test, ts_result[1])
            observed_test_stats.add_conditional_best_fit(mu_test, ts_result[2])


@export
class IntervalCalculator():
    """NOTE: currently works for a single dataset only.
    Class for obtaining frequentist confidence intervals from test statistic distributions
    and observed test statistics.

    Arguments:
        - signal_source_names: tuple of names for signal sources (e.g. WIMPs of different
            masses)
        - observed_test_stats: dictionary {sourcename: ObservedTestStatistics} returned
            by running TSEvaluation routine to get observed test statistics
        - test_stat_dists_SB: dictionary {sourcename: TestStatisticDistributions} returned
            by running TSEvaluation routine to get test statistic distirbutions under
            the S+B hypothesis
        - test_stat_dists_B: dictionary {sourcename: TestStatisticDistributions} returned
            by running TSEvaluation routine to get test statistic distirbutions under
            the B-only hypothesis
    """
    def __init__(
            self,
            signal_source_names: ty.Tuple[str],
            observed_test_stats: ObservedTestStatistics,
            test_stat_dists_SB: TestStatisticDistributions,
            test_stat_dists_B: TestStatisticDistributions,
            test_stat_dists_SB_disco: TestStatisticDistributions=None):

        self.signal_source_names = signal_source_names
        self.observed_test_stats = observed_test_stats
        self.test_stat_dists_SB = test_stat_dists_SB
        self.test_stat_dists_B = test_stat_dists_B
        self.test_stat_dists_SB_disco = test_stat_dists_SB_disco

    @staticmethod
    def interp_helper(x, y, crossing_points, crit_val,
                      rising_edge=False, inverse=False):
        if rising_edge:
            x_left = x[crossing_points[0]]
            x_right = x[crossing_points[0] + 1]
            y_left = y[crossing_points[0]]
            y_right = y[crossing_points[0] + 1]
        else:
            x_left = x[crossing_points[-1]]
            x_right = x[crossing_points[-1] + 1]
            y_left = y[crossing_points[-1]]
            y_right = y[crossing_points[-1] + 1]
        gradient = (y_right - y_left) / (x_right - x_left)

        if inverse:
            return (crit_val - y_left) / gradient + x_left
        else:
            return (crit_val - x_left) * gradient + y_left

    def get_p_vals(self, conf_level, use_CLs=False, asymptotic=False):
        """Internal function to get p-value curves.
        """
        p_sb_collection = dict()
        powers_collection = dict()
        p_b_collection = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            if asymptotic:
                p_sb_collection[signal_source] = self.observed_test_stats[signal_source]
                continue

            # Get test statistic distribitions and observed test statistics
            test_stat_dists_SB = self.test_stat_dists_SB[signal_source]
            test_stat_dists_B = self.test_stat_dists_B[signal_source]
            observed_test_stats = self.observed_test_stats[signal_source]

            p_sb = test_stat_dists_SB.get_p_vals(observed_test_stats)
            p_sb_collection[signal_source] = p_sb

            if use_CLs:
                p_b = test_stat_dists_B.get_p_vals(observed_test_stats, inverse=True)
                p_b_collection[signal_source] = p_b
            else:
                crit_vals = test_stat_dists_SB.get_crit_vals(conf_level)
                powers = test_stat_dists_B.get_p_vals(crit_vals)
                powers_collection[signal_source] = powers

        if asymptotic:
            return p_sb_collection

        if use_CLs:
            return p_sb_collection, p_b_collection
        else:
            return p_sb_collection, powers_collection

    def get_interval(self, conf_level=0.1, pcl_level=0.16,
                     use_CLs=False,
                     asymptotic=False):
        """Get frequentist confidence interval.

        Arguments:
            - conf_level: confidence level to be used
            - pcl_level: level at which to power constrain the upper limits
            - use_CLs: if False, limits will be power constrained
                (https://arxiv.org/abs/1105.3166), and the final return value will be
                the powers under H1. If True, the CLs method will be used
                (https://inspirehep.net/literature/599622), and the final return value
                will be the p-value curves under H1
        """
        if not asymptotic:
            if use_CLs:
                p_sb, p_b = self.get_p_vals(conf_level, use_CLs=True)
            else:
                p_sb, powers = self.get_p_vals(conf_level, use_CLs=False)
        else:
            p_sb = self.get_p_vals(conf_level, use_CLs=True, asymptotic=True)

        lower_lim_all = dict()
        upper_lim_all = dict()
        upper_lim_all_raw = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            if not asymptotic:
                these_p_sb = p_sb[signal_source]
            else:
                these_p_sb = p_sb[signal_source].test_stats
            mus = np.array(list(these_p_sb.keys()))
            p_vals = np.array(list(these_p_sb.values()))

            if not asymptotic:
                if use_CLs:
                    these_p_b = p_b[signal_source]
                    p_vals_b = np.array(list(these_p_b.values()))
                    p_vals = p_vals / (1. - p_vals_b + 1e-10)
                else:
                    these_powers = powers[signal_source]
                    pws = np.array(list(these_powers.values()))

            # Find points where the p-value curve cross the critical value, decreasing
            upper_lims = np.argwhere(np.diff(np.sign(p_vals - np.ones_like(p_vals) * conf_level)) < 0.).flatten()
            # Find points where the p-value curve cross the critical value, increasing
            lower_lims = np.argwhere(np.diff(np.sign(p_vals - np.ones_like(p_vals) * conf_level)) > 0.).flatten()

            if len(lower_lims > 0):
                # Take the lowest increasing crossing point, and interpolate to get an upper limit
                lower_lim = self.interp_helper(mus, p_vals, lower_lims, conf_level,
                                               rising_edge=True, inverse=True)
            else:
                # We have no lower limit
                lower_lim = None

            assert len(upper_lims) > 0, 'No upper limit found!'
            # Take the highest decreasing crossing point, and interpolate to get an upper limit
            upper_lim = self.interp_helper(mus, p_vals, upper_lims, conf_level,
                                           rising_edge=False, inverse=True)
            upper_lim_raw = upper_lim

            if use_CLs is False and not asymptotic:
                M0 = self.interp_helper(mus, pws, upper_lims, upper_lim,
                                        rising_edge=False, inverse=False)
                if M0 < pcl_level:
                    # Find points where the power curve cross the critical value, increasing
                    upper_lims = np.argwhere(np.diff(np.sign(pws - np.ones_like(pws) * pcl_level)) > 0.).flatten()
                    # Take the lowest increasing crossing point, and interpolate to get an upper limit
                    upper_lim = self.interp_helper(mus, pws, upper_lims, pcl_level,
                                                   rising_edge=True, inverse=True)

            lower_lim_all[signal_source] = lower_lim
            upper_lim_all[signal_source] = upper_lim
            upper_lim_all_raw[signal_source] = upper_lim_raw

        if asymptotic:
            return lower_lim_all, upper_lim_all
        if use_CLs is False:
            return lower_lim_all, upper_lim_all, upper_lim_all_raw, p_sb, powers
        else:
            return lower_lim_all, upper_lim_all, p_sb, p_b

    def upper_lims_bands(self, pval_curve, mus, conf_level):
        try:
            upper_lims = np.argwhere(np.diff(np.sign(pval_curve - np.ones_like(pval_curve) * conf_level)) < 0.).flatten()
            return self.interp_helper(mus, pval_curve, upper_lims, conf_level,
                                    rising_edge=False, inverse=True)
        except Exception:
            return 0.

    def critical_disco_value(self, disco_pot_curve, mus, discovery_sigma):
        crossing_point = np.argwhere(np.diff(np.sign(disco_pot_curve - np.ones_like(disco_pot_curve) * discovery_sigma)) > 0.).flatten()
        return self.interp_helper(mus, disco_pot_curve, crossing_point, discovery_sigma,
                                  rising_edge=True, inverse=True)

    def get_bands(self, conf_level=0.1, quantiles=[0, 1, -1, 2, -2],
                  use_CLs=False, asymptotic=False):
        """
        """
        bands = dict()
        all_mus = dict()
        all_p_val_curves = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get test statistic distribitions
            if not asymptotic:
                test_stat_dists_SB = self.test_stat_dists_SB[signal_source]
            test_stat_dists_B = self.test_stat_dists_B[signal_source]

            mus = []
            p_val_curves = []
            # Loop over signal rate multipliers
            for mu_test, ts_values in test_stat_dists_B.ts_dists.items():
                if asymptotic:
                    these_p_vals = ts_values
                else:
                    these_p_vals = (100. - stats.percentileofscore(test_stat_dists_SB.ts_dists[mu_test],
                                                                ts_values,
                                                                kind='weak')) / 100.
                if use_CLs:
                    these_p_vals_b = stats.percentileofscore(test_stat_dists_B.ts_dists[mu_test],
                                                             ts_values,
                                                             kind='weak') / 100.
                    these_p_vals = these_p_vals / (1. - these_p_vals_b + 1e-10)
                mus.append(mu_test)
                p_val_curves.append(these_p_vals)

            p_val_curves = np.transpose(np.stack(p_val_curves, axis=0))
            upper_lims_bands = np.apply_along_axis(self.upper_lims_bands, 1, p_val_curves, mus, conf_level)

            if len(upper_lims_bands[upper_lims_bands == 0.]) > 0.:
                print(f'Found {len(upper_lims_bands[upper_lims_bands == 0.])} failed toy for {signal_source}; removing...')
                upper_lims_bands = upper_lims_bands[upper_lims_bands > 0.]

            these_bands = dict()
            for quantile in quantiles:
                these_bands[quantile] = np.quantile(np.sort(upper_lims_bands), stats.norm.cdf(quantile))
            bands[signal_source] = these_bands
            all_mus[signal_source] = mus
            all_p_val_curves[signal_source] = p_val_curves

        return bands, all_mus, all_p_val_curves

    def get_disco_sig(self):
        """
        """
        disco_sigs = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get observed (mu = 0) test statistic and B (m = 0) test statistic distribition
            try:
                observed_test_stat = self.observed_test_stats[signal_source].test_stats[0.]
                test_stat_dist_B = self.test_stat_dists_B[signal_source].ts_dists[0.]
            except Exception:
                raise RuntimeError("Error: did you scan over mu = 0?")

            p_val = (100. - stats.percentileofscore(test_stat_dist_B,
                                                    observed_test_stat,
                                                    kind='weak')) / 100.
            disco_sig = stats.norm.ppf(1. - p_val)
            disco_sig = np.where(disco_sig > 0., disco_sig, 0.)
            disco_sigs[signal_source] = disco_sig

        return disco_sigs

    def get_median_disco_asymptotic(self, sigma_level=3):
        """
        """
        medians = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get test statistic distribitions
            test_stat_dists_SB_disco = self.test_stat_dists_SB_disco[signal_source]

            mus = []
            disco_sig_curves = []
            # Loop over signal rate multipliers
            for mu_test, ts_values in test_stat_dists_SB_disco.ts_dists.items():
                these_disco_sigs = np.sqrt(ts_values)

                mus.append(mu_test)
                disco_sig_curves.append(these_disco_sigs)

            disco_sig_curves = np.stack(disco_sig_curves, axis=0)
            median_disco_sigs = [np.median(disco_sigs) for disco_sigs in disco_sig_curves]

            median_crossing_point = self.critical_disco_value(median_disco_sigs, mus, 3)
            medians[signal_source] = median_crossing_point

        return medians
