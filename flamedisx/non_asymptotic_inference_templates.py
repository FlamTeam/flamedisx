import flamedisx as fd
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
import typing as ty

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

    def __call__(self, mu_test, signal_source_name, guess_dict):
        # To fix the signal RM in the conditional fit
        fix_dict = {f'{signal_source_name}_rate_multiplier': mu_test}

        guess_dict_nuisance = guess_dict.copy()
        guess_dict_nuisance.pop(f'{signal_source_name}_rate_multiplier')

        # Conditional fit
        bf_conditional = self.likelihood.bestfit(fix=fix_dict, guess=guess_dict_nuisance, suppress_warnings=True)
        # Uncnditional fit
        bf_unconditional = self.likelihood.bestfit(guess=guess_dict, suppress_warnings=True)

        # Return the test statistic, unconditional fit and conditional fit
        return self.evaluate(bf_unconditional, bf_conditional), bf_unconditional, bf_conditional


@export
class TestStatisticTMuTilde(TestStatistic):
    """Evaluate the test statistic of equation 11 in https://arxiv.org/abs/1007.1727.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, bf_unconditional, bf_conditional):
        ll_conditional = self.likelihood(**bf_conditional)
        ll_unconditional = self.likelihood(**bf_unconditional)

        return -2. * (ll_conditional - ll_unconditional)


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
            f'POI values for observed test statistics and test statistic distributions ' \
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
        - sources: dictionary {sourcename: class} of all signal and background source classes
        - arguments: dictionary {sourcename: {kwarg1: value, ...}, ...}, for
            passing keyword arguments to source constructors
        - expected_background_counts: dictionary of expected counts for background sources
        - gaussian_constraint_widths: dictionary giving the constraint width for all sources
            using Gaussian constraints for their rate nuisance parameters
        - sample_other_constraints: dictionary of functions to sample constraint means
            in the toys for any sources using non-Gaussian constraints for their rate nuisance
            parameters. Argument to the function will be either the prior expected counts,
            or the number of counts at the conditional MLE, depending on the mode
        - rm_bounds: dictionary {sourcename: (lower, upper)} to set fit bounds on the rate multipliers
        - log_constraint_fn: logarithm of the constraint function used in the likelihood. Any arguments
            which aren't fit parameters, such as those determining constraint means for toys, will need
            passing via the set_constraint_extra_args() function
        - ntoys: number of toys that will be run to get test statistic distributions
        - batch_size: batch size that will be used for the RM fits
    """
    def __init__(
            self,
            test_statistic: TestStatistic.__class__,
            signal_source_names: ty.Tuple[str],
            background_source_names: ty.Tuple[str],
            sources: ty.Dict[str, fd.Source.__class__],
            arguments: ty.Dict[str, ty.Dict[str, ty.Union[int, float]]] = None,
            expected_background_counts: ty.Dict[str, float] = None,
            gaussian_constraint_widths: ty.Dict[str, float] = None,
            sample_other_constraints: ty.Dict[str, ty.Callable] = None,
            rm_bounds: ty.Dict[str, ty.Tuple[float, float]] = None,
            log_constraint_fn: ty.Callable = None,
            ntoys=1000,
            batch_size=10000):

        for key in sources.keys():
            if key not in arguments.keys():
                arguments[key] = dict()

        if gaussian_constraint_widths is None:
            gaussian_constraints_widths = dict()

        if sample_other_constraints is None:
            sample_other_constraints = dict()

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

        self.ntoys = ntoys
        self.batch_size = batch_size

        self.test_statistic = test_statistic

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.sources = sources
        self.arguments = arguments

        self.expected_background_counts = expected_background_counts
        self.gaussian_constraint_widths = gaussian_constraint_widths
        self.sample_other_constraints = sample_other_constraints
        self.rm_bounds = rm_bounds

    def run_routine(self, mus_test, save_fits=False,
                    observed_data=None, observed_test_stats=None):
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
        """
        if observed_test_stats is not None:
            self.observed_test_stats = observed_test_stats
        else:
            self.observed_test_stats = None

        if observed_data is None:
            self.background_only_toys = []

        observed_test_stats_collection = dict()
        test_stat_dists_SB_collection = dict()
        test_stat_dists_B_collection = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            observed_test_stats = ObservedTestStatistics()
            test_stat_dists_SB = TestStatisticDistributions()
            test_stat_dists_B = TestStatisticDistributions()

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
            for background_source in self.background_source_names:
                if background_source in self.rm_bounds.keys():
                    rm_bounds[background_source] = self.rm_bounds[background_source]

            # Pass rate multiplier bounds to likelihood
            likelihood.set_rate_multiplier_bounds(**rm_bounds)

            # Pass constraint function to likelihood
            likelihood.set_log_constraint(self.log_constraint_fn)

            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                # Case where we want observed test statistics
                if observed_data is not None:
                    self.get_observed_test_stat(observed_test_stats, observed_data,
                                                mu_test, signal_source, likelihood, save_fits=save_fits)
                # Case where we want test statistic distributions
                else:
                    self.toy_test_statistic_dist(test_stat_dists_SB, test_stat_dists_B,
                                                 mu_test, signal_source, likelihood, save_fits=save_fits)

            if observed_data is not None:
                observed_test_stats_collection[signal_source] = observed_test_stats
            else:
                test_stat_dists_SB_collection[signal_source] = test_stat_dists_SB
                test_stat_dists_B_collection[signal_source] = test_stat_dists_B

        if observed_data is not None:
            return observed_test_stats_collection
        else:
            return test_stat_dists_SB_collection, test_stat_dists_B_collection

    def toy_test_statistic_dist(self, test_stat_dists_SB, test_stat_dists_B,
                                mu_test, signal_source_name, likelihood, save_fits=False):
        """Internal function to get test statistic distribution.
        """
        ts_values_SB = []
        ts_values_B = []
        if save_fits:
            unconditional_bfs_SB = []
            conditional_bfs_SB = []
            unconditional_bfs_B = []
            conditional_bfs_B = []

        # Loop over toys
        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            simulate_dict = dict()
            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                # Case where we use the conditional best fits as constraint centers and simulated values
                if self.observed_test_stats is not None:
                    try:
                        conditional_bfs_observed = self.observed_test_stats[signal_source_name].conditional_best_fits
                        expected_background_counts = conditional_bfs_observed[mu_test][f'{background_source}_rate_multiplier']
                    except Exception:
                        raise RuntimeError("Could not find observed conditional best fits")
                # Case where we use the prior expected counts as constraint centers and simualted values
                else:
                    expected_background_counts = self.expected_background_counts[background_source]

                # Sample constraint centers
                if background_source in self.gaussian_constraint_widths:
                    draw = stats.norm.rvs(loc=expected_background_counts,
                                          scale = self.gaussian_constraint_widths[background_source])
                    constraint_extra_args[f'{background_source}_expected_counts'] = tf.cast(draw, fd.float_type())

                elif background_source in self.sample_other_constraints:
                    draw = self.sample_other_constraints[background_source](expected_background_counts)
                    constraint_extra_args[f'{background_source}_expected_counts'] = tf.cast(draw, fd.float_type())

                simulate_dict[f'{background_source}_rate_multiplier'] = expected_background_counts

            # Shift the constraint in the likelihood based on the background RMs we drew
            likelihood.set_constraint_extra_args(**constraint_extra_args)

            # S+B toys

            simulate_dict[f'{signal_source_name}_rate_multiplier'] = mu_test
            # Simulate and set data
            toy_data_SB = likelihood.simulate(**simulate_dict)
            likelihood.set_data(toy_data_SB)
            # Create test statistic
            test_statistic_SB = self.test_statistic(likelihood)
            # Guesses for fit
            guess_dict_SB = simulate_dict.copy()
            for key, value in guess_dict_SB.items():
                if value < 0.1:
                    guess_dict_SB[key] = 0.1
            # Evaluate test statistic
            ts_result_SB = test_statistic_SB(mu_test, signal_source_name, guess_dict_SB)
            # Save test statistic, and possibly fits
            ts_values_SB.append(ts_result_SB[0])
            if save_fits:
                unconditional_bfs_SB.append(ts_result_SB[1])
                conditional_bfs_SB.append(ts_result_SB[2])

            # B-only toys

            # Ensure we use the same background-only toy dataset for every signal model in this toy
            try:
                toy_data_B = self.background_only_toys[toy]
            except IndexError:
                simulate_dict[f'{signal_source_name}_rate_multiplier'] = 0.
                # Simulate and set data
                toy_data_B = likelihood.simulate(**simulate_dict)
                self.background_only_toys.append(toy_data_B)

            likelihood.set_data(toy_data_B)
            # Create test statistic
            test_statistic_B = self.test_statistic(likelihood)
            # Guesses for fit
            guess_dict_B = simulate_dict.copy()
            for key, value in guess_dict_B.items():
                if value < 0.1:
                    guess_dict_B[key] = 0.1
            # Evaluate test statistic
            ts_result_B = test_statistic_B(mu_test, signal_source_name, guess_dict_B)
            # Save test statistic, and possibly fits
            ts_values_B.append(ts_result_B[0])
            if save_fits:
                unconditional_bfs_B.append(ts_result_SB[1])
                conditional_bfs_B.append(ts_result_SB[2])

        # Add to the test statistic distributions
        test_stat_dists_SB.add_ts_dist(mu_test, ts_values_SB)
        test_stat_dists_B.add_ts_dist(mu_test, ts_values_B)

        # Possibly save the fits
        if save_fits:
            test_stat_dists_SB.add_unconditional_best_fit(mu_test, unconditional_bfs_SB)
            test_stat_dists_SB.add_conditional_best_fit(mu_test, conditional_bfs_SB)
            test_stat_dists_B.add_unconditional_best_fit(mu_test, unconditional_bfs_B)
            test_stat_dists_B.add_conditional_best_fit(mu_test, conditional_bfs_B)

    def get_observed_test_stat(self, observed_test_stats, observed_data,
                               mu_test, signal_source_name, likelihood, save_fits=False):
        """Internal function to evaluate observed test statistic.
        """
        # The constraints are centered on the expected values
        constraint_extra_args = dict()
        for background_source in self.background_source_names:
            constraint_extra_args[f'{background_source}_expected_counts'] = self.expected_background_counts[background_source]

        likelihood.set_constraint_extra_args(**constraint_extra_args)

        # Set data
        likelihood.set_data(observed_data)
        # Create test statistic
        test_statistic = self.test_statistic(likelihood)
        # Guesses for fit
        guess_dict = {f'{signal_source_name}_rate_multiplier': mu_test}
        for background_source in self.background_source_names:
            guess_dict[f'{background_source}_rate_multiplier'] = self.expected_background_counts[background_source]
        for key, value in guess_dict.items():
            if value < 0.1:
                guess_dict[key] = 0.1
        # Evaluate test statistic
        ts_result = test_statistic(mu_test, signal_source_name, guess_dict)

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
            test_stat_dists_B: TestStatisticDistributions):

        self.signal_source_names = signal_source_names
        self.observed_test_stats = observed_test_stats
        self.test_stat_dists_SB = test_stat_dists_SB
        self.test_stat_dists_B = test_stat_dists_B

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

    def get_p_vals(self, conf_level, use_CLs=False):
        """Internal function to get p-value curves.
        """
        p_sb_collection = dict()
        powers_collection = dict()
        p_b_collection = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
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

        if use_CLs:
            return p_sb_collection, p_b_collection
        else:
            return p_sb_collection, powers_collection

    def get_interval(self, conf_level=0.1, pcl_level=0.16,
                     use_CLs=False):
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
        if use_CLs:
            p_sb, p_b = self.get_p_vals(conf_level, use_CLs=True)
        else:
            p_sb, powers = self.get_p_vals(conf_level, use_CLs=False)

        lower_lim_all = dict()
        upper_lim_all = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            these_p_sb = p_sb[signal_source]
            mus = np.array(list(these_p_sb.keys()))
            p_vals = np.array(list(these_p_sb.values()))

            if use_CLs:
                these_p_b = p_b[signal_source]
                p_vals_b = np.array(list(these_p_b.values()))
                p_vals = p_vals / (1. - p_vals_b)
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

            if use_CLs is False:
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

        if use_CLs is False:
            return lower_lim_all, upper_lim_all, p_sb, powers
        else:
            return lower_lim_all, upper_lim_all, p_sb, p_b

    def get_bands_OLD(self, conf_level=0.1):
        """
        """
        bands = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get test statistic distribitions
            test_stat_dists_SB = self.test_stat_dists_SB[signal_source]
            test_stat_dists_B = self.test_stat_dists_B[signal_source]

            mus = []
            p_val_quantiles = {0: [], 1: [], -1: [], 2: []}

            # Loop over signal rate multipliers
            for mu_test, ts_values in test_stat_dists_B.ts_dists.items():
                these_p_vals = (100. - stats.percentileofscore(test_stat_dists_SB.ts_dists[mu_test],
                                                               ts_values,
                                                               kind='weak')) / 100.
                mus.append(mu_test)

                for key, value in p_val_quantiles.items():
                    value.append(np.quantile(these_p_vals, stats.norm.cdf(key)))

            these_bands = dict()
            for key, value in p_val_quantiles.items():
                upper_lims = np.argwhere(np.diff(np.sign(value - np.ones_like(value) * conf_level)) < 0.).flatten()
                these_bands[key] = self.interp_helper(mus, value, upper_lims, conf_level,
                                                      rising_edge=False, inverse=True)

            bands[signal_source] = these_bands

        return bands
