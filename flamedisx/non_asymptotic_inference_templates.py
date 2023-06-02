import flamedisx as fd
import numpy as np
from scipy import stats
import pickle as pkl
from tqdm.auto import tqdm
import typing as ty

import tensorflow as tf

export, __all__ = fd.exporter()


@export
class TestStatistic():
    """
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
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, bf_unconditional, bf_conditional):
        """Evaluate the test statistic of equation 11 in
        https://arxiv.org/abs/1007.1727.
        """
        ll_conditional = self.likelihood(**bf_conditional)
        ll_unconditional = self.likelihood(**bf_unconditional)

        return -2. * (ll_conditional - ll_unconditional)


@export
class TestStatisticDistributions():
    """
    """
    def __init__(self):
        self.ts_dists = dict()

    def add_ts_dist(self, mu_test, ts_values):
        self.ts_dists[mu_test] = ts_values


@export
class ToyTSDists():
    """NOTE: currently works for a single dataset only.

    Arguments:
        - signal_source_names: tuple of names for signal sources (e.g. WIMPs of different
            masses)
        - background_source_names: tuple of names for background sources
        - sources: dictionary {sourcename: class} of all signal and background source classes
        - arguments: dictionary {sourcename: {kwarg1: value, ...}, ...}, for
            passing keyword arguments to source constructors
        - batch_size: batch size that will be used for the RM fits
        - expected_background_counts: dictionary of expected counts for background sources
        - gaussian_constraint_widths: dictionary giving the constraint width for all sources
            using Gaussian constraints for their rate nuisance parameters
        - sample_other_constraints: dictionary of functions to sample constraint means
            in the toys for any sources using non-Gaussian constraints for their rate nuisance
            parameters. Argument to the function will be either the prior expected counts,
            or the number of counts at the conditional MLE, depending on the mode
        - rm_bounds: dictionary {sourcename: (lower, upper)} to set fit bounds on the rate multipliers
        - ntoys: number of toys that will be run to get test statistic distributions
        - log_constraint_fn: logarithm of the constraint function used in the likelihood. Any arguments
            which aren't fit parameters, such as those determining constraint means for toys, will need
            passing via the set_constraint_extra_args() function
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

    def get_test_stat_dists(self, mus_test=None, conditional_best_fits=None):
        """Get test statistic distributions.

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate multipliers
                to be tested for each signal source
            - conditional_best_fits: pass in conditional MLEs from the data if you wish to
                center rate nuisance parameter constraints on these, rather than prior expected counts
        """
        if conditional_best_fits is not None:
            self.conditional_best_fits = pkl.load(open(conditional_best_fits, 'rb'))
        else:
            self.conditional_best_fits = None

        test_stat_dists_SB_collection = dict()
        test_stat_dists_B_collection = dict()
        unconditional_best_fits = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            test_stat_dists_SB = TestStatisticDistributions()
            test_stat_dists_B = TestStatisticDistributions()
            unconditional_bfs = dict()

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

            # Save the test statistic values and unconditional best fits for each toy, for each
            # signal RM scanned over, for this signal source
            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                ts_dist = self.toy_test_statistic_dist(mu_test, signal_source, likelihood)
                test_stat_dists_SB.add_ts_dist(mu_test, ts_dist[0])
                test_stat_dists_B.add_ts_dist(mu_test, ts_dist[2])
                unconditional_bfs[mu_test] = ts_dist[1]

            test_stat_dists_SB_collection[signal_source] = test_stat_dists_SB
            test_stat_dists_B_collection[signal_source] = test_stat_dists_B
            unconditional_best_fits[signal_source] = unconditional_bfs

        return test_stat_dists_SB_collection, test_stat_dists_B_collection, unconditional_best_fits

    def toy_test_statistic_dist(self, mu_test, signal_source_name, likelihood):
        """Internal function to get a test statistic distribution for a given signal source
        and signal RM.
        """
        ts_values_SB = []
        ts_values_B = []
        unconditional_bfs = []

        # Loop over toys
        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            simulate_dict = dict()
            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                # Case where we use the conditional best fits as constraint centers and simulated values
                if self.conditional_best_fits is not None:
                    expected_background_counts = self.conditional_best_fits[signal_source_name][mu_test][f'{background_source}_rate_multiplier']
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
            # Save test statistic
            ts_values_SB.append(ts_result_SB[0])
            unconditional_bfs.append(ts_result_SB[1])

            # B-only toys

            simulate_dict[f'{signal_source_name}_rate_multiplier'] = 0.
            # Simulate and set data
            toy_data_B = likelihood.simulate(**simulate_dict)
            likelihood.set_data(toy_data_B)
            # Create test statistic
            test_statistic_B = self.test_statistic(likelihood)
            # Guesses for fit
            guess_dict_B = simulate_dict.copy()
            for key, value in guess_dict_B.items():
                if value < 0.1:
                    guess_dict_B[key] = 0.1
            ts_result_B = test_statistic_B(mu_test, signal_source_name, guess_dict_B)
            ts_values_B.append(ts_result_B[0])

        # Return the test statistic and unconditional best fit
        return ts_values_SB, unconditional_bfs, ts_values_B


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
        - expected_background_counts: dictionary of expected counts for background sources
        - gaussian_constraint_widths: dictionary giving the constraint width for all sources
            using Gaussian constraints for their rate nuisance parameters
        - sample_other_constraints: dictionary of functions to sample constraint means
            in the toys for any sources using non-Gaussian constraints for their rate nuisance
            parameters. Argument to the function will be either the prior expected counts,
            or the number of counts at the conditional MLE, depending on the mode
        - rm_bounds: dictionary {sourcename: (lower, upper)} to set fit bounds on the rate multipliers
        - ntoys: number of toys that will be run to get test statistic distributions
        - log_constraint_fn: logarithm of the constraint function used in the likelihood. Any arguments
            which aren't fit parameters, such as those determining constraint means for toys, will need
            passing via the set_constraint_extra_args() function
    """

    def __init__(
            self,
            test_statistic: TestStatistic.__class__,
            signal_source_names: ty.Tuple[str],
            background_source_names: ty.Tuple[str],
            sources: ty.Dict[str, fd.Source.__class__],
            arguments: ty.Dict[str, ty.Dict[str, ty.Union[int, float]]] = None,
            batch_size=10000,
            expected_background_counts: ty.Dict[str, float] = None,
            gaussian_constraint_widths: ty.Dict[str, float] = None,
            sample_other_constraints: ty.Dict[str, ty.Callable] = None,
            rm_bounds: ty.Dict[str, ty.Tuple[float, float]] = None,
            ntoys=1000,
            log_constraint_fn: ty.Callable = None):

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

        self.test_statistic = test_statistic

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.ntoys = ntoys
        self.batch_size = batch_size

        self.expected_background_counts = expected_background_counts
        self.gaussian_constraint_widths = gaussian_constraint_widths
        self.sample_other_constraints = sample_other_constraints
        self.rm_bounds = rm_bounds

        self.test_stat_dists = dict()
        self.test_stat_dists_pcl = dict()
        self.observed_test_stats = dict()
        self.conditional_best_fits = dict()
        self.p_vals = dict()
        self.powers = dict()
        self.p_bs = dict()

        self.sources = sources
        self.arguments = arguments

    def get_test_stat_dists(self, mus_test=None, input_dists=None, input_dists_pcl=None,
                            input_conditional_best_fits=None,
                            dists_output_name=None, use_expected_nuisance=False):
        """Get test statistic distributions.

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate multipliers
                to be tested for each signal source
            - input_dists: path to input test statistic distributions, if we wish to pass this
            - input_dist_pcl:
            - input_conditional_best_fits:
            - dists_output_name: name of file in which to save test statistic distributions,
                if this is desired
            - use_expected_nuisance: if set to true, center rate nuisance parameter constraints on the
                prior expected counts, rather than conditional MLEs
        """
        if input_dists is not None:
            # Read in test statistic distributions
            self.test_stat_dists = pkl.load(open(input_dists, 'rb'))
            if input_dists_pcl is not None:
                # Read in PCL test statistic distributions
                self.test_stat_dists_pcl = pkl.load(open(input_dists_pcl, 'rb'))
            return

        if input_conditional_best_fits is not None:
            # Read in conditional best fits
            self.conditional_best_fits = pkl.load(open(input_conditional_best_fits, 'rb'))

        self.test_stat_dists = dict()
        unconditional_best_fits = dict()
        constraint_central_values = dict()
        self.test_stat_dists_pcl = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            test_stat_dists = dict()
            unconditional_bfs = dict()
            constraint_vals = dict()
            test_stat_dists_pcl = dict()

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

            # Save the test statistic values and unconditional best fits for each toy, for each
            # signal RM scanned over, for this signal source
            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                ts_dist = self.toy_test_statistic_dist(mu_test, signal_source, likelihood, use_expected_nuisance)
                test_stat_dists[mu_test] = ts_dist[0]
                unconditional_bfs[mu_test] = ts_dist[1]
                constraint_vals[mu_test] = ts_dist[2]
                test_stat_dists_pcl[mu_test] = ts_dist[3]

            self.test_stat_dists[signal_source] = test_stat_dists
            unconditional_best_fits[signal_source] = unconditional_bfs
            constraint_central_values[signal_source] = constraint_vals
            self.test_stat_dists_pcl[signal_source] = test_stat_dists_pcl

        # Output test statistic distributions and fits
        if dists_output_name is not None:
            pkl.dump(self.test_stat_dists, open(dists_output_name, 'wb'))
            pkl.dump(unconditional_best_fits, open(dists_output_name[:-4] + '_unconditional_fits.pkl', 'wb'))
            pkl.dump(constraint_central_values, open(dists_output_name[:-4] + '_constraint_central_values.pkl', 'wb'))
            pkl.dump(self.test_stat_dists_pcl, open(dists_output_name[:-4] + '_pcl.pkl', 'wb'))

    def toy_test_statistic_dist(self, mu_test, signal_source_name, likelihood, use_expected_nuisance):
        """Internal function to get a test statistic distribution for a given signal source
        and signal RM.
        """
        ts_values = []
        unconditional_bfs = []
        constraint_vals = []
        ts_values_pcl = []

        # Loop over toys
        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            simulate_dict = dict()
            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                # Case where we use the conditional best fits as constraint centers and simulated values
                if use_expected_nuisance is False:
                    expected_background_counts = self.conditional_best_fits[signal_source_name][mu_test][f'{background_source}_rate_multiplier']
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

            simulate_dict[f'{signal_source_name}_rate_multiplier'] = mu_test

            # Shift the constraint in the likelihood based on the background RMs we drew
            likelihood.set_constraint_extra_args(**constraint_extra_args)

            # Simulate and set data
            toy_data = likelihood.simulate(**simulate_dict)

            likelihood.set_data(toy_data)

            this_test_statistic = self.test_statistic(likelihood)

            guess_dict = simulate_dict.copy()

            for key, value in guess_dict.items():
                if value < 0.1:
                    guess_dict[key] = 0.1

            ts_result = this_test_statistic(mu_test, signal_source_name, guess_dict)
            ts_values.append(ts_result[0])
            unconditional_bfs.append(ts_result[1])

            # Now repeat for PCL
            simulate_dict[f'{signal_source_name}_rate_multiplier'] = 0.

            # Simulate and set data
            toy_data_pcl = likelihood.simulate(**simulate_dict)

            likelihood.set_data(toy_data_pcl)

            guess_dict_pcl = simulate_dict.copy()

            for key, value in guess_dict_pcl.items():
                if value < 0.1:
                    guess_dict_pcl[key] = 0.1

            ts_result_pcl = this_test_statistic(mu_test, signal_source_name, guess_dict_pcl)
            ts_values_pcl.append(ts_result_pcl[0])

            constraint_vals_dict = dict()
            for key, value in constraint_extra_args.items():
                constraint_vals_dict[key] = fd.tf_to_np(value)
            constraint_vals.append(constraint_vals_dict)

        # Return the test statistic and unconditional best fit
        return ts_values, unconditional_bfs, constraint_vals, ts_values_pcl

    def get_observed_test_stats(self, mus_test=None, data=None, input_test_stats=None, test_stats_output_name=None):
        """Get observed test statistics.

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate multipliers
                to be tested for each signal source
            - data: observed dataset
            - input_test_stats: path to input observed test statistics, if we wish to pass this
            - test_stats_output_name: name of file in which to save observed test statistics,
                if this is desired
        """
        if input_test_stats is not None:
            # Read in observed test statistics
            self.observed_test_stats = pkl.load(open(input_test_stats, 'rb'))
            return

        assert data is not None, 'Must pass in data'

        self.observed_test_stats = dict()
        self.conditional_best_fits = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            observed_test_stats = dict()
            conditional_best_fits = dict()

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

            # The constraints are centered on the expected values
            constraint_extra_args = dict()
            for background_source in self.background_source_names:
                constraint_extra_args[f'{background_source}_expected_counts'] = self.expected_background_counts[background_source]

            likelihood.set_constraint_extra_args(**constraint_extra_args)

            # Set data
            likelihood.set_data(data)

            these_mus_test = mus_test[signal_source]
            # Loop over signal rate multipliers
            for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                # Set the guesses
                guess_dict = {f'{signal_source}_rate_multiplier': mu_test}
                for background_source in self.background_source_names:
                    guess_dict[f'{background_source}_rate_multiplier'] = self.expected_background_counts[background_source]

                for key, value in guess_dict.items():
                    if value < 0.1:
                        guess_dict[key] = 0.1

                ts_result = self.test_statistic_tmu_tilde(mu_test, signal_source, likelihood, guess_dict)
                observed_test_stats[mu_test] = ts_result[0]
                conditional_best_fits[mu_test] = ts_result[2]

            self.observed_test_stats[signal_source] = observed_test_stats
            self.conditional_best_fits[signal_source] = conditional_best_fits

            # Output observed test statistics
            if test_stats_output_name is not None:
                pkl.dump(self.observed_test_stats, open(test_stats_output_name, 'wb'))
                pkl.dump(self.conditional_best_fits, open(test_stats_output_name[:-4] + '_conditional_fits.pkl', 'wb'))

    def get_p_vals(self, conf_level):
        """Internal function to get p-value curves.
        """
        self.p_vals = dict()
        self.powers = dict()
        self.p_bs = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get test statistic distribitions and observed test statistics
            test_stat_dists = self.test_stat_dists[signal_source]
            test_stat_dists_pcl = self.test_stat_dists_pcl[signal_source]
            observed_test_stats = self.observed_test_stats[signal_source]

            assert test_stat_dists.keys() == observed_test_stats.keys(), \
                f'Must get test statistic distributions and observed test statistics for {signal_source} with ' \
                'the same mu values'

            p_vals = dict()
            powers = dict()
            p_bs = dict()
            # Loop over signal rate multipliers
            for mu_test in observed_test_stats.keys():
                # Compute the p-value from the observed test statistic and the S+B distribition
                p_vals[mu_test] = (100. - stats.percentileofscore(test_stat_dists[mu_test],
                                                                  observed_test_stats[mu_test],
                                                                  kind='weak')) / 100.
                # Get the critical TS value under the S+B distribution
                ts_crit = np.quantile(test_stat_dists[mu_test], 1. - conf_level)
                # Compute the power from the critical TS value and the B distribition
                powers[mu_test] = (100. - stats.percentileofscore(test_stat_dists_pcl[mu_test],
                                                                  ts_crit,
                                                                  kind='weak')) / 100.
                # Compute the p_b value from the observed test statistic and the B distribition
                p_bs[mu_test] = stats.percentileofscore(test_stat_dists_pcl[mu_test],
                                                        observed_test_stats[mu_test],
                                                        kind='weak') / 100.

            # Record p-value, power, p_b curves
            self.p_vals[signal_source] = p_vals
            self.powers[signal_source] = powers
            self.p_bs[signal_source] = p_bs

    @staticmethod
    def inverse_interp_rising_edge(x, y, crossing_points, y_crit):
        x_left = x[crossing_points[0]]
        x_right = x[crossing_points[0] + 1]
        y_left = y[crossing_points[0]]
        y_right = y[crossing_points[0] + 1]

        gradient = (y_right - y_left) / (x_right - x_left)
        crossing_point = (y_crit - y_left) / gradient + x_left

        return crossing_point

    @staticmethod
    def inverse_interp_falling_edge(x, y, crossing_points, y_crit):
        x_left = x[crossing_points[-1]]
        x_right = x[crossing_points[-1] + 1]
        y_left = y[crossing_points[-1]]
        y_right = y[crossing_points[-1] + 1]

        gradient = (y_right - y_left) / (x_right - x_left)
        crossing_point = (y_crit - y_left) / gradient + x_left

        return crossing_point

    @staticmethod
    def interp_falling_edge(x, y, crossing_points, x_crit):
        x_left = x[crossing_points[-1]]
        x_right = x[crossing_points[-1] + 1]
        y_left = y[crossing_points[-1]]
        y_right = y[crossing_points[-1] + 1]

        gradient = (y_right - y_left) / (x_right - x_left)
        y_val = (x_crit - x_left) * gradient + y_left

        return y_val

    def get_interval(self, conf_level=0.1, pcl_level=0.16, return_p_vals=False,
                     use_CLs=False):
        """Get either upper limit, or possibly upper and lower limits.
        Before using this get_test_stat_dists() and get_observed_test_stats() must
        have bene called.

        Arguments:
            - conf_level: confidence level to be used for the limit/interval
            - return_p_vals: whether or not to output the p-value curves
        """
        # Get the p-value curves
        self.get_p_vals(conf_level)

        lower_lim_all = dict()
        upper_lim_all = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            these_pvals = self.p_vals[signal_source]

            mus = list(these_pvals.keys())
            pvals = list(these_pvals.values())

            if use_CLs is True:
                these_pbs = self.p_bs[signal_source]
                pbs = list(these_pbs.values())

                pvals = np.array(pvals) / (1. - np.array(pbs))
            else:
                these_powers = self.powers[signal_source]
                powers = list(these_powers.values())

            # Find points where the p-value curve cross the critical value, decreasing
            upper_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) < 0.).flatten()
            # Find points where the p-value curve cross the critical value, increasing
            lower_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) > 0.).flatten()

            if len(lower_lims > 0):
                # Take the lowest increasing crossing point, and interpolate to get an upper limit
                lower_lim = self.inverse_interp_rising_edge(mus, pvals, lower_lims, conf_level)
            else:
                # We have no lower limit
                lower_lim = None

            assert len(upper_lims) > 0, 'No upper limit found!'
            # Take the highest decreasing crossing point, and interpolate to get an upper limit
            upper_lim = self.inverse_interp_falling_edge(mus, pvals, upper_lims, conf_level)

            if use_CLs is False:
                if lower_lim is not None:
                    raise RuntimeError("Current not handling PCL for interval, just upper limit")

                M0 = self.interp_falling_edge(mus, powers, upper_lims, upper_lim)
                if M0 < pcl_level:
                    # Find points where the power curve cross the critical value, increasing
                    upper_lims = np.argwhere(np.diff(np.sign(powers - np.ones_like(powers) * pcl_level)) > 0.).flatten()
                    # Take the lowest increasing crossing point, and interpolate to get an upper limit
                    upper_lim = self.inverse_interp_rising_edge(mus, powers, upper_lims, pcl_level)

            lower_lim_all[signal_source] = lower_lim
            upper_lim_all[signal_source] = upper_lim

        if return_p_vals is True:
            if use_CLs is True:
                extra_return = self.p_bs
            else:
                extra_return = self.powers
            # Return p-value curves, and intervals/limits
            return self.p_vals, extra_return, lower_lim_all, upper_lim_all
        else:
            # Return intervals/limits
            return lower_lim_all, upper_lim_all
