import flamedisx as fd
import numpy as np
from scipy import stats
import pickle as pkl
from tqdm.auto import tqdm
import typing as ty

import tensorflow as tf

export, __all__ = fd.exporter()


@export
class FrequentistSensitivityRatesOnlyWilks():
    """NOTE: currently works for a single dataset only.

    Arguments:
        - signal_source_names: tuple of names for signal sources (e.g. WIMPs of different
            masses) of which we want to get limits/intervals
        - background_source_names: tuple of names for background sources
        - sources: dictionary {sourcename: class} of all signal and background source classes
        - arguments: dictionary {sourcename: {kwarg1: value, ...}, ...}, for
            passing keyword arguments to source constructors
        - batch_size_rates: batch size that will be used for the RM fits
        - expected_counts:
        - gaussian_constraint_widths: dictionary giving the constraint width for all sources
            using Gaussian constraints for their rate nuisance parameters
        - ndatasets: number of background-only datasets we will generate to estimate median sensitivity
        - log_constraint_fn: logarithm of the constraint function used in the likelihood. Any arguments
            which aren't fit parameters, such as those determining constraint means for toys, will need
            passing via the set_constraint_extra_args() function
        - input_reservoir:
        - skip_reservoir:
    """

    def __init__(
            self,
            signal_source_names: ty.Tuple[str],
            background_source_names: ty.Tuple[str],
            sources: ty.Dict[str, fd.Source.__class__],
            batch_size_rates=10000,
            expected_counts: ty.Dict[str, float] = None,
            gaussian_constraint_widths: ty.Dict[str, float] = None,
            ndatasets=100,
            log_constraint_fn: ty.Callable = None,
            input_reservoir=None,
            skip_reservoir=False):

        if gaussian_constraint_widths is None:
            gaussian_constraints_widths = dict()

        if log_constraint_fn is None:
            def log_constraint_fn(**kwargs):
                return 0.
            self.log_constraint_fn = log_constraint_fn
        else:
            self.log_constraint_fn = log_constraint_fn

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.ndatasets = ndatasets
        self.batch_size_rates = batch_size_rates

        self.expected_counts = expected_counts

        self.gaussian_constraint_widths = gaussian_constraint_widths

        self.test_stats = dict()
        self.median_p_vals = dict()

        self.sources = sources

        if not skip_reservoir:
            assert input_reservoir is not None, "Currently only support pre-computed reservoirs"
            # Read in frozen source reservoir
            self.reservoir = pkl.load(open(input_reservoir, 'rb'))

    def test_statistic_qmu(self, mu_test, signal_source_name, likelihood, guess_dict):
        """Internal function to evaluate the test statistic of equation 14 in
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
        if bf_unconditional[f'{signal_source_name}_rate_multiplier'] > mu_test:
            q_mu = 0.
        else:
            q_mu = -2. * (ll_conditional - ll_unconditional)

        return q_mu

    def get_test_stats_background_only(self, mus_test=None, input_test_stats=None, test_stats_output_name=None):
        """
        """
        if input_test_stats is not None:
            # Read in test statistics
            self.test_stats = pkl.load(open(input_test_stats, 'rb'))
            return

        self.test_stats = dict()

        # Loop over background-only datasets
        for toy in tqdm(range(self.ndatasets), desc='Running over datasets'):
            toy_data = None
            # Loop over signal sources
            for signal_source in self.signal_source_names:
                if signal_source not in self.test_stats.keys():
                    self.test_stats[signal_source] = dict()

                sources = dict()
                for background_source in self.background_source_names:
                    sources[background_source] = self.sources[background_source]
                sources[signal_source] = self.sources[signal_source]

                # Create likelihood of FrozenReservoirSources
                likelihood = fd.LogLikelihood(sources={sname: fd.FrozenReservoirSource for sname in sources.keys()},
                                              arguments={sname: {'source_type': sclass,
                                                                 'source_name': sname,
                                                                 'reservoir': self.reservoir,
                                                                 'input_mus': self.expected_counts,
                                                                 'rescale_mu': True,
                                                                 'ignore_events_check': True}
                                                         for sname, sclass in sources.items()},
                                              progress=False,
                                              batch_size=self.batch_size_rates,
                                              free_rates=tuple([sname for sname in sources.keys()]))

                # Pass constraint function to likelihood
                likelihood.set_log_constraint(self.log_constraint_fn)

                # Ensure we use the same toy dataset for every signal model in this toy
                if toy_data is None:
                    simulate_dict = dict()
                    constraint_extra_args = dict()
                    for background_source in self.background_source_names:
                        expected_background_counts = self.expected_counts[background_source]

                        # Sample constraint centers
                        if background_source in self.gaussian_constraint_widths:
                            draw = stats.norm.rvs(loc=expected_background_counts,
                                                  scale = self.gaussian_constraint_widths[background_source])
                            constraint_extra_args[f'{background_source}_expected_counts'] = tf.cast(draw, fd.float_type())

                        simulate_dict[f'{background_source}_rate_multiplier'] = expected_background_counts

                    simulate_dict[f'{signal_source}_rate_multiplier'] = 0.

                    # Simulate
                    toy_data = likelihood.simulate(**simulate_dict)

                # Shift the constraint in the likelihood based on the background RMs we drew
                likelihood.set_constraint_extra_args(**constraint_extra_args)

                # Set data
                likelihood.set_data(toy_data)

                guess_dict = dict()
                for background_source in self.background_source_names:
                    guess_dict[f'{background_source}_rate_multiplier'] = simulate_dict[f'{background_source}_rate_multiplier']
                guess_dict[f'{signal_source}_rate_multiplier'] = 0.

                for key, value in guess_dict.items():
                    if value < 0.1:
                        guess_dict[key] = 0.1

                # Save the test statistic values for each background-only dataset, for each
                # signal RM scanned over, for this signal source
                these_mus_test = mus_test[signal_source]
                # Loop over signal rate multipliers
                for mu_test in tqdm(these_mus_test, desc='Scanning over mus'):
                    if mu_test not in self.test_stats[signal_source].keys():
                        self.test_stats[signal_source][mu_test] = [self.test_statistic_qmu(mu_test, signal_source, likelihood, guess_dict)]
                    else:
                        self.test_stats[signal_source][mu_test].append(self.test_statistic_qmu(mu_test, signal_source, likelihood, guess_dict))

        # Output test statistics
        if test_stats_output_name is not None:
            pkl.dump(self.test_stats, open(test_stats_output_name, 'wb'))

    def get_limits(self, conf_level=0.1, return_p_vals=False):
        """
        """
        median_upper_lims = dict()
        # Loop over signal sources
        for signal_source in self.signal_source_names:
            these_test_stats = self.test_stats[signal_source]

            these_pvals = dict()
            for key, value in these_test_stats.items():
                these_pvals[key] = np.median(1. - stats.norm.cdf(np.sqrt(value)))

            self.median_p_vals[signal_source] = these_pvals

            mus = list(these_pvals.keys())
            pvals = list(these_pvals.values())

            # Find points where the p-value curve cross the critical value
            upper_lims = np.argwhere(np.diff(np.sign(pvals - np.ones_like(pvals) * conf_level)) < 0.).flatten()

            assert len(upper_lims) > 0, 'No upper limit found!'
            # Take the highest decreasing crossing point, and interpolate to get an upper limit
            upper_mu_left = mus[upper_lims[-1]]
            upper_mu_right = mus[upper_lims[-1] + 1]
            upper_pval_left = pvals[upper_lims[-1]]
            upper_pval_right = pvals[upper_lims[-1] + 1]

            upper_gradient = (upper_pval_right - upper_pval_left) / (upper_mu_right - upper_mu_left)
            upper_lim = (conf_level - upper_pval_left) / upper_gradient + upper_mu_left

            median_upper_lims[signal_source] = upper_lim

        if return_p_vals is True:
            # Return p-value curves, and upper limits
            return self.median_p_vals, median_upper_lims
        else:
            # Return upper limits
            return median_upper_lims
