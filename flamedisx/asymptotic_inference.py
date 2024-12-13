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

        # Return the asymptotic p-value
        return self.evaluate_asymptotic_pval(bf_unconditional, bf_conditional,
                                                 mu_test)


@export
class TestStatisticTMu(TestStatistic):
    """Evaluate the test statistic of equation 11 in https://arxiv.org/abs/1007.1727.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_asymptotic_pval(self, bf_unconditional, bf_conditional, mu_test):
        ll_conditional = self.likelihood(**bf_conditional)
        ll_unconditional = self.likelihood(**bf_unconditional)

        ts = max([-2. * (ll_conditional - ll_unconditional), 0.])

        F = 2. * stats.norm.cdf(np.sqrt(ts)) - 1.

        pval = 1. - F
        return pval
    

@export
class pValDistributions():
    """ Class to store p-values (pass in as a list),
    for a range of values of the parameter of interest being tested ('mu').
    """
    def __init__(self):
        self.pval_dists = dict()

    def add_pval_dist(self, mu_test, pvals):
        self.pval_dists[mu_test] = np.array(pvals)


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
        - likelihood_container: BLAH
        - ntoys: number of toys that will be run to get test statistic distributions
    """
    def __init__(
            self,
            test_statistic: TestStatistic.__class__,
            signal_source_names: ty.Tuple[str],
            background_source_names: ty.Tuple[str],
            expected_background_counts: ty.Dict[str, float] = None,
            gaussian_constraint_widths: ty.Dict[str, float] = None,
            likelihood_container=None,
            ntoys=1000):

        if gaussian_constraint_widths is None:
            gaussian_constraint_widths = dict()

        self.ntoys = ntoys

        self.likelihood_container = likelihood_container
        self.test_statistic = test_statistic

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names

        self.expected_background_counts = expected_background_counts
        self.gaussian_constraint_widths = gaussian_constraint_widths

    def run_routine(self, mus_test=None,
                    generate_B_toys=False,
                    simulate_dict_B=None, toy_data_B=None, constraint_extra_args_B=None,
                    toy_batch=0,
                    mode='sensitivity'):
        """BLAH

        Arguments:
            - mus_test: dictionary {sourcename: np.array([mu1, mu2, ...])} of signal rate
                multipliers to be tested for each signal source
            - generate_B_toys: if true, the routine run will be a generation of background-only
                datasets
            - simulate_dict_B: first return argument of the result of calling this function with
                generate_B_toys=True)
            - toy_data_B: second return argument of the result of calling this function with
                generate_B_toys=True)
            - constraint_extra_args_B: third return argument of the result of calling this function with
                generate_B_toys=True)
            - toy_batch: if parallelising toys, this should correspond to the parallel batch index
                (starting at 0) being run, to ensure the correct background-only toys are accessed
            - mode: BLAH
        """
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

        pval_dists_collection = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            pval_dists= pValDistributions()

            sources = dict()
            arguments = dict()
            for sname, source in self.likelihood_container.sources.items():
                if (sname == signal_source) or (sname in self.background_source_names):
                    sources[sname] = source
                    arguments[sname] = self.likelihood_container.arguments[sname]

            likelihood = fd.LogLikelihood(sources=sources,
                                          arguments=arguments,
                                          batch_size=self.likelihood_container.batch_size,
                                          free_rates=tuple(sources.keys()),
                                          log_constraint=self.likelihood_container.log_constraint)

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
                self.toy_test_statistic_dist(pval_dists,
                                             mu_test, signal_source, likelihood,
                                             mode=mode)

            pval_dists_collection[signal_source] = pval_dists

        return pval_dists_collection

    def sample_data_constraints(self, mu_test, signal_source_name, likelihood):
        """Internal function to sample the toy data and constraint central values
        following a frequentist procedure. Method taken depends on whether conditional
        best fits were passed.
        """
        simulate_dict = dict()
        constraint_extra_args = dict()
        for background_source in self.background_source_names:
            expected_background_counts = self.expected_background_counts[background_source]

            # Sample constraint centers
            if background_source in self.gaussian_constraint_widths:
                draw = stats.norm.rvs(loc=expected_background_counts,
                                      scale=self.gaussian_constraint_widths[background_source])
                constraint_extra_args[f'{background_source}_expected_counts'] = tf.cast(draw, fd.float_type())

            simulate_dict[f'{background_source}_rate_multiplier'] = tf.cast(expected_background_counts, fd.float_type())
            simulate_dict[f'{signal_source_name}_rate_multiplier'] = tf.cast(mu_test, fd.float_type())

        toy_data = likelihood.simulate(**simulate_dict)

        return simulate_dict, toy_data, constraint_extra_args

    def toy_test_statistic_dist(self, pval_dist,
                                mu_test, signal_source_name, likelihood,
                                mode='sensitivity'):
        """Internal function to get test statistic distribution.
        """
        pvals = []

        # Loop over toys
        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            if mode == 'discovery':
                # S+B toys

                simulate_dict_SB, toy_data_SB, constraint_extra_args_SB = \
                    self.sample_data_constraints(mu_test, signal_source_name, likelihood)
                # Guesses for fit
                guess_dict_SB = simulate_dict_SB.copy()
                for key, value in guess_dict_SB.items():
                    if value < 0.1:
                        guess_dict_SB[key] = 0.1

                # Shift the constraint in the likelihood based on the background RMs we drew
                likelihood.set_constraint_extra_args(**constraint_extra_args_SB)
                # Set data
                likelihood.set_data(toy_data_SB)
                # Create test statistic
                test_statistic_SB = self.test_statistic(likelihood)

                # Evaluate discovery p-value
                pval_disco_SB = test_statistic_SB(0., signal_source_name, guess_dict_SB)
                pvals.append(pval_disco_SB)

            elif mode == 'sensitivity':
                # B-only toys

                try:
                    toy_data_B = self.toy_data_B[toy+(self.toy_batch*self.ntoys)]
                    constraint_extra_args_B = self.constraint_extra_args_B[toy+(self.toy_batch*self.ntoys)]
                    # Guesses for fit
                    guess_dict_B = self.simulate_dict_B.copy()
                    guess_dict_B[f'{signal_source_name}_rate_multiplier'] = 0.
                    for key, value in guess_dict_B.items():
                        if value < 0.1:
                            guess_dict_B[key] = 0.1
                except Exception:
                    raise RuntimeError("Could not find background-only datasets")

                # Shift the constraint in the likelihood based on the background RMs we drew
                likelihood.set_constraint_extra_args(**constraint_extra_args_B)
                # Set data
                likelihood.set_data(toy_data_B)
                # Create test statistic
                test_statistic_B = self.test_statistic(likelihood)

                # Evaluate p-value
                pval_B = test_statistic_B(mu_test, signal_source_name, guess_dict_B)
                pvals.append(pval_B)

        pval_dist.add_pval_dist(mu_test, pvals)
        return


@export
class IntervalCalculator():
    """BLAH

    Arguments:
        - signal_source_names: tuple of names for signal sources (e.g. WIMPs of different
            masses)
        - pval_dists: dictionary {sourcename: pValDistributions} returned
            by running TSEvaluation routine to get p-values under either the S+B or
            the B hypothesis
    """
    def __init__(
            self,
            signal_source_names: ty.Tuple[str],
            pval_dists: pValDistributions):

        self.signal_source_names = signal_source_names
        self.pval_dists = pval_dists

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

    def upper_lims_bands(self, pval_curve, mus, conf_level):
        try:
            upper_lims = np.argwhere(np.diff(np.sign(pval_curve - np.ones_like(pval_curve) * conf_level)) < 0.).flatten()
            return self.interp_helper(mus, pval_curve, upper_lims, conf_level,
                                    rising_edge=False, inverse=True)
        except Exception:
            return 0.

    def get_bands_sensitivity(self, conf_level=0.1,
                              quantiles=[0, 1, -1, 2, -2]):
        """
        """
        bands = dict()
        all_mus = dict()
        all_p_val_curves = dict()

        # Loop over signal sources
        for signal_source in self.signal_source_names:
            # Get p-value distribitions
            pval_dists = self.pval_dists[signal_source]

            mus = []
            p_val_curves = []
            # Loop over signal rate multipliers
            for mu_test, these_p_vals in pval_dists.pval_dists.items():
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

    # def get_bands_discovery(self, quantiles=[0, 1, -1, 2, -2]):
    #     """
    #     """
    #     bands = dict()
    #     # Loop over signal sources
    #     for signal_source in self.signal_source_names:
    #         # Get test statistic distribitions
    #         test_stat_dists_SB_disco = self.test_stat_dists_SB_disco[signal_source]
    #         assert len(test_stat_dists_SB_disco.ts_dists.keys()) == 1, 'Currently only support a single signal strength'
    #         these_disco_sigs = np.sqrt(list(test_stat_dists_SB_disco.ts_dists.values())[0])
    #         these_bands = dict()
    #         for quantile in quantiles:
    #             these_bands[quantile] = np.quantile(np.sort(these_disco_sigs), stats.norm.cdf(quantile))
    #         bands[signal_source] = these_bands
    #     return bands
