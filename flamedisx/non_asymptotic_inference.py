import flamedisx as fd
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
import typing as ty

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
            defaults=None,
            ntoys=1000):

        if arguments is None:
            arguments = dict()

        for key in sources.keys():
            if key not in arguments.keys():
                arguments[key] = dict()

        if defaults is None:
            defaults = dict()

        self.signal_source_names = signal_source_names
        self.background_source_names = background_source_names
        self.ntoys = ntoys
        self.batch_size = batch_size
        self.test_stat_dists = dict()
        self.observed_test_stats = dict()
        self.p_vals = dict()

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

    def get_test_stat_dists(self, mus_test=None):
        assert mus_test is not None, 'Must pass in mus to be scanned over'

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

            default_rm_bounds = {signal_source: (-5., 50.)}
            for source_name in self.background_source_names:
                default_rm_bounds[source_name] = (None, None)
            likelihood_fast.set_rate_multiplier_bounds(**default_rm_bounds)

            for mu_test in tqdm(mus_test, desc='Scanning over mus'):
                ts_dist = self.toy_test_statistic_dist(mu_test, signal_source, likelihood_fast)
                test_stat_dists[mu_test] = ts_dist

            self.test_stat_dists[signal_source] = test_stat_dists

    def toy_test_statistic_dist(self, mu_test, signal_source_name, likelihood_fast):
        rm_value_dict = {f'{signal_source_name}_rate_multiplier': mu_test}
        for source_name in self.background_source_names:
            rm_value_dict[f'{source_name}_rate_multiplier'] = 1.

        ts_values = []

        for toy in tqdm(range(self.ntoys), desc='Doing toys'):
            toy_data = likelihood_fast.simulate(**rm_value_dict)
            likelihood_fast.set_data(toy_data)

            ts_values.append(self.test_statistic_tmu_tilde(mu_test, signal_source_name, likelihood_fast))

        return ts_values

    def get_interval(self, mus_test=None, data=None, conf_level=0.1, return_p_vals=False):
        self.log_likelihood_full = fd.LogLikelihood(sources=sources,
                                                    progress=False,
                                                    batch_size=self.batch_size,
                                                    free_rates=tuple([sname for sname in sources.keys()]))

        self.log_likelihood_full.set_rate_multiplier_bounds(**default_rm_bounds)

        self.get_test_stat_dists(mus_test=mus_test)
        self.get_observed_test_stats(mus_test=mus_test, data=data)
        self.get_p_vals()

        mus = list(self.p_vals.keys())
        pvals = list(self.p_vals.values())

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

        if return_p_vals is True:
            return self.p_vals, lower_lim, upper_lim
        else:
            return lower_lim, upper_lim

    def get_observed_test_stats(self, mus_test=None, data=None):
        assert mus_test is not None, 'Must pass in mus to be scanned over'
        assert data is not None, 'Must pass in data'

        self.log_likelihood_full.set_data(data)

        self.observed_test_stats = dict()
        for mu_test in tqdm(mus_test, desc='Scanning over mus'):
            self.observed_test_stats[mu_test] = self.test_statistic_tmu_tilde(mu_test, observed=True)

    def get_p_vals(self):
        assert len(self.test_stat_dists) > 0, 'Must generate test statistic distributions first'
        assert len(self.observed_test_stats) > 0, 'Must calculate observed test statistics first'
        assert self.test_stat_dists.keys() == self.observed_test_stats.keys(), \
            'Must get test statistic distributions and observed test statistics with the same mu values'

        self.p_vals = dict()
        for mu_test in self.observed_test_stats.keys():
            self.p_vals[mu_test] = (100. - stats.percentileofscore(self.test_stat_dists[mu_test],
                                                                  self.observed_test_stats[mu_test])) / 100.
