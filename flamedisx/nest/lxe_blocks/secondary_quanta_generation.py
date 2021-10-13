import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS2Photons(fd.Block):
    dimensions = ('electrons_detected', 's2_photons_produced')

    model_functions = ('electron_gain_mean', 'electron_gain_std')

    def _compute(self, data_tensor, ptensor,
                 electrons_detected, s2_photons_produced):
        mean_per_q = self.gimme('electron_gain_mean',
                                data_tensor=data_tensor,
                                ptensor=ptensor)[:, o, o]
        std_per_q = self.gimme('electron_gain_std',
                               data_tensor=data_tensor,
                               ptensor=ptensor)[:, o, o]

        mean = electrons_detected * mean_per_q
        std = electrons_detected ** 0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        # Don't forget continuity correction!
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).cdf(s2_photons_produced + 0.5) - \
            tfp.distributions.Normal(loc=mean, scale=std + 1e-10).cdf(
                s2_photons_produced - 0.5)

        return result

    def _simulate(self, d):
        d['s2_photons_produced'] = np.round(stats.norm.rvs(
            loc=(d['electrons_detected']
                 * self.gimme_numpy('electron_gain_mean')),
            scale=(d['electrons_detected']**0.5
                   * self.gimme_numpy('electron_gain_std')))).astype(int)

    def _annotate(self, d):
        for suffix, bound in (('_min', 'lower'),
                              ('_max', 'upper')):
            out_bounds = d['s2_photons_produced' + suffix]
            supports = [np.linspace(np.floor(out_bound / self.gimme_numpy('electron_gain_mean')[0] * 0.9),
                        np.ceil(out_bound / self.gimme_numpy('electron_gain_mean')[0] * 1.1), 1000).astype(int)
                        for out_bound in out_bounds]
            mus = supports * self.gimme_numpy('electron_gain_mean')
            sigmas = np.sqrt(supports * self.gimme_numpy('electron_gain_std')**2)
            rvs = [out_bound * np.ones_like(support)
                   for out_bound, support in zip(out_bounds, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim='electrons_detected',
                                   bounds_prob=self.source.bounds_prob, bound=bound,
                                   bound_type='normal', supports=supports,
                                   rvs_normal=rvs, mus_normal=mus, sigmas_normal=sigmas)
