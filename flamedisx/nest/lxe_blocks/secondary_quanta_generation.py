import numpy as np
from scipy import stats
import scipy.special as sp
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS2Photons(fd.Block):
    dimensions = ('electrons_detected', 's2_photons_produced')

    model_functions = ('electron_gain_mean', 'electron_gain_std')

    # MC_annotate = True
    #
    # MC_annotate_dimensions = ('electrons_detected',)

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
        out_mles = np.round(d['s2_photons_produced_min']).astype(int)
        means = self.gimme_numpy('electron_gain_mean') * np.ones(len(out_mles))
        stds = self.gimme_numpy('electron_gain_std') * np.ones(len(out_mles))
        xs = [np.linspace(np.floor(out_mle / mean * 0.9), np.ceil(out_mle / mean * 1.1), 1000).astype(int) for out_mle, mean in zip(out_mles, means)]

        mus = [x * mean for x, mean in zip(xs, means)]
        sigmas = [np.sqrt(x * std * std) for x, std in zip(xs, stds)]

        pdfs = [(1 / np.sqrt(sigma)) * np.exp(-0.5 * (out_mle - mu)**2 / sigma**2) for mu, sigma, out_mle, x in zip(mus, sigmas, out_mles, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        lower_lims = [x[np.where(cdf < 0.00135)[0][-1]] if len(np.where(cdf < 0.00135)[0]) > 0 else out_mle for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        out_mles = np.round(d['s2_photons_produced_max']).astype(int)
        means = self.gimme_numpy('electron_gain_mean') * np.ones(len(out_mles))
        stds = self.gimme_numpy('electron_gain_std') * np.ones(len(out_mles))
        xs = [np.linspace(np.floor(out_mle / mean * 0.9), np.ceil(out_mle / mean * 1.1), 1000).astype(int) for out_mle, mean in zip(out_mles, means)]

        mus = [x * mean for x, mean in zip(xs, means)]
        sigmas = [np.sqrt(x * std * std) for x, std in zip(xs, stds)]

        pdfs = [(1 / np.sqrt(sigma)) * np.exp(-0.5 * (out_mle - mu)**2 / sigma**2) for mu, sigma, out_mle, x in zip(mus, sigmas, out_mles, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        upper_lims = [x[np.where(cdf > (1. - 0.00135))[0][0]] if len(np.where(cdf > (1. - 0.00135))[0]) > 0 else np.ceil(out_mle / mean * 10).astype(int) for x, cdf, out_mle, mean in zip(xs, cdfs, out_mles, means)]

        d['electrons_detected_mle'] = d['s2_photons_produced_mle'] / means
        d['electrons_detected_min'] = lower_lims
        d['electrons_detected_max'] = upper_lims
