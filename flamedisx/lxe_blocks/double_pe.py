import numpy as np
from scipy import stats
import scipy.special as sp
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS1Photoelectrons(fd.Block):
    dimensions = ('photons_detected', 'photoelectrons_detected')

    model_functions = ('double_pe_fraction',)

    double_pe_fraction = 0.219

    def _compute(self, data_tensor, ptensor,
                 photons_detected, photoelectrons_detected):
        p_dpe = self.gimme('double_pe_fraction',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Double-pe emission only creates additional photoelectrons.
        # Invalid values will get assigned p=0 later.
        extra_pe = photoelectrons_detected - photons_detected
        invalid = extra_pe < 0

        # Negative arguments would mess up tfp's Binomial
        extra_pe = tf.where(invalid,
                            tf.zeros_like(extra_pe),
                            extra_pe)

        # (N_pe - N_photons) distributed as Binom(N_photons, p=pdpe)
        result = tfp.distributions.Binomial(
                total_count=photons_detected,
                probs=tf.cast(p_dpe, dtype=fd.float_type())
            ).prob(extra_pe)

        # Set probability of extra_pe < 0 cases to 0
        return tf.where(invalid,
                        tf.zeros_like(photoelectrons_detected),
                        result)

    def _simulate(self, d):
        d['photoelectrons_detected'] = stats.binom.rvs(
            n=d['photons_detected'],
            p=self.gimme_numpy('double_pe_fraction')) + d['photons_detected']

    def _annotate(self, d):
        out_mles = np.round(d['photoelectrons_detected_mle']).astype(int)
        xs = [np.arange(np.ceil(out_mle / 2.), out_mle + 1.).astype(int) for out_mle in out_mles]
        ps = self.gimme_numpy('double_pe_fraction')

        pdfs = [sp.binom(x, out_mle - x) * pow(p, out_mle - x) * pow(1. - p, 2. * x - out_mle) for out_mle, x, p in zip(out_mles, xs, ps)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        lower_lims = [x[np.where(cdf < 0.00135)[0][-1]] if len(np.where(cdf < 0.00135)[0]) > 0 else np.ceil(out_mle / 2.).astype(int) for x, cdf, out_mle in zip(xs, cdfs, out_mles)]
        upper_lims = [x[np.where(cdf > (1. - 0.00135))[0][0]] if len(np.where(cdf > (1. - 0.00135))[0]) > 0 else out_mle for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        d['photons_detected_mle'] = d['photoelectrons_detected_mle'].values / (1 + ps)
        d['photons_detected_min'] = lower_lims
        d['photons_detected_max'] = upper_lims
