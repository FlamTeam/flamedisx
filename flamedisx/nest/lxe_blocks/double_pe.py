import numpy as np
from scipy import stats
import scipy.special as sp
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


class MakePhotoelectrons(fd.Block):
    model_functions = ('double_pe_fraction',)

    quanta_in_name: str
    quanta_out_name: str

    def _compute(self, data_tensor, ptensor,
                 quanta_in, quanta_out):
        p_dpe = self.gimme('double_pe_fraction',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Double-pe emission only creates additional photoelectrons.
        # Invalid values will get assigned p=0 later.
        extra_pe = quanta_out - quanta_in
        invalid = extra_pe < 0

        # Negative arguments would mess up tfp's Binomial
        extra_pe = tf.where(invalid,
                            tf.zeros_like(extra_pe),
                            extra_pe)

        # (N_pe - N_photons) distributed as Binom(N_photons, p=pdpe)
        result = tfp.distributions.Binomial(
                total_count=quanta_in,
                probs=tf.cast(p_dpe, dtype=fd.float_type())
            ).prob(extra_pe)

        # Set probability of extra_pe < 0 cases to 0
        return tf.where(invalid,
                        tf.zeros_like(quanta_out),
                        result)

    def _simulate(self, d):
        d[self.quanta_out_name] = stats.binom.rvs(
            n=d[self.quanta_in_name],
            p=self.gimme_numpy('double_pe_fraction')) + d[self.quanta_in_name]

    def _annotate(self, d):
        out_mles = np.round(d[self.quanta_out_name + '_min']).astype(int)
        ps = self.gimme_numpy('double_pe_fraction')
        xs = [np.arange(np.ceil(out_mle / 2.), out_mle + 1.).astype(int) for out_mle in out_mles]

        pdfs = [sp.binom(x, out_mle - x) * pow(p, out_mle - x) * pow(1. - p, 2. * x - out_mle) for out_mle, p, x in zip(out_mles, ps, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        lower_lims = [x[np.where(cdf < 0.00135)[0][-1]] if len(np.where(cdf < 0.00135)[0]) > 0 else np.ceil(out_mle / 2.).astype(int) for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        out_mles = np.round(d[self.quanta_out_name + '_max']).astype(int)
        ps = self.gimme_numpy('double_pe_fraction')
        xs = [np.arange(np.ceil(out_mle / 2.), out_mle + 1.).astype(int) for out_mle in out_mles]

        pdfs = [sp.binom(x, out_mle - x) * pow(p, out_mle - x) * pow(1. - p, 2. * x - out_mle) for out_mle, p, x in zip(out_mles, ps, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        upper_lims = [x[np.where(cdf > (1. - 0.00135))[0][0]] if len(np.where(cdf > (1. - 0.00135))[0]) > 0 else out_mle for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        d[self.quanta_in_name + '_mle'] = d[self.quanta_out_name + '_mle'].values / (1 + ps)
        d[self.quanta_in_name + '_min'] = lower_lims
        d[self.quanta_in_name + '_max'] = upper_lims


@export
class MakeS1Photoelectrons(MakePhotoelectrons):
    dimensions = ('photons_detected', 's1_photoelectrons_produced')

    quanta_in_name = 'photons_detected'
    quanta_out_name = 's1_photoelectrons_produced'

    # MC_annotate = True
    #
    # MC_annotate_dimensions = ('photons_detected',)

    def _compute(self, data_tensor, ptensor,
                 photons_detected, s1_photoelectrons_produced):
        return super()._compute(
            quanta_in=photons_detected,
            quanta_out=s1_photoelectrons_produced,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2Photoelectrons(MakePhotoelectrons):
    dimensions = ('s2_photons_detected', 's2_photoelectrons_detected')

    quanta_in_name = 's2_photons_detected'
    quanta_out_name = 's2_photoelectrons_detected'

    # MC_annotate = True
    #
    # MC_annotate_dimensions = ('s2_photons_detected',)

    def _compute(self, data_tensor, ptensor,
                 s2_photons_detected, s2_photoelectrons_detected):
        return super()._compute(
            quanta_in=s2_photons_detected,
            quanta_out=s2_photoelectrons_detected,
            data_tensor=data_tensor, ptensor=ptensor)

    def _annotate(self, d):
        out_mles = np.round(d[self.quanta_out_name + '_min']).astype(int)
        ps = self.gimme_numpy('double_pe_fraction')
        xs = [np.arange(np.ceil(out_mle / 2.), out_mle + 1.).astype(int) for out_mle in out_mles]

        mus = [x * p for x, p in zip(xs, ps)]
        sigmas = [np.sqrt(x * p * (1 - p)) for x, p in zip(xs, ps)]

        pdfs = [(1 / np.sqrt(sigma)) * np.exp(-0.5 * (out_mle - x - mu)**2 / sigma**2) for mu, sigma, out_mle, x in zip(mus, sigmas, out_mles, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        lower_lims = [x[np.where(cdf < 0.00135)[0][-1]] if len(np.where(cdf < 0.00135)[0]) > 0 else np.ceil(out_mle / 2.).astype(int) for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        out_mles = np.round(d[self.quanta_out_name + '_max']).astype(int)
        ps = self.gimme_numpy('double_pe_fraction')
        xs = [np.arange(np.ceil(out_mle / 2.), out_mle + 1.).astype(int) for out_mle in out_mles]

        mus = [x * p for x, p in zip(xs, ps)]
        sigmas = [np.sqrt(x * p * (1 - p)) for x, p in zip(xs, ps)]

        pdfs = [(1 / np.sqrt(sigma)) * np.exp(-0.5 * (out_mle - x - mu)**2 / sigma**2) for mu, sigma, out_mle, x in zip(mus, sigmas, out_mles, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        upper_lims = [x[np.where(cdf > (1. - 0.00135))[0][0]] if len(np.where(cdf > (1. - 0.00135))[0]) > 0 else out_mle for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        d[self.quanta_in_name + '_mle'] = d[self.quanta_out_name + '_mle'].values / (1 + ps)
        d[self.quanta_in_name + '_min'] = lower_lims
        d[self.quanta_in_name + '_max'] = upper_lims
