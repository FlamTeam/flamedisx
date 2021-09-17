import numpy as np
from scipy import stats
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
        for suffix, bound in (('_min', 'lower'),
                               ('_max', 'upper'),
                               ('_mle', 'mle')):
            out_bounds = d['photoelectrons_detected' + suffix]
            supports = [np.linspace(np.ceil(out_bound / 2.), out_bound + 1., 1000).astype(int) for out_bound in out_bounds]
            ns = supports
            ps = [p * np.ones_like(support) for p, support in zip(self.gimme_numpy('double_pe_fraction'), supports)]
            rvs = [out_bound - support for out_bound, support in zip(out_bounds, supports)]

            self.bayes_bounds_binomial(d, 'photons_detected', supports=supports,
                                       rvs_binom=rvs, ns_binom=ns, ps_binom=ps, bound=bound)
