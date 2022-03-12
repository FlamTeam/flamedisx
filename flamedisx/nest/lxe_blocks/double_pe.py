import numpy as np
from scipy import stats
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
        for suffix, bound in (('_min', 'lower'),
                              ('_max', 'upper')):
            out_bounds = d[self.quanta_out_name + suffix]
            supports = [np.linspace(np.ceil(out_bound / 2.), out_bound + 1., 1000).astype(int)
                        for out_bound in out_bounds]
            ns = supports
            ps = [p * np.ones_like(support) for p, support in zip(self.gimme_numpy('double_pe_fraction'), supports)]
            rvs = [out_bound - support for out_bound, support in zip(out_bounds, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim=self.quanta_in_name,
                                   bounds_prob=self.source.bounds_prob, bound=bound,
                                   bound_type='binomial', supports=supports,
                                   rvs_binom=rvs, ns_binom=ns, ps_binom=ps)


@export
class MakeS1Photoelectrons(MakePhotoelectrons):
    dimensions = ('photons_detected', 's1_photoelectrons_produced')

    quanta_in_name = 'photons_detected'
    quanta_out_name = 's1_photoelectrons_produced'

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

    def _compute(self, data_tensor, ptensor,
                 s2_photons_detected, s2_photoelectrons_detected):
        return super()._compute(
            quanta_in=s2_photons_detected,
            quanta_out=s2_photoelectrons_detected,
            data_tensor=data_tensor, ptensor=ptensor)
