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
        # TODO: this assumes the spread from the double PE effect is subdominant
        dpe_fraction = self.gimme_numpy('double_pe_fraction')
        for suffix, intify in (('_min', np.floor),
                               ('_max', np.ceil),
                               ('_mle', lambda x: x)):
            d[self.quanta_in_name + suffix] = \
                intify(d[self.quanta_out_name + suffix].values
                       / (1 + dpe_fraction))


@export
class MakeS1Photoelectrons(MakePhotoelectrons):
    dimensions = ('photons_detected', 's1_photoelectrons_produced')

    quanta_in_name = 'photons_detected'
    quanta_out_name = 's1_photoelectrons_produced'

    MC_annotate = True

    MC_annotate_dimensions = ('photons_detected',)

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

    MC_annotate = True

    MC_annotate_dimensions = ('s2_photons_detected',)

    def _compute(self, data_tensor, ptensor,
                 s2_photons_detected, s2_photoelectrons_detected):
        return super()._compute(
            quanta_in=s2_photons_detected,
            quanta_out=s2_photoelectrons_detected,
            data_tensor=data_tensor, ptensor=ptensor)
