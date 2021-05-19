import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


class MakePhotoelectrons(fd.Block):
    model_functions = ('double_pe_fraction',)

    in_quanta_name: str
    out_quanta_name: str

    def _compute(self, data_tensor, ptensor,
                 quanta_detected, quanta_produced):
        p_dpe = self.gimme('double_pe_fraction',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Double-pe emission only creates additional photoelectrons.
        # Invalid values will get assigned p=0 later.
        extra_pe = quanta_produced - quanta_detected
        invalid = extra_pe < 0

        # Negative arguments would mess up tfp's Binomial
        extra_pe = tf.where(invalid,
                            tf.zeros_like(extra_pe),
                            extra_pe)

        # (N_pe - N_photons) distributed as Binom(N_photons, p=pdpe)
        result = tfp.distributions.Binomial(
                total_count=quanta_detected,
                probs=tf.cast(p_dpe, dtype=fd.float_type())
            ).prob(extra_pe)

        # Set probability of extra_pe < 0 cases to 0
        return tf.where(invalid,
                        tf.zeros_like(quanta_produced),
                        result)

    def _simulate(self, d):
        d[self.out_quanta_name + '_produced'] = stats.binom.rvs(
            n=d[self.in_quanta_name + '_detected'],
            p=self.gimme_numpy('double_pe_fraction')) + \
            d[self.in_quanta_name + '_detected']

    def _annotate(self, d):
        # TODO: this assumes the spread from the double PE effect is subdominant
        dpe_fraction = self.gimme_numpy('double_pe_fraction')
        for suffix, intify in (('min', np.floor),
                               ('max', np.ceil),
                               ('mle', lambda x: x)):
            d[self.in_quanta_name + '_detected_' + suffix] = \
                intify(d[self.out_quanta_name + '_produced_' + suffix].values
                       / (1 + dpe_fraction))


@export
class MakeS1Photoelectrons(MakePhotoelectrons):
    dimensions = ('photons_detected', 's1_photoelectrons_produced')
    extra_dimensions = ()

    in_quanta_name = 'photons'
    out_quanta_name = 's1_photoelectrons'

    def _compute(self, data_tensor, ptensor,
                 photons_detected, s1_photoelectrons_produced):
        return super()._compute(
            quanta_detected=photons_detected,
            quanta_produced=s1_photoelectrons_produced,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2Photoelectrons(MakePhotoelectrons):
    dimensions = ('s2_photons_detected', 's2_photoelectrons_produced')
    extra_dimensions = ()

    in_quanta_name = 's2_photons'
    out_quanta_name = 's2_photoelectrons'

    def _compute(self, data_tensor, ptensor,
                 s2_photons_detected, s2_photoelectrons_produced):
        return super()._compute(
            quanta_detected=s2_photons_detected,
            quanta_produced=s2_photoelectrons_produced,
            data_tensor=data_tensor, ptensor=ptensor)
