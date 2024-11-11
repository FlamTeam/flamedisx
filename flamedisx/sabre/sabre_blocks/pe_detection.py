import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class DetectPhotoelectrons(fd.Block):

    dimensions = ('photoelectrons_detected', 'photons_produced')

    special_model_functions = ('pe_detection_eff',)
    model_functions = special_model_functions

    special_model_functions = ()
    model_functions = ('pe_detection_eff',) + special_model_functions

    pe_detection_eff = 0.25

    def _compute(self, data_tensor, ptensor,
                 photons_produced, photoelectrons_detected):
        p = self.gimme('pe_detection_eff',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        
        return tfp.distributions.Binomial(
                total_count=photons_produced,
                probs=tf.cast(p, dtype=fd.float_type())
            ).prob(photoelectrons_detected)

    def _simulate(self, d):
        p = self.gimme_numpy('pe_detection_eff')

        d['photoelectrons_detected'] = stats.binom.rvs(
            n=d['photons_produced'],
            p=p)

    def _annotate(self, d):
        eff = self.gimme_numpy('pe_detection_eff')

        # Estimate produced quanta
        n_prod_mle = d['photons_produced_mle'] = \
            d['photoelectrons_detected_mle'] / eff

        # Estimating the spread in number of produced quanta is tricky since
        # the number of detected quanta is itself uncertain.
        # TODO: where did this derivation come from again?
        q = (1 - eff) / eff
        _std = (q + (q ** 2 + 4 * n_prod_mle * q) ** 0.5) / 2

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            d['photons_produced_' + bound] = intify(
                n_prod_mle + sign * self.source.max_sigma * _std
            ).clip(0, None).astype(int)
