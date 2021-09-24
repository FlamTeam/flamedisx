import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class DetectS1Photoelectrons(fd.Block):
    dimensions = ('s1_photoelectrons_produced', 's1_photoelectrons_detected')
    extra_dimensions = ()

    special_model_functions = ('photoelectron_detection_eff',)
    model_functions = special_model_functions

    def _compute(self, data_tensor, ptensor,
                 s1_photoelectrons_produced, s1_photoelectrons_detected):
        p_det = self.gimme('photoelectron_detection_eff',
                           bonus_arg=s1_photoelectrons_produced,
                           data_tensor=data_tensor, ptensor=ptensor)

        result = tfp.distributions.Binomial(
                total_count=s1_photoelectrons_produced,
                probs=tf.cast(p_det, dtype=fd.float_type())
            ).prob(s1_photoelectrons_detected)

        result = tf.where(tf.math.is_nan(result),
                          tf.zeros_like(result, dtype=fd.float_type()),
                          result)

        return result

    def _simulate(self, d):
        d['s1_photoelectrons_detected'] = stats.binom.rvs(
            n=d['s1_photoelectrons_produced'],
            p=self.gimme_numpy(
                'photoelectron_detection_eff',
                d['s1_photoelectrons_produced']))

    def _annotate(self, d):
        # Estimate the mle of the detection probability via interpolation
        # _nprod_temp = np.logspace(-1., 8., 1000)
        # _pdet_temp = self.gimme_numpy(
        #     'photoelectron_detection_eff',
        #     _nprod_temp)
        # p_det_mle = np.interp(
        #     d['s1_photoelectrons_detected_mle'],
        #     _nprod_temp * _pdet_temp,
        #     _pdet_temp)
        # TODO: this assumes the spread from the PE detection efficiency is subdominant
        # TODO: come back and fix thing with p_det_mle
        for suffix, intify in (('min', np.floor),
                               ('max', np.ceil),
                               ('mle', np.round)):
            d['s1_photoelectrons_produced_' + suffix] = \
                intify(d['s1_photoelectrons_detected_' + suffix].values)
