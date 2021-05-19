import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis

import configparser
import os


@export
class DetectS1Photoelectrons(fd.Block):
    dimensions = ('s1_photoelectrons_produced', 's1_photoelectrons_detected')
    extra_dimensions = ()

    model_functions = ('photoelectron_detection_eff',)

    def _compute(self, data_tensor, ptensor,
                 s1_photoelectrons_produced, s1_photoelectrons_detected):
        p_det = self.gimme('photoelectron_detection_eff',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        result = tfp.distributions.Binomial(
                total_count=s1_photoelectrons_produced,
                probs=tf.cast(p_det, dtype=fd.float_type())
            ).prob(s1_photoelectrons_detected)

        return result

    def _simulate(self, d):
        d['s1_photoelectrons_detected'] = stats.binom.rvs(
            n=d['s1_photoelectrons_produced'],
            p=self.gimme_numpy('photoelectron_detection_eff'))

    def _annotate(self, d):
        # TODO: this assumes the spread from the PE detection efficiency is subdominant
        p_det = self.gimme_numpy('photoelectron_detection_eff')
        for suffix, intify in (('min', np.floor),
                               ('max', np.ceil),
                               ('mle', np.round)):
            d['s1_photoelectrons_produced_' + suffix] = \
                intify(d['s1_photoelectrons_detected_' + suffix].values
                       / p_det)
