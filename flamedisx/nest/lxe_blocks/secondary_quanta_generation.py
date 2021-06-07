import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS2Photons(fd.Block):
    dimensions = ('electrons_detected', 's2_photons_produced')
    extra_dimensions = ()

    model_functions = ('electron_gain_mean', 'electron_gain_std')

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
        d['s2_photons_produced'] = tf.cast(tf.math.round(stats.norm.rvs(
            loc=(d['electrons_detected']
                 * self.gimme_numpy('electron_gain_mean')),
            scale=(d['electrons_detected']**0.5
                   * self.gimme_numpy('electron_gain_std')))), dtype=fd.int_type())

    def _annotate(self, d):
        m = self.gimme_numpy('electron_gain_mean')
        s = self.gimme_numpy('electron_gain_std')

        mle = d['electrons_detected_mle'] = \
            (d['s2_photons_produced_mle'] / m).clip(0, None)

        scale = mle**0.5 * s / m

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            d['electrons_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)
