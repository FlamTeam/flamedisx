import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS2AfterLoss(fd.Block):
    dimensions = ('s2_raw', 's2_raw_after_loss')

    model_functions = ('s2_survival_p',)

    max_dim_size = {'s2_raw': 240}

    s2_survival_p = 1.

    def _compute(self, data_tensor, ptensor,
                 s2_raw, s2_raw_after_loss):
        s2_survival_probability = self.gimme('s2_survival_p',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # s2_raw_after_loss distributed as Binom(s2_raw, p=s2_survival_probability)
        s2_raw_after_loss = tf.clip_by_value(s2_raw_after_loss, 1e-15, tf.float32.max)
        s2_raw = tf.clip_by_value(tf.cast(s2_raw, dtype=fd.int_type()), 0, tf.int32.max)
        s2_survival_probability = tf.clip_by_value(tf.cast(s2_survival_probability, dtype=fd.float_type()), 0.,1.)
        result = tfp.distributions.Binomial(
                total_count=tf.cast(s2_raw, dtype=fd.float_type()),
                probs=s2_survival_probability
            ).prob(s2_raw_after_loss)

        return result

    def _simulate(self, d):
        d['s2_raw_after_loss'] = stats.binom.rvs(
            n=np.clip(d['s2_raw'].astype(dtype=np.int32),0,np.iinfo(np.int32).max),
            p=np.nan_to_num(self.gimme_numpy('s2_survival_p')).clip(0., 1.))

    def _annotate(self, d):
        # TODO: copied from double PE effect
        s2_survival_probability = self.gimme_numpy('s2_survival_p')

        mle = d['s2_raw' + '_mle'] = \
            (d['s2_raw_after_loss_' + '_mle'] / s2_survival_probability).clip(0, None)
        s = d['s2_raw'] * s2_survival_probability*(1-s2_survival_probability)
        scale = mle**0.5 * s / s2_survival_probability

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d['s2_raw'  + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(int)
