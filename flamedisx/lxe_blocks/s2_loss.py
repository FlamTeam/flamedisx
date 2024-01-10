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

    s2_survival_p = 1

    def _compute(self, data_tensor, ptensor,
                 s2_raw, s2_raw_after_loss):
        s2_survival_probability = self.gimme('s2_survival_p',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # s2_raw_after_loss distributed as Binom(s2_raw, p=s2_survival_probability)
        s2_raw_after_loss = tf.clip_by_value(s2_raw_after_loss, 1e-15, tf.float32.max)
        result = tfp.distributions.Binomial(
                total_count=tf.clip_by_value(tf.cast(s2_raw, dtype=fd.int_type()),1e-15, tf.int32.max),
                probs=tf.clip_by_value(tf.cast(s2_survival_probability, dtype=fd.float_type()), 0.,1.)
            ).prob(s2_raw_after_loss)

        return result

    def _simulate(self, d):
        d['s2_raw_after_loss'] = stats.binom.rvs(
            n=np.clip(d['s2_raw'].astype(dtype=np.int32),0,np.iinfo(np.int32).max),
            p=np.nan_to_num(self.gimme_numpy('s2_survival_p')).clip(0., 1.))

    def _annotate(self, d):
        # TODO: copied from double PE effect
        s2_survival_probability = self.gimme_numpy('s2_survival_p')
        for suffix, intify in (('min', np.floor),
                               ('max', np.ceil),
                               ('mle', lambda x: x)):
            d['s2_raw_' + suffix] = \
                intify(d['s2_raw_after_loss_' + suffix].values
                       / (1 + s2_survival_probability))
