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
        invalid = s2_raw_after_loss > s2_raw
        # s2_raw_after_loss distributed as Binom(s2_raw, p=s2_survival_probability)
        result = tfp.distributions.Binomial(
                total_count=s2_raw,
                probs=tf.clip_by_value(s2_survival_probability, 0.,1.)
            ).prob(s2_raw_after_loss)

        return tf.where(invalid,
                        tf.zeros_like(s2_raw),
                        result)

    def _simulate(self, d):
        d['s2_raw_after_loss'] = tfp.distributions.Binomial(
                total_count=d['s2_raw'],
                probs=tf.clip_by_value(self.gimme_numpy('s2_survival_p'), 0.,1.)#.astype(np.float64)
            ).sample()
        
    def _annotate(self, d):
        # TODO: copied from double PE effect
        s2_survival_probability = self.gimme_numpy('s2_survival_p')

        mle = d['s2_raw' + '_mle'] = \
            (d['s2_raw_after_loss' + '_mle'] / s2_survival_probability).clip(0, None)
        scale = mle*s2_survival_probability*(1-s2_survival_probability)
        
        d['s2_raw'  + '_min'] = np.floor(mle-self.source.max_sigma*scale).clip(0, None)
        d['s2_raw'  + '_max'] = np.ceil(mle+self.source.max_sigma*scale).clip(0, None)
