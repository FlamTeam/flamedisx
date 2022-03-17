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
        for suffix, bound in (('_min', 'lower'),
                              ('_max', 'upper')):
            out_bounds = d['s1_photoelectrons_detected' + suffix]
            supports = [np.linspace(out_bound, out_bound * 2., 1000).astype(int)
                        for out_bound in out_bounds]
            ns = supports
            ps = [self.gimme_numpy('photoelectron_detection_eff', support) for support in supports]
            rvs = [out_bound * np.ones_like(support)
                   for out_bound, support in zip(out_bounds, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim='s1_photoelectrons_produced',
                                   bounds_prob=self.source.bounds_prob, bound=bound,
                                   bound_type='binomial', supports=supports,
                                   rvs_binom=rvs, ns_binom=ns, ps_binom=ps)
