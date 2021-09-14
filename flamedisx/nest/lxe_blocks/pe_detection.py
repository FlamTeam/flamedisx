import numpy as np
from scipy import stats
import scipy.special as sp
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
        out_mles = np.round(d['s1_photoelectrons_detected_min']).astype(int)
        xs = [np.linspace(out_mle, out_mle * 2, 1000).astype(int) for out_mle in out_mles]

        eff = [self.gimme_numpy('photoelectron_detection_eff', x) for x in xs]
        ps = eff

        pdfs = [sp.binom(x, out_mle) * pow(p, out_mle) * pow(1. - p, x - out_mle) for out_mle, p, x in zip(out_mles, ps, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        lower_lims = [x[np.where(cdf < 0.00135)[0][-1]] if len(np.where(cdf < 0.00135)[0]) > 0 else out_mle for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        out_mles = np.round(d['s1_photoelectrons_detected_max']).astype(int)
        xs = [np.linspace(out_mle, out_mle * 2, 1000).astype(int) for out_mle in out_mles]

        eff = [self.gimme_numpy('photoelectron_detection_eff', x) for x in xs]
        ps = eff

        pdfs = [sp.binom(x, out_mle) * pow(p, out_mle) * pow(1. - p, x - out_mle) for out_mle, p, x in zip(out_mles, ps, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        upper_lims = [x[np.where(cdf > (1. - 0.00135))[0][0]] if len(np.where(cdf > (1. - 0.00135))[0]) > 0 else out_mle * 10 for x, cdf, out_mle in zip(xs, cdfs, out_mles)]

        out_mles = np.round(d['s1_photoelectrons_detected_mle']).astype(int)
        xs = [np.linspace(out_mle, out_mle * 2, 1000).astype(int) for out_mle in out_mles]

        eff = [self.gimme_numpy('photoelectron_detection_eff', x) for x in xs]
        ps = eff

        pdfs = [sp.binom(x, out_mle) * pow(p, out_mle) * pow(1. - p, x - out_mle) for out_mle, p, x in zip(out_mles, ps, xs)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        mles = [x[np.argmin(np.abs(cdf - 0.5))] for x, cdf in zip(xs, cdfs)]

        d['s1_photoelectrons_produced_mle'] = mles
        d['s1_photoelectrons_produced_min'] = lower_lims
        d['s1_photoelectrons_produced_max'] = upper_lims
