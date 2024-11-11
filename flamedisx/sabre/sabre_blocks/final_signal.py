import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeFinalSignal(fd.Block):
    model_attributes = ()  # leave it explicitly empty

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    dimensions = ('integrated_charge', 'photoelectrons_detected')
    special_model_functions = ()
    model_functions = (
        'photoelectron_gain_mean',
        'photoelectron_gain_std',) + special_model_functions

    max_dim_size = {'photoelectrons_detected': 120}

    photoelectron_gain_mean = 1.
    photoelectron_gain_std = 0.5

    def _simulate(self, d):
        d['integrated_charge'] = stats.norm.rvs(
            loc=(d['photoelectrons_detected']
                 * self.gimme_numpy('photoelectron_gain_mean')),
            scale=(d['photoelectrons_detected']**0.5
                   * self.gimme_numpy('photoelectron_gain_std')))

    def _annotate(self, d):
        m = self.gimme_numpy('photoelectron_gain_mean')
        s = self.gimme_numpy('photoelectron_gain_std')

        mle = d['photoelectrons_detected_mle'] = \
            (d['integrated_charge'] / m).clip(0, None)
        scale = mle**0.5 * s / m

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d['photoelectrons_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(int)

    def _compute(self,
                 data_tensor, ptensor,
                 photoelectrons_detected, integrated_charge):
        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme('photoelectron_gain_mean',
                                data_tensor=data_tensor,
                                ptensor=ptensor)[:, o, o]
        std_per_q = self.gimme('photoelectron_gain_std',
                               data_tensor=data_tensor,
                               ptensor=ptensor)[:, o, o]

        mean = photoelectrons_detected * mean_per_q
        std = photoelectrons_detected ** 0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(integrated_charge)
        return result