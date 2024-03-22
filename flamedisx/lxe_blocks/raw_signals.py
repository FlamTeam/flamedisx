import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


SIGNAL_NAMES = dict(photoelectron='s1_raw', electron='s2_raw')


class MakeFinalSignals(fd.Block):
    """Common code for MakeS1 and MakeS2"""

    model_attributes = ()  # leave it explicitly empty

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    quanta_name: str
    signal_name: str

    def _simulate(self, d):
        d[self.signal_name] = stats.norm.rvs(
            loc=(d[self.quanta_name + 's_detected']
                 * self.gimme_numpy(self.quanta_name + '_gain_mean')),
            scale=(d[self.quanta_name + 's_detected']**0.5
                   * self.gimme_numpy(self.quanta_name + '_gain_std')))

    def _annotate(self, d):
        m = self.gimme_numpy(self.quanta_name + '_gain_mean')
        s = self.gimme_numpy(self.quanta_name + '_gain_std')

        mle = d[self.quanta_name + 's_detected_mle'] = \
            (d[self.signal_name + '_mle'] / m).clip(0, None)
        scale = mle**0.5 * s / m

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d[self.quanta_name + 's_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(int)

    def _compute(self,
                 quanta_detected, s_observed,
                 data_tensor, ptensor):
        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme(self.quanta_name + '_gain_mean',
                                data_tensor=data_tensor,
                                ptensor=ptensor)[:, o, o]
        std_per_q = self.gimme(self.quanta_name + '_gain_std',
                               data_tensor=data_tensor,
                               ptensor=ptensor)[:, o, o]

        mean = quanta_detected * mean_per_q
        std = quanta_detected ** 0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(s_observed)
        return result


@export
class MakeS1(MakeFinalSignals):

    quanta_name = 'photoelectron'
    signal_name = 's1_raw'

    dimensions = ('photoelectrons_detected', 's1_raw')
    special_model_functions = ()
    model_functions = (
        'photoelectron_gain_mean',
        'photoelectron_gain_std',) + special_model_functions

    max_dim_size = {'photoelectrons_detected': 120}

    photoelectron_gain_mean = 1.
    photoelectron_gain_std = 0.5

    def _compute(self, data_tensor, ptensor,
                 photoelectrons_detected, s1_raw):
        return super()._compute(
            quanta_detected=photoelectrons_detected,
            s_observed=s1_raw,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2(MakeFinalSignals):

    quanta_name = 'electron'
    signal_name = 's2_raw'

    dimensions = ('electrons_detected', 's2_raw')
    special_model_functions = ()
    model_functions = (
        ('electron_gain_mean',
         'electron_gain_std',)
        + special_model_functions)

    max_dim_size = {'electrons_detected': 120}

    @staticmethod
    def electron_gain_mean(z, *, g2=20):
        return g2 * tf.ones_like(z)

    electron_gain_std = 5.

    def _compute(self, data_tensor, ptensor,
                 electrons_detected, s2_raw):
        return super()._compute(
            quanta_detected=electrons_detected,
            s_observed=s2_raw,
            data_tensor=data_tensor, ptensor=ptensor)
