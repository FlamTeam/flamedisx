import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


SIGNAL_NAMES = dict(photoelectron='s1', electron='s2')


class MakeFinalSignals(fd.Block):
    """Common code for MakeS1 and MakeS2"""

    model_attributes = ('check_acceptances',)

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

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

        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    def _annotate(self, d):
        m = self.gimme_numpy(self.quanta_name + '_gain_mean')
        s = self.gimme_numpy(self.quanta_name + '_gain_std')

        mle = d[self.quanta_name + 's_detected_mle'] = \
            (d[self.signal_name] / m).clip(0, None)
        scale = mle**0.5 * s / m

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d[self.quanta_name + 's_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)

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

        # Add detection/selection efficiency
        result *= self.gimme(SIGNAL_NAMES[self.quanta_name] + '_acceptance',
                             data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        return result

    def check_data(self):
        if not self.check_acceptances:
            return
        s_acc = self.gimme_numpy(self.signal_name + '_acceptance')
        if np.any(s_acc <= 0):
            raise ValueError(f"Found event with non-positive {self.signal_name} "
                             f"acceptance: did you apply and configure "
                             "your cuts correctly?")


@export
class MakeS1(MakeFinalSignals):

    quanta_name = 'photoelectron'
    signal_name = 's1'

    dimensions = ('photoelectrons_detected', 's1')
    special_model_functions = ('reconstruction_bias_s1',)
    model_functions = (
        'photoelectron_gain_mean',
        'photoelectron_gain_std',
        's1_acceptance') + special_model_functions

    max_dim_size = {'photoelectrons_detected': 120}

    photoelectron_gain_mean = 1.
    photoelectron_gain_std = 0.5

    def s1_acceptance(self, s1, s1_min=2, s1_max=70):
        return tf.where((s1 < s1_min) | (s1 > s1_max),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def reconstruction_bias_s1(sig):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    def _compute(self, data_tensor, ptensor,
                 photoelectrons_detected, s1):
        return super()._compute(
            quanta_detected=photoelectrons_detected,
            s_observed=s1,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2(MakeFinalSignals):

    quanta_name = 'electron'
    signal_name = 's2'

    dimensions = ('electrons_detected', 's2')
    special_model_functions = ('reconstruction_bias_s2',)
    model_functions = (
        ('electron_gain_mean',
         'electron_gain_std',
         's2_acceptance')
        + special_model_functions)

    max_dim_size = {'electrons_detected': 120}

    @staticmethod
    def electron_gain_mean(z, *, g2=20):
        return g2 * tf.ones_like(z)

    electron_gain_std = 5.

    def s2_acceptance(self, s2, s2_min=2, s2_max=6000):
        return tf.where((s2 < s2_min) | (s2 > s2_max),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    @staticmethod
    def reconstruction_bias_s2(sig):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    def _compute(self, data_tensor, ptensor,
                 electrons_detected, s2):
        return super()._compute(
            quanta_detected=electrons_detected,
            s_observed=s2,
            data_tensor=data_tensor, ptensor=ptensor)
