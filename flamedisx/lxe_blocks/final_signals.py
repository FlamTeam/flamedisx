import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


SIGNAL_NAMES = dict(photon='s1', electron='s2')


class MakeFinalSignals:
    """Common code for MakeS1 and MakeS2"""

    static_attributes = ('check_efficiencies',)

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    quanta_name: str
    s_name: str

    def _simulate(self, d):
        d[self.s_name] = stats.norm.rvs(
            loc=(d[self.quanta_name + 's_detected']
                 * self.gimme_numpy(self.quanta_name + '_gain_mean')),
            scale=(d[self.quanta_name + 's_detected']**0.5
                   * self.gimme_numpy(self.quanta_name + '_gain_std')))
        d['p_accepted'] *= self.gimme_numpy(self.s_name + '_acceptance')

    def _annotate(self, d):
        m = self.gimme_numpy(self.quanta_name + '_gain_mean').values
        s = self.gimme_numpy(self.quanta_name + '_gain_std').values

        mle = d[self.quanta_name + 's_detected_mle'] = \
            (d[self.s_name] / d[self.quanta_name + '_gain_mean']).clip(0, None)
        scale = mle**0.5 * s / m

        for bound, sign in (('min', -1), ('max', +1)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE
            d[self.quanta_name + 's_detected_' + bound] = (
                    mle + sign * self.source.max_sigma * scale
            ).round().clip(0, None).astype(np.int)

    def __compute(self,
                  quanta_detected, s_observed,
                  data_tensor, ptensor):
        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme(self.quanta_name + '_gain_mean',
                                data_tensor=data_tensor, ptensor=ptensor)[:, o]
        std_per_q = self.gimme(self.quanta_name + '_gain_std',
                               data_tensor=data_tensor, ptensor=ptensor)[:, o]

        mean = quanta_detected * mean_per_q
        std = quanta_detected ** 0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(s_observed[:, o])

        # Add detection/selection efficiency
        result *= self.gimme(SIGNAL_NAMES[self.quanta_name] + '_acceptance',
                             data_tensor=data_tensor, ptensor=ptensor)[:, o]
        return result

    def check_data(self):
        if not self.check_acceptances:
            return
        s_acc = self.gimme_numpy(self.s_name + '_acceptance')
        if np.any(s_acc <= 0):
            raise ValueError(f"Found event with non-positive {self.s_name} "
                             f"acceptance: did you apply and configure "
                             "your cuts correctly?")


@export
class MakeS1(fd.Block, MakeFinalSignals):

    dimensions = ('photoelectrons_detected', 's1')
    model_functions = ('photelectron_gain_mean', 'photelectron_gain_std',
                       's1_acceptance')

    photoelectron_gain_mean = 1.
    photoelectron_gain_std = 0.5

    s1_acceptance = 1.

    quanta_name = 'photoelectron'
    s_name = 's1'

    def _compute(self, data_tensor, ptensor,
                 photoelectrons_detected, s1):
        return self.__compute(
            quanta_detected=photoelectrons_detected,
            s_observed=s1,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2(fd.Block, MakeFinalSignals):

    dimensions = ('electrons_detected', 's2')
    model_functions = ('electron_gain_mean', 'electron_gain_std',
                       's2_acceptance')

    @staticmethod
    def electron_gain_mean(z, *, g2=20):
        return g2 * tf.ones_like(z)

    electron_gain_std = 5.

    s2_acceptance = 1.

    quanta_name = 'electron'
    s_name = 's2'

    def _compute(self, data_tensor, ptensor,
                 electrons_detected, s2):
        return self.__compute(
            quanta_detected=electrons_detected,
            s_observed=s2,
            data_tensor=data_tensor, ptensor=ptensor)
