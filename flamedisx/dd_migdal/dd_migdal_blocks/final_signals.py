import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


SIGNAL_NAMES = dict(photoelectron='s1', electron='s2')


@export
class MakeS1S2(fd.Block):
    """
    """

    # model_attributes = ('check_acceptances',)

    dimensions = ('energy', 's1s2')
    model_functions = ('signal_means', 'signal_covs')

    # # Whether to check acceptances are positive at the observed events.
    # # This is recommended, but you'll have to turn it off if your
    # # likelihood includes regions where only anomalous sources make events.
    # check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        means = self.gimme_numpy('signal_means')
        covs = self.gimme_numpy('signal_covs')

        shape = np.broadcast_shapes(means.shape, covs.shape[:-1])

        # Sample instead independently X from Normal(0, Id), then transform to
        # LX + mu, where L is the Cholesky decomposition of the covariance
        # matrix and mu is the mean vector
        X = np.random.standard_normal((*shape, 1))
        L = np.linalg.cholesky(covs)

        s1s2 = (L @ X).reshape(shape) + means

        d['s1s2'] = list(s1s2)

        # d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    # def _compute(self,
    #              quanta_detected, s_observed,
    #              data_tensor, ptensor):
    #     # Lookup signal gain mean and std per detected quanta
    #     mean_per_q = self.gimme(self.quanta_name + '_gain_mean',
    #                             data_tensor=data_tensor,
    #                             ptensor=ptensor)[:, o, o]
    #     std_per_q = self.gimme(self.quanta_name + '_gain_std',
    #                            data_tensor=data_tensor,
    #                            ptensor=ptensor)[:, o, o]
    #
    #     mean = quanta_detected * mean_per_q
    #     std = quanta_detected ** 0.5 * std_per_q
    #
    #     # add offset to std to avoid NaNs from norm.pdf if std = 0
    #     result = tfp.distributions.Normal(
    #         loc=mean, scale=std + 1e-10
    #     ).prob(s_observed)
    #
    #     # Add detection/selection efficiency
    #     result *= self.gimme(SIGNAL_NAMES[self.quanta_name] + '_acceptance',
    #                          data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
    #     return result

    def _annotate(self, d):
        pass

    # def check_data(self):
    #     if not self.check_acceptances:
    #         return
    #     s_acc = self.gimme_numpy(self.signal_name + '_acceptance')
    #     if np.any(s_acc <= 0):
    #         raise ValueError(f"Found event with non-positive {self.signal_name} "
    #                          f"acceptance: did you apply and configure "
    #                          "your cuts correctly?")
