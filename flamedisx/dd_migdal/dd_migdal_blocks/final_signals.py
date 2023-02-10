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
    depends_on = ((('energy',), 'rate_vs_energy'),)

    special_model_functions = ('signal_means', 'signal_covs')
    model_functions = ('s1s2_acceptance',) + special_model_functions

    array_columns = (('s1s2', 2),)

    # # Whether to check acceptances are positive at the observed events.
    # # This is recommended, but you'll have to turn it off if your
    # # likelihood includes regions where only anomalous sources make events.
    # check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def s1s2_acceptance(self, s1s2):
        return tf.ones_like(s1s2, dtype=fd.float_type())[:, 0]

    def _simulate(self, d):
        energies = d['energy'].values

        means = self.gimme_numpy('signal_means', bonus_arg=energies)
        means = np.array(means).transpose()

        covs = self.gimme_numpy('signal_covs', bonus_arg=energies)
        covs = np.array(covs).transpose(2, 0, 1)

        shape = np.broadcast_shapes(means.shape, covs.shape[:-1])

        # Sample instead independently X from Normal(0, Id), then transform to
        # LX + mu, where L is the Cholesky decomposition of the covariance
        # matrix and mu is the mean vector
        X = np.random.standard_normal((*shape, 1))
        L = np.linalg.cholesky(covs)

        s1s2 = (L @ X).reshape(shape) + means

        d['s1s2'] = list(s1s2)

        # d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1s2,
                 # Dependency domain and value
                 energy, rate_vs_energy,):
        energies = energy[0, :, 0]

        means = self.gimme('signal_means', bonus_arg=energies,
                           data_tensor=data_tensor,
                           ptensor=ptensor)
        means = tf.transpose(means)

        covs = self.gimme('signal_covs', bonus_arg=energies,
                           data_tensor=data_tensor,
                           ptensor=ptensor)
        covs = tf.transpose(covs, perm=[2, 0, 1])

        scale = tf.linalg.cholesky(covs)

        means = tf.repeat(means[o, :, :], self.source.batch_size, axis=0)
        scale = tf.repeat(scale[o, :, :], self.source.batch_size, axis=0)
        result = tfp.distributions.MultivariateNormalTriL(loc=means, scale_tril=scale).prob(s1s2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                 data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(result)[1], axis=1)
        result *= acceptance

        return result[:, o, :]

    # def check_data(self):
    #     if not self.check_acceptances:
    #         return
    #     s_acc = self.gimme_numpy(self.signal_name + '_acceptance')
    #     if np.any(s_acc <= 0):
    #         raise ValueError(f"Found event with non-positive {self.signal_name} "
    #                          f"acceptance: did you apply and configure "
    #                          "your cuts correctly?")
