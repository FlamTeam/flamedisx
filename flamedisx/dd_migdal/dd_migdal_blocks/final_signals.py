import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS1S2MSU(fd.Block):
    """
    """

    dimensions = ('energy_first', 's1s2')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_second',), 'rate_vs_energy'))

    special_model_functions = ('signal_means', 'signal_covs')
    model_functions = special_model_functions

    array_columns = (('s1s2', 2),)

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values

        s1_mean_first, s1_mean_second, s2_mean_first, s2_mean_second = self.gimme_numpy('signal_means', (energies_first, energies_second))
        means = [s1_mean_first + s1_mean_second, s2_mean_first + s2_mean_second]
        means = np.array(means).transpose()

        covs = self.gimme_numpy('signal_covs', (s1_mean_first, s1_mean_second, s2_mean_first, s2_mean_second))
        covs = np.array(covs).transpose(2, 0, 1)

        shape = np.broadcast_shapes(means.shape, covs.shape[:-1])

        # Sample instead independently X from Normal(0, Id), then transform to
        # LX + mu, where L is the Cholesky decomposition of the covariance
        # matrix and mu is the mean vector
        X = np.random.standard_normal((*shape, 1))
        L = np.linalg.cholesky(covs)

        s1s2 = (L @ X).reshape(shape) + means

        d['s1s2'] = list(s1s2)

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1s2,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_second, rate_vs_energy):
        energies_first = energy_first[0, :, 0]
        energies_second = energy_second[0, :]

        energies_first = tf.repeat(energies_first[:, o], tf.shape(energy_second[0, :]), axis=1)
        energies_second = tf.repeat(energies_second[o, :], tf.shape(energy_first[0, :, 0]), axis=0)

        s1_mean_first, s1_mean_second, s2_mean_first, s2_mean_second = self.gimme('signal_means',
                                                                                  bonus_arg=(energies_first, energies_second),
                                                                                  data_tensor=data_tensor,
                                                                                  ptensor=ptensor)
        means = [s1_mean_first + s1_mean_second, s2_mean_first + s2_mean_second]
        means = tf.transpose(means, perm=[1, 2, 0])

        covs = self.gimme('signal_covs', bonus_arg=(s1_mean_first, s1_mean_second, s2_mean_first, s2_mean_second),
                           data_tensor=data_tensor,
                           ptensor=ptensor)
        covs = tf.transpose(covs, perm=[2, 3, 0, 1])

        scale = tf.linalg.cholesky(covs)

        means = tf.repeat(means[o, :, :], self.source.batch_size, axis=0)
        scale = tf.repeat(scale[o, :, :], self.source.batch_size, axis=0)

        s1s2 = tf.repeat(s1s2[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)

        probs = tfp.distributions.MultivariateNormalTriL(loc=means, scale_tril=scale).prob(s1s2)

        R_E1E2 = probs * rate_vs_energy
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        return R_E1[:, o, :]


@export
class MakeS1S2SS(fd.Block):
    """
    """

    dimensions = ('energy_first', 's1s2')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),)

    special_model_functions = ('signal_means', 'signal_covs')
    model_functions = special_model_functions

    array_columns = (('s1s2', 2),)

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies = d['energy_first'].values

        s1_mean, s2_mean = self.gimme_numpy('signal_means', energies)
        means = [s1_mean, s2_mean]
        means = np.array(means).transpose()

        covs = self.gimme_numpy('signal_covs', (s1_mean, s2_mean))
        covs = np.array(covs).transpose(2, 0, 1)

        shape = np.broadcast_shapes(means.shape, covs.shape[:-1])

        # Sample instead independently X from Normal(0, Id), then transform to
        # LX + mu, where L is the Cholesky decomposition of the covariance
        # matrix and mu is the mean vector
        X = np.random.standard_normal((*shape, 1))
        L = np.linalg.cholesky(covs)

        s1s2 = (L @ X).reshape(shape) + means

        d['s1s2'] = list(s1s2)

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1s2,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):
        energies = energy_first[0, :, 0]

        s1_mean, s2_mean = self.gimme('signal_means',
                                      bonus_arg=energies,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)
        means = [s1_mean, s2_mean]
        means = tf.transpose(means)

        covs = self.gimme('signal_covs', bonus_arg=(s1_mean, s2_mean),
                           data_tensor=data_tensor,
                           ptensor=ptensor)
        covs = tf.transpose(covs, perm=[2, 0, 1])

        scale = tf.linalg.cholesky(covs)

        means = tf.repeat(means[o, :, :], self.source.batch_size, axis=0)
        scale = tf.repeat(scale[o, :, :], self.source.batch_size, axis=0)

        probs = tfp.distributions.MultivariateNormalTriL(loc=means, scale_tril=scale).prob(s1s2)

        return probs[:, o, :]
