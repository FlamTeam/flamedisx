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

    dimensions = ('energy_first', 's1s2')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_second',), 'rate_vs_energy_second'))

    special_model_functions = ('signal_means_double', 'signal_covs_double')
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
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values

        means = self.gimme_numpy('signal_means_double', (energies_first, energies_second))
        means = np.array(means).transpose()

        covs = self.gimme_numpy('signal_covs_double', (energies_first, energies_second))
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
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_second, rate_vs_energy_second):
        energies_first = energy_first[0, :, 0]
        energies_second = energy_second[0, :]

        energies_first = tf.repeat(energies_first[:, o], tf.shape(energy_second[0, :]), axis=1)
        energies_second = tf.repeat(energies_second[o, :], tf.shape(energy_first[0, :, 0]), axis=0)

        means = self.gimme('signal_means_double', bonus_arg=(energies_first, energies_second),
                           data_tensor=data_tensor,
                           ptensor=ptensor)
        means = tf.transpose(means, perm=[1, 2, 0])

        covs = self.gimme('signal_covs_double', bonus_arg=(energies_first, energies_second),
                           data_tensor=data_tensor,
                           ptensor=ptensor)
        covs = tf.transpose(covs, perm=[2, 3, 0, 1])

        scale = tf.linalg.cholesky(covs)

        means = tf.repeat(means[o, :, :], self.source.batch_size, axis=0)
        scale = tf.repeat(scale[o, :, :], self.source.batch_size, axis=0)

        s1s2 = tf.repeat(s1s2[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)

        result_all_energies = tfp.distributions.MultivariateNormalTriL(loc=means, scale_tril=scale).prob(s1s2)

        result_energies_first = (result_all_energies @ rate_vs_energy_second[..., None])[..., 0]

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                 data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(result_energies_first)[1], axis=1)
        result_energies_first *= acceptance

        return result_energies_first[:, o, :]

    # def check_data(self):
    #     if not self.check_acceptances:
    #         return
    #     s_acc = self.gimme_numpy(self.signal_name + '_acceptance')
    #     if np.any(s_acc <= 0):
    #         raise ValueError(f"Found event with non-positive {self.signal_name} "
    #                          f"acceptance: did you apply and configure "
    #                          "your cuts correctly?")
