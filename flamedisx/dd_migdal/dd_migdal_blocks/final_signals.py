import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import math as m
pi = tf.constant(m.pi)

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeS1S2MSU(fd.Block):
    """
    """
    model_attributes = ('check_acceptances',)

    dimensions = ('energy_first', 's1')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_second',), 'rate_vs_energy'))

    special_model_functions = ('signal_means', 'signal_vars', 'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        s1_var_first, s2_var_first = self.gimme_numpy('signal_vars', (s1_mean_first, s2_mean_first))
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second
        anti_corr = self.gimme_numpy('signal_corr', energies_first)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_second, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_second[0, :]), axis=2)
        energies_second = tf.repeat(energy_second[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)[:, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)

        s1_mean_first, s2_mean_first = self.gimme('signal_means',
                                                  bonus_arg=energies_first,
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second)
        s2_mean = (s2_mean_first + s2_mean_second)

        s1_var_first, s2_var_first = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_first, s2_mean_first),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second)
        s2_var = (s2_var_first + s2_var_second)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)
        anti_corr = self.gimme('signal_corr',
                               bonus_arg=energies_first,
                               data_tensor=data_tensor,
                               ptensor=ptensor)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2 = probs * rate_vs_energy[:, :, :, o]
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])

    def check_data(self):
        if not self.check_acceptances:
         return
        s_acc = self.gimme_numpy('s1s2_acceptance')
        if np.any(s_acc <= 0):
         raise ValueError(f"Found event with non-positive signal "
                          f"acceptance: did you apply and configure "
                          "your cuts correctly?")


@export
class MakeS1S2MSU3(fd.Block):
    """
    """
    model_attributes = ('check_acceptances',)

    dimensions = ('energy_first', 's1')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_others',), 'rate_vs_energy'))

    special_model_functions = ('signal_means', 'signal_vars', 'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        energies_third= d['energy_third'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean_third, s2_mean_third = self.gimme_numpy('signal_means', energies_third)
        s1_mean = s1_mean_first + s1_mean_second + s1_mean_third
        s2_mean = s2_mean_first + s2_mean_second + s2_mean_third

        s1_var_first, s2_var_first = self.gimme_numpy('signal_vars', (s1_mean_first, s2_mean_first))
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var_third, s2_var_third = self.gimme_numpy('signal_vars', (s1_mean_third, s2_mean_third))
        s1_var = s1_var_first + s1_var_second + s1_var_third
        s2_var = s2_var_first + s2_var_second + s2_var_third
        anti_corr = self.gimme_numpy('signal_corr', energies_first)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_others, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_others[0, :,]), axis=2)
        energies_first = tf.repeat(energies_first[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)

        energies_second= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_second= tf.repeat(energies_second[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)[:, :, :, :, o]

        energies_third= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_third= tf.repeat(energies_third[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)[:, :, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s1 = tf.repeat(s1[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)

        s1_mean_first, s2_mean_first = self.gimme('signal_means',
                                                  bonus_arg=energies_first,
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean_third, s2_mean_third = self.gimme('signal_means',
                                                 bonus_arg=energies_third,
                                                 data_tensor=data_tensor,
                                                 ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second + s1_mean_third)
        s2_mean = (s2_mean_first + s2_mean_second + s2_mean_third)

        s1_var_first, s2_var_first = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_first, s2_mean_first),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var_third, s2_var_third = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_third, s2_mean_third),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second + s1_var_third)
        s2_var = (s2_var_first + s2_var_second + s2_var_third)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)
        anti_corr = self.gimme('signal_corr',
                               bonus_arg=energies_first,
                               data_tensor=data_tensor,
                               ptensor=ptensor)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2E3 = probs * rate_vs_energy[:, :, :, :, o]
        R_E1E2 = tf.reduce_sum(R_E1E2E3, axis=3)
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])

    def check_data(self):
        if not self.check_acceptances:
         return
        s_acc = self.gimme_numpy('s1s2_acceptance')
        if np.any(s_acc <= 0):
         raise ValueError(f"Found event with non-positive signal "
                          f"acceptance: did you apply and configure "
                          "your cuts correctly?")


@export
class MakeS1S2SS(fd.Block):
    """
    """
    model_attributes = ('check_acceptances',)

    dimensions = ('energy_first', 's1')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),)

    special_model_functions = ('signal_means', 'signal_vars', 'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance') + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies = d['energy_first'].values

        s1_mean, s2_mean = self.gimme_numpy('signal_means', energies)

        s1_var, s2_var = self.gimme_numpy('signal_vars', (s1_mean, s2_mean))
        anti_corr = self.gimme_numpy('signal_corr', energies)

        X = np.random.normal(size=len(energies))
        Y = np.random.normal(size=len(energies))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1_mean, s2_mean = self.gimme('signal_means',
                                      bonus_arg=energy_first,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)

        s1_var, s2_var = self.gimme('signal_vars',
                                    bonus_arg=(s1_mean, s2_mean),
                                    data_tensor=data_tensor,
                                    ptensor=ptensor)
        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)
        anti_corr = self.gimme('signal_corr',
                               bonus_arg=energy_first,
                               data_tensor=data_tensor,
                               ptensor=ptensor)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(probs)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(probs)[2], axis=2)
        probs *= acceptance

        return tf.transpose(probs, perm=[0, 2, 1])

    def check_data(self):
        if not self.check_acceptances:
         return
        s_acc = self.gimme_numpy('s1s2_acceptance')
        if np.any(s_acc <= 0):
         raise ValueError(f"Found event with non-positive signal "
                          f"acceptance: did you apply and configure "
                          "your cuts correctly?")


@export
class MakeS1S2Migdal(fd.Block):
    """
    """
    model_attributes = ('check_acceptances',)

    dimensions = ('energy_first', 's1')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_second',), 'rate_vs_energy'))

    special_model_functions = ('signal_means', 'signal_vars',
                               'signal_means_ER', 'signal_vars_ER',
                               'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', energies_first)
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second

        s1s2_corr_second = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_second = s1s2_corr_second  * np.sqrt(s1_var_second * s2_var_second)

        s1s2_cov = s1s2_cov_first + s1s2_cov_second
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_second, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_second[0, :]), axis=2)
        energies_second = tf.repeat(energy_second[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)[:, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)

        s1_mean_first = self.source.s1_mean_ER_tf
        s1_mean_first = tf.repeat(s1_mean_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s2_mean_first = self.source.s2_mean_ER_tf
        s2_mean_first = tf.repeat(s2_mean_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second)
        s2_mean = (s2_mean_first + s2_mean_second)

        s1_var_first = self.source.s1_var_ER_tf
        s1_var_first = tf.repeat(s1_var_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s2_var_first = self.source.s2_var_ER_tf
        s2_var_first = tf.repeat(s2_var_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second)
        s2_var = (s2_var_first + s2_var_second)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_cov_first = self.source.s1s2_cov_ER_tf
        s1s2_cov_first = tf.repeat(s1s2_cov_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s1s2_corr_second = self.gimme('signal_corr',
                                      bonus_arg=energies_first,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)
        s1s2_cov_second = s1s2_corr_second  * tf.sqrt(s1_var_second * s2_var_second)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second
        anti_corr = s1s2_cov / (s1_std * s2_std)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2 = probs * rate_vs_energy[:, :, :, o]
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])

    def check_data(self):
        if not self.check_acceptances:
         return
        s_acc = self.gimme_numpy('s1s2_acceptance')
        if np.any(s_acc <= 0):
         raise ValueError(f"Found event with non-positive signal "
                          f"acceptance: did you apply and configure "
                          "your cuts correctly?")


@export
class MakeS1S2MigdalMSU(fd.Block):
    """
    """
    model_attributes = ('check_acceptances',)

    dimensions = ('energy_first', 's1')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_others',), 'rate_vs_energy'))

    special_model_functions = ('signal_means', 'signal_vars',
                               'signal_means_ER', 'signal_vars_ER',
                               'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        energies_third= d['energy_third'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean_third, s2_mean_third = self.gimme_numpy('signal_means', energies_third)
        s1_mean = s1_mean_first + s1_mean_second + s1_mean_third
        s2_mean = s2_mean_first + s2_mean_second + s2_mean_third

        s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', energies_first)
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var_third, s2_var_third = self.gimme_numpy('signal_vars', (s1_mean_third, s2_mean_third))
        s1_var = s1_var_first + s1_var_second + s1_var_third
        s2_var = s2_var_first + s2_var_second + s2_var_third

        s1s2_corr_others = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_second = s1s2_corr_others * np.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_others * np.sqrt(s1_var_third * s2_var_third)

        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_others, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_others[0, :,]), axis=2)
        energies_first = tf.repeat(energies_first[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)

        energies_second= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_second= tf.repeat(energies_second[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)[:, :, :, :, o]

        energies_third= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_third= tf.repeat(energies_third[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)[:, :, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s1 = tf.repeat(s1[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)

        s1_mean_first = self.source.s1_mean_ER_tf
        s1_mean_first = tf.repeat(s1_mean_first[o, :, :], tf.shape(s1)[0], axis=0)
        s1_mean_first = tf.repeat(s1_mean_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s2_mean_first = self.source.s2_mean_ER_tf
        s2_mean_first = tf.repeat(s2_mean_first[o, :, :], tf.shape(s1)[0], axis=0)
        s2_mean_first = tf.repeat(s2_mean_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean_third, s2_mean_third = self.gimme('signal_means',
                                                 bonus_arg=energies_third,
                                                 data_tensor=data_tensor,
                                                 ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second + s1_mean_third)
        s2_mean = (s2_mean_first + s2_mean_second + s2_mean_third)

        s1_var_first = self.source.s1_var_ER_tf
        s1_var_first = tf.repeat(s1_var_first[o, :, :], tf.shape(s1)[0], axis=0)
        s1_var_first = tf.repeat(s1_var_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s2_var_first = self.source.s2_var_ER_tf
        s2_var_first = tf.repeat(s2_var_first[o, :, :], tf.shape(s1)[0], axis=0)
        s2_var_first = tf.repeat(s2_var_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var_third, s2_var_third = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_third, s2_mean_third),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second + s1_var_third)
        s2_var = (s2_var_first + s2_var_second + s2_var_third)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_cov_first = self.source.s1s2_cov_ER_tf
        s1s2_cov_first = tf.repeat(s1s2_cov_first[o, :, :], tf.shape(s1)[0], axis=0)
        s1s2_cov_first = tf.repeat(s1s2_cov_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s1s2_corr_others = self.gimme('signal_corr',
                                      bonus_arg=energies_first,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)
        s1s2_cov_second = s1s2_corr_others * tf.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_others  * tf.sqrt(s1_var_third * s2_var_third)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / (s1_std * s2_std)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2E3 = probs * rate_vs_energy[:, :, :, :, o]
        R_E1E2 = tf.reduce_sum(R_E1E2E3, axis=3)
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])

    def check_data(self):
        if not self.check_acceptances:
         return
        s_acc = self.gimme_numpy('s1s2_acceptance')
        if np.any(s_acc <= 0):
         raise ValueError(f"Found event with non-positive signal "
                          f"acceptance: did you apply and configure "
                          "your cuts correctly?")
