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

        s1s2_corr_nr = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_first = s1s2_corr_nr  * np.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * np.sqrt(s1_var_second * s2_var_second)
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

        s1s2_corr_nr = self.gimme('signal_corr',
                                  bonus_arg=energies_first,
                                  data_tensor=data_tensor,
                                  ptensor=ptensor)
        s1s2_cov_first = s1s2_corr_nr  * tf.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * tf.sqrt(s1_var_second * s2_var_second)
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
class MakeS1S2MSU3(MakeS1S2MSU):
    """
    """
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_others',), 'rate_vs_energy'))

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

        s1s2_corr_nr = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_first = s1s2_corr_nr  * np.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * np.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_nr  * np.sqrt(s1_var_third * s2_var_third)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

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

        s1s2_corr_nr = self.gimme('signal_corr',
                                  bonus_arg=energies_first,
                                  data_tensor=data_tensor,
                                  ptensor=ptensor)
        s1s2_cov_first = s1s2_corr_nr  * tf.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * tf.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_nr  * tf.sqrt(s1_var_third * s2_var_third)
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


@export
class MakeS1S2SS(MakeS1S2MSU):
    """
    """
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),)

    def _simulate(self, d): 
        energies = d['energy_first'].values

        s1_mean, s2_mean = self.gimme_numpy('signal_means', energies)
        s1_var, s2_var = self.gimme_numpy('signal_vars', (s1_mean, s2_mean))
        anti_corr = self.gimme_numpy('signal_corr', energies)
        s1_skew, s2_skew = self.gimme('signal_skews', energies) 

        skewa = [s1_skew, s2_skew]
        final_std = [np.sqrt(s1_var),np.sqrt(s2_var)]
        final_mean = [s1_mean,s2_mean]
        dim=2

        
        cov = np.array([[1., anti_corr],[anti_corr,1.]])
        
        aCa = skewa @ cov @ skewa
        delta = (1. / np.sqrt(1. + aCa)) * cov @ skewa

        cov_star = np.block([[np.ones(1),     delta],
                                 [delta[:, None], cov]])

        x = scipy.stats.multivariate_normal(np.zeros(dim+1), cov_star).rvs(size=size)
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]

        scale = final_std / np.sqrt(1. - 2 * delta**2 / np.pi)
        loc = final_mean - scale * delta * np.sqrt(2/np.pi)

        samples = x1*scale+loc 
        
        d['s1'] = samples[:,0]
        d['s2'] = samples[:,1]
        
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):

        s1_mean, s2_mean = self.gimme('signal_means',
                                      bonus_arg=energy_first,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)

        s1_var, s2_var = self.gimme('signal_vars',
                                    bonus_arg=(s1_mean, s2_mean),
                                    data_tensor=data_tensor,
                                    ptensor=ptensor)

        anti_corr = self.gimme('signal_corr',
                               bonus_arg=energy_first,
                               data_tensor=data_tensor,
                               ptensor=ptensor)
        
        s1_skew, s2_skew = self.gimme('signal_skews', # 240213 - AV added
                                        bonus_arg=energy_first,
                                        data_tensor=data_tensor,
                                        ptensor=ptensor)
        
        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)
 
        
        # Define a Bivariate Normal PDF: 
        ## https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr) 

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))
        
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (s1_skew * s1) + (s2_skew * s2)
        Erf = tf.math.erf( Erf_arg / tf.sqrt(2) )
        norm_cdf = ( 1 + Erf ) / 2
        
        # skew(s1,s2) = mvn_pdf(s1,s2) * 2 * norm_cdf(alpha1*s1 + alpha2*s2)
        probs = mvn_pdf * 2 * norm_cdf
        

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(probs)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(probs)[2], axis=2)
        probs *= acceptance

        return tf.transpose(probs, perm=[0, 2, 1])


@export
class MakeS1S2Migdal(MakeS1S2MSU):
    """
    """
    special_model_functions = ('signal_means', 'signal_vars',
                               'signal_means_ER', 'signal_vars_ER',
                               'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

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


@export
class MakeS1S2MigdalMSU(MakeS1S2MSU3):
    """
    """
    special_model_functions = ('signal_means', 'signal_vars',
                               'signal_means_ER', 'signal_vars_ER',
                               'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

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


@export
class MakeS1S2ER(MakeS1S2SS):
    """
    """
    special_model_functions = ('signal_means_ER', 'signal_vars_ER')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies = d['energy_first'].values

        s1_mean, s2_mean = self.gimme_numpy('signal_means_ER', energies)
        s1_var, s2_var, s1s2_cov = self.gimme_numpy('signal_vars_ER', energies)
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies))
        Y = np.random.normal(size=len(energies))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1_mean = self.source.s1_mean_ER_tf
        s1_mean = tf.repeat(s1_mean[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        s2_mean = self.source.s2_mean_ER_tf
        s2_mean = tf.repeat(s2_mean[o, :], tf.shape(s1)[0], axis=0)[:, :, o]

        s1_var = self.source.s1_var_ER_tf
        s1_var = tf.repeat(s1_var[o,:], tf.shape(s1)[0], axis=0)[:, :, o]
        s2_var = self.source.s2_var_ER_tf
        s2_var = tf.repeat(s2_var[o,:], tf.shape(s1)[0], axis=0)[:, :, o]

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_cov = self.source.s1s2_cov_ER_tf
        s1s2_cov = tf.repeat(s1s2_cov[o,:], tf.shape(s1)[0], axis=0)[:, :, o]
        anti_corr = s1s2_cov / (s1_std * s2_std)

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
