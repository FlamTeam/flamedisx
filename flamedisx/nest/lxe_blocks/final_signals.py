import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


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

    signal_name: str

    def _simulate(self, d):
        d[self.signal_name] = stats.norm.rvs(
            loc=(d[self.signal_name + '_photoelectrons_detected']),
            scale=(self.gimme_numpy(self.signal_name + '_spe_smearing',
                   d[self.signal_name + '_photoelectrons_detected'])))

        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    def _annotate(self, d):
        for bound in ('lower', 'upper'):
            observed_signals = d[self.signal_name].clip(0, None)
            supports = [np.linspace(np.floor(observed_signal / 2.),
                                    np.ceil(observed_signal * 2.), 1000).astype(int)
                        for observed_signal in observed_signals]
            mus = supports
            sigmas = [self.gimme_numpy(self.signal_name + '_spe_smearing', support) for support in supports]
            rvs = [observed_signal * np.ones_like(support)
                   for observed_signal, support in zip(observed_signals, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim=self.signal_name + '_photoelectrons_detected',
                                   bounds_prob=self.source.bounds_prob_outer, bound=bound,
                                   bound_type='normal', supports=supports,
                                   rvs_normal=rvs, mus_normal=mus, sigmas_normal=sigmas)

    def _compute(self,
                 photoelectrons_detected, s_observed,
                 data_tensor, ptensor):
        mean = photoelectrons_detected
        std = self.gimme(
            self.signal_name + '_spe_smearing',
            bonus_arg=photoelectrons_detected,
            data_tensor=data_tensor,
            ptensor=ptensor)

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(s_observed)

        # Add detection/selection efficiency
        result *= self.gimme(self.signal_name + '_acceptance',
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

    signal_name = 's1'

    dimensions = ('s1_photoelectrons_detected', 's1')
    special_model_functions = ('s1_spe_smearing', 'reconstruction_bias_s1')
    model_functions = ('s1_acceptance',) + special_model_functions

    max_dim_size = {'s1_photoelectrons_detected': 120}

    def s1_acceptance(self, s1):
        return tf.where((s1 < self.source.S1_min) | (s1 > self.source.S1_max),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def reconstruction_bias_s1(sig):
        """ Dummy method for pax s1 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    def _compute(self, data_tensor, ptensor,
                 s1_photoelectrons_detected, s1):
        return super()._compute(
            photoelectrons_detected=s1_photoelectrons_detected,
            s_observed=s1,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2(MakeFinalSignals):

    signal_name = 's2'

    dimensions = ('s2_photoelectrons_detected', 's2')
    special_model_functions = ('s2_spe_smearing', 'reconstruction_bias_s2')
    model_functions = ('s2_acceptance',) + special_model_functions

    max_dim_size = {'s2_photoelectrons_detected': 120}

    def s2_acceptance(self, s2):
        return tf.where((s2 < self.source.S2_min) | (s2 > self.source.S2_max),
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
                 s2_photoelectrons_detected, s2):
        return super()._compute(
            photoelectrons_detected=s2_photoelectrons_detected,
            s_observed=s2,
            data_tensor=data_tensor, ptensor=ptensor)
