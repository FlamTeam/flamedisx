import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


SIGNAL_NAMES = dict(photoelectron='s1', electron='s2')


import pdb as pdb
class ReconstructSignals(fd.Block):
    """Common code for ReconstructS1 and ReconstructS2"""

    model_attributes = ('check_acceptances',)

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    raw_signal_name: str
    signal_name: str

    def _simulate(self, d):
        d[self.signal_name] = stats.norm.rvs(
            loc=(d[self.raw_signal_name] *
                 self.gimme_numpy('reconstruction_bias_simulate_' + self.signal_name)),
            scale=(self.gimme_numpy('reconstruction_smear_simulate_' + self.signal_name)))

        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    def _annotate(self, d):
        m = self.gimme_numpy(self.raw_signal_name + '_gain_mean')
        pdb.set_trace()

        mle = d[self.raw_signal_name + 's_detected_mle'] = \
            (d[self.signal_name] / m).clip(0, None)

        # The amount that you could have been smeared by from the raw signals
        scale = self.gimme_numpy('reconstruction_smear_simulate_' + self.signal_name)

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d[self.raw_signal_name + '_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)

    def _compute(self,
                 quanta_detected, s_observed,
                 data_tensor, ptensor):
        # ASK JORAN PROPERLY 
        bias = self.gimme('reconstruction_bias_compute_' + self.signal_name,
                          data_tensor=data_tensor,
                          ptensor=ptensor)[:, o, o]

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        smear = self.gimme('reconstruction_smear_compute_' + self.signal_name,
                           data_tensor=data_tensor,
                           ptensor=ptensor)[:, o, o] + 1e-10

        result = tfp.distributions.Normal(
            loc=(s_observed/bias), scale=smear).prob(s_observed)

        # Add detection/selection efficiency
        result *= self.gimme(SIGNAL_NAMES[self.signal_name] + '_acceptance',
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
class ReconstructS1(ReconstructSignals):

    raw_signal_name = 's1_raw'
    signal_name = 's1'

    dimensions = ('s1_raw', 's1')
    special_model_functions = ()
    model_functions = (
        's1_acceptance',
        'reconstruction_bias_simulate_s1',
        'reconstruction_smear_simulate_s1',
        'reconstruction_bias_compute_s1',
        'reconstruction_smear_compute_s1',
        ) + special_model_functions

    max_dim_size = {'s1_raw': 120}

    def s1_acceptance(self, s1, s1_min=2, s1_max=70):
        return tf.where((s1 < s1_min) | (s1 > s1_max),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    # Getting from s1_raw -> s1
    def reconstruction_bias_simulate_s1(self, s1_raw):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s1_raw, dtype=fd.float_type())

    def reconstruction_smear_simulate_s1(self, s1_raw):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        # Trying to simulate dirac delta
        # TODO: find what number to put here safely
        return tf.ones_like(s1_raw, dtype=fd.float_type())*1e-45

    # Getting from s1 -> s1_raw
    def reconstruction_bias_compute_s1(self, s1):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s1, dtype=fd.float_type())

    def reconstruction_smear_compute_s1(self, s1):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        # Trying to simulate dirac delta
        # TODO: find what number to put here safely
        return tf.ones_like(s1, dtype=fd.float_type())*1e-45

    def _compute(self, data_tensor, ptensor,
                 s1_raw, s1):
        return super()._compute(
            quanta_detected=s1_raw,
            s_observed=s1,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class ReconstructS2(ReconstructSignals):

    raw_signal_name = 's2_raw'
    signal_name = 's2'

    dimensions = ('s2_raw', 's2')
    special_model_functions = ()
    model_functions = (
        ('s2_acceptance',
        'reconstruction_bias_simulate_s2',
        'reconstruction_smear_simulate_s2',
        'reconstruction_bias_compute_s2',
        'reconstruction_smear_compute_s2',
        )
        + special_model_functions)

    max_dim_size = {'s2_raw': 120}

    def s2_acceptance(self, s2, s2_min=2, s2_max=6000):
        return tf.where((s2 < s2_min) | (s2 > s2_max),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    # Getting from s2_raw -> s2
    def reconstruction_bias_simulate_s2(self, s2_raw):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s2_raw, dtype=fd.float_type())

    def reconstruction_smear_simulate_s2(self, s2_raw):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        # Trying to simulate dirac delta
        # TODO: find what number to put here safely
        return tf.ones_like(s2_raw, dtype=fd.float_type())*1e-45

    # Getting from s2 -> s2_raw
    def reconstruction_bias_compute_s2(self, s2):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s2, dtype=fd.float_type())

    def reconstruction_smear_compute_s2(self, s2):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        # Trying to simulate dirac delta
        # TODO: find what number to put here safely
        return tf.ones_like(s2, dtype=fd.float_type())*1e-45

    def _compute(self, data_tensor, ptensor,
                 s2_raw, s2):
        return super()._compute(
            quanta_detected=s2_raw,
            s_observed=s2,
            data_tensor=data_tensor, ptensor=ptensor)
