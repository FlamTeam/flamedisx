import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


import pdb as pdb

this_load_s1 = 0.001 
this_load_s2 = 0.001
class ReconstructSignals(fd.Block):
    """Common code for ReconstructS1 and ReconstructS2"""

    model_attributes = ('check_acceptances',)
    non_integer_dimensions = ('s1_raw', 's2_raw',)

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
        bias = self.gimme_numpy('reconstruction_bias_simulate_' + self.signal_name,
                     bonus_arg=d[self.raw_signal_name].values)
        mu = d[self.raw_signal_name] * bias

        # loading this with 1e-15 to be symmetric with _compute
        relative_smear = self.gimme_numpy('reconstruction_smear_simulate_' + self.signal_name,
                     bonus_arg=d[self.raw_signal_name].values)
        #smear = np.clip(d[self.raw_signal_name] * relative_smear, 1e-15, None)
        smear = np.clip(relative_smear, 1e-15, None)
        # TODO: why some raw signals <=0?
        # checked 1e7 events and didn't see any raw_signals<=0..

        d[self.signal_name] = stats.norm.rvs(
            loc=mu,
            scale=smear)
        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    def _annotate(self, d):
        bias = self.gimme_numpy('reconstruction_bias_simulate_' + self.signal_name,
                     bonus_arg=d[self.raw_signal_name].values)
        smear = self.gimme_numpy('reconstruction_smear_simulate_' + self.signal_name,
                     bonus_arg=d[self.raw_signal_name].values)
        mle = d[self.raw_signal_name + '_mle'] = \
            (d[self.signal_name] / bias).clip(0, None)

        # The amount that you could have been smeared by from the raw signals
        scale = mle**0.5 * smear/bias

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d[self.raw_signal_name + '_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)

    def _compute(self,
                 s_raw,
                 s_observed,
                 data_tensor, ptensor):
        # bias = reconstructed_area/raw_area
        bias = self.gimme('reconstruction_bias_simulate_' + self.signal_name,
                          data_tensor=data_tensor,
                          bonus_arg=s_raw,
                          ptensor=ptensor)
        mu = s_raw * bias

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        relative_smear = self.gimme('reconstruction_smear_simulate_' + self.signal_name,
                           data_tensor=data_tensor,
                           bonus_arg=s_raw,
                           ptensor=ptensor) + 1e-15
        smear = tf.clip_by_value(relative_smear,
                                 clip_value_min=1e-15,
                                 clip_value_max=tf.float32.max)

        result = tfp.distributions.Normal(
            loc=mu, scale=smear).prob(s_observed)

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
class ReconstructS1(ReconstructSignals):

    raw_signal_name = 's1_raw'
    signal_name = 's1'

    dimensions = ('s1_raw', 's1')
    special_model_functions = (
        'reconstruction_bias_simulate_s1',
        'reconstruction_smear_simulate_s1',)
    model_functions = (
        's1_acceptance',
        'reconstruction_bias_annotate_s1',
        'reconstruction_smear_annotate_s1',
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

        Simulating dirac delta. Loading of this number done in main _compute
        function. Keeping this number as zero here to avoid loading done in
        multiple places.
        """
        return tf.zeros_like(s1_raw, dtype=fd.float_type())+this_load_s1

    # Getting from s1 -> s1_raw
    def reconstruction_bias_annotate_s1(self, s1):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s1, dtype=fd.float_type())

    def reconstruction_smear_annotate_s1(self, s1):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        
        Simulating dirac delta. Loading of this number done in main _compute
        function. Keeping this number as zero here to avoid loading done in
        multiple places.
        """
        return tf.zeros_like(s1, dtype=fd.float_type())+this_load_s1

    def _compute(self, data_tensor, ptensor,
                 s1_raw, s1):
        return super()._compute(
            s_raw=s1_raw,
            s_observed=s1,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class ReconstructS2(ReconstructSignals):

    raw_signal_name = 's2_raw'
    signal_name = 's2'

    dimensions = ('s2_raw', 's2')
    special_model_functions = (
        'reconstruction_bias_simulate_s2',
        'reconstruction_smear_simulate_s2',
 )
    model_functions = (
        ('s2_acceptance',
        'reconstruction_bias_annotate_s2',
        'reconstruction_smear_annotate_s2',
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

        Simulating dirac delta. Loading of this number done in main _compute
        function. Keeping this number as zero here to avoid loading done in
        multiple places.
        """
        return tf.zeros_like(s2_raw, dtype=fd.float_type())+this_load_s2

    # Getting from s2 -> s2_raw
    def reconstruction_bias_annotate_s2(self, s2):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s2, dtype=fd.float_type())

    def reconstruction_smear_annotate_s2(self, s2):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.

        Simulating dirac delta. Loading of this number done in main _compute
        function. Keeping this number as zero here to avoid loading done in
        multiple places.
        """
        return tf.zeros_like(s2, dtype=fd.float_type())+this_load_s2

    def _compute(self, data_tensor, ptensor,
                 s2_raw, s2):
        return super()._compute(
            s_raw=s2_raw,
            s_observed=s2,
            data_tensor=data_tensor, ptensor=ptensor)
