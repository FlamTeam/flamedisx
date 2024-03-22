import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


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
        bias = self.gimme_numpy(f'reconstruction_bias_{self.signal_name}_simulate',
                                bonus_arg=d[self.raw_signal_name].values)
        mu = d[self.raw_signal_name] * bias

        # clipping this to (1e-15, float32max) to be symmetric with _compute
        smear = self.gimme_numpy(f'reconstruction_smear_{self.signal_name}_simulate',
                                 bonus_arg=d[self.raw_signal_name].values)
        smear = np.clip(smear, 1e-15, tf.float32.max)
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
        bias = self.gimme_numpy(f'reconstruction_bias_{self.signal_name}_annotate',
                                bonus_arg=d[self.signal_name].values)
        smear = self.gimme_numpy(f'reconstruction_smear_{self.signal_name}_annotate',
                                 bonus_arg=d[self.signal_name].values)
        mle = d[self.raw_signal_name + '_mle'] = \
            (d[self.signal_name] / bias).clip(0, None)

        # The amount that you could have been smeared by from the raw signals
        scale = mle**0.5 * smear/bias

        for bound, sign in (('min', -1),
                            ('max', +1)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)

            d[self.raw_signal_name + '_' + bound] = (
                mle + sign * 2. * scale
            ).clip(0, None)

    def _compute(self,
                 s_raw,
                 s_observed,
                 data_tensor, ptensor):
        bias = self.gimme(f'reconstruction_bias_{self.signal_name}_simulate',
                          data_tensor=data_tensor,
                          bonus_arg=s_raw,
                          ptensor=ptensor)
        mu = s_raw * bias  # reconstructed_area = bias*raw_area

        relative_smear = self.gimme(f'reconstruction_smear_{self.signal_name}_simulate',
                                    data_tensor=data_tensor,
                                    bonus_arg=s_raw,
                                    ptensor=ptensor)
        # add offset to std to avoid NaNs from norm.pdf if std = 0
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
        'reconstruction_bias_s1_simulate',
        'reconstruction_smear_s1_simulate',
        'reconstruction_bias_s1_annotate',
        'reconstruction_smear_s1_annotate',)
    model_functions = (
        's1_acceptance',
        ) + special_model_functions

    max_dim_size = {'s1_raw': 120}
    s1_smear_load = 0.01

    def s1_acceptance(self, s1, s1_min=2, s1_max=70):
        return tf.where((s1 < s1_min) | (s1 > s1_max),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    # Getting from s1_raw -> s1
    def reconstruction_bias_s1_simulate(self, s1_raw):
        """ Dummy method for pax s1 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        s1_raw is a float64, output is fd.float_type() which is a float32
        """
        return tf.ones_like(s1_raw, dtype=fd.float_type())

    def reconstruction_smear_s1_simulate(self, s1_raw):
        """ Dummy method for pax s1 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.

        Not quite a dirac delta, which will make this block an identity matrix
        but computationally intractible or at least not trivially. Need to smear
        this dirac delta out by a small loading term of 0.001. A larger s1_raw
        max_dim_size would need a smaller loading term.
        """
        return tf.zeros_like(s1_raw, dtype=fd.float_type())+self.s1_smear_load

    # Getting from s1 -> s1_raw
    def reconstruction_bias_s1_annotate(self, s1_raw):
        """ Dummy method for pax s1 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        s1_raw is a float64, output is fd.float_type() which is a float32
        """
        return tf.ones_like(s1_raw, dtype=fd.float_type())

    def reconstruction_smear_s1_annotate(self, s1_raw):
        """ Dummy method for pax s1 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.

        Not quite a dirac delta, which will make this block an identity matrix
        but computationally intractible or at least not trivially. Need to smear
        this dirac delta out by a small loading term of 0.001. A larger s1_raw
        max_dim_size would need a smaller loading term.
        """
        return tf.zeros_like(s1_raw, dtype=fd.float_type())+self.s1_smear_load

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
        'reconstruction_bias_s2_simulate',
        'reconstruction_smear_s2_simulate',
        'reconstruction_bias_s2_annotate',
        'reconstruction_smear_s2_annotate',
        )
    model_functions = (
        ('s2_acceptance',)
        + special_model_functions)

    max_dim_size = {'s2_raw': 240}
    s2_smear_load = 3e-3

    def s2_acceptance(self, s2, s2_min=2, s2_max=6000):
        return tf.where((s2 < s2_min) | (s2 > s2_max),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    # Getting from s2_raw -> s2
    def reconstruction_bias_s2_simulate(self, s2_raw):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s2_raw, dtype=fd.float_type())

    def reconstruction_smear_s2_simulate(self, s2_raw):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.

        Not quite a dirac delta, which will make this block an identity matrix
        but computationally intractible or at least not trivially. Need to smear
        this dirac delta out by a small loading term of 0.001. A larger s2_raw
        max_dim_size would need a smaller loading term.
        """
        return tf.zeros_like(s2_raw, dtype=fd.float_type())+self.s2_smear_load

    # Getting from s2 -> s2_raw
    def reconstruction_bias_s2_annotate(self, s2_raw):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        return tf.ones_like(s2_raw, dtype=fd.float_type())

    def reconstruction_smear_s2_annotate(self, s2_raw):
        """ Dummy method for pax s2 reconstruction bias spread. Overwrite
        it in source specific class. See x1t_sr1.py for example.

        Not quite a dirac delta, which will make this block an identity matrix
        but computationally intractible or at least not trivially. Need to smear
        this dirac delta out by a small loading term of 0.001. A larger s2_raw
        max_dim_size would need a smaller loading term.
        """
        return tf.zeros_like(s2_raw, dtype=fd.float_type())+self.s2_smear_load

    def _compute(self, data_tensor, ptensor,
                 s2_raw, s2):
        return super()._compute(
            s_raw=s2_raw,
            s_observed=s2,
            data_tensor=data_tensor, ptensor=ptensor)
