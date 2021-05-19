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

    dimensions = ('photoelectrons_detected', 's1')
    extra_dimensions = ()
    special_model_functions = ('reconstruction_bias_s1',)
    model_functions = ('spe_res', 's1_acceptance',) + special_model_functions

    def s1_acceptance(self, s1):
        return tf.where((s1 < self.source.S1_min) | (s1 > self.source.S1_max),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def reconstruction_bias_s1(sig):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    def _simulate(self, d):
        d['s1'] = stats.norm.rvs(
            loc=(d['photoelectrons_detected']),
            scale=(d['photoelectrons_detected']**0.5
                   * self.gimme_numpy('spe_res')))

        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy('s1_acceptance')

    def _annotate(self, d):
        mle = d['photoelectrons_detected_mle'] = d['s1'].clip(0, None)
        scale = mle**0.5 * self.gimme_numpy('spe_res')

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d['photoelectrons_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)

    def _compute(self, data_tensor, ptensor,
                 photoelectrons_detected, s1):
        mean = photoelectrons_detected
        std = photoelectrons_detected ** 0.5 * self.gimme('spe_res',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(s1)

        # Add detection/selection efficiency
        result *= self.gimme('s1_acceptance',
                             data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        return result


@export
class MakeS2(MakeFinalSignals):

    signal_name = 's2'

    dimensions = ('s2_photons_detected', 's2')
    extra_dimensions = ()
    special_model_functions = ('reconstruction_bias_s2',)
    model_functions = (
        ('dpe_factor',
         'spe_res',
         's2_acceptance')
        + special_model_functions)

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

    def _simulate(self, d):
        d['s2'] = self.gimme_numpy('dpe_factor') * stats.norm.rvs(
            loc=(d['s2_photons_detected']),
            scale=(d['s2_photons_detected']**0.5
                   * self.gimme_numpy('spe_res')))

        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy('s2_acceptance')

    def _annotate(self, d):
        m = self.gimme_numpy('dpe_factor')
        s = self.gimme_numpy('dpe_factor')

        mle = d['s2_photons_detected_mle'] = \
            (d['s2'] / m).clip(0, None)

        scale = mle**0.5 * s / m * self.gimme_numpy('spe_res')

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d['s2_photons_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)

    def _compute(self, data_tensor, ptensor,
                 s2_photons_detected, s2):
        dpe_factor = self.gimme('dpe_factor',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        mean = s2_photons_detected * dpe_factor
        std = s2_photons_detected ** 0.5 * dpe_factor * self.gimme('spe_res',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(s2)

        # Add detection/selection efficiency
        result *= self.gimme('s2_acceptance',
                             data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        return result
