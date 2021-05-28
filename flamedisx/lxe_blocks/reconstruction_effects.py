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

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    signal_name: str

    def _simulate(self, d):
        # Simulating events
        # reconstruction_bias_mean = (bias+1)*true_area
        # `bias` and `reconstruction_bias_std` are functions of true_area and are
        # read in and interpolated from external files
        #
        # The actual reconstructed_area is then sampled from a Gaussian
        # with mean = reconstructed_area_mean and 
        # standard deviation = reconstructed_bias_std

        d[self.signal_name] = stats.norm.rvs(
                loc=d[self.signal_name+'_true']*self.gimme_numpy('reconstruction_bias_mean_'+self.signal_name,
                    bonus_arg=d[self.signal_name+'_true']),
                scale=self.gimme_numpy('reconstruction_bias_std_'+self.signal_name,
                    bonus_arg=d[self.signal_name+'_true']))

        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')

    def _annotate(self, d):
        m = self.gimme_numpy('reconstruction_bias_mean_'+self.signal_name,
                             bonus_arg=d[self.signal_name+'_true'])
        s = self.gimme_numpy('reconstruction_bias_std_'+self.signal_name,
                             bonus_arg=d[self.signal_name+'_true'])

        mle = d[self.signal_name + '_true_mle'] = \
            (d[self.signal_name] / m).clip(0, None)
        scale = mle**0.5 * s / m

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            # For detected quanta the MLE is quite accurate
            # (since fluctuations are tiny)
            # so let's just use the relative error on the MLE)
            d[self.signal_name + '_true_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)
    
    def _compute(self,
                 data_tensor, ptensor,
                 s_observed,
                 s_true):
        # Computing pdf given data
        # true_area = reverse_reconstruction_bias_mean/(reverse_bias+1)
        # 
        # `reverse_bias` and `reverse_reconstruction_bias_std` are functions of
        # reconstructed_area and are read in and interpolated from external
        # files
        #
        # Computing the probability of observing signal of size
        # reconstructed_area given that it is drawn from a Gaussian with 
        # mean = true_area = reconstruction_bias_mean/(reverse_bias+1) and
        # standard deviation = reverse_reconstruction_bias_std

        '''
        recon_mean = s_observed/self.gimme('reverse_reconstruction_bias_mean_'+self.signal_name,
                             data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=s_observed)
        recon_std = self.gimme('reverse_reconstruction_bias_std_'+self.signal_name,
                             data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=s_observed)
        '''
        # so actually conceptually you should be evaluating, at s_observed, the pdf of the
        # gaussian with mean and standard deviation from s_true. but you can't
        # get s_true as it it, so it's more like s_true_hat.
        recon_mean = s_observed
        recon_std = self.gimme('reconstruction_bias_std_'+self.signal_name,
                             data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=s_observed)

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=recon_mean, scale=recon_std + 1e-10
        ).prob(s_observed)

        # Add detection/selection efficiency, which is a function of
        # reconstructed_area
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

    signal_name = 's1'

    dimensions = ('s1_true', 's1')

    special_model_functions = ('reconstruction_bias_mean_s1',
            'reconstruction_bias_std_s1',
            'reverse_reconstruction_bias_mean_s1',
            'reverse_reconstruction_bias_std_s1')
    model_functions = ('s1_acceptance',) + special_model_functions
    
    def _compute(self, data_tensor, ptensor, s1, s1_true):
        return super()._compute(
            data_tensor=data_tensor, ptensor=ptensor,
            s_observed=s1,
            s_true=s1_true)
            
    @staticmethod
    def s1_acceptance(s1):
        return tf.where((s1 < 2) | (s1 > 70),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def reconstruction_bias_mean_s1(sig):
        """ Dummy method for pax s1 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias_mean = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias_mean

    @staticmethod
    def reconstruction_bias_std_s1(sig):
        """ Dummy method for pax s1 reconstruction bias smear. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias_std = tf.zeros_like(sig, dtype=fd.float_type())
        return reconstruction_bias_std

    @staticmethod
    def reverse_reconstruction_bias_mean_s1(sig):
        """ Dummy method for pax s1 reverse_reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reverse_reconstruction_bias_mean = tf.ones_like(sig, dtype=fd.float_type())
        return reverse_reconstruction_bias_mean

    @staticmethod
    def reverse_reconstruction_bias_std_s1(sig):
        """ Dummy method for pax s1 reverse_reconstruction bias smear. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reverse_reconstruction_bias_std = tf.zeros_like(sig, dtype=fd.float_type())
        return reverse_reconstruction_bias_std


@export
class ReconstructS2(ReconstructSignals):

    signal_name = 's2'

    dimensions = ('s2_true', 's2')

    special_model_functions = ('reconstruction_bias_mean_s2',
            'reconstruction_bias_std_s2',
            'reverse_reconstruction_bias_mean_s2',
            'reverse_reconstruction_bias_std_s2')
    model_functions = ('s2_acceptance',) + special_model_functions
    
    def _compute(self, data_tensor, ptensor, s2, s2_true):
        return super()._compute(
            data_tensor=data_tensor, ptensor=ptensor,
            s_observed=s2,
            s_true=s2_true)
            
    @staticmethod
    def s2_acceptance(s2):
        return tf.where((s2 < 200) | (s2 > 6000),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    @staticmethod
    def reconstruction_bias_mean_s2(sig):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    @staticmethod
    def reconstruction_bias_std_s2(sig):
        """ Dummy method for pax s2 reconstruction bias smear. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias_std = tf.zeros_like(sig, dtype=fd.float_type())
        return reconstruction_bias_std

    @staticmethod
    def reverse_reconstruction_bias_mean_s2(sig):
        """ Dummy method for pax s2 reverse_reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reverse_reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reverse_reconstruction_bias

    @staticmethod
    def reverse_reconstruction_bias_std_s2(sig):
        """ Dummy method for pax s2 reverse_reconstruction bias smear. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reverse_reconstruction_bias_std = tf.zeros_like(sig, dtype=fd.float_type())
        return reverse_reconstruction_bias_std
