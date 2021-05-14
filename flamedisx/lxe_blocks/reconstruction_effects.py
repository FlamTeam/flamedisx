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

    signal_name: str

    def _simulate(self, d):
        d[self.signal_name+'_observed'] = stats.norm.rvs(
                loc=d[self.signal_name]*self.gimme_numpy('reconstruction_bias_'+self.signal_name,
                    bonus_arg=d[self.signal_name]),
                scale=d[self.signal_name]*self.gimme_numpy('reconstruction_smear_'+self.signal_name,
                    bonus_arg=d[self.signal_name]))

        ''' Will uncomment later, don't wanna touch finals_signal.py in this commit yet
        # Call add_extra_columns now, since s1 and s2 are known and derived
        # observables from it (cs1, cs2) might be used in the acceptance.
        # TODO: This is a bit of a kludge
        self.source.add_extra_columns(d)
        d['p_accepted'] *= self.gimme_numpy(self.signal_name + '_acceptance')
        '''

    def _annotate(self, d):
        tf.print('PASSED: Dunno what to do here.')
        pass
    
    def _compute(self,
                 s_observed,
                 data_tensor, ptensor):
        # Computing pdf given data
        # true area = reconstructed area/(bias+1)
        # reconstruction mean and smear
        recon_mean = self.gimme('reconstruction_bias_'+self.signal_name,
                             data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=s_observed)
        recon_std = self.gimme('reconstruction_smear_'+self.signal_name,
                             data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=s_observed)
        s_true = s_observed

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=recon_mean, scale=recon_std
        ).prob(s_true)

        return result

    
@export
class ReconstructS1(ReconstructSignals):

    signal_name = 's1'

    dimensions = ('s1',)
    special_model_functions = ('reconstruction_bias_s1',
            'reconstruction_smear_s1')
    model_functions = special_model_functions

    @staticmethod
    def reconstruction_bias_s1(sig):
        """ Dummy method for pax s1 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    @staticmethod
    def reconstruction_smear_s1(sig):
        """ Dummy method for pax s1 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_smear = 0.1*tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_smear


@export
class ReconstructS2(ReconstructSignals):

    signal_name = 's2'

    dimensions = ('s2',)
    special_model_functions = ('reconstruction_bias_s2',
            'reconstruction_smear_s2')
    model_functions = special_model_functions

    @staticmethod
    def reconstruction_bias_s2(sig):
        """ Dummy method for pax s2 reconstruction bias mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_bias = tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_bias

    @staticmethod
    def reconstruction_smear_s2(sig):
        """ Dummy method for pax s2 reconstruction smear mean. Overwrite
        it in source specific class. See x1t_sr1.py for example.
        """
        reconstruction_smear = 0.1*tf.ones_like(sig, dtype=fd.float_type())
        return reconstruction_smear
