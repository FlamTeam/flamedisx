"""XENON1T SR1 implementation
"""
import numpy as np
import tensorflow as tf

from multihist import Hist1d
import wimprates

import flamedisx as fd

export, __all__ = fd.exporter()

o = tf.newaxis

##
# Yield maps
##


s1_map, s2_map = [
    fd.InterpolatingMap(fd.get_resource(fd.pax_file(x)))
    for x in ('XENON1T_s1_xyz_ly_kr83m-SR1_pax-664_fdc-adcorrtpf.json',
              'XENON1T_s2_xy_ly_SR1_v2.2.json')]


##
# Flamedisx sources
##


@export
class SR1Source:
    drift_velocity = 1.335 * 1e-4   # cm/ns
    extra_needed_columns = tuple(
        list(fd.ERSource.extra_needed_columns)
        + ['x_observed', 'y_observed'])

    def random_truth(self, n_events, fix_truth=None, **params):
        d = super().random_truth(n_events, fix_truth=fix_truth, **params)

        # Add extra needed columns
        # TODO: Add FDC maps instead of posrec resolution
        d['x_observed'] = np.random.normal(d['x'].values,
                                           scale=2)  # 2cm resolution)
        d['y_observed'] = np.random.normal(d['y'].values,
                                           scale=2)  # 2cm resolution)
        return d

    @staticmethod
    def add_extra_columns(d):
        d['s2_relative_ly'] = s2_map(
             np.transpose([d['x_observed'].values,
                          d['y_observed'].values]))
        d['s1_relative_ly'] = s1_map(
            np.transpose([d['x'].values,
                          d['y'].values,
                          d['z'].values]))

    @staticmethod
    def electron_detection_eff(drift_time, *, elife=650e3, extraction_eff=0.96):
        #TODO: include function for elife time dependency
        return extraction_eff * tf.exp(-drift_time / elife)

    @staticmethod
    def electron_gain_mean(s2_relative_ly, *, g2=11.4/(1.-0.63)/0.96):
        return g2 * s2_relative_ly

    @staticmethod
    def electron_gain_std(s2_relative_ly, *, g2=11.4/(1.-0.63)/0.96):
        return g2*0.96*0.25+0.*s2_relative_ly    

    #TODO: implement better the double_pe_fraction or photon_detection_efficiency as parameter
    @staticmethod
    def photon_detection_eff(s1_relative_ly, g1 =0.123): 
        #g1 = 0.142 from paper
        mean_eff= g1 / (1. + 0.219)
        return mean_eff * s1_relative_ly
    


# ER Source for SR1
@export
class SR1ERSource(SR1Source,fd.ERSource):

    @staticmethod
    def p_electron(nq, W=13.8e-3, mean_nexni=0.135,  q0=1.13, q1=0.47, 
                   gamma_er=0.031 , omega_er=31.): 
        # gamma_er from paper 0.124/4

        F = tf.constant(81.,dtype=fd.float_type())
        
        e_kev = nq * W
        fi = 1. / (1. + mean_nexni)
        ni, nex = nq * fi, nq * (1. - fi)
        wiggle_er = gamma_er * tf.exp(-e_kev / omega_er) * F ** (-0.24) 
        # delta_er and gamma_er are highly correlated
        # F **(-delta_er) set to constant
        r_er = 1. - tf.math.log(1. + ni * wiggle_er) / (ni * wiggle_er)
        r_er /= (1. + tf.exp(-(e_kev - q0) / q1))
        p_el = ni * (1. - r_er) / nq
        
        return fd.safe_p(p_el)

    @staticmethod
    def p_electron_fluctuation(nq, q2=0.034, q3_nq=123. ):
        # From SR0, BBF model, right?
        # q3 = 1.7 keV ~= 123 quanta
        # For SR1:
        return tf.clip_by_value(q2 * (tf.constant(1.,dtype=fd.float_type()) - tf.exp(-nq / q3_nq)),
                                tf.constant(1e-4,dtype=fd.float_type()),
                                float('inf'))
    @staticmethod
    def s2_acceptance(s2):
        return tf.where((s2 < 500) | (s2 > 6000),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

class SR1NRSource(SR1Source, fd.NRSource):
    extra_needed_columns = tuple(set(
        list(SR1Source.extra_needed_columns) + 
        list(fd.NRSource.extra_needed_columns)))

# TODO: Modify the SR1NRSource to fit AmBe data better and add WIMPSource
