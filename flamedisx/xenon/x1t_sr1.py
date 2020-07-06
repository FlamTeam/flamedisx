"""XENON1T SR1 implementation
"""
import numpy as np
import tensorflow as tf

from multihist import Hist1d
import wimprates

import flamedisx as fd
import json
import scipy.interpolate as itp

import matplotlib.pyplot as plt
import pdb
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
# Parameters
##
def_g1 = 0.142
def_g2 = 11.4 # don't divide by weird shits. will explode downstream.

def_p_dpe = 0.219
def_extract_eff = 0.96

def_elife = 641e3
def_drift_vel = 1.34 * 1e-4   # cm/ns, from analysis paper II

def_field = 81.

##
# Loading Pax reconstruction bias
##
pathBagS1 = ['/home/peaelle42/software/bbf/bbf/data/ReconstructionS1BiasMeanLowers_SR1_v2.json',
         '/home/peaelle42/software/bbf/bbf/data/ReconstructionS1BiasMeanUppers_SR1_v2.json']
pathBagS2 = ['/home/peaelle42/software/bbf/bbf/data/ReconstructionS2BiasMeanLowers_SR1_v2.json',
         '/home/peaelle42/software/bbf/bbf/data/ReconstructionS2BiasMeanUppers_SR1_v2.json']
def read_bias(path):
    with open(path) as fid:
        data = json.load(fid)
        y = np.asarray(data['map'])
        x = np.asarray(np.linspace(*data['coordinate_system'][0][1]))
    return x, y

def itp_bias(pathBag):
    xx, yy = read_bias(pathBag[0])
    xx1, yy1 = read_bias(pathBag[1])

    if sum(xx-xx1)!=0 :
        print('FATAL: Bias maps evaluated at different S1 or S2 areas.')
        raise

    yy_avg = (yy+yy1)/2.
    f = itp.interp1d(xx, yy_avg, kind='linear')

    return f, min(xx), max(xx)
####################

def cal_bias(s, fmap, fmap_min, fmap_max):
    aa = s.numpy()
    bb = np.argsort(aa)
    aa_sorted = aa[bb]

    cc = np.argwhere((aa_sorted>fmap_min) &
            (aa_sorted<fmap_max))
    n_low = np.size(np.argwhere(aa_sorted<=fmap_min))
    n_high = np.size(np.argwhere(aa_sorted>=fmap_max))

    aa_sel = aa_sorted[cc]

    dd = fmap(aa_sel)
    ee = np.concatenate((np.ones((n_low,1))*dd[0], dd,
        np.ones((n_high,1))*dd[-1]))

    ff = np.ones((len(aa), 1))
    ff[bb] = ee + 1.

    return tf.convert_to_tensor(np.squeeze(ff), dtype=fd.float_type())

#####################
recon_map_s1, recon_min_s1, recon_max_s1 = itp_bias(pathBagS1)
recon_map_s2, recon_min_s2, recon_max_s2 = itp_bias(pathBagS2)

##
# Flamedisx sources
##


class SR1Source:
    drift_velocity = def_drift_vel

    def random_truth(self, n_events, fix_truth=None, **params):
        d = super().random_truth(n_events, fix_truth=fix_truth, **params)

        # Add extra needed columns
        # TODO: Add FDC maps instead of posrec resolution
        d['x_observed'] = np.random.normal(d['x'].values,
                                           scale=2)  # 2cm resolution)
        d['y_observed'] = np.random.normal(d['y'].values,
                                           scale=2)  # 2cm resolution)
        return d

    def add_extra_columns(self, d):
        super().add_extra_columns(d)
        d['s2_relative_ly'] = s2_map(
             np.transpose([d['x_observed'].values,
                          d['y_observed'].values]))
        d['s1_relative_ly'] = s1_map(
            np.transpose([d['x'].values,
                          d['y'].values,
                          d['z'].values]))

    @staticmethod
    def electron_detection_eff(drift_time, *, elife=def_elife, extraction_eff=def_extract_eff):
        #TODO: include function for elife time dependency
        return extraction_eff * tf.exp(-drift_time / elife)

    @staticmethod
    def electron_gain_mean(s2_relative_ly, *, g2=def_g2/(1.-0.63)/def_extract_eff):
        return g2 * s2_relative_ly

    @staticmethod
    def electron_gain_std(s2_relative_ly, *, g2=def_g2/(1.-0.63)/def_extract_eff):
        return g2*def_extract_eff*0.25+0.*s2_relative_ly

    #TODO: implement better the double_pe_fraction or photon_detection_efficiency as parameter
    @staticmethod
    def photon_detection_eff(s1_relative_ly, g1 =def_g1):
        #g1 = 0.142 from paper
        mean_eff= g1 / (1. + def_p_dpe)
        return mean_eff * s1_relative_ly

    @staticmethod
    def s2_acceptance(s2):
        return tf.where((s2 < 100) | (s2 > 6000),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))


# ER Source for SR1
@export
class SR1ERSource(SR1Source,fd.ERSource):

    @staticmethod
    def p_electron(nq, W=13.8e-3, mean_nexni=0.15,  q0=1.13, q1=0.47,
                   gamma_er=0.031 , omega_er=31.):
        # gamma_er from paper 0.124/4
        F = tf.constant(def_field,dtype=fd.float_type())

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
    def s1_acceptance(s1, photon_detection_eff, photon_gain_mean, mean_eff=def_g1 / (1 + def_p_dpe), 
            cs1_min=3, cs1_max=70):
        #cs1_min=0., cs1_max=np.inf):
        print('s1_acceptance: cs1 min = %i, cs1 max = %f' % (cs1_min, cs1_max))
        
        recon_bias_s1 = cal_bias(s1, recon_map_s1, recon_min_s1, recon_max_s1)
        cs1 = mean_eff * recon_bias_s1  * s1 / (photon_detection_eff * photon_gain_mean)
        #cs1 = mean_eff * s1 / (photon_detection_eff * photon_gain_mean)
        return tf.where((cs1 > cs1_min) & (cs1 < cs1_max),
                        tf.ones_like(s1, dtype=fd.float_type()),
                        tf.zeros_like(s1, dtype=fd.float_type()))

    @staticmethod
    def s2_acceptance(s2, electron_detection_eff, electron_gain_mean,
        cs2b_min=50.1, cs2b_max=7940):
            #cs2b_min=0., cs2b_max=np.inf):
        print('s2_acceptance: cs2b min = %i, cs2b max = %f' % (cs2b_min, cs2b_max))
        #cs2 = (11.4/(1-0.63)/0.96) * s2 / (electron_detection_eff*electron_gain_mean)
        #cs2 = mean_eff * s2 / (electron_detection_eff*electron_gain_mean)
        print('5:03')

        #cs2 = (def_g2/def_extract_eff) * s2 / (electron_detection_eff*electron_gain_mean)

        recon_bias_s2 = cal_bias(s2, recon_map_s2, recon_min_s2, recon_max_s2)
        cs2 = (def_g2*recon_bias_s2/def_extract_eff) * s2 / (electron_detection_eff*electron_gain_mean)

        return tf.where((cs2 > cs2b_min) & (cs2 < cs2b_max),
                        tf.ones_like(s2, dtype=fd.float_type()),
                        tf.zeros_like(s2, dtype=fd.float_type()))

@export
class SR1NRSource(SR1Source, fd.NRSource):
    # TODO: Define the proper nr spectrum
    # TODO: Modify the SR1NRSource to fit AmBe data better

    def p_electron(self, nq, *,
            alpha=1.280, zeta=0.045, beta=273 * .9e-4,
            gamma=0.0141, delta=0.062,
            drift_field=def_field):
        """Fraction of detectable NR quanta that become electrons,
        slightly adjusted from Lenardo et al.'s global fit
        (https://arxiv.org/abs/1412.4417).

        Penning quenching is accounted in the photon detection efficiency.
        """
        # TODO: so to make field pos-dependent, override this entire f?
        # could be made easier...

        # prevent /0  # TODO can do better than this
        nq = nq + 1e-9

        # Note: final term depends on nq now, not energy
        # this means beta is different from lenardo et al
        nexni = alpha * drift_field ** -zeta * (1 - tf.exp(-beta * nq))
        ni = nq * 1 / (1 + nexni)

        # Fraction of ions NOT participating in recombination
        squiggle = gamma * drift_field ** -delta
        fnotr = tf.math.log(1 + ni * squiggle) / (ni * squiggle)

        # Finally, number of electrons produced..
        n_el = ni * fnotr

        return fd.safe_p(n_el / nq)


@export
class SR1WIMPSource(SR1NRSource, fd.WIMPSource):
    pass
