"""XENON1T SR0 implementation

"""
import numpy as np
import pandas as pd
import tensorflow as tf

import flamedisx as fd

export, __all__ = fd.exporter()


##
# Electron probability
##

def p_el_sr0(e_kev):
    """Return probability of created quanta to become an electron
    for different deposited energies e_kev.

    This uses the charge yield model for XENON1T SR0,
    as published in https://arxiv.org/abs/1902.11297
    (median posterior).
    """
    e_kev = tf.convert_to_tensor(e_kev, dtype=fd.float_type())

    # Parameters from Table II, for SR0
    mean_nexni = 0.15
    w_bbf = 13.8e-3
    q0 = 1.13
    q1 = 0.47
    gamma_er = 0.124 / 4
    omega_er = 31
    f_dr = 120
    delta_er = 0.24

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')

        fi = 1 / (1 + mean_nexni)
        nq = e_kev / w_bbf
        ni, nex = nq * fi, nq * (1 - fi)
        wiggle_er = gamma_er * tf.exp(-e_kev / omega_er) * f_dr ** (-delta_er)
        r_er = 1 - tf.math.log(1 + ni * wiggle_er) / (ni * wiggle_er)
        r_er /= (1 + tf.exp(-(e_kev - q0) / q1))
        p_el = ni * (1 - r_er) / nq

    # placeholder value for e = 0 (better than NaN)
    p_el = tf.where(e_kev == 0, tf.ones_like(p_el), p_el)

    return p_el


##
# Yield maps
##


s1_map, s2_map = [
    fd.InterpolatingMap(fd.get_resource(fd.pax_file(x)))
    for x in ('XENON1T_s1_xyz_ly_kr83m_SR0_pax-642_fdc-AdCorrTPF.json',
              'XENON1T_s2_xy_ly_SR0_24Feb2017.json')]


##
# Flamedisx sources
##


class SR0Source:
    # TODO: add p_el_sr0

    def random_truth(self, energies, fix_truth=None, **params):
        d = super().random_truth(energies, fix_truth=fix_truth, **params)
        # TODO: Add field distortion maps
        d['x_observed'] = d['x']
        d['y_observed'] = d['y']
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
    def electron_gain_mean(s2_relative_ly,
                           g2=11.4 / (1 - 0.63) / 0.96):
        return g2 * s2_relative_ly

    electron_gain_std = 11.4 * 0.25 / (1 - 0.63)

    @staticmethod
    def photon_detection_eff(s1_relative_ly,
                             mean_eff=0.142 / (1 + 0.219)):
        return mean_eff * s1_relative_ly

    @staticmethod
    def s1_acceptance(s1, photon_detection_eff, photon_gain_mean,
                      mean_eff=0.142 / (1 + 0.219)):
        # Both cS1 and S1 acceptance
        cs1 = mean_eff * s1 / (
            photon_detection_eff * photon_gain_mean)
        return tf.where((s1 < 2) | (s1 > 70) | (cs1 < 2),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))


@export
class SR0ERSource(SR0Source, fd.ERSource):
    pass


@export
class SR0NRSource(SR0Source, fd.NRSource):
    pass


@export
class SR0WIMPSource(SR0Source, fd.WIMPSource):
    """
    This describes an SR0-like WIMP source, not THE
    SR0 source. The time range is not changed from the default.
    """
    # WIMP settings
    es = np.geomspace(0.7, 50, 100)  # [keV]
    mw = 1e3  # GeV
    sigma_nucleon = 1e-45  # cm^2
