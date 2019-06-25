"""XENON1T SR0 implementation

"""
import numpy as np
import tensorflow as tf

from multihist import Hist1d
import wimprates

import flamedisx as fd

export, __all__ = fd.exporter()


##
# Electron probability
##

def p_el_thesis(e_kev, a=15, b=-27.7, c=32.5, e0=5.):
    eps = fd.tf_log10(e_kev / e0 + 1e-9)
    qy = a * eps ** 2 + b * eps + c
    return qy * 13.8e-3


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
    F = 120
    delta_er = 0.24

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')

        fi = 1 / (1 + mean_nexni)
        nq = e_kev / w_bbf
        ni, nex = nq * fi, nq * (1 - fi)
        wiggle_er = gamma_er * tf.exp(-e_kev / omega_er) * F ** (-delta_er)
        r_er = 1 - tf.math.log(1 + ni * wiggle_er) / (ni * wiggle_er)
        r_er /= (1 + tf.exp(-(e_kev - q0) / q1))
        p_el = ni * (1 - r_er) / nq

    # placeholder value for e = 0 (better than NaN)
    p_el = tf.where(e_kev == 0, tf.ones_like(p_el), p_el)

    return p_el


def p_electron_nr(
        nq,
        alpha=1.280, zeta=0.045, beta=273 * .9e-4,
        gamma=0.0141, delta=0.062,
        drift_field=120):
    # From lenardo et al global fit
    # TODO: account for Penning quenching in photon detection efficiency
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

    return n_el / nq


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


@export
class SR0Source:
    extra_needed_columns = ('x_observed', 'y_observed',
                            'x', 'y', 'z')

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
    def electron_detection_eff(drift_time,
                               *, elife=452e3, extraction_eff=0.96):
        return extraction_eff * tf.exp(-drift_time / elife)

    @staticmethod
    def electron_gain_mean(s2_relative_ly,
                           g2=11.4 / (1 - 0.63) / 0.96):
        return g2 * s2_relative_ly

    electron_gain_std = 11.4 * 0.25 / (1 - 0.63)

    @staticmethod
    def photon_detection_eff(s1_relative_ly,
                             mean_eff=0.142 / (1 + 0.219)):
        return mean_eff * s1_relative_ly


@export
class SR0ERSource(SR0Source, fd.ERSource):

    @staticmethod
    def p_electron(nq, erqy_a=15, erqy_b=-27.7, erqy_c=32.5, erqy_e0=5.):
        return fd.safe_p(p_el_thesis(nq * 13.8e-3,
                                     a=erqy_a, b=erqy_b, c=erqy_c,
                                     e0=erqy_e0))

    @staticmethod
    def p_electron_fluctuation(nq):
        # q3 = 1.7 keV ~= 123 quanta
        return tf.clip_by_value(0.041 * (1. - tf.exp(-nq / 123.)),
                                1e-4,
                                float('inf'))


# Compute events/bin spectrum for a WIMP
example_wimp_es = np.geomspace(1, 50, 100)
example_wimp_rs = wimprates.rate_wimp_std(
    example_wimp_es,
    mw=1e3, sigma_nucleon=1e-45)
example_sp = Hist1d.from_histogram(
    example_wimp_rs[:-1] * np.diff(example_wimp_es),
    example_wimp_es)


@export
class SR0NRSource(SR0Source, fd.NRSource):

    def energy_spectrum(self, drift_time):
        n_evts = len(drift_time)
        return (
            example_sp.bin_centers[np.newaxis,:].repeat(n_evts, axis=0),
            example_sp.histogram[np.newaxis,:].repeat(n_evts, axis=0))

    @staticmethod
    def p_electron(nq):
        return fd.safe_p(p_electron_nr(nq))
