"""XENON1T SR0 implementation

"""
import numpy as np
import straxen

##
# Electron probability
##

def p_el_thesis(e_kev, a=15, b=-27.7, c=32.5, e0=5):
    eps = np.log10(e_kev / e0)
    qy = a * eps ** 2 + b * eps + c
    pel = qy * 13.8e-3
    return pel.clip(1e-9, 1 - 1e-9)


def p_el_sr0(e_kev):
    """Return probability of created quanta to become an electron
    for different deposited energies e_kev.

    This uses the charge yield model for XENON1T SR0,
    as published in https://arxiv.org/abs/1902.11297
    (median posterior).
    """
    e_kev = np.asarray(e_kev)

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
        wiggle_er = gamma_er * np.exp(-e_kev / omega_er) * F ** (-delta_er)
        r_er = 1 - np.log(1 + ni * wiggle_er) / (ni * wiggle_er)
        r_er /= (1 + np.exp(-(e_kev - q0) / q1))
        p_el = ni * (1 - r_er) / nq

    # placeholder value for e = 0 (better than NaN)
    p_el[e_kev == 0] = 1
    return p_el


def p_electron_nr(
        nq,
        alpha=1.280, zeta=0.045, beta=273 * .9e-4,
        gamma=0.0141, delta=0.061,
        drift_field=120):
    # From lenardo et al global fit
    # TODO: account for Penning quenching in photon detection efficiency
    # TODO: so to make field pos-dependent, override this entire f?
    # could be made easier...

    # Note: final term depends on nq now, not energy
    # this means beta is different from lenardo et al
    nexni = alpha * drift_field ** -zeta * (1 - np.exp(-beta * nq))
    ni = nq * 1 / (1 + nexni)

    # Fraction of ions NOT participating in recombination
    squiggle = gamma * drift_field ** -delta
    fnotr = np.log(1 + ni * squiggle) / (ni * squiggle)

    # Finally, number of electrons produced..
    n_el = ni * fnotr

    return n_el / nq


##
# Yield maps
##


s1_map = straxen.InterpolatingMap(
        straxen.get_resource(
            straxen.pax_file(
                'XENON1T_s1_xyz_ly_kr83m_SR0_pax-642_fdc-AdCorrTPF.json')))

s2_map = straxen.InterpolatingMap(
        straxen.get_resource(
            straxen.pax_file(
                'XENON1T_s2_xy_ly_SR0_24Feb2017.json')))


##
# Flamedisx sources
##

from flamedisx import ERSource, NRSource


class SR0ERSource(ERSource):

    @staticmethod
    def p_electron(nq):
        # result = np.ones_like(nq) * 0.9
        # mask = nq != 0
        # result[mask] = p_el_sr1(nq[mask] * 13.7e-3)
        # return result
        return p_el_thesis(nq * 13.7e-3)

    @staticmethod
    def p_electron_fluctuation(nq):
        return 0.01 * np.ones_like(nq)

    @staticmethod
    def electron_detection_eff(drift_time,
                               *, elife=452e3, extraction_eff=0.96):
        return extraction_eff * np.exp(-drift_time / elife)

    @staticmethod
    def electron_gain_mean(x_observed, y_observed, g2=11.4 / (1 - 0.63)):
        return g2 * s2_map(np.transpose([x_observed, y_observed]))

    electron_gain_std = 11.4 * 0.25 / (1 - 0.63)

    @staticmethod
    def photon_detection_eff(x, y, z, g1=0.142):
        return g1 * s1_map(np.transpose([x, y, z]))

    photon_gain_mean = 1
    photon_gain_std = 0.5



class SR0NRSource(NRSource, SR0ERSource):

    @staticmethod
    def p_electron(nq):
        return p_electron_nr(nq)