"""Basic ER/NR source implementations

"""


from multihist import Hist1d
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

import flamedisx as fd
export, __all__ = fd.exporter()

from .source import data_methods, special_data_methods

o = tf.newaxis


@export
class ERSource(fd.Source):

    tpc_radius = 47.9   # cm
    tpc_length = 97.6   # cm
    drift_velocity = 1.335 * 1e-4   # cm/ns

    work = 13.7e-3

    def energy_spectrum(self, drift_time):
        """Return (energies in keV, rate at these energies),
        both (n_events, n_energies) tensors.
        """
        # TODO: doesn't depend on drift_time...
        n_evts = len(drift_time)
        return (fd.repeat(tf.cast(tf.linspace(0., 10., 1000)[o, :],
                                 dtype=fd.float_type()),
                         n_evts, axis=0),
                fd.repeat(tf.ones(1000, dtype=fd.float_type())[o, :],
                         n_evts, axis=0))

    def energy_spectrum_hist(self):
        # TODO: fails if e is pos/time dependent
        es, rs = self.gimme('energy_spectrum', numpy_out=True)
        return Hist1d.from_histogram(rs[0, :-1], es[0, :])

    def simulate_es(self, n):
        return self.energy_spectrum_hist().get_random(n)

    @staticmethod
    def p_electron(nq, *, er_pel_a=15, er_pel_b=-27.7, er_pel_c=32.5,
                   er_pel_e0=5.):
        """Fraction of ER quanta that become electrons
        Simplified form from Jelle's thesis
        """
        # The original model depended on energy, but in flamedisx
        # it has to be a direct function of nq.
        e_kev_sortof = nq * 13.7e-3
        eps = fd.tf_log10(e_kev_sortof / er_pel_e0 + 1e-9)
        qy = (
            er_pel_a * eps ** 2
            + er_pel_b * eps
            + er_pel_c)
        return fd.safe_p(qy * 13.7e-3)

    @staticmethod
    def p_electron_fluctuation(nq):
        # From SR0, BBF model, right?
        # q3 = 1.7 keV ~= 123 quanta
        return tf.clip_by_value(0.041 * (1. - tf.exp(-nq / 123.)),
                                1e-4,
                                float('inf'))

    @staticmethod
    def penning_quenching_eff(nph):
        return tf.ones_like(nph, dtype=fd.float_type())

    # Detection efficiencies
    @staticmethod
    def electron_detection_eff(drift_time, *, elife=452e3, extraction_eff=0.96):
        return extraction_eff * tf.exp(-drift_time / elife)

    photon_detection_eff = 0.1

    # Acceptance of selection/detection on photons/electrons detected
    # The min_xxx attributes are also used in the bound computations
    min_s1_photons_detected = 3.
    min_s2_electrons_detected = 3.

    def electron_acceptance(self, electrons_detected):
        return tf.where(
            electrons_detected < self.min_s2_electrons_detected,
            tf.zeros_like(electrons_detected, dtype=fd.float_type()),
            tf.ones_like(electrons_detected, dtype=fd.float_type()))

    def photon_acceptance(self, photons_detected):
        return tf.where(
            photons_detected < self.min_s1_photons_detected,
            tf.zeros_like(photons_detected, dtype=fd.float_type()),
            tf.ones_like(photons_detected, dtype=fd.float_type()))

    # Acceptance of selections on S1/S2 directly

    @staticmethod
    def s1_acceptance(s1):
        return tf.where((s1 < 2) | (s1 > 70),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def s2_acceptance(s2):
        return tf.where((s2 < 200) | (s2 > 6000),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    @staticmethod
    def electron_gain_mean(z, *, g2=20):
        return g2 * tf.ones_like(z)

    electron_gain_std = 5.

    photon_gain_mean = 1.
    photon_gain_std = 0.5
    double_pe_fraction = 0.219

    def _simulate_nq(self):
        work = self.gimme('work', numpy_out=True)
        self.data['nq'] = np.floor(self.data['energy'].values / work).astype(np.int)

    @classmethod
    def simulate_aux(cls, n_events):
        data = dict()
        data['r'] = (np.random.rand(n_events) * cls.tpc_radius)**0.5
        data['theta'] = np.random.rand(n_events)
        data['x'] = data['r'] * np.cos(data['theta'])
        data['y'] = data['r'] * np.sin(data['theta'])
        data['z'] = - np.random.rand(n_events) * cls.tpc_length
        data['drift_time'] = - data['z']/ cls.drift_velocity
        return pd.DataFrame(data)


@export
class NRSource(ERSource):
    do_pel_fluct = False
    data_methods = tuple(data_methods + ['lindhard_l'])
    special_data_methods = tuple(special_data_methods + ['lindhard_l'])

    def p_electron(self, nq, *,
            alpha=1.280, zeta=0.045, beta=273 * .9e-4,
            gamma=0.0141, delta=0.062,
            drift_field=120):
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

    @staticmethod
    def lindhard_l(e, lindhard_k=0.138):
        """Return Lindhard quenching factor at energy e in keV"""
        eps = 11.5 * e * 54**(-7/3)             # Xenon: Z = 54
        g = 3. * eps**0.15 + 0.7 * eps**0.6 + eps
        res = lindhard_k * g/(1. + lindhard_k * g)
        return res

    def energy_spectrum(self, drift_time):
        """Return (energies in keV, events at these energies),
        both (n_events, n_energies) tensors.
        """
        e = fd.repeat(tf.cast(tf.linspace(0.7, 150., 100)[o, :],
                              fd.float_type()),
                      len(drift_time), axis=0)
        return e, tf.ones_like(e, dtype=fd.float_type())

    def rate_nq(self, nq_1d, i_batch=None):
        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum', i_batch=i_batch)
        mean_q_produced = (
                es
                * self.gimme('lindhard_l', es, i_batch=i_batch)
                / self.gimme('work', i_batch=i_batch)[:, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tfp.distributions.Poisson(
            mean_q_produced[:, o, :]).prob(nq_1d[:, :, o])

        return tf.reduce_sum(p_nq_e * rate_e[:, o, :], axis=2)

    @staticmethod
    def penning_quenching_eff(nph, eta=8.2e-5 * 3.3, labda=0.8 * 1.15):
        return 1. / (1. + eta * nph ** labda)

    def _simulate_nq(self):
        work = self.gimme('work', numpy_out=True)
        lindhard_l = self.gimme('lindhard_l', self.data['energy'].values,
                                numpy_out=True)
        self.data['nq'] = stats.poisson.rvs(
            self.data['energy'].values * lindhard_l / work)
