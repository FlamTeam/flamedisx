from multihist import Hist1d
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

import flamedisx as fd
export, __all__ = fd.exporter()

from .source import data_methods, special_data_methods

o = tf.newaxis


@export
class ERSource(fd.Source):

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
    def p_electron(nq):
        return 0.5 * tf.ones_like(nq, dtype=fd.float_type())

    @staticmethod
    def p_electron_fluctuation(nq):
        return 0.01 * tf.ones_like(nq, dtype=fd.float_type())

    @staticmethod
    def penning_quenching_eff(nph):
        return tf.ones_like(nph, dtype=fd.float_type())

    # Detection efficiencies

    @staticmethod
    def electron_detection_eff(drift_time, *, elife=600e3, extraction_eff=1.):
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
        return tf.where(s1 < 2,
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def s2_acceptance(s2):
        return tf.where(s2 < 200,
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    electron_gain_mean = 20.
    electron_gain_std = 5.

    photon_gain_mean = 1.
    photon_gain_std = 0.5
    double_pe_fraction = 0.219

    def _simulate_nq(self):
        work = self.gimme('work', numpy_out=True)
        self.data['nq'] = np.floor(self.data['energy'].values / work).astype(np.int)


@export
class NRSource(ERSource):
    # TODO: needs batching as well!!
    do_pel_fluct = False
    data_methods = tuple(data_methods + ['lindhard_l'])
    special_data_methods = tuple(special_data_methods + ['lindhard_l'])

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
