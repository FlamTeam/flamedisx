import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


DEFAULT_WORK_PER_QUANTUM = 13.7e-3


@export
class MakeERQuanta(fd.Block):

    dimensions = ('quanta_produced', 'energy')
    depends_on = ((('deposited_energy',), 'rate_vs_energy'),)
    model_functions = ('work',)

    work = DEFAULT_WORK_PER_QUANTUM

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 produced_quanta,
                 # Dependency domain and value
                 deposited_energy, rate_vs_energy):

        # Assume the intial number of quanta is always the same for each energy
        work = self.gimme('work', data_tensor=data_tensor, ptensor=ptensor)
        produced_quanta_real = tf.cast(
            tf.floor(deposited_energy / work[:, o]),
            dtype=fd.float_type())

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        return tf.cast(tf.equal(produced_quanta[:, :, o],
                                produced_quanta_real[:, o, :]),
                       dtype=fd.float_type())

    def _simulate(self, d):
        work = self.gimme_numpy('work')
        d['quanta_produced_mle'] = np.floor(d['energy'].values
                                            / work).astype(np.int)

    def _annotate(self, d):
        # No bounds need to be estimated; we will consider the entire
        # energy spectrum for each event.

        # Nonetheless, it's useful to reconstruct the 'visible' energy
        # via the combined energy scale (CES
        work = self.gimme_numpy('work')
        d['e_charge_vis'] = work * (
            d['electron_detected_mle'] / d['electron_detection_eff_mle'])
        d['e_light_vis'] = work * (
            d['photon_detected_mle'] / (
                d['photon_detection_eff'] / d['penning_quenching_eff_mle']))
        d['e_vis'] = d['e_charge_vis'] + d['e_light_vis']


@export
class MakeNRQuanta(fd.Block):

    dimensions = ('quanta_produced', 'energy')
    depends_on = ((('deposited_energy',), 'rate_vs_energy'),)

    data_methods = ('work',)
    special_data_methods = ('lindhard_l',)

    work = DEFAULT_WORK_PER_QUANTUM

    @staticmethod
    def lindhard_l(e, lindhard_k=tf.constant(0.138, dtype=fd.float_type())):
        """Return Lindhard quenching factor at energy e in keV"""
        eps = e * tf.constant(11.5 * 54.**(-7./3.), dtype=fd.float_type())  # Xenon: Z = 54

        n0 = tf.constant(3., dtype=fd.float_type())
        n1 = tf.constant(0.7, dtype=fd.float_type())
        n2 = tf.constant(1.0, dtype=fd.float_type())
        p0 = tf.constant(0.15, dtype=fd.float_type())
        p1 = tf.constant(0.6, dtype=fd.float_type())

        g = n0 * tf.pow(eps, p0) + n1 * tf.pow(eps, p1) + eps
        res = lindhard_k * g/(n2 + lindhard_k * g)
        return res

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 produced_quanta,
                 # Dependency domain and value
                 deposited_energy, rate_vs_energy):

        work = self.gimme('work', data_tensor=data_tensor, ptensor=ptensor)
        mean_q_produced = (
                deposited_energy
                * self.gimme('lindhard_l', bonus_arg=deposited_energy,
                             data_tensor=data_tensor, ptensor=ptensor)
                / work[:, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        return tfp.distributions.Poisson(
            mean_q_produced[:, o, :]).prob(produced_quanta[:, :, o])

    def _simulate(self, d):
        # If you forget the .values here, you may get a Python core dump...
        energies = d['energy'].values
        # OK to use None, simulator has set defaults
        work = self.gimme_numpy('work', data_tensor=None, ptensor=None)
        lindhard_l = self.gimme_numpy('lindhard_l',
                                      bonus_arg=energies,
                                      data_tensor=None, ptensor=None)
        d['quanta_produced'] = stats.poisson.rvs(energies * lindhard_l / work)
        return d
