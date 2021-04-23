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
    extra_dimensions = (('quanta_produced_noStep', False),)
    depends_on = ((('energy',), 'rate_vs_energy'),)
    model_functions = ('work',)

    work = DEFAULT_WORK_PER_QUANTUM

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 quanta_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy,
                 # Extra tensors for internal use
                 quanta_produced_noStep, energy_noStep):

        # Assume the intial number of quanta is always the same for each energy
        work = self.gimme('work', data_tensor=data_tensor, ptensor=ptensor)
        quanta_produced_real = tf.cast(
            tf.floor(energy_noStep / work[:, o, o]),
            dtype=fd.float_type())

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        result = tf.cast(tf.equal(quanta_produced_noStep,
        quanta_produced_real), dtype=fd.float_type())

        # Chunks to average result in, to get to same dimension as
        # quanta_produced
        chunks = int(result.shape[1] / quanta_produced.shape[1])

        # Take the average - will later multiply by the stepping, so divide
        # out by it here
        result_temp = tf.reshape(result, [result.shape[0],
        int(result.shape[1] / chunks), chunks, result.shape[2]])
        result = tf.reduce_sum(result_temp, axis=2) \
        / (self.source.steps['quanta_produced'][:, : ,o] \
        * self.source.steps['quanta_produced'][:, : ,o])

        return result

    def _simulate(self, d):
        work = self.gimme_numpy('work')
        d['quanta_produced'] = np.floor(d['energy'].values
                                        / work).astype(np.int)

    def _annotate(self, d):
        for bound in ('min', 'max'):
            d['quanta_produced_noStep_' + bound] = (
                    d['electrons_produced_' + bound]
                    + d['photons_produced_' + bound])
        annotate_ces(self, d)

    def _domain_dict_bonus(self, d):
        return domain_dict_bonus(self, d)


@export
class MakeNRQuanta(fd.Block):

    dimensions = ('quanta_produced', 'energy')
    depends_on = ((('energy',), 'rate_vs_energy'),)

    special_model_functions = ('lindhard_l',)
    model_functions = ('work',) + special_model_functions

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
                 quanta_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy):

        work = self.gimme('work', data_tensor=data_tensor, ptensor=ptensor)
        mean_q_produced = (
                energy
                * self.gimme('lindhard_l', bonus_arg=energy,
                             data_tensor=data_tensor, ptensor=ptensor)
                / work[:, o, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        return tfp.distributions.Poisson(mean_q_produced).prob(quanta_produced)

    def _simulate(self, d):
        # If you forget the .values here, you may get a Python core dump...
        energies = d['energy'].values
        work = self.gimme_numpy('work')
        lindhard_l = self.gimme_numpy('lindhard_l', bonus_arg=energies)
        d['quanta_produced'] = stats.poisson.rvs(energies * lindhard_l / work)

    def _annotate(self, d):
        annotate_ces(self, d)


def annotate_ces(self, d):
    # No bounds need to be estimated; we will consider the entire
    # energy spectrum for each event.

    # Nonetheless, it's useful to reconstruct the 'visible' energy
    # via the combined energy scale (CES
    work = self.gimme_numpy('work')
    d['e_charge_vis'] = work * d['electrons_produced_mle']
    d['e_light_vis'] = work * d['photons_produced_mle']
    d['e_vis'] = d['e_charge_vis'] + d['e_light_vis']

    for bound in ('min', 'max'):
        d['quanta_produced_' + bound] = (
                d['electrons_produced_' + bound]
                + d['photons_produced_' + bound])

def domain_dict_bonus(self, d):
    # When we sum slice of the result using quanta_produced_noStep, need it to
    # be the same size as quanta_produced
    dimsize = self.source.dimsizes['quanta_produced_noStep']
    dimsize += (self.source.dimsizes['quanta_produced'] -
    (dimsize % self.source.dimsizes['quanta_produced']))

    # Calculate cross_domains
    mi = self.source._fetch('quanta_produced_noStep_min',data_tensor=d)[:, o]
    quanta_produced_noStep_domain = mi + tf.cast(tf.range(dimsize),
    dtype=fd.float_type())
    energy_domain = self.source.domain('energy', d)

    quanta_produced_noStep = tf.repeat(quanta_produced_noStep_domain[:, :, o],
    energy_domain.shape[1], axis=2)
    energy_noStep = tf.repeat(energy_domain[:, o, :],
    quanta_produced_noStep_domain.shape[1], axis=1)

    # Return as domain_dict
    return dict({'quanta_produced_noStep': quanta_produced_noStep,
    'energy_noStep': energy_noStep})
