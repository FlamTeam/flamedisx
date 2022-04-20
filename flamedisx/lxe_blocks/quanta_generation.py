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
    bonus_dimensions = (('quanta_produced_noStep', False),)
    depends_on = ((('energy',), 'rate_vs_energy'),)
    model_functions = ('work',)

    max_dim_size = {'quanta_produced': 100}

    work = DEFAULT_WORK_PER_QUANTUM

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 quanta_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy,
                 # Extra domains for internal use
                 quanta_produced_noStep, energy_noStep):

        # We will use the tensors without stepping througout, then reduce
        # to the correct dimensions for the stepped domains via an averaging

        # Assume the intial number of quanta is always the same for each energy
        work = self.gimme('work', data_tensor=data_tensor, ptensor=ptensor)
        quanta_produced_real = tf.cast(
            tf.floor(energy_noStep / work[:, o, o]),
            dtype=fd.float_type())

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        result = tf.cast(
            tf.equal(quanta_produced_noStep,
                     quanta_produced_real), dtype=fd.float_type())

        # Padding needed to correctly average over the unstepped quanta
        # domain to the stepped quanta domain: sum slices in equal chunks with
        # the necessary padding on either side, then average the two
        padding = int(
            tf.floor(
                tf.shape(quanta_produced_noStep)[1] /
                (tf.shape(quanta_produced)[1]-1))
        ) - tf.shape(quanta_produced_noStep)[1] % \
            (tf.shape(quanta_produced)[1]-1)

        # Do the padding
        result_pad_left = tf.pad(result, [[0, 0], [padding, 0], [0, 0]])
        result_pad_right = tf.pad(result, [[0, 0], [0, padding], [0, 0]])

        # Chunks to reshape into, to allow for summation of slices via a
        # reduce_sum
        chunks = int(tf.shape(result_pad_left)[1] / tf.shape(quanta_produced)[1])
        steps = self.source._fetch('quanta_produced_steps',
                                   data_tensor=data_tensor)

        # Average slices padding from the left, diving again by the step size
        # as this will be multiplied by later (once more than it needs to be)
        result_temp_left = tf.reshape(
            result_pad_left,
            [tf.shape(result_pad_left)[0],
                int(tf.shape(result_pad_left)[1] / chunks),
                chunks,
                tf.shape(result_pad_left)[2]])
        result_left = tf.reduce_sum(result_temp_left, axis=2) \
            / (steps[:, o, o] * steps[:, o, o])

        # Average slices padding from the right, diving again by the step size
        # as this will be multiplied by later (once more than it needs to be)
        result_temp_right = tf.reshape(
            result_pad_right,
            [tf.shape(result_pad_right)[0],
                int(tf.shape(result_pad_right)[1] / chunks),
                chunks,
                tf.shape(result_pad_right)[2]])
        result_right = tf.reduce_sum(result_temp_right, axis=2) \
            / (steps[:, o, o] * steps[:, o, o])

        # Return the average of the two
        return (result_left + result_right) / 2

    def _simulate(self, d):
        work = self.gimme_numpy('work')
        d['quanta_produced'] = np.floor(d['energy'].values
                                        / work).astype(np.int)

    def _annotate(self, d):
        d['quanta_produced_noStep_min'] = (
                d['electrons_produced_min']
                + d['photons_produced_min']).clip(1, None)
        annotate_ces(self, d)

    def _domain_dict_bonus(self, d):
        return domain_dict_bonus(self, d)

    def _calculate_dimsizes_special(self):
        return calculate_dimsizes_special(self)


@export
class MakeNRQuanta(fd.Block):

    dimensions = ('quanta_produced', 'energy')
    bonus_dimensions = (('quanta_produced_noStep', False),)
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
                 energy, rate_vs_energy,
                 # Extra domains for internal use
                 quanta_produced_noStep, energy_noStep):

        # We will use the tensors without stepping througout, then reduce
        # to the correct dimensions for the stepped domains via an averaging

        work = self.gimme('work', data_tensor=data_tensor, ptensor=ptensor)
        mean_q_produced = (
                energy_noStep
                * self.gimme('lindhard_l', bonus_arg=energy_noStep[:, 0, :],
                             data_tensor=data_tensor, ptensor=ptensor)[:, o, :]
                / work[:, o, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        result = tfp.distributions.Poisson(mean_q_produced).prob(
            quanta_produced_noStep)

        # Padding needed to correctly average over the unstepped quanta
        # domain to the stepped quanta domain: sum slices in equal chunks with
        # the necessary padding on either side, then average the two
        padding = int(
            tf.floor(
                tf.shape(quanta_produced_noStep)[1] /
                (tf.shape(quanta_produced)[1]-1))
        ) - tf.shape(quanta_produced_noStep)[1] % \
            (tf.shape(quanta_produced)[1]-1)

        # Do the padding
        result_pad_left = tf.pad(result, [[0, 0], [padding, 0], [0, 0]])
        result_pad_right = tf.pad(result, [[0, 0], [0, padding], [0, 0]])

        # Chunks to reshape into, to allow for summation of slices via a
        # reduce_sum
        chunks = int(tf.shape(result_pad_left)[1] / tf.shape(quanta_produced)[1])
        steps = self.source._fetch('quanta_produced_steps',
                                   data_tensor=data_tensor)

        # Average slices padding from the left, diving again by the step size
        # as this will be multiplied by later (once more than it needs to be)
        result_temp_left = tf.reshape(
            result_pad_left,
            [tf.shape(result_pad_left)[0],
                int(tf.shape(result_pad_left)[1] / chunks),
                chunks,
                tf.shape(result_pad_left)[2]])
        result_left = tf.reduce_sum(result_temp_left, axis=2) \
            / (steps[:, o, o] * steps[:, o, o])

        # Average slices padding from the right, diving again by the step size
        # as this will be multiplied by later (once more than it needs to be)
        result_temp_right = tf.reshape(
            result_pad_right,
            [tf.shape(result_pad_right)[0],
                int(tf.shape(result_pad_right)[1] / chunks),
                chunks,
                tf.shape(result_pad_right)[2]])
        result_right = tf.reduce_sum(result_temp_right, axis=2) \
            / (steps[:, o, o] * steps[:, o, o])

        # Return the average of the two
        return (result_left + result_right) / 2

    def _simulate(self, d):
        # If you forget the .values here, you may get a Python core dump...
        energies = d['energy'].values
        work = self.gimme_numpy('work')
        lindhard_l = self.gimme_numpy('lindhard_l', bonus_arg=energies)
        d['quanta_produced'] = stats.poisson.rvs(energies * lindhard_l / work)

    def _annotate(self, d):
        d['quanta_produced_noStep_min'] = (
                d['electrons_produced_min']
                + d['photons_produced_min']).clip(1, None)
        annotate_ces(self, d)

    def _domain_dict_bonus(self, d):
        return domain_dict_bonus(self, d)

    def _calculate_dimsizes_special(self):
        return calculate_dimsizes_special(self)


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
                + d['photons_produced_' + bound]).clip(1, None)


def domain_dict_bonus(self, d):
    # Calculate cross_domains from quanta_produced and energy
    quanta_produced_domain = self.source.domain('quanta_produced', d)
    energy_domain = self.source.domain('energy', d)
    quanta_produced = tf.repeat(quanta_produced_domain[:, :, o],
                                tf.shape(energy_domain)[1],
                                axis=2)

    # Calculate cross_domains from quanta_produced_noStep and energy
    mi = self.source._fetch('quanta_produced_noStep_min', data_tensor=d)[:, o]
    quanta_produced_noStep_domain = mi + tf.range(tf.reduce_max(
        self.source._fetch('quanta_produced_noStep_dimsizes', data_tensor=d)))

    quanta_produced_noStep = tf.repeat(
        quanta_produced_noStep_domain[:, :, o],
        tf.shape(energy_domain)[1],
        axis=2)
    energy_noStep = tf.repeat(
        energy_domain[:, o, :],
        tf.shape(quanta_produced_noStep_domain)[1],
        axis=1)

    # Return as domain_dict
    return dict({'quanta_produced': quanta_produced,
                 'quanta_produced_noStep': quanta_produced_noStep,
                 'energy_noStep': energy_noStep})


def calculate_dimsizes_special(self):
    d = self.source.data

    # Want electrons and photons to have the same stepping: choose the minimum
    # of the two for each event
    quanta_steps = (d['electrons_produced_steps'] <=
                    d['photons_produced_steps']) * \
        d['electrons_produced_steps'] \
        + (d['photons_produced_steps'] <
           d['electrons_produced_steps']) * d['photons_produced_steps']

    batch_size = self.source.batch_size
    n_batches = self.source.n_batches

    # Need the electrons/photons steps to be the same within a batch for the
    # averaging procedure in _compute to work correctly
    for i in range(n_batches):
        quanta_steps[i * batch_size: (i + 1) * batch_size + 1] = \
            max(quanta_steps[i * batch_size: (i + 1) * batch_size + 1])

    d['electrons_produced_steps'] = quanta_steps
    d['photons_produced_steps'] = quanta_steps
    d['quanta_produced_steps'] = quanta_steps
    d['quanta_produced_noStep_steps'] = 1

    # Ensure that we still cover the full electrons_produced bounds, even if
    # the stepping has changed
    electrons_produced_dimsizes = np.ceil(
        (d['electrons_produced_max'].to_numpy()
            - d['electrons_produced_min'].to_numpy()) / quanta_steps) + 1
    self.source.dimsizes['electrons_produced'] = electrons_produced_dimsizes

    # Ensure that we still cover the full photons_produced bounds, even if
    # the stepping has changed
    photons_produced_dimsizes = np.ceil(
        (d['photons_produced_max'].to_numpy()
            - d['photons_produced_min'].to_numpy()) / quanta_steps) + 1
    self.source.dimsizes['photons_produced'] = photons_produced_dimsizes

    # Correct dimsize for quanta_produced, to cover the full range that the sum
    # of electrons_produced + photons_produced can take
    quanta_produced_dimsizes = electrons_produced_dimsizes \
        + photons_produced_dimsizes - 1

    # Need the quanta_produced dimsizes to be the same within a batch for the
    # averaging procedure in _compute to work correctly
    for i in range(n_batches):
        quanta_produced_dimsizes[i * batch_size: (i + 1) * batch_size + 1] = \
            max(quanta_produced_dimsizes[i * batch_size:
                (i + 1) * batch_size + 1])

    self.source.dimsizes['quanta_produced'] = quanta_produced_dimsizes

    # Correct dimsizes for quanta_produced_noStep, to cover the full range of
    # quanta_produced with integer steps
    self.source.dimsizes['quanta_produced_noStep'] = quanta_produced_dimsizes \
        + (quanta_steps - 1) * (quanta_produced_dimsizes - 1)
