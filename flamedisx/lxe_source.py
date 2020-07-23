import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis
SIGNAL_NAMES = dict(photon='s1', electron='s2')
DEFAULT_WORK_PER_QUANTUM = 13.7e-3


@export
class UniformConstantEnergy(fd.Block):

    # TODO: how to make this easily overridable?
    # Maybe energy should come directly from the source, not from a block?

    # The fiducial volume bounds for a cylindrical volume
    # default to full (2t) XENON1T dimensions
    fv_radius = 47.9   # cm
    fv_high = 0  # cm
    fv_low = -97.6  # cm

    drift_velocity = 1.335 * 1e-4   # cm/ns

    # The default boundaries are at points where the WIMP wind is at its
    # average speed.
    # This will then also be true at the midpoint of these times.
    t_start = pd.to_datetime('2019-09-01T08:28:00')
    t_stop = pd.to_datetime('2020-09-01T08:28:00')

    dimensions = ('deposited_energy',)

    energies: tf.linspace(0., 10., 1000)
    rates: tf.ones(1000, dtype=fd.float_type())

    def _compute(self,
                 data_tensor, ptensor,
                 energy):
        return fd.repeat(self.rates[o, :], self.batch_size, axis=0)

    def domain(self, data_tensor, ptensor):
        return fd.repeat(self.energies[o, :], self.batch_size, axis=0)

    def random_truth_observables(self, n_events):
        """Return dictionary with x, y, z, r, theta, drift_time
        and event_time randomly drawn.
        """
        data = self.draw_positions(n_events)

        # Draw uniform time
        data['event_time'] = np.random.uniform(
            self.t_start.value,
            self.t_stop.value,
            size=n_events)
        return data

    def draw_positions(self, n_events):
        data = dict()
        data['r'] = (np.random.rand(n_events) * self.fv_radius**2)**0.5
        data['theta'] = np.random.uniform(0, 2*np.pi, size=n_events)
        data['z'] = np.random.uniform(self.fv_low, self.fv_high,
                                      size=n_events)
        data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])

        data['drift_time'] = - data['z'] / self.drift_velocity
        return data


@export
class MakeERQuanta(fd.Block):

    dimensions = ('produced_quanta', 'energy')
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


@export
class MakeNRQuanta(fd.Block):

    dimensions = ('produced_quanta', 'energy')
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


@export
class MakePhotonsElectronsBinomial(fd.Block):

    do_pel_fluct = False

    depends_on = ((('produced_quanta',), 'rate_vs_quanta'),)
    dimensions = ('electrons_produced', 'photons_produced')

    special_model_functions = ('p_electron')

    p_electron = 0.5   # Nonsense, ER and NR sources will provide specifics

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 electrons_produced, photons_produced,
                 # Dependency domain and value
                 produced_quanta, rate_vs_quanta):
        pel = self.source.gimme('p_electron', bonus_arg=produced_quanta,
                                data_tensor=data_tensor, ptensor=ptensor)

        # Create tensors with the dimensions of our final result
        # i.e. (n_events, |photons_produced|, |electrons_produced|),
        # containing:
        # ... numbers of total quanta produced
        nq = electrons_produced + photons_produced
        # ... indices in nq arrays
        _nq_ind = nq - self.source._fetch(
            'nq_min', data_tensor=data_tensor)[:, o, o]
        # ... differential rate
        rate_nq = fd.lookup_axis1(rate_vs_quanta, _nq_ind)
        # ... probability of a quantum to become an electron
        pel = fd.lookup_axis1(pel, _nq_ind)
        # Finally, the main computation is simple:
        pel = tf.where(tf.math.is_nan(pel),
                       tf.zeros_like(pel, dtype=fd.float_type()),
                       pel)
        pel = tf.clip_by_value(pel, 1e-6, 1. - 1e-6)

        if self.do_pel_fluct:
            pel_fluct = self.gimme('p_electron_fluctuation',
                                   bonus_arg=produced_quanta,
                                   data_tensor=data_tensor,
                                   ptensor=ptensor)
            pel_fluct = fd.lookup_axis1(pel_fluct, _nq_ind)
            pel_fluct = tf.clip_by_value(pel_fluct, fd.MIN_FLUCTUATION_P, 1.)
            # See issue #37 for why we use 1 - p and photons here
            return rate_nq * fd.beta_binom_pmf(
                photons_produced,
                n=nq,
                p_mean=1. - pel,
                p_sigma=pel_fluct)

        else:
            return rate_nq * tfp.distributions.Binomial(
                total_count=nq, probs=pel).prob(electrons_produced)


class MakePhotonsElectronsBetaBinomial(MakePhotonsElectronsBinomial):
    do_pel_fluct = True

    special_model_functions = tuple(
        list(MakePhotonsElectronsBinomial.special_model_functions)
        + ['p_electron_fluctuation'])

    @staticmethod
    def p_electron_fluctuation(nq):
        # From SR0, BBF model, right?
        # q3 = 1.7 keV ~= 123 quanta
        return tf.clip_by_value(0.041 * (1. - tf.exp(-nq / 123.)),
                                fd.MIN_FLUCTUATION_P,
                                1.)


class DetectPhotons(fd.Block):
    dimensions = ('photons_produced', 'photons_detected')

    model_functions = ('photon_detection_eff', 'penning_quenching_eff')
    special_model_functions = ('photon_acceptance',)

    photon_detection_eff = 0.1
    photon_acceptance = 1.

    @staticmethod
    def penning_quenching_eff(nph):
        return 1. + 0. * nph

    def _compute(self, data_tensor, ptensor,
                 photons_produced, photons_detected):
        p = self.gimme('photon_detection_eff',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Note *= doesn't work, p will get reshaped
        p = p * self.gimme('penning_quenching_eff', bonus_arg=photons_produced,
                           data_tensor=data_tensor, ptensor=ptensor)

        result = tfp.distributions.Binomial(
            total_count=photons_produced,
            probs=tf.cast(p, dtype=fd.float_type())
        ).prob(photons_detected)
        return result * self.gimme('photon_acceptance',
                                   bonus_arg=photons_detected,
                                   data_tensor=data_tensor, ptensor=ptensor)


# TODO: Can we avoid duplication in a nice way?
class DetectElectrons(fd.Block):
    dimensions = ('electrons_produced', 'electrons_detected')

    model_functions = ('electron_detection_eff',)
    special_model_functions = ('electron_acceptance',)

    electron_detection_eff = 1.
    electron_acceptance = 1.

    def _compute(self, data_tensor, ptensor,
                 electrons_produced, electrons_detected):
        p = self.gimme('electron_detection_eff',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        result = tfp.distributions.Binomial(
            total_count=electrons_produced,
            probs=tf.cast(p, dtype=fd.float_type())
        ).prob(electrons_detected)
        return result * self.gimme('electron_acceptance',
                                   bonus_arg=electrons_detected,
                                   data_tensor=data_tensor, ptensor=ptensor)


class MakeS1Photoelectrons(fd.Block):
    dimensions = ('photons_detected', 'photoelectrons_detected')

    def _compute(self, data_tensor, ptensor,
                 photons_detected, photoelectrons_detected):
        p_dpe = self.gimme('double_pe_fraction',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Double-pe emission only creates additional photoelectrons.
        # Invalid values will get assigned p=0 later.
        extra_pe = photoelectrons_detected - photons_detected
        invalid = extra_pe < 0

        # Negative arguments would mess up tfp's Binomial
        extra_pe = tf.where(invalid,
                            tf.zeros_like(extra_pe),
                            extra_pe)

        # (N_pe - N_photons) distributed as Binom(N_photons, p=pdpe)
        result = tfp.distributions.Binomial(
                total_count=photons_detected,
                probs=tf.cast(p_dpe, dtype=fd.float_type())
            ).prob(extra_pe)

        # Set probability of extra_pe < 0 cases to 0
        return tf.where(invalid,
                        tf.zeros_like(photoelectrons_detected),
                        result)


class MakeS1(fd.Block):

    dimensions = ('photoelectrons_detected', 's1')
    model_functions = ('photon_gain_mean', 'photon_gain_std',
                       's1_acceptance')

    # TODO: Since #78, this is the gain per photo-electron, not per photon.
    # We should refactor this, probably when revisiting annotate / introduce
    # a block model structure
    photon_gain_mean = 1.
    photon_gain_std = 0.5

    s1_acceptance = 1.

    def _compute(self, data_tensor, ptensor,
                 photoelectrons_detected, s1):
        return gaussian_smearing(self,
                                 quanta_type='photon',
                                 data_tensor=data_tensor,
                                 quanta_detected=photoelectrons_detected,
                                 s_observed=s1,
                                 ptensor=ptensor)


class MakeS2(fd.Block):

    dimensions = ('electrons_detected', 's2')
    model_functions = ('electron_gain_mean', 'electron_gain_std',
                       's2_acceptance')

    @staticmethod
    def electron_gain_mean(z, *, g2=20):
        return g2 * tf.ones_like(z)

    electron_gain_std = 5.

    s2_acceptance = 1.

    def _compute(self, data_tensor, ptensor,
                 electrons_detected, s2):
        return gaussian_smearing(self,
                                 quanta_type='electrons',
                                 data_tensor=data_tensor,
                                 quanta_detected=electrons_detected,
                                 s_observed=s2,
                                 ptensor=ptensor)


def gaussian_smearing(self,
                      quanta_type,
                      quanta_detected, s_observed,
                      data_tensor, ptensor):
    # Lookup signal gain mean and std per detected quanta
    mean_per_q = self.gimme(quanta_type + '_gain_mean',
                            data_tensor=data_tensor, ptensor=ptensor)[:, o]
    std_per_q = self.gimme(quanta_type + '_gain_std',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o]

    mean = quanta_detected * mean_per_q
    std = quanta_detected**0.5 * std_per_q

    # add offset to std to avoid NaNs from norm.pdf if std = 0
    result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(s_observed[:, o])

    # Add detection/selection efficiency
    result *= self.gimme(SIGNAL_NAMES[quanta_type] + '_acceptance',
                         data_tensor=data_tensor, ptensor=ptensor)[:, o]
    return result


class ERSource(fd.BlockModelSource):
    model_blocks = (
        SimpleEnergySpectrum,
        MakeERQuanta,
        MakePhotonsElectronsBetaBinomial,
        DetectPhotons,
        MakeS1Photoelectrons,
        MakeS1,
        DetectElectrons,
        MakeS2)

    observables = tuple(SIGNAL_NAMES.values())


class NRSource(fd.BlockModelSource):
    model_blocks = (
        SimpleEnergySpectrum,
        MakeNRQuanta,
        MakePhotonsElectronsBinomial,
        DetectPhotons,
        MakeS1Photoelectrons,
        MakeS1,
        DetectElectrons,
        MakeS2)

    observables = tuple(SIGNAL_NAMES.values())
