import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


class DetectPhotonsOrElectrons(fd.Block):
    """Common code for DetectPhotons and DetectElectrons"""

    model_attributes = ('check_efficiencies',)

    # Whether to check if all events have a positive detection efficiency.
    # As with check_acceptances in MakeFinalSignals, you may have to
    # turn this off, depending on your application.
    check_efficiencies = True

    quanta_name: str

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _compute(self, data_tensor, ptensor,
                 quanta_produced, quanta_detected):
        p = self.gimme(self.quanta_name + '_detection_eff',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        if self.quanta_name == 'photon':
            # Note *= doesn't work, p will get reshaped
            p = p * self.gimme('s1_posDependence',
                               data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        elif self.quanta_name == 's2_photon':
            p = p * self.gimme('s2_posDependence',
                               data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        result = tfp.distributions.Binomial(
                total_count=quanta_produced,
                probs=tf.cast(p, dtype=fd.float_type())
            ).prob(quanta_detected)
        acceptance = self.gimme(self.quanta_name + '_acceptance',
                                bonus_arg=quanta_detected,
                                data_tensor=data_tensor, ptensor=ptensor)
        return result * acceptance

    def _simulate(self, d):
        p = self.gimme_numpy(self.quanta_name + '_detection_eff')

        if self.quanta_name == 'photon':
            p *= self.gimme_numpy(
                's1_posDependence')
        elif self.quanta_name == 's2_photon':
            p *= self.gimme_numpy(
                's2_posDependence')

        d[self.quanta_name + 's_detected'] = stats.binom.rvs(
            n=d[self.quanta_name + 's_produced'],
            p=p)
        d['p_accepted'] *= self.gimme_numpy(
            self.quanta_name + '_acceptance',
            d[self.quanta_name + 's_detected'].values)

    def _annotate(self, d):
        # Get efficiency
        eff = self.gimme_numpy(self.quanta_name + '_detection_eff')
        if self.quanta_name == 'photon':
            eff *= self.gimme_numpy('s1_posDependence')
        elif self.quanta_name == 's2_photon':
            eff *= self.gimme_numpy('s2_posDependence')

        # Check for bad efficiencies
        if self.check_efficiencies and np.any(eff <= 0):
            raise ValueError(f"Found event with nonpositive {self.quanta_name} "
                             "detection efficiency: did you apply and "
                             "configure your cuts correctly?")

        # Estimate produced quanta
        n_prod_mle = d[self.quanta_name + 's_produced_mle'] = \
            d[self.quanta_name + 's_detected_mle'] / eff

        # Estimating the spread in number of produced quanta is tricky since
        # the number of detected quanta is itself uncertain.
        # TODO: where did this derivation come from again?
        q = (1 - eff) / eff
        _std = (q + (q ** 2 + 4 * n_prod_mle * q) ** 0.5) / 2

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            d[self.quanta_name + 's_produced_' + bound] = intify(
                n_prod_mle + sign * self.source.max_sigma * _std
            ).clip(0, None).astype(np.int)


@export
class DetectPhotons(DetectPhotonsOrElectrons):
    dimensions = ('photons_produced', 'photons_detected')

    special_model_functions = ('photon_acceptance',)
    model_functions = ('photon_detection_eff',
                       's1_posDependence') + special_model_functions

    MC_annotate = True

    MC_annotate_dimensions = ('photons_produced',)

    def s1_posDependence(self, r, z):
        """
        Override for specific detector.
        """
        return tf.ones_like(r, dtype=fd.float_type())

    def photon_acceptance(self, photons_detected):
        return tf.where(
            photons_detected < self.source.min_photons,
            tf.zeros_like(photons_detected, dtype=fd.float_type()),
            tf.ones_like(photons_detected, dtype=fd.float_type()))

    quanta_name = 'photon'

    def _compute(self, data_tensor, ptensor,
                 photons_produced, photons_detected):
        return super()._compute(quanta_produced=photons_produced,
                                quanta_detected=photons_detected,
                                data_tensor=data_tensor, ptensor=ptensor)


@export
class DetectElectrons(DetectPhotonsOrElectrons):
    dimensions = ('electrons_produced', 'electrons_detected')

    special_model_functions = ('electron_acceptance',)
    model_functions = ('electron_detection_eff',) + special_model_functions

    MC_annotate = True

    MC_annotate_dimensions = ('electrons_produced',)

    electron_acceptance = 1.

    quanta_name = 'electron'

    def _compute(self, data_tensor, ptensor,
                 electrons_produced, electrons_detected):
        return super()._compute(quanta_produced=electrons_produced,
                                quanta_detected=electrons_detected,
                                data_tensor=data_tensor, ptensor=ptensor)


@export
class DetectS2Photons(DetectPhotonsOrElectrons):
    dimensions = ('s2_photons_produced', 's2_photons_detected')

    special_model_functions = ('s2_photon_acceptance',)
    model_functions = ('s2_photon_detection_eff',
                       's2_posDependence') + special_model_functions

    def s2_posDependence(self, r):
        """
        Override for specific detector.
        """
        return tf.ones_like(r, dtype=fd.float_type())

    s2_photon_acceptance = 1.

    quanta_name = 's2_photon'

    def _compute(self, data_tensor, ptensor,
                 s2_photons_produced, s2_photons_detected):
        return super()._compute(quanta_produced=s2_photons_produced,
                                quanta_detected=s2_photons_detected,
                                data_tensor=data_tensor, ptensor=ptensor)
