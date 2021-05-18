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
    extra_dimensions = ()

    special_model_functions = ('photon_acceptance',)
    model_functions = ('photon_detection_eff',
                       's1_posDependence') + special_model_functions

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
    extra_dimensions = ()

    special_model_functions = ('electron_acceptance',)
    model_functions = ('electron_detection_eff',) + special_model_functions

    electron_acceptance = 1.

    quanta_name = 'electron'

    def _compute(self, data_tensor, ptensor,
                 electrons_produced, electrons_detected):
        return super()._compute(quanta_produced=electrons_produced,
                                quanta_detected=electrons_detected,
                                data_tensor=data_tensor, ptensor=ptensor)


@export
class DetectS2Photons(fd.Block):
    dimensions = ('electrons_detected', 'S2_photons_produced')
    extra_dimensions = ()

    model_functions = ('electron_gain_mean', 'electron_gain_std')

    def _compute(self, data_tensor, ptensor,
                 electrons_detected, S2_photons_produced):
        mean_per_q = self.gimme('electron_gain_mean',
                                data_tensor=data_tensor,
                                ptensor=ptensor)[:, o, o]
        std_per_q = self.gimme('electron_gain_std',
                               data_tensor=data_tensor,
                               ptensor=ptensor)[:, o, o]

        mean = electrons_detected * mean_per_q
        std = electrons_detected ** 0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(S2_photons_produced)

        return result

    def _simulate(self, d):
        d['S2_photons_produced'] = stats.norm.rvs(
            loc=(d['electrons_detected']
                 * self.gimme_numpy('electron_gain_mean')),
            scale=(d['electrons_detected']**0.5
                   * self.gimme_numpy('electron_gain_std')))

    def _annotate(self, d):
        m = self.gimme_numpy('electron_gain_mean')
        s = self.gimme_numpy('electron_gain_std')

        mle = d['electrons_detected_mle'] = \
            (d['S2_photons_produced_mle'] / m).clip(0, None)

        scale = mle**0.5 * s / m

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            d['electrons_detected_' + bound] = intify(
                mle + sign * self.source.max_sigma * scale
            ).clip(0, None).astype(np.int)
