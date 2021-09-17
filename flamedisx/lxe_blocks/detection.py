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
            p = p * self.gimme('penning_quenching_eff',
                               bonus_arg=quanta_produced,
                               data_tensor=data_tensor, ptensor=ptensor)

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
                'penning_quenching_eff', d['photons_produced'].values)

        d[self.quanta_name + 's_detected'] = stats.binom.rvs(
            n=d[self.quanta_name + 's_produced'],
            p=p)
        d['p_accepted'] *= self.gimme_numpy(
            self.quanta_name + '_acceptance',
            d[self.quanta_name + 's_detected'].values)

    def _annotate(self, d):
        # Get efficiency
        effs = self.gimme_numpy(self.quanta_name + '_detection_eff')
        if self.quanta_name == 'photon':
            effs *= self.gimme_numpy('penning_quenching_eff',
                                    d['photons_detected_mle'].values / effs)

        # Check for bad efficiencies
        if self.check_efficiencies and np.any(effs <= 0):
            raise ValueError(f"Found event with nonpositive {self.quanta_name} "
                             "detection efficiency: did you apply and "
                             "configure your cuts correctly?")

        for suffix, bound in (('_min', 'lower'),
                               ('_max', 'upper'),
                               ('_mle', 'mle')):
            out_bounds = d[self.quanta_name + 's_detected' + suffix]
            supports = [np.linspace(out_bound, np.ceil(out_bound / eff * 10),
                                                       1000).astype(int) for out_bound, eff in zip(out_bounds, effs)]
            ns = supports
            ps = [eff * np.ones_like(support) for eff, support in zip(effs, supports)]
            rvs = [out_bound * np.ones_like(support) for out_bound, support in zip (out_bounds, supports)]

            self.bayes_bounds_binomial(d, self.quanta_name + 's_produced', supports=supports,
                                       rvs_binom=rvs, ns_binom=ns, ps_binom=ps, bound=bound)


@export
class DetectPhotons(DetectPhotonsOrElectrons):
    dimensions = ('photons_produced', 'photons_detected')

    special_model_functions = ('photon_acceptance', 'penning_quenching_eff')
    model_functions = ('photon_detection_eff',) + special_model_functions

    photon_detection_eff = 0.1

    def photon_acceptance(self, photons_detected, min_photons=3):
        return tf.where(
            photons_detected < min_photons,
            tf.zeros_like(photons_detected, dtype=fd.float_type()),
            tf.ones_like(photons_detected, dtype=fd.float_type()))

    quanta_name = 'photon'

    @staticmethod
    def penning_quenching_eff(nph):
        return 1. + 0. * nph

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

    @staticmethod
    def electron_detection_eff(drift_time, *,
                               elife=452e3, extraction_eff=0.96):
        return extraction_eff * tf.exp(-drift_time / elife)

    electron_acceptance = 1.

    quanta_name = 'electron'

    def _compute(self, data_tensor, ptensor,
                 electrons_produced, electrons_detected):
        return super()._compute(quanta_produced=electrons_produced,
                                quanta_detected=electrons_detected,
                                data_tensor=data_tensor, ptensor=ptensor)
