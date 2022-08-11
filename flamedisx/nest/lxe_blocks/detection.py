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
        effs = self.gimme_numpy(self.quanta_name + '_detection_eff')
        if self.quanta_name == 'photon':
            effs *= self.gimme_numpy('s1_posDependence')
        elif self.quanta_name == 's2_photon':
            effs *= self.gimme_numpy('s2_posDependence')

        # Check for bad efficiencies
        if self.check_efficiencies and np.any(effs <= 0):
            raise ValueError(f"Found event with nonpositive {self.quanta_name} "
                             "detection efficiency: did you apply and "
                             "configure your cuts correctly?")

        for suffix, bound in (('_min', 'lower'),
                              ('_max', 'upper')):
            out_bounds = d[self.quanta_name + 's_detected' + suffix]
            supports = [np.linspace(out_bound, np.ceil(out_bound / eff * 10.),
                                    1000).astype(int) for out_bound, eff in zip(out_bounds, effs)]
            ns = supports
            ps = [eff * np.ones_like(support) for eff, support in zip(effs, supports)]
            rvs = [out_bound * np.ones_like(support)
                   for out_bound, support in zip(out_bounds, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim=self.quanta_name + 's_produced',
                                   bounds_prob=self.source.bounds_prob, bound=bound,
                                   bound_type='binomial', supports=supports,
                                   rvs_binom=rvs, ns_binom=ns, ps_binom=ps)

    def _annotate_special(self, d):
        # Here we obtain improved bounds on photons and electrons detected with a non-flat prior
        if self.quanta_name not in ('photon', 'electron'):
            return False

        for batch in range(self.source.n_batches):
            d_batch = d[batch * self.source.batch_size:(batch + 1) * self.source.batch_size]

            # Get efficiency
            effs = self.gimme_numpy(self.quanta_name + '_detection_eff')[
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size]
            if self.quanta_name == 'photon':
                effs *= self.gimme_numpy('s1_posDependence')[
                    batch * self.source.batch_size:(batch + 1) * self.source.batch_size]

            for suffix, bound in (('_min', 'lower'),
                                  ('_max', 'upper')):
                out_bounds = d_batch[self.quanta_name + 's_detected' + suffix]
                supports = [np.linspace(out_bound, np.ceil(out_bound / eff * 10.),
                                        1000).astype(int) for out_bound, eff in zip(out_bounds, effs)]
                ns = supports
                ps = [eff * np.ones_like(support) for eff, support in zip(effs, supports)]
                rvs = [out_bound * np.ones_like(support)
                       for out_bound, support in zip(out_bounds, supports)]

                fd.bounds.bayes_bounds_priors(source=self.source, batch=batch,
                                              df=d, in_dim=self.quanta_name + 's_produced',
                                              bounds_prob=self.source.bounds_prob, bound=bound,
                                              bound_type='binomial', supports=supports,
                                              rvs_binom=rvs, ns_binom=ns, ps_binom=ps)

            return True


@export
class DetectPhotons(DetectPhotonsOrElectrons):
    dimensions = ('photons_produced', 'photons_detected')

    special_model_functions = ('photon_acceptance',)
    model_functions = ('photon_detection_eff',
                       's1_posDependence') + special_model_functions

    def s1_posDependence(self, r, z):
        """
        Override for specific detector.
        """
        return tf.ones_like(r, dtype=fd.float_type())

    def photon_acceptance(self, photons_detected):
        BigPhi_alpha_SPE = 0.5 * (1. + tf.math.erf(-1. / self.source.spe_res / tf.sqrt(2.)))
        BigPhi_xi_SPE = 0.5 * (1. + tf.math.erf((self.source.S1_min - 1.) / self.source.spe_res / tf.sqrt(2.)))
        sPE_belowThresh_percentile = (BigPhi_xi_SPE - BigPhi_alpha_SPE) / (1. - BigPhi_alpha_SPE)

        BigPhi_alpha_DPE = 0.5 * (1. + tf.math.erf(-2. / (tf.sqrt(2.) * self.source.spe_res) / tf.sqrt(2.)))
        BigPhi_xi_DPE = 0.5 * (1. + tf.math.erf((self.source.S1_min - 2.) / (tf.sqrt(2.) * self.source.spe_res) / tf.sqrt(2.)))
        dPE_belowThresh_percentile = (BigPhi_xi_DPE - BigPhi_alpha_DPE) / (1. - BigPhi_alpha_DPE)

        belowThresh_percentile = sPE_belowThresh_percentile * (1. - self.source.double_pe_fraction) + \
            dPE_belowThresh_percentile * self.source.double_pe_fraction

        eff = tf.where(self.source.spe_eff < 1.,
                       self.source.spe_eff + (1. - self.source.spe_eff) / (2. * self.source.num_pmts) * photons_detected,
                       self.source.spe_eff)
        eff_trunc = tf.cast(tf.where(eff > 1., 1., eff), fd.float_type())

        p = eff_trunc * (1. - belowThresh_percentile)

        coin_table = self.source.coin_table

        probs = tfp.distributions.Binomial(total_count=tf.cast(photons_detected, fd.float_type()), probs=tf.cast(p, dtype=fd.float_type())).prob(1) * coin_table[0] \
            + tfp.distributions.Binomial(total_count=tf.cast(photons_detected, fd.float_type()), probs=tf.cast(p, dtype=fd.float_type())).prob(2) * coin_table[1] \
            + tfp.distributions.Binomial(total_count=tf.cast(photons_detected, fd.float_type()), probs=tf.cast(p, dtype=fd.float_type())).prob(3) * coin_table[2] \
            + tfp.distributions.Binomial(total_count=tf.cast(photons_detected, fd.float_type()), probs=tf.cast(p, dtype=fd.float_type())).prob(4) * coin_table[3] \
            + tfp.distributions.Binomial(total_count=tf.cast(photons_detected, fd.float_type()), probs=tf.cast(p, dtype=fd.float_type())).prob(5) * coin_table[4] \

        return tf.where(
            photons_detected <= 5,
            probs,
            tf.ones_like(photons_detected, dtype=fd.float_type()))

        # return tf.where(
        #     photons_detected < self.source.min_photons,
        #     tf.zeros_like(photons_detected, dtype=fd.float_type()),
        #     tf.ones_like(photons_detected, dtype=fd.float_type()))

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
