import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakePhotonsElectronsBinomial(fd.Block):

    do_pel_fluct = False

    depends_on = ((('quanta_produced',), 'rate_vs_quanta'),)
    dimensions = ('electrons_produced', 'photons_produced')

    special_model_functions = ('p_electron',)
    model_functions = special_model_functions

    p_electron = 0.5   # Nonsense, ER and NR sources provide specifics

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 electrons_produced, photons_produced,
                 # Dependency domain and value
                 quanta_produced, rate_vs_quanta):
        pel = self.source.gimme('p_electron', bonus_arg=quanta_produced,
                                data_tensor=data_tensor, ptensor=ptensor)

        # Create tensors with the dimensions of our final result
        # i.e. (n_events, |photons_produced|, |electrons_produced|),
        # containing:
        # ... numbers of total quanta produced
        nq = electrons_produced + photons_produced
        # ... indices in nq arrays: make sure stepping is accounted for!
        _nq_ind = tf.round(
            (nq - self.source._fetch(
                'quanta_produced_min', data_tensor=data_tensor
            )[:, o, o]) / self.source._fetch(
                'quanta_produced_steps', data_tensor=data_tensor
            )[:, o, o])
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
                                   bonus_arg=quanta_produced,
                                   data_tensor=data_tensor,
                                   ptensor=ptensor)
            pel_fluct = fd.lookup_axis1(pel_fluct, _nq_ind)
            # See issue #37 for why we use 1 - p and photons here
            return rate_nq * fd.beta_binom_pmf(
                photons_produced,
                n=nq,
                p_mean=1. - pel,
                p_sigma=pel_fluct)

        else:
            return rate_nq * tfp.distributions.Binomial(
                total_count=nq, probs=pel).prob(electrons_produced)

    def _simulate(self, d):
        d['p_el_mean'] = self.gimme_numpy('p_electron',
                                          d['quanta_produced'].values)

        if self.do_pel_fluct:
            d['p_el_fluct'] = self.gimme_numpy(
                'p_electron_fluctuation', d['quanta_produced'].values)
            d['p_el_actual'] = 1. - stats.beta.rvs(
                *fd.beta_params(1. - d['p_el_mean'], d['p_el_fluct']))
        else:
            d['p_el_fluct'] = 0.
            d['p_el_actual'] = d['p_el_mean']

        d['p_el_actual'] = np.nan_to_num(d['p_el_actual']).clip(0, 1)
        d['electrons_produced'] = stats.binom.rvs(
            n=d['quanta_produced'],
            p=d['p_el_actual'])
        d['photons_produced'] = d['quanta_produced'] - d['electrons_produced']

    def _annotate(self, d):
        for suffix in ('min', 'max', 'mle'):
            d['quanta_produced_' + suffix] = (
                d['photons_produced_' + suffix]
                + d['electrons_produced_' + suffix])


@export
class MakePhotonsElectronsBetaBinomial(MakePhotonsElectronsBinomial):
    do_pel_fluct = True

    special_model_functions = tuple(
        list(MakePhotonsElectronsBinomial.special_model_functions)
        + ['p_electron_fluctuation'])
    model_functions = special_model_functions

    @staticmethod
    def p_electron_fluctuation(nq):
        # From SR0, BBF model, right?
        # q3 = 1.7 keV ~= 123 quanta
        return 0.041 * (1. - tf.exp(-nq / 123.))
