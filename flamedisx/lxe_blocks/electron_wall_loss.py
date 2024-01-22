import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis

@export
class ElectronWallLoss(fd.Block):
    dimensions = ('electrons_produced', 'electrons_survived')

    #special_model_functions = ('electron_survival_probability')
    model_functions = ('electron_survival_probability',) #+ special_model_functions

    max_dim_size = {'electrons_produced': 100}

    quanta_name = 'electron'

    electron_survival_probability = 1.


    def _compute(self, data_tensor, ptensor,
                 electrons_produced, electrons_survived):
        p = self.gimme('electron_survival_probability',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        result = tfp.distributions.Binomial(
                total_count=electrons_produced,
                probs=tf.cast(p, dtype=fd.float_type())
            ).prob(electrons_survived)

        return result

    def _simulate(self, d):
        p = self.gimme_numpy('electron_survival_probability')

        d['electrons_survived'] = stats.binom.rvs(
            n=d['electrons_produced'], p=p)

    
    def _annotate(self, d):
        # Get efficiency
        eff = self.gimme_numpy('electron_survival_probability')

        # Estimate produced electrons
        n_prod_mle = d[self.quanta_name + 's_produced_mle'] = \
            d['electrons_survived_mle'] / eff

        # Estimating the spread in number of produced quanta is tricky since
        # the number of produced_after_loss quanta is itself uncertain.
        # TODO: where did this derivation come from again?
        q = (1 - eff) / eff
        _std = (q + (q ** 2 + 4 * n_prod_mle * q) ** 0.5) / 2

        for bound, sign, intify in (('min', -1, np.floor),
                                    ('max', +1, np.ceil)):
            d[self.quanta_name + 's_produced_' + bound] = intify(
                n_prod_mle + sign * self.source.max_sigma * _std
            ).clip(0, None).astype(int)

