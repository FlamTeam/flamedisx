import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakePhotons(fd.Block):

    depends_on = ((('energy',), 'rate_vs_energy'),)
    dimensions = ('photons_produced', 'energy')

    special_model_functions = ('light_yield',)
    model_functions = special_model_functions

    light_yield = 1.   # Nonsense, source provides specifics

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 photons_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy):
        ly = self.source.gimme('light_yield', bonus_arg=energy,
                                data_tensor=data_tensor, ptensor=ptensor)
        mean_yield = energy * ly

        return rate_vs_energy[:, o, :] * tfp.distributions.Poisson(
                rate=mean_yield).prob(photons_produced)

    def _simulate(self, d):
        d['ly'] = self.gimme_numpy('light_yield', bonus_arg=d['energy'].values)

        d['mean_yield'] = d['ly'] * d['energy']
        d['photons_produced'] = stats.poisson.rvs(mu=d['mean_yield'])

    def _annotate(self, d):
        pass
