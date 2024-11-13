import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class PhotonsPhotoelectrons(fd.Block):
    dimensions = ('photoelectrons_detected', 'energy')

    special_model_functions = ('eff_light_yield',)
    model_functions = special_model_functions

    eff_light_yield = 1.   # Nonsense, source provides specifics

    def _compute(self,
                 data_tensor, ptensor,
                 energy, photoelectrons_detected):
        eff_ly = self.source.gimme('eff_light_yield', bonus_arg=energy,
                                data_tensor=data_tensor, ptensor=ptensor)
        mean_yield = energy * eff_ly

        return tfp.distributions.Poisson(rate=mean_yield).prob(photoelectrons_detected)

    def _simulate(self, d):
        d['eff_ly'] = self.gimme_numpy('eff_light_yield', bonus_arg=d['energy'].values)

        d['mean_yield'] = d['eff_ly'] * d['energy']
        d['photoelectrons_detected'] = stats.poisson.rvs(mu=d['mean_yield'])

    def _annotate(self, d):
        pass
