import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakeFinalSignal(fd.Block):
    model_attributes = ()  # leave it explicitly empty

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    dimensions = ('reconstructed_energy', 'energy')
    special_model_functions = ('smearing_resolution',)
    model_functions = special_model_functions

    def _simulate(self, d):
        d['reconstructed_energy'] = stats.norm.rvs(
            loc=(d['energy']),
            scale=(self.gimme_numpy('smearing_resolution', bonus_arg=d['energy'].values)))

    def _annotate(self, d):
        pass

    def _compute(self,
                 data_tensor, ptensor,
                 energy, reconstructed_energy):
        mean = energy
        std = self.gimme('smearing_resolution', bonus_arg=energy,
                         data_tensor=data_tensor,
                         ptensor=ptensor)

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
            loc=mean, scale=std + 1e-10
        ).prob(reconstructed_energy)
        return result