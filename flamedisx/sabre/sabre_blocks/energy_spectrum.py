import numpy as np
import pandas as pd
import tensorflow as tf

import os

from multihist import Histdd
import pickle as pkl

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


class EnergySpectrum(fd.FirstBlock):
    dimensions = ('energy',)
    model_attributes = ('energies',)

    #: Tensor listing energies this source can produce.
    #: Approximate the energy spectrum as a sequence of delta functions.
    energies = tf.cast(tf.linspace(1., 100., 100),
                       dtype=fd.float_type())

    def domain(self, data_tensor):
        assert isinstance(self.energies, tf.Tensor)
        return {self.dimensions[0]: tf.repeat(fd.np_to_tf(self.energies)[o, :],
                                              self.source.batch_size,
                                              axis=0)}

    def _annotate(self, d):
        pass

    def random_truth(self, n_events, fix_truth=None, **params):
        """Return pandas dataframe with event positions and times
        randomly drawn.
        """
        data = pd.DataFrame()

        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy)
        assert len(spectrum_numpy) == len(self.energies), \
            "Energies and spectrum have different length"

        data['energy'] = np.random.choice(
            fd.tf_to_np(self.energies),
            size=n_events,
            p=spectrum_numpy / spectrum_numpy.sum(),
            replace=True)
        assert np.all(data['energy'] >= 0), "Generated negative energies??"

        # For a constant-shape spectrum, fixing truth values is easy:
        # we just overwrite the simulated values.
        # Fixing one does not constrain any of the others.
        self.source._overwrite_fixed_truths(data, fix_truth, n_events)

        return data

    def validate_fix_truth(self, d):
        pass

    def _compute(self, data_tensor, ptensor, **kwargs):
        raise NotImplementedError

    def mu_before_efficiencies(self, **params):
        raise NotImplementedError


@export
class FixedShapeEnergySpectrum(EnergySpectrum):
    """For a source whose energy spectrum has the same shape
    throughout space and time.

    If you add a rate variation with space, you must override draw_positions
     and mu_before_efficiencies.
     If you add a rate variation with time, you must override draw_times.
     If you add a rate variation depending on both space and time, you must
     override all of random_truth!

    By default, this uses a flat 0 - 100 keV spectrum, sampled at 100 points.
    """

    model_attributes = ('rates_vs_energy',) + EnergySpectrum.model_attributes
    model_functions = ()

    #: Tensor listing the number of events for each energy the souce produces
    #: Recall we approximate energy spectra by a sequence of delta functions.
    rates_vs_energy = tf.ones(100, dtype=fd.float_type())

    def _compute(self, data_tensor, ptensor, *, energy):
        spectrum = tf.repeat(self.rates_vs_energy[o, :],
                             self.source.batch_size,
                             axis=0)
        return spectrum

    def mu_before_efficiencies(self, **params):
        return np.sum(fd.np_to_tf(self.rates_vs_energy))