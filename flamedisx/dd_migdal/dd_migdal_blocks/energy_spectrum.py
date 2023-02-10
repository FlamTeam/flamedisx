import numpy as np
import pandas as pd
import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class EnergySpectrum(fd.FirstBlock):
    dimensions = ('energy',)
    model_attributes = ('energies', 'rates_vs_energy')

    #: Tensor listing energies this source can produce.
    #: Approximate the energy spectrum as a sequence of delta functions.
    energies = tf.cast(tf.linspace(2., 8., 1000),
                       dtype=fd.float_type())
    #: Tensor listing the number of events for each energy the souce produces
    #: Recall we approximate energy spectra by a sequence of delta functions.
    rates_vs_energy = tf.ones(1000, dtype=fd.float_type())

    def _compute(self, data_tensor, ptensor, *, energy):
        spectrum = tf.repeat(self.rates_vs_energy[o, :],
                             self.source.batch_size,
                             axis=0)
        return spectrum

    def mu_before_efficiencies(self, **params):
        return np.sum(fd.np_to_tf(self.rates_vs_energy))

    def domain(self, data_tensor):
        return {self.dimensions[0]: tf.repeat(fd.np_to_tf(self.energies)[o, :],
                                              self.source.batch_size,
                                              axis=0)}

    def _annotate(self, d):
        pass

    def random_truth(self, n_events, fix_truth=None, **params):
        """Return pandas dataframe with event positions and times
        randomly drawn.
        """
        # The energy spectrum is represented as a 'comb' of delta functions
        # in the differential rate calculation. To get simulator-data agreement,
        # we have to do the same thing here.
        # TODO: can we fix both to be better?
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy)
        assert len(spectrum_numpy) == len(self.energies), \
            "Energies and spectrum have different length"

        data = dict()
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

        data = pd.DataFrame(data)
        return data

    def validate_fix_truth(self, d):
        """
        """
        if d is not None:
            raise RuntimeError("Currently don't support fix_truth for this source")
        return dict()
