import numpy as np
import pandas as pd
import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class EnergySpectrumFirst(fd.FirstBlock):
    dimensions = ('energy_first',)
    model_attributes = ('energies_first', 'rates_vs_energy_first')

    #: Tensor listing energies this source can produce.
    #: Approximate the energy spectrum as a sequence of delta functions.
    energies_first= tf.cast(tf.linspace(2., 8., 100),
                            dtype=fd.float_type())
    #: Tensor listing the number of events for each energy the souce produces
    #: Recall we approximate energy spectra by a sequence of delta functions.
    rates_vs_energy_first = tf.ones(100, dtype=fd.float_type())

    def _compute(self, data_tensor, ptensor, *, energy_first):
        spectrum = tf.repeat(self.rates_vs_energy_first[o, :],
                             self.source.batch_size,
                             axis=0)
        return spectrum

    def mu_before_efficiencies(self, **params):
        return np.sum(fd.np_to_tf(self.rates_vs_energy_first))

    def domain(self, data_tensor):
        return {self.dimensions[0]: tf.repeat(fd.np_to_tf(self.energies_first)[o, :],
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
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy_first)
        assert len(spectrum_numpy) == len(self.energies_first), \
            "Energies and spectrum have different length"

        data = dict()
        data['energy_first'] = np.random.choice(
            fd.tf_to_np(self.energies_first),
            size=n_events,
            p=spectrum_numpy / spectrum_numpy.sum(),
            replace=True)
        assert np.all(data['energy_first'] >= 0), "Generated negative energies??"

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


@export
class EnergySpectrumSecond(fd.Block):
    dimensions = ('energy_second',)
    model_attributes = ('energies_second', 'rates_vs_energy_second')

    #: Tensor listing energies this source can produce.
    #: Approximate the energy spectrum as a sequence of delta functions.
    energies_second= tf.cast(tf.linspace(2., 8., 100),
                            dtype=fd.float_type())
    #: Tensor listing the number of events for each energy the souce produces
    #: Recall we approximate energy spectra by a sequence of delta functions.
    rates_vs_energy_second = tf.ones(100, dtype=fd.float_type())

    def _compute(self, data_tensor, ptensor, *, energy_second):
        spectrum = tf.repeat(self.rates_vs_energy_second[o, :],
                             self.source.batch_size,
                             axis=0)
        return spectrum

    def _simulate(self, d):
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy_second)
        assert len(spectrum_numpy) == len(self.energies_second), \
            "Energies and spectrum have different length"

        d['energy_second'] = np.random.choice(
            fd.tf_to_np(self.energies_second),
            size=len(d),
            p=spectrum_numpy / spectrum_numpy.sum(),
            replace=True)
        assert np.all(d['energy_first'] >= 0), "Generated negative energies??"

    def _annotate(self, d):
        d['energy_second_min'] = fd.tf_to_np(self.energies_second)[0]
        d['energy_second_max'] = fd.tf_to_np(self.energies_second)[-1]

    def _calculate_dimsizes_special(self):
        d = self.source.data

        self.source.dimsizes['energy_second'] = len(self.energies_second)

        d_energy = np.diff(self.energies_second)
        d['energy_second_steps'] = d_energy[0]

        assert np.isclose(self.energies_second[0] + (len(self.energies_second) - 1) * d_energy[0],
                          self.energies_second[-1]), "Logic only works with constant stepping in energy spectrum"
