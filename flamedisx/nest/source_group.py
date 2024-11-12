import numpy as np
import tensorflow as tf

from copy import deepcopy

import glob

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class SourceGroup:
    """Compute the probability of events in a dataset given a set of energies in an underlying base_source.
    These probabilities will then be stored and scaled by the corresponding energy spectrum values for a real
    source to quickly calculate differential rates given a set of source sharing the underlying liquid xenon
    response model.
    """
    def __init__(self, data=None, source_group_type=None):
        """Initialize the source group.

        :param data: Dataframe with events to use in the inference
        :param source_group_type: Instantiated class from lxe_source_groups.py, containing the relevant model
        blocks to be used in the base_source
        """
        assert isinstance(source_group_type, fd.Source), "Must pass a flamedisx source to use as the base type"

        # This would be a SpatialRateEnergySpectrum
        assert not hasattr(source_group_type, 'spatial_hist'), \
            "Logic here is best suited to a source_group_type with a flat FixedShapeEnergySpectrum"
        # This would be a VariableEnergySpectrum
        assert not hasattr(source_group_type, 'energy_spectrum'), \
            "Logic here is best suited to a source_group_type with a flat FixedShapeEnergySpectrum"
        # This would be a non-flat FixedShapeEnergySpectrum
        assert np.all(source_group_type.rates_vs_energy.numpy() == source_group_type.rates_vs_energy.numpy()[0]), \
            "Logic here is best suited to a source_group_type with a flat FixedShapeEnergySpectrum"

        self.base_source = deepcopy(source_group_type)

        # We replace the central block with a specialised version to be used in the source group computation
        if type(self.base_source.model_blocks[1]) is fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronsNR:
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronsNR(self.base_source,
                 ignore_shape_assertion=True),) + \
                self.base_source.model_blocks[2:]
        elif type(self.base_source.model_blocks[1]) is fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronER:
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronER(self.base_source,
                 ignore_shape_assertion=True),) + \
                self.base_source.model_blocks[2:]
        else:
            raise RuntimeError(f"Cannot handle the current block logic passing {type(source_group_type).__name__} "
                               "to SourceGroup")

        if data is not None:
            self.set_data(data)

    def set_data(self, data=None, data_is_annotated=False):
        assert data is not None, "Must pass data when calling set_data()"

        self.base_source.set_data(data, ignore_priors=True, data_is_annotated=data_is_annotated)

    def get_diff_rates(self):
        """Compute the probabilities of the events in our dataset for all energies that remain in the
        base_source spectrum after trimming/stepping for each batch.
        """
        # Run the computation
        energies, diff_rates = self.base_source.batched_differential_rate()

        # Compute [(energy_1, probability(event | energy_1)), (energy_2, probability(event | energy_2)), ...]
        # for every event
        energies_diff_rates_all = []
        for es, drs in zip(energies[:self.base_source.n_events], diff_rates[:self.base_source.n_events]):
            energies_diff_rates = [(energy, diff_rate) for energy, diff_rate in zip(es, drs)]
            energies_diff_rates_all.append(energies_diff_rates)

        self.base_source.data = self.base_source.data[:self.base_source.n_events]

        # Store the results in the base_source data attribute
        self.base_source.data['energies_diff_rates'] = energies_diff_rates_all

    @staticmethod
    def scale_by_spectrum(energies_diff_rates, spectrum_values):
        diff_rates = []
        for es_drs, spectrum in zip(energies_diff_rates, spectrum_values):
            this_diff_rate = 0.
            for e_dr, spectrum_value in zip(es_drs, spectrum):
                this_diff_rate += (e_dr[1] * spectrum_value)
            diff_rates.append(this_diff_rate)

        return diff_rates

    def get_diff_rate_source(self, source):
        """Compute the differential rates for all events under a real source.

        :param source: the real source the differential rates should be computed under
        """
        this_source = deepcopy(source)

        assert self.base_source.batch_size == this_source.batch_size, \
            "source_group_type and source must have the same batch size"
        assert len(fd.tf_to_np(self.base_source.energies)) == len(fd.tf_to_np(source.energies)), \
            "source_group_type and source must have the same energies in their spectra"
        assert (fd.tf_to_np(self.base_source.energies) == fd.tf_to_np(source.energies)).all(), \
            "source_group_type and source must have the same energies in their spectra"

        this_source.set_data(self.base_source.data, data_is_annotated=True)

        diff_rates = []
        for i_batch in range(this_source.n_batches):
            q = this_source.data_tensor[i_batch]
            # Grab the probabilities of events in this batch under its set of trimmed/stepped energies.
            # Grab also the spectrum values of the source under those energies
            if i_batch == this_source.n_batches - 1:
                energies_diff_rates = \
                    self.base_source.data['energies_diff_rates'][i_batch * self.base_source.batch_size::].values
                spectrum_values = fd.tf_to_np(this_source.model_blocks[0]._compute(q, None, energy=None))

            else:
                energies_diff_rates = \
                    self.base_source.data['energies_diff_rates'][i_batch *
                                                                 self.base_source.batch_size:(i_batch + 1) *
                                                                 self.base_source.batch_size]
                spectrum_values = fd.tf_to_np(this_source.model_blocks[0]._compute(q, None, energy=None))

            # Multiply the probabilities and spectrum values together and sum to obtain the differential rates
            diff_rates.extend(self.scale_by_spectrum(energies_diff_rates, spectrum_values))

        return diff_rates
