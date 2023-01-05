import typing as ty

import numpy as np
import pandas as pd
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
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronsNR(self.base_source, ignore_shape_assertion=True),) + \
                self.base_source.model_blocks[2:]
        elif type(self.base_source.model_blocks[1]) is fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronER:
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronER(self.base_source, ignore_shape_assertion=True),) + \
                self.base_source.model_blocks[2:]
        else:
            raise RuntimeError(f"Cannot handle the current block logic passing {type(source_group_type).__name__} to SourceGroup")

        if data is not None:
            self.set_data(data)

    def set_data(self, data=None, data_is_annotated=False):
        assert data is not None, "Must pass data when calling set_data()"

        self.base_source.set_data(data, ignore_priors=True, data_is_annotated=data_is_annotated)

    def get_diff_rates(self, read_in_dir=None):
        """Compute the probabilities of the events in our dataset for all energies that remain in the
        base_source spectrum after trimming/stepping for each batch.

        :param read_in_dir: If we will be reading in central block values, path to the directory
        containing these
        """
        if read_in_dir is not None:
            quanta_tensor_dict = dict()
            electrons_full_dict = dict()
            photons_full_dict = dict()

            # We store the saved quanta tensor values and corresponding electron/photon domains
            # for every energy they were saved for
            for file in glob.glob(f'{read_in_dir}/*'):
                parts = file.split('/')[-1].split('_')

                energy = parts[3]

                electrons_min = int(parts[5])
                electrons_steps = int(parts[6])
                electrons_dimsize = int(parts[7])

                photons_min = int(parts[9])
                photons_steps = int(parts[10])
                photons_dimsize = int(parts[11])

                electrons_full = tf.repeat(((tf.range(electrons_dimsize) * electrons_steps) + electrons_min)[o, :], self.base_source.batch_size, axis=0)
                photons_full = tf.repeat(((tf.range(photons_dimsize) * photons_steps) + photons_min)[o, :], self.base_source.batch_size, axis=0)

                electrons_full_dict[energy] = electrons_full
                photons_full_dict[energy] = photons_full

                tensor_in = \
                    tf.data.TFRecordDataset(file).map(lambda x: tf.io.parse_tensor(x, out_type=fd.float_type()))
                for tensor in tensor_in:
                    quanta_tensor = tensor

                quanta_tensor_dict[energy] = quanta_tensor

            # Run the computation reading in the central block values
            energies, diff_rates = self.base_source.batched_differential_rate(quanta_tensor_dict=quanta_tensor_dict,
                                                                              electrons_full_dict=electrons_full_dict,
                                                                              photons_full_dict=photons_full_dict)
        else:
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

    def cache_central_block(self, central_block_class, energy, electrons_min, electrons_max, photons_min, photons_max):
        """Cache the P(electrons_produced, photons_produced | energy) values of the central block
        for a particular energy.

        :param central_block_class: Class corresponding to the central block for this source group's base_source
        :param energy: energy this central block is being saved for
        :param electrons_min: minimum value of the electrons_produced domain the block should be saved for
        :param electrons_max: maximum value of the electrons_produced domain the block should be saved for
        :param photons_min: minimum value of the photons_produced domain the block should be saved for
        :param photons_max: maximum value of the photons_produced domain the block should be saved for
        """
        assert self.base_source.batch_size == 1, "Need the batch size of the base source to be 1"

        # Dummy event to do dummy annotation for blocks we won't be computing
        while True:
            data = self.base_source.simulate(1)
            if len(data) > 0:
                break

        self.base_source.set_data(data, _skip_tf_init=True, ignore_priors=True)

        # Set the central block bounds to the input values
        self.base_source.data['electrons_produced_min'] = electrons_min
        self.base_source.data['electrons_produced_max'] = electrons_max
        self.base_source.data['photons_produced_min'] = photons_min
        self.base_source.data['photons_produced_max'] = photons_max
        self.base_source.data['energy_min'] = energy
        self.base_source.data['energy_max'] = energy

        # Ensure the data tensor is updated to reflect the changes above
        self.base_source.model_blocks[1]._annotate_special(self.base_source.data)
        self.base_source._calculate_dimsizes()

        self.base_source.set_data(self.base_source.data, data_is_annotated=True)

        # Store also the steps and dimsizes for electrons_produced and photons_produced to
        # be able to recreate the domains when reading in the quanta tensors
        electrons_steps = int(self.base_source.data['electrons_produced_steps'].iloc[0])
        electrons_dimsize = int(self.base_source.data['electrons_produced_dimsizes'].iloc[0])

        photons_steps = int(self.base_source.data['photons_produced_steps'].iloc[0])
        photons_dimsize = int(self.base_source.data['photons_produced_dimsizes'].iloc[0])

        # Ensure we only compute the central block
        for b in self.base_source.model_blocks:
            if b.__class__ != central_block_class:
                continue

            # Gather the kwargs for the central block computation
            kwargs = dict()
            kwargs.update(self.base_source._domain_dict(('energy',), self.base_source.data_tensor[0]))
            kwargs['rate_vs_energy'] = self.base_source.model_blocks[0]._compute(self.base_source.data_tensor[0], None, energy=None)
            kwargs.update(self.base_source._domain_dict(b.dimensions, self.base_source.data_tensor[0]))
            kwargs.update(b._domain_dict_bonus(self.base_source.data_tensor[0]))

            # Create the filename the block will be saved under
            write_out = f'central_block_energy_{energy}_electrons_{electrons_min}_{electrons_steps}_{electrons_dimsize}_photons_{photons_min}_{photons_steps}_{photons_dimsize}'
            kwargs['write_out'] = write_out

            # Compute and save the block
            b._compute(self.base_source.data_tensor[0], None, **kwargs)

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

        assert self.base_source.batch_size == this_source.batch_size, "source_group_type and source must have the same batch size"
        assert (fd.tf_to_np(self.base_source.energies) == fd.tf_to_np(source.energies)).all(), "source_group_type and source must have the same energies in their spectra"

        this_source.set_data(self.base_source.data, data_is_annotated=True)

        diff_rates = []
        for i_batch in range(this_source.n_batches):
            q = this_source.data_tensor[i_batch]
            # Grab the probabilities of events in this batch under its set of trimmed/stepped energies.
            # Grab also the spectrum values of the source under those energies
            if i_batch == this_source.n_batches - 1:
                energies_diff_rates = self.base_source.data['energies_diff_rates'][i_batch * self.base_source.batch_size::].values
                spectrum_values = fd.tf_to_np(this_source.model_blocks[0]._compute(q, None, energy=None))

            else:
                energies_diff_rates = self.base_source.data['energies_diff_rates'][i_batch * self.base_source.batch_size:(i_batch + 1) * self.base_source.batch_size]
                spectrum_values = fd.tf_to_np(this_source.model_blocks[0]._compute(q, None, energy=None))

            # Multiply the probabilities and spectrum values together and sum to obtain the differential rates
            diff_rates.extend(self.scale_by_spectrum(energies_diff_rates, spectrum_values))

        return diff_rates
