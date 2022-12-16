import typing as ty

import numpy as np
import pandas as pd

from copy import deepcopy

import flamedisx as fd
export, __all__ = fd.exporter()


@export
class SourceGroup:
    """"""
    def __init__(self, data=None, source_group_type=None):
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

        if type(self.base_source.model_blocks[1]) is fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronsNR:
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronsNR(self.base_source),) + \
                self.base_source.model_blocks[2:]
        elif type(self.base_source.model_blocks[1]) is fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronER:
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronER(self.base_source),) + \
                self.base_source.model_blocks[2:]
        else:
            raise RuntimeError(f"Cannot handle the current block logic passing {type(source_group_type).__name__} to SourceGroup")

        if data is not None:
            self.set_data(data)

    def set_data(self, data=None, data_is_annotated=False):
        assert data is not None, "Must pass data when calling set_data()"

        self.base_source.set_data(data, ignore_priors=True, data_is_annotated=data_is_annotated)

    def get_diff_rates(self, read_in=None):
        if read_in is not None:
            # assert set(('photons_produced', 'electrons_produced', 'energy')).issubset(self.base_source.no_step_dimensions)

            parts = read_in.split('_')
            self.base_source.data['electrons_produced_min'] = int(parts[6])
            self.base_source.data['electrons_produced_max'] = int(parts[7])
            self.base_source.data['photons_produced_min'] = int(parts[9])
            self.base_source.data['photons_produced_max'] = int(parts[10])
            self.base_source.data['energy_min'] = float(parts[12])
            self.base_source.data['energy_max'] = float(parts[13])

            self.base_source.model_blocks[1]._annotate_special(self.base_source.data)
            self.base_source._calculate_dimsizes()

            data = self.base_source.data[:self.base_source.n_events].copy()
            self.base_source.set_data(data, data_is_annotated=True)

            energies, diff_rates = self.base_source.batched_differential_rate(read_in=read_in, autograph=False)

        else:
            energies, diff_rates = self.base_source.batched_differential_rate()

        energies_diff_rates_all = []
        for es, drs in zip(energies[:self.base_source.n_events], diff_rates[:self.base_source.n_events]):
            energies_diff_rates = [(energy, diff_rate) for energy, diff_rate in zip(es, drs)]
            energies_diff_rates_all.append(energies_diff_rates)

        self.base_source.data = self.base_source.data[:self.base_source.n_events]

        if read_in is None:
            self.base_source.data['energies_diff_rates'] = energies_diff_rates_all
        else:
            return energies_diff_rates_all

    def cache_central_block(self, central_block_class, central_energy, electrons_min, electrons_max, photons_min, photons_max, energy_min, energy_max):
        assert self.base_source.batch_size == 1, "Need the batch size of the base source to be 1"
        # assert set(('photons_produced', 'electrons_produced', 'energy')).issubset(self.base_source.no_step_dimensions)

        while True:
            data = self.base_source.simulate(1)
            if len(data) > 0:
                break

        self.base_source.set_data(data, _skip_tf_init=True)

        self.base_source.data['electrons_produced_min'] = electrons_min
        self.base_source.data['electrons_produced_max'] = electrons_max
        self.base_source.data['photons_produced_min'] = photons_min
        self.base_source.data['photons_produced_max'] = photons_max
        self.base_source.data['energy_min'] = energy_min
        self.base_source.data['energy_max'] = energy_max

        self.base_source.model_blocks[1]._annotate_special(self.base_source.data)
        self.base_source._calculate_dimsizes()

        self.base_source.set_data(self.base_source.data, data_is_annotated=True)


        for b in self.base_source.model_blocks:
            if b.__class__ != central_block_class:
                continue

            kwargs = dict()
            kwargs.update(self.base_source._domain_dict(('energy',), self.base_source.data_tensor[0]))
            kwargs['rate_vs_energy'] = self.base_source.model_blocks[0]._compute(self.base_source.data_tensor[0], None, energy=None)
            kwargs.update(self.base_source._domain_dict(b.dimensions, self.base_source.data_tensor[0]))
            kwargs.update(b._domain_dict_bonus(self.base_source.data_tensor[0]))

            write_out = f'central_block_central_energy_{central_energy}_electrons_{electrons_min}_{electrons_max}_photons_{photons_min}_{photons_max}_energy_{energy_min}_{energy_max}'
            kwargs['write_out'] = write_out

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
        this_source = deepcopy(source)

        assert self.base_source.batch_size == this_source.batch_size

        this_source.set_data(self.base_source.data, data_is_annotated=True)

        diff_rates = []
        for i_batch in range(this_source.n_batches):
            q = this_source.data_tensor[i_batch]
            if i_batch == this_source.n_batches - 1:
                energies_diff_rates = self.base_source.data['energies_diff_rates'][i_batch * self.base_source.batch_size::].values
                spectrum_values = fd.tf_to_np(this_source.model_blocks[0]._compute(q, None, energy=None))

            else:
                energies_diff_rates = self.base_source.data['energies_diff_rates'][i_batch * self.base_source.batch_size:(i_batch + 1) * self.base_source.batch_size]
                spectrum_values = fd.tf_to_np(this_source.model_blocks[0]._compute(q, None, energy=None))

            diff_rates.extend(self.scale_by_spectrum(energies_diff_rates, spectrum_values))

        return diff_rates
