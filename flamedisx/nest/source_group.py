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

        if isinstance(self.base_source.model_blocks[1], fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronsNR):
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronsNR(self.base_source),) + \
                self.base_source.model_blocks[2:]
        elif isinstance(self.base_source.model_blocks[1], fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronER):
            self.base_source.model_blocks = (self.base_source.model_blocks[0],) + \
                (fd.nest.lxe_blocks.quanta_splitting_source_group.SGMakePhotonsElectronER(self.base_source),) + \
                self.base_source.model_blocks[2:]
        else:
            raise RuntimeError(f"Cannot handle the current block logic passing {type(source_group_type).__name__} to SourceGroup")

        if data is not None:
            self.set_data(data)

    def set_data(self, data=None):
        assert data is not None, "Must pass data when calling set_data()"

        self.base_source.set_data(data)

    def get_diff_rates(self):
        energies, diff_rates = self.base_source.batched_differential_rate()

        energies_diff_rates_all = []
        for es, drs in zip(np.concatenate(energies)[:self.base_source.n_events], np.transpose(np.concatenate(diff_rates, axis=1))[:self.base_source.n_events]):
            energies_diff_rates = [(energy, diff_rate) for energy, diff_rate in zip(es, drs)]
            energies_diff_rates_all.append(energies_diff_rates)

        self.base_source.data = self.base_source.data[:self.base_source.n_events]
        self.base_source.data['energies_diff_rates'] = energies_diff_rates_all

    def get_diff_rate_source(self, source):
        this_source = deepcopy(source)

        this_source.set_data(self.base_source.data, data_is_annotated=True)
        for i_batch in range(this_source.n_batches):
            q = this_source.data_tensor[i_batch]
            print(this_source.model_blocks[0]._compute(q, None, energy=None))
