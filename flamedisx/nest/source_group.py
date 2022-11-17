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
        self.base_source.batched_differential_rate()
