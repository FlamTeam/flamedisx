import tensorflow as tf

import flamedisx as fd
from .. import sabre as fd_sabre

export, __all__ = fd.exporter()


@export
class SABRESource(fd.BlockModelSource):
    model_blocks = (
        fd_sabre.FixedShapeEnergySpectrum,
        fd_sabre.MakePhotons,
        fd_sabre.MakeFinalSignal)

    @staticmethod
    def light_yield(energy, *, abs_ly=10.):
        """
        """
        ly = abs_ly
        return ly

    final_dimensions = ('integrated_charge',)
    no_step_dimensions = ()