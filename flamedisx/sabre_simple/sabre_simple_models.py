import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
from .. import sabre_simple as fd_sabre_simple

export, __all__ = fd.exporter()


@export
class SABRESimpleSource(fd.BlockModelSource):
    model_blocks = (
        fd_sabre_simple.FixedShapeEnergySpectrum,
        fd_sabre_simple.MakeFinalSignal)

    def __init__(self, *args, energies=None, rates_vs_energy=None, **kwargs):
        self.energies= tf.cast(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.cast(rates_vs_energy, dtype=fd.float_type())

        super().__init__(*args, **kwargs)

    def smearing_resolution(self, energy, *, resolution=0.014):
        """
        """
        return resolution * tf.sqrt(energy / 1000.) * 1000.

    final_dimensions = ('reconstructed_energy',)
    no_step_dimensions = ()