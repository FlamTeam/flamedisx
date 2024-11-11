import tensorflow as tf
import pickle as pkl

import flamedisx as fd
from .. import sabre as fd_sabre

export, __all__ = fd.exporter()


@export
class SABRESource(fd.BlockModelSource):
    model_blocks = (
        fd_sabre.FixedShapeEnergySpectrum,
        fd_sabre.MakePhotons,
        fd_sabre.DetectPhotoelectrons,
        fd_sabre.MakeFinalSignal)

    def __init__(self, *args, spectrum_path=None, **kwargs):
        energy_spectrum = pkl.load(open(spectrum_path, 'rb'))
        self.energies= tf.cast(energy_spectrum[0], dtype=fd.float_type())
        self.rates_vs_energy = tf.cast(energy_spectrum[1], dtype=fd.float_type())

        super().__init__(*args, **kwargs)

    @staticmethod
    def light_yield(energy, *, abs_ly=45.):
        """
        """
        ly = abs_ly
        return ly

    final_dimensions = ('integrated_charge',)
    no_step_dimensions = ()