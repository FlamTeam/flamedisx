import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis


@export
class ERSource(fd.BlockModelSource):
    model_blocks = (
        fd.UniformConstantEnergy,
        fd.MakeERQuanta,
        fd.MakePhotonsElectronsBetaBinomial,
        fd.DetectPhotons,
        fd.MakeS1Photoelectrons,
        fd.MakeS1,
        fd.DetectElectrons,
        fd.MakeS2)


@export
class NRSource(fd.BlockModelSource):
    model_blocks = (
        fd.UniformConstantEnergy,
        fd.MakeNRQuanta,
        fd.MakePhotonsElectronsBinomial,
        fd.DetectPhotons,
        fd.MakeS1Photoelectrons,
        fd.MakeS1,
        fd.DetectElectrons,
        fd.MakeS2)

    observables = ('s1', 's2')
