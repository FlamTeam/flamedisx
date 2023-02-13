import numpy as np
import tensorflow as tf

import flamedisx as fd
from .. import dd_migdal as fd_dd_migdal

export, __all__ = fd.exporter()


@export
class TestSource(fd.BlockModelSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirst,
        fd_dd_migdal.EnergySpectrumSecond,
        fd_dd_migdal.MakeS1S2)

    @staticmethod
    def signal_means(energy):
        s1_mean = energy
        s2_mean = 10. * energy

        return [s1_mean, s2_mean]

    @staticmethod
    def signal_covs(energy):
        s1_var = tf.sqrt(energy)
        s2_var = tf.sqrt(10. * energy)

        s1s2_cov = -0.1 * tf.sqrt(s1_var * s2_var)

        return [[s1_var, s1s2_cov], [s1s2_cov, s2_var]]

    final_dimensions = ('s1s2',)
