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
    def signal_means_single(energy):
        s1_mean = energy
        s2_mean = 10. * energy

        return [s1_mean, s2_mean]

    @staticmethod
    def signal_covs_single(energy):
        s1_var = tf.sqrt(energy)
        s2_var = tf.sqrt(10. * energy)

        s1s2_cov = -0.1 * tf.sqrt(s1_var * s2_var)

        return [[s1_var, s1s2_cov], [s1s2_cov, s2_var]]

    @staticmethod
    def signal_means_double(*args):
        energy_first = args[0]
        energy_second = args[1]

        s1_mean_first = energy_first
        s1_mean_second = 0.5 * energy_second

        s2_mean_first= 10. * energy_first
        s2_mean_second = 5. * energy_second

        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        return [s1_mean, s2_mean]

    @staticmethod
    def signal_covs_double(*args):
        energy_first = args[0]
        energy_second = args[1]

        s1_var_first = tf.sqrt(energy_first)
        s1_var_second = tf.sqrt(0.5 * energy_second)

        s2_var_first = tf.sqrt(10. * energy_first)
        s2_var_second = tf.sqrt(5. * energy_second)

        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second

        s1s2_cov = -0.1 * tf.sqrt(s1_var * s2_var)

        return [[s1_var, s1s2_cov], [s1s2_cov, s2_var]]

    final_dimensions = ('s1s2',)
    no_step_dimensions = ('energy_second')
