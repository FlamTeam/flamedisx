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
    def signal_means_double(*args, a=11., b=1.1, c_s2_mean_0=755.5, c_s2_mean_1=0.605, g1=0.1131, g2=47.35):
        energy_first = args[0]
        energy_second = args[1]

        s2_mean_first = 10**(np.log10(c_s2_mean_0) + c_s2_mean_1 * np.log10(energy_first))
        s2_mean_second = 10**(np.log10(c_s2_mean_0) + c_s2_mean_1 * np.log10(energy_second))

        s1_mean_first = (a * energy_first**b - s2_mean_first / g2) * g1
        s1_mean_second = (a * energy_second**b - s2_mean_second / g2) * g1

        s1_mean_first = np.where(s1_mean_first < 0.01, 0.01, s1_mean_first)
        s1_mean_second = np.where(s1_mean_second < 0.01, 0.01, s1_mean_second)

        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        return s1_mean_first, s1_mean_second, s2_mean_first, s2_mean_second

    @staticmethod
    def signal_covs_double(*args, c_s1_var=1.2, c_s2_var=14.):
        s1_mean_first = args[0]
        s1_mean_second = args[1]

        s2_mean_first = args[2]
        s2_mean_second = args[3]

        s1_var_first = c_s1_var * s1_mean_first
        s1_var_second = c_s1_var * s1_mean_second

        s2_var_first = c_s2_var * s2_mean_first
        s2_var_second = c_s2_var * s2_mean_second

        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second

        s1s2_cov = -0.1 * tf.sqrt(s1_var * s2_var)

        return [[s1_var, s1s2_cov], [s1s2_cov, s2_var]]

    final_dimensions = ('s1s2',)
    no_step_dimensions = ('energy_second')
