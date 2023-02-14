import numpy as np
import tensorflow as tf

import flamedisx as fd
from .. import dd_migdal as fd_dd_migdal

export, __all__ = fd.exporter()


@export
class NRNRSource(fd.BlockModelSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMSU,
        fd_dd_migdal.EnergySpectrumSecondMSU,
        fd_dd_migdal.MakeS1S2MSU)

    @staticmethod
    def signal_means(*args, a=11., b=1.1, c_s2_0=755.5, c_s2_1=0.605, g1=0.1131, g2=47.35):
        energy_first = args[0]
        energy_second = args[1]

        s2_mean_first = 10**(fd.tf_log10(c_s2_0) + c_s2_1 * fd.tf_log10(energy_first))
        s2_mean_second = 10**(fd.tf_log10(c_s2_0) + c_s2_1 * fd.tf_log10(energy_second))

        s1_mean_first = (a * energy_first**b - s2_mean_first / g2) * g1
        s1_mean_second = (a * energy_second**b - s2_mean_second / g2) * g1

        s1_mean_first = tf.where(s1_mean_first < 0.01, 0.01 * tf.ones_like(s1_mean_first, dtype=fd.float_type()), s1_mean_first)
        s1_mean_second = tf.where(s1_mean_second < 0.01, 0.01 * tf.ones_like(s1_mean_second, dtype=fd.float_type()), s1_mean_second)

        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        return s1_mean_first, s1_mean_second, s2_mean_first, s2_mean_second

    @staticmethod
    def signal_covs(*args, d_s1=1.2, d_s2=14., anti_corr=-0.1):
        s1_mean_first = args[0]
        s1_mean_second = args[1]

        s2_mean_first = args[2]
        s2_mean_second = args[3]

        s1_var_first = d_s1 * s1_mean_first
        s1_var_second = d_s1 * s1_mean_second

        s2_var_first = d_s2 * s2_mean_first
        s2_var_second = d_s2 * s2_mean_second

        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second

        s1s2_cov = anti_corr * tf.sqrt(s1_var * s2_var)

        return [[s1_var, s1s2_cov], [s1s2_cov, s2_var]]

    final_dimensions = ('s1s2',)
    no_step_dimensions = ('energy_second')


@export
class NRSource(fd.BlockModelSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstSS,
        fd_dd_migdal.MakeS1S2SS)

    @staticmethod
    def signal_means(energy, a=11., b=1.1, c_s2_0=755.5, c_s2_1=0.605, g1=0.1131, g2=47.35):
        s2_mean = 10**(fd.tf_log10(c_s2_0) + c_s2_1 * fd.tf_log10(energy))

        s1_mean = (a * energy**b - s2_mean / g2) * g1
        s1_mean= tf.where(s1_mean < 0.01, 0.01 * tf.ones_like(s1_mean, dtype=fd.float_type()), s1_mean)

        return s1_mean, s2_mean

    @staticmethod
    def signal_covs(*args, d_s1=1.2, d_s2=14., anti_corr=-0.1):
        s1_mean = args[0]
        s2_mean = args[1]

        s1_var = d_s1 * s1_mean

        s2_var = d_s2 * s2_mean

        s1s2_cov = anti_corr * tf.sqrt(s1_var * s2_var)

        return [[s1_var, s1s2_cov], [s1s2_cov, s2_var]]

    final_dimensions = ('s1s2',)
