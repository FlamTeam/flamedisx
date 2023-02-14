import numpy as np
import tensorflow as tf

import flamedisx as fd
from .. import dd_migdal as fd_dd_migdal

export, __all__ = fd.exporter()


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
    def signal_vars(*args, d_s1=1.2, d_s2=14.):
        s1_mean = args[0]
        s2_mean = args[1]

        s1_var = d_s1 * s1_mean

        s2_var = d_s2 * s2_mean

        return s1_var, s2_var

    @staticmethod
    def signal_cov(*args, anti_corr=-0.1):
        s1_var = args[0]
        s2_var = args[1]

        s1s2_cov = anti_corr * tf.sqrt(s1_var * s2_var)

        return s1s2_cov

    final_dimensions = ('s1s2',)


@export
class NRNRSource(NRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMSU,
        fd_dd_migdal.EnergySpectrumSecondMSU,
        fd_dd_migdal.MakeS1S2MSU)

    no_step_dimensions = ('energy_second')
