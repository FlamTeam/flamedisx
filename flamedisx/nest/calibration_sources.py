"""Calibration sources for LXe TPCs

"""
import tensorflow as tf

import os
import pandas as pd

import flamedisx as fd
from . import lxe_sources as fd_nest

import math as m
pi = tf.constant(m.pi)

export, __all__ = fd.exporter()


##
# Flamedisx sources
##


@export
class CH3TSource(fd_nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'
        super().__init__(*args, **kwargs)

        m_e = 510.9989461
        aa = 0.0072973525664
        ZZ = 2.
        qValue = 18.5898

        energies = tf.linspace(0.01, qValue, 1000)

        B = tf.sqrt(energies**2 + 2. * energies * m_e) / (energies + m_e)
        x = (2. * pi * ZZ * aa) * (energies + m_e) / tf.sqrt(energies**2 + 2. * energies * m_e)
        spectrum = tf.sqrt(2. * energies * m_e) * (energies + m_e) * (qValue - energies) * (qValue - energies) * x * (1. / (1. - tf.exp(-x))) * (1.002037 - 0.001427 * B)
        spectrum = spectrum / tf.math.reduce_max(spectrum)

        df_tritium = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'calibration_spectra/tritium_spectrum.pkl'))
        self.energies = tf.cast(energies, fd.float_type())
        self.rates_vs_energy = tf.cast(spectrum, fd.float_type())
