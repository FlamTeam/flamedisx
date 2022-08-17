"""Calibration sources for LXe TPCs

"""
import tensorflow as tf

import os
import pandas as pd

import flamedisx as fd
from . import lxe_sources as fd_nest

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

        df_tritium = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'calibration_spectra/tritium_spectrum.pkl'))
        self.energies = tf.convert_to_tensor(df_tritium['energies'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_tritium['rates_vs_energy'].values, dtype=fd.float_type())
