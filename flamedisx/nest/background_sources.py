"""Background sources for LXe TPCs

"""
import tensorflow as tf

import os
import numpy as np
import pandas as pd

import flamedisx as fd
from . import lxe_sources as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##


@export
class vERSource(fd_nest.nestERSource):
    """ER background source from solar neutrinos (PP+7Be+CNO).
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    Normalise such that the spectrum predicts 30.76 events in 1 tonne year.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_vER = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/vER_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_vER['energy_keV'].values, dtype=fd.float_type())
        scale = fid_mass * livetime * 30.76
        self.rates_vs_energy = tf.convert_to_tensor(df_vER['spectrum_value_norm'].values * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)
