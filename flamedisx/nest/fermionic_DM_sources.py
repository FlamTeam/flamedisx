"""
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
class FermionicDMSource(fd_nest.nestNRSource):
    def __init__(self, *args, mass_MeV=57.966, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_fermionic_DM = pd.read_pickle(os.path.join(os.path.dirname(__file__), f'fermionic_DM_spectra/{mass_MeV:.3f}_MeV.pkl'))

        self.energies = tf.convert_to_tensor(df_fermionic_DM['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_fermionic_DM['R_tonne_yr'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)
