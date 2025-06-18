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
class EFTScalarO6Source(fd_nest.nestNRSource):
    """
    """

    def __init__(self, *args, mass_GeV=1000, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'signal_spectra/O6_NREFT_scalar_spectra.pkl'))[mass_GeV]

        self.energies = tf.convert_to_tensor(df['energy_keV'].values, dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor(df['spectrum_value_norm'].values * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)