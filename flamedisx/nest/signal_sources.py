"""Signal sources for LXe TPCs

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


@export
class ALPGalacticDMSource(fd_nest.nestERSource):
    """
    """

    def __init__(self, *args, mass_keV=10., g_ae=1e-12, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        # See arXiv:2102.11740, Eq. 5
        rate = 1.2e19 / 131.293 * g_ae**2 * self.get_PE_xsec_Xe(mass_keV) * mass_keV * 365.25 * 1000.

        self.energies = tf.convert_to_tensor([mass_keV], dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor([rate * scale], dtype=fd.float_type())

        super().__init__(*args, **kwargs)

    @staticmethod
    def get_PE_xsec_Xe(energy_keV):
        photoelectric_Xsec_path = os.path.join(os.path.dirname(__file__), 'signal_spectra/PhotoElectricXsecXe.txt')
        E_pe, xsec_pe = np.loadtxt(photoelectric_Xsec_path, skiprows=1, unpack=True)
        PE_xsec = np.interp(energy_keV, E_pe, xsec_pe)
        return PE_xsec


@export
class HiddenPhotonSource(fd_nest.nestERSource):
    """
    """

    def __init__(self, *args, mass_keV=10., kappa=1e-12, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        # See arXiv:2102.11740, Eq. 9
        rate = 4.0e23 / 131.293 * kappa**2 * self.get_PE_xsec_Xe(mass_keV) / mass_keV * 365.25 * 1000.

        self.energies = tf.convert_to_tensor([mass_keV], dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor([rate * scale], dtype=fd.float_type())

        super().__init__(*args, **kwargs)

    @staticmethod
    def get_PE_xsec_Xe(energy_keV):
        photoelectric_Xsec_path = os.path.join(os.path.dirname(__file__), 'signal_spectra/PhotoElectricXsecXe.txt')
        E_pe, xsec_pe = np.loadtxt(photoelectric_Xsec_path, skiprows=1, unpack=True)
        PE_xsec = np.interp(energy_keV, E_pe, xsec_pe)
        return PE_xsec