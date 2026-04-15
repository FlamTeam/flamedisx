"""Background sources for LXe TPCs

"""
import tensorflow as tf

import os
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import yaml

import flamedisx as fd
from . import lxe_sources as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##


@export
class vERSource(fd_nest.nestERSource):
    """ER background source from solar neutrinos (PP + 7Be + CNO).
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., energy_max=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/vER_spectrum.pkl'))

        energies = df['energy_keV'].values
        rates_vs_energy = df['spectrum_value_norm'].values

        scale = fid_mass * livetime

        if energy_max is not None:
            rates_vs_energy  = np.transpose(rates_vs_energy[np.argwhere(energies < energy_max)])[0]
            energies  = np.transpose(energies[np.argwhere(energies < energy_max)])[0]

        self.energies = tf.convert_to_tensor(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(rates_vs_energy * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Xe136Source(fd_nest.nestERSource):
    """"Beta background source from the 2-neutrino double beta decay of 136Xe.
    Reads in energy spectrum from .pkl file.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., energy_max=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/136Xe_spectrum.pkl'))

        energies = df['energy_keV'].values
        rates_vs_energy = df['spectrum_value_norm'].values

        scale = fid_mass * livetime

        if energy_max is not None:
            rates_vs_energy  = np.transpose(rates_vs_energy[np.argwhere(energies < energy_max)])[0]
            energies  = np.transpose(energies[np.argwhere(energies < energy_max)])[0]

        self.energies = tf.convert_to_tensor(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(rates_vs_energy * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Xe124Source(fd_nest.nestERSource):
    """"EC background source from the 2-neutrino double electron capture of 124Xe.
    Reads in energy spectrum from .pkl file.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/124Xe_spectrum.pkl'))

        energies = df['energy_keV'].values
        rates_vs_energy = df['spectrum_value_norm'].values

        scale = fid_mass * livetime

        self.energies = tf.convert_to_tensor(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(rates_vs_energy * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Pb214Source(fd_nest.nestERSource):
    """Beta background source from 214Pb.
    Reads in energy spectrum from .pkl file.
    Normalised to 0.1 uBq/kg.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., energy_max=None, activity_uBq_kg=0.1, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/214Pb_spectrum.pkl'))

        energies = df['energy_keV'].values
        rates_vs_energy = df['spectrum_value_norm'].values

        scale = fid_mass * livetime * (activity_uBq_kg / 0.1)

        if energy_max is not None:
            rates_vs_energy  = np.transpose(rates_vs_energy[np.argwhere(energies < energy_max)])[0]
            energies  = np.transpose(energies[np.argwhere(energies < energy_max)])[0]

        self.energies = tf.convert_to_tensor(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(rates_vs_energy * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Kr85Source(fd_nest.nestERSource):
    """Beta background source from 85Kr.
    Reads in energy spectrum from .pkl file.
    Normalised to 0.1 ppt.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., energy_max=None, activity_ppt=0.1, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/85Kr_spectrum.pkl'))

        energies = df['energy_keV'].values
        rates_vs_energy = df['spectrum_value_norm'].values

        scale = fid_mass * livetime * (activity_ppt / 0.1)

        if energy_max is not None:
            rates_vs_energy  = np.transpose(rates_vs_energy[np.argwhere(energies < energy_max)])[0]
            energies  = np.transpose(energies[np.argwhere(energies < energy_max)])[0]

        self.energies = tf.convert_to_tensor(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(rates_vs_energy * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class vNRSolarSource(fd_nest.nestNRSource):
    """CEvNS background source from B8 + HEP neutrinos.
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_CEvNS_solar = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/CEvNS_solar_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_CEvNS_solar['energy_keV'].values, dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor(df_CEvNS_solar['spectrum_value_norm'].values * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class vNROtherLNGSSource(fd_nest.nestNRSource):
    """CEvNS background source from Atmospheric (LNGS flux) + DSNB neutrinos.
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_CEvNS_other = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/CEvNS_other_LNGS_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_CEvNS_other['energy_keV'].values, dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor(df_CEvNS_other['spectrum_value_norm'].values * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class vNROtherSURFSource(fd_nest.nestNRSource):
    """CEvNS background source from Atmospheric (SURF flux) + DSNB neutrinos.
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_CEvNS_other = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/CEvNS_other_SURF_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_CEvNS_other['energy_keV'].values, dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor(df_CEvNS_other['spectrum_value_norm'].values * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)



@export
class NeutronSource(fd_nest.nestNRSource):
    """NR background source from external neutrons.
    Reads in energy spectrum from .pkl file, generated with XLZD GEANT4 simulations.
    Normalise such that the spectrum predicts 1 event in 1 tonne year.
    """

    def __init__(self, *args, fid_mass=1., livetime=1., **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        template_path = os.path.join(os.path.dirname(__file__), 'background_spectra/60t_downloads/')

        components_file = os.path.join(os.path.dirname(__file__), 'config/components_60.yaml')

        with open(components_file) as f:
            components = yaml.safe_load(f)["components"]


        energy_range = (6, 30)

        combined_energy_spectra = None
        energy_spectrum_bins = None

        for comp in components:

            path = comp["path"]

            if "mass" in comp:
                scale = comp["mass"] * comp["rate"]
            elif "units" in comp:
                scale = comp["units"] * comp["rate"]
            else:
                continue

            if scale == 0:
                continue

            neutron_file = uproot.open(template_path + path)

            if "SSneutronClusterInformation;1" not in neutron_file:
                print(f"Skipping {path} (no key)")
                continue

            neutroncluster = neutron_file["SSneutronClusterInformation;1"].arrays()
            energy_data = ak.to_numpy(neutroncluster["Edep"])

            energy_spectrum, energy_bins = np.histogram(energy_data, bins=100, range=energy_range)


            # Applying scaling

            scaled_energy_spectrum = energy_spectrum * scale


            # Summing the spectra
            if combined_energy_spectra is None:
                combined_energy_spectra = scaled_energy_spectrum
            else:
                combined_energy_spectra += scaled_energy_spectrum

            if energy_spectrum_bins is None:
                energy_spectrum_bins = energy_bins
                

        norm_energy_spectrum = combined_energy_spectra / np.sum(combined_energy_spectra)
        energy_bin_centres = 0.5 * (energy_spectrum_bins[:-1] + energy_spectrum_bins[1:])



        self.energies = tf.convert_to_tensor(energy_bin_centres, dtype=fd.float_type())
        scale = fid_mass * livetime
        self.rates_vs_energy = tf.convert_to_tensor(norm_energy_spectrum * scale, dtype=fd.float_type())

        super().__init__(*args, **kwargs)