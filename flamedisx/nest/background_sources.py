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
class Pb214Source(fd_nest.nestERSource):
    """Beta background source from 214Pb decay.
    Reads in energy spectrum from .pkl file. Normalise such that the sum of
    rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_214Pb = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/214Pb_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_214Pb['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_214Pb['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class DetERSource(fd_nest.nestERSource):
    """ER background source from detector components.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        self.energies = tf.cast(np.arange(0.6, 22.01, 0.1), fd.float_type())
        self.rates_vs_energy = tf.ones_like(self.energies, fd.float_type()) / sum(np.ones_like(self.energies))

        super().__init__(*args, **kwargs)


@export
class BetaSource(fd_nest.nestERSource):
    """Beta background source combining 212Pb and 85Kr.
    Reads in energy spectra from .pkl files. Normalise such that the sum of
    rates_vs_energy is 1.

    Arguments:
        - weights: tuple (length 2) of weights to apply to the spectra,
        in the order given above. 212Pb relative to 1 uBq / kg,
        85Kr relative to 0.3ppt.
    """

    def __init__(self, *args, weights=(1, 1,), **kwargs):
        assert len(weights) == 2, "Weights must be a tuple of length 3"
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_212Pb = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/212Pb_spectrum.pkl'))
        df_85Kr = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/85Kr_spectrum.pkl'))

        assert (df_212Pb['energy_keV'].values == df_85Kr['energy_keV'].values).all(), \
            "Energy spectrum components must have equal energies"

        # Weight the spectra according to the weights provided, then combine
        df_212Pb_values = df_212Pb['spectrum_value_norm'].values * weights[0]
        df_85Kr_values = df_85Kr['spectrum_value_norm'].values * weights[1]

        combined_rates_vs_energy = df_212Pb_values + df_85Kr_values

        # Re-normalise the summed spectra
        combined_rates_vs_energy = combined_rates_vs_energy / sum(combined_rates_vs_energy)

        self.energies = tf.convert_to_tensor(df_212Pb['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(combined_rates_vs_energy, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Xe136Source(fd_nest.nestERSource):
    """Beta background source from the 2-neutrino double beta decay of 136Xe.
    Reads in energy spectrum from .pkl file. Normalise such that the sum of
    rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_136Xe = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/136Xe_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_136Xe['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_136Xe['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class vERSource(fd_nest.nestERSource):
    """ER background source from solar neutrinos (PP+7Be+CNO).
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    Normalise such that the sum of rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_vER = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/vER_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_vER['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_vER['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Ar37Source(fd_nest.nestERSource):
    """Background source for the electron capture decay of 37Ar.
    Reads in energy spectrum from .pkl file. Normalise such that the sum of
    rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_37Ar = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/37Ar_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_37Ar['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_37Ar['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Xe124Source(fd_nest.nestERGammaWeightedSource):
    """Background source for the electron capture decay of 124Xe.
    Reads in energy spectrum from .pkl file. Normalise such that the sum of
    rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_124Xe = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/124Xe_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_124Xe['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_124Xe['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class Xe127Source(fd_nest.nestERGammaWeightedSource):
    """Background source for the electron capture decay of 127Xe.
    Reads in energy spectrum from .pkl file. Normalise such that the sum of
    rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_127Xe = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/127Xe_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_127Xe['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_127Xe['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class B8Source(fd_nest.nestNRSource):
    """NR background source from 8B solar neutrinos.
    Reads in energy spectrum from .pkl file, generated with LZ's DMCalc.
    Normalise such that the sum of rates_vs_energy is 1.
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_8B = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/8B_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_8B['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_8B['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)



@export
class SpatialERSource(fd_nest.nestSpatialRateERSource):
    """Spatially dependent ER source
    """ 
    def __init__(self,mh, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'
        self.spatial_hist=mh

        if not hasattr(self,'energies'):    
            raise( self.__class__.__name__+" Has No Energy Spectra")
        
        super().__init__(*args, **kwargs)

@export
class SpatialNRSource(fd_nest.nestSpatialRateNRSource):
    """Spatially dependent NR source
    """   
    def __init__(self,mh, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'
        self.spatial_hist=mh
        #add energy spectra
        if not hasattr(self,'energies'):    
            raise( self.__class__.__name__+" Has No Energy Spectra")
        super().__init__(*args, **kwargs)
