"""Background sources for LXe TPCs

"""
import tensorflow as tf

import os
import numpy as np
import pandas as pd
import pickle
from multihist import Histdd
import flamedisx as fd
from . import lxe_sources as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##
def read_position_spectra(FilePath):
    """Reads in energy spectrum from .pkl file, generated with LZ's LZLAMA+BACCARAT->BGSKimmer/Assmbler."""
    #extract background position data
    with open(FilePath,'rb') as handle:
        data_dict=pickle.load(handle)
    keys=data_dict.keys()
    
    i=data_dict['x[cm]']
    j=data_dict['y[cm]']
    weight=data_dict['weight']

    k_key= list(keys)[2]
    dt_ns=data_dict[k_key]
    if k_key=='drift time[ns]':
        z=(146.1 - dt_ns*0.00015334659984647314) 
    elif k_key=='drift time[us]':
        z=(146.1 - dt_ns*1000*0.00015334659984647314) 
    elif k_key=='z[cm]':
        z=dt_ns
    else:
        raise KeyError("Key error for coords: %s in LZdetNRSource: Require x/y/z[cm] for lenghts, theta[rad],  and drift time[ns] or [us] "%coord)
    k=z
    if 'energy_keV' and 'spectrum_value_norm' in keys:
        value_energy=data_dict['energy_keV']
        value_norm=data_dict['spectrum_value_norm']
    
    return [i,j,k,weight,value_energy,value_norm]
@export
class BetaSource(fd_nest.nestERSource):
    """Beta background source combining 214Pb, 212Pb and 85Kr.
    Reads in energy spectra from .pkl files. Normalise such that the sum of
    rates_vs_energy is 1.

    Arguments:
        - weights: tuple (length 3) of weights to apply to the spectra,
        in the order given above. 214Pb and 212Pb relative to 1 uBq / kg,
        85Kr relative to 0.3ppt.
    """

    def __init__(self, *args, weights=(1, 1, 1,), **kwargs):
        assert len(weights) == 3, "Weights must be a tuple of length 3"
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_214Pb = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/214Pb_spectrum.pkl'))
        df_212Pb = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/212Pb_spectrum.pkl'))
        df_85Kr = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/85Kr_spectrum.pkl'))

        assert np.logical_and((df_214Pb['energy_keV'].values == df_212Pb['energy_keV'].values).all(),
                              (df_212Pb['energy_keV'].values == df_85Kr['energy_keV'].values).all()), \
            "Energy spectrum components must have equal energies"

        # Weight the spectra according to the weights provided, then combine
        df_214Pb_values = df_214Pb['spectrum_value_norm'].values * weights[0]
        df_212Pb_values = df_212Pb['spectrum_value_norm'].values * weights[1]
        df_85Kr_values = df_85Kr['spectrum_value_norm'].values * weights[2]

        combined_rates_vs_energy = df_214Pb_values + df_212Pb_values + df_85Kr_values

        # Re-normalise the summed spectra
        combined_rates_vs_energy = combined_rates_vs_energy / sum(combined_rates_vs_energy)

        self.energies = tf.convert_to_tensor(df_214Pb['energy_keV'].values, dtype=fd.float_type())
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
class SpatialNRSource(fd_nest.nestSpatialRateNRSource):
    """Spatially dependent NR source
    """
    
    def __init__(self,FilePath,nbins, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'
        #create histogram
        
        [i,j,k,weight,value_energy,value_norm]=read_position_spectra(FilePath)
        offset=1
        if nbins==None:
            nbins=int(np.sqrt(len(i))/5)
        mh = Histdd(
            bins=nbins,
            range=[[min(i)-offset,max(i)+offset],[min(j)-offset,max(j)+offset],[min(k)-offset,max(k)+offset]],
            axis_names=['x','y','z']
            )
        mh.add(i,j,k,weights=weight)
        self.spatial_hist=mh
        #add energy spectra
        if not hasattr(self,'energies'):    
            print( self.__class__.__name__+" Energy spectrum assigned from position file: "+str(FilePath))
            self.energies = tf.convert_to_tensor(value_energy, dtype=fd.float_type())
            self.rates_vs_energy = tf.convert_to_tensor(value_norm, dtype=fd.float_type())
        super().__init__(*args, **kwargs)
        
@export
class DetNRSource(SpatialNRSource):
    "Derived from incorrect assembled file"
    
    def __init__(self,FilePath=os.path.join(os.path.dirname(__file__), '../lz/Det_NR_test_position_data.pkl'),nbins=30,*args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(FilePath, nbins,*args, **kwargs)

        
        
@export
class SpatialERSource(fd_nest.nestSpatialRateERSource):
    """Spatially dependent ER source
    """
    
    def __init__(self,FilePath,nbins, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'
        #create histogram
        [i,j,k,weight,value_energy,value_norm]=read_position_spectra(FilePath)
        offset=1
        if nbins==None:
            nbins=int(np.sqrt(len(i))/5)
        mh = Histdd(
            bins=nbins,
            range=[[min(i)-offset,max(i)+offset],[min(j)-offset,max(j)+offset],[min(k)-offset,max(k)+offset]],
            axis_names=['x','y','z']
            )
        mh.add(i,j,k,weights=weight)
        self.spatial_hist=mh
        #add energy spectra
        if not hasattr(self,'energies'):    
            print( self.__class__.__name__+" Energy spectrum assigned from position file: "+str(FilePath))
            self.energies = tf.convert_to_tensor(value_energy, dtype=fd.float_type())
            self.rates_vs_energy = tf.convert_to_tensor(value_norm, dtype=fd.float_type())
        
        super().__init__(*args, **kwargs)

@export
class SpatialXe127Source(SpatialERSource):
    """Spatially dependent Xe127 source Derived from correct skimmed/assembled files 
    """
    
    def __init__(self,FilePath=os.path.join(os.path.dirname(__file__), '../lz/Xe127_test_position_data.pkl'),nbins=None,*args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        df_127Xe = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/127Xe_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_127Xe['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_127Xe['spectrum_value_norm'].values, dtype=fd.float_type())
        super().__init__(FilePath, nbins,*args, **kwargs)
        