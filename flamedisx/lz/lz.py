"""lz detector implementation

"""
import numpy as np
import tensorflow as tf
import pickle
import configparser
import os
import pandas as pd
from multihist import Histdd

import flamedisx as fd

import pickle as pkl

from multihist import Histdd

export, __all__ = fd.exporter()


##
# Useful functions
##


def interpolate_acceptance(arg, domain, acceptances):
    """ Function to interpolate signal acceptance curves
    :param arg: argument values for domain interpolation
    :param domain: domain values from interpolation map
    :param acceptances: acceptance values from interpolation map
    :return: Tensor of interpolated map values (same shape as x)
    """
    return np.interp(x=arg, xp=domain, fp=acceptances)

def build_position_map_from_data(map_file, axis_names, bins):
    """
    """
    map_df= fd.get_lz_file(map_file)
    assert isinstance(map_df, pd.DataFrame), 'Must pass in a dataframe to build position map hisotgram'

    mh = Histdd(bins=bins, axis_names=axis_names)

    add_args = []
    for axis_name in axis_names:
        add_args.append(map_df[axis_name].values)

    try:
        weights = map_df['weight'].values
    except Exception:
        weights = None

    mh.add(*add_args, weights=weights)

    return mh





def create_spatial_hist(file_path,is_histdd=False,**kwargs):
    """
        Function to handle event-level data: 
            data={x,y,z,dt[ns]/dt[us]/z[cm],energy spectra bin centers, eneregy spectre bin heights}
        and a pickled Histdd
        :param file_path: path to data 
        :param is_histdd: whether event data or pickled Histdd
        :param n_bins: Optional, customises binning of event data
        :return: spatial_hist, energy spectra bin centers, eneregy spectra bin heights
    """
    def read_position_pkl(FilePath):
        """
        Reads in energy spectrum from .pkl file, 
        e.g those generated with LZ's LZLAMA+BACCARAT->BGSKimmer/Assmbler
        :param FilePath: path to the .pkl data file 
        :return: array with neccessary data
        """
        #extract background position data
        print("Loading Data From: %s"%FilePath)
        with open(FilePath,'rb') as handle:
            data_dict=pickle.load(handle)
        keys=data_dict.keys()

        i=data_dict['x[cm]']
        j=data_dict['y[cm]']
        weight=data_dict['weight']

        k_key= list(keys)[2]
        dt_ns=data_dict[k_key]
        if k_key=='drift time[ns]':
            z=(self.z_topDrift - dt_ns*self.drift_velocity)
        elif k_key=='drift time[us]':
            z=(self.z_topDrift - dt_ns*1000*self.drift_velocity)
        elif k_key=='z[cm]':
            z=dt_ns
        else:
            raise KeyError("Key error for coords: %s in LZdetNRSource: Require x/y/z[cm] for lenghts, theta[rad],  and drift time[ns] or [us] "%coord)
        k=z
        if 'energy_keV' and 'spectrum_value_norm' in keys:
            value_energy=data_dict['energy_keV']
            value_norm=data_dict['spectrum_value_norm']
        return [i,j,k,weight,value_energy,value_norm]
    if 'n_bins' not in kwargs:
        kwargs['n_bins']=None
    if ~is_histdd:
        [i,j,k,weight,value_energy,value_norm]=read_position_pkl(file_path)
        #create histogram
        if kwargs['n_bins']==None:
            kwargs['n_bins']=int(np.sqrt(len(i))/5)
        mh = Histdd(
            bins=kwargs['n_bins'],
            range=[[min(i),max(i)],[min(j),max(j)],[min(k),max(k)]],
            axis_names=['x','y','z']
            )
        mh.add(i,j,k,weights=weight)
        return mh,value_energy,value_norm
    else: 
        return pickle.load(open(file_path, 'rb')), None,None
        


##
# Flamedisx sources
##

##
# Common to all LZ sources
##


class LZSource:
    path_s1_corr_LZAP = 's1_map_22Apr22.json'
    path_s2_corr_LZAP = 's2_map_30Mar22.json'
    path_s1_corr_latest = 's1_map_latest.json'
    path_s2_corr_latest = 's2_map_latest.json'

    path_s1_acc_curve = 'cS1_acceptance_curve.pkl'
    path_s2_acc_curve = 'cS2_acceptance_curve.pkl'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('lz',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.drift_velocity = self.drift_velocity * 0.96 / 0.95

        self.cS1_min = config.getfloat('NEST', 'cS1_min_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.cS1_max = config.getfloat('NEST', 'cS1_max_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.S2_min = config.getfloat('NEST', 'S2_min_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.cS2_max = config.getfloat('NEST', 'cS2_max_config') * (1 + self.double_pe_fraction)  # phd to phe

        try:
            self.s1_map_LZAP = fd.InterpolatingMap(fd.get_lz_file(self.path_s1_corr_LZAP))
            self.s2_map_LZAP = fd.InterpolatingMap(fd.get_lz_file(self.path_s2_corr_LZAP))
            self.s1_map_latest = fd.InterpolatingMap(fd.get_lz_file(self.path_s1_corr_latest))
            self.s2_map_latest = fd.InterpolatingMap(fd.get_lz_file(self.path_s2_corr_latest))
        except Exception:
            print("Could not load maps; setting position corrections to 1")
            self.s1_map_LZAP = None
            self.s2_map_LZAP = None
            self.s1_map_latest = None
            self.s2_map_latest = None

        try:
            df_S1_acc = fd.get_lz_file(self.path_s1_acc_curve)
            df_S2_acc =fd.get_lz_file(self.path_s2_acc_curve)

            self.cs1_acc_domain = df_S1_acc['cS1_phd'].values * (1 + self.double_pe_fraction)  # phd to phe
            self.cs1_acc_curve = df_S1_acc['cS1_acceptance'].values

            self.log10_cs2_acc_domain = df_S2_acc['log10_cS2_phd'].values + \
                np.log10(1 + self.double_pe_fraction)  # log_10(phd) to log_10(phe)
            self.log10_cs2_acc_curve = df_S2_acc['cS2_acceptance'].values
        except Exception:
            print("Could not load acceptance curves; setting to 1")

            self.cs1_acc_domain = None
            self.log10_cs2_acc_domain = None

    @staticmethod
    def photon_detection_eff(z, *, g1=0.113569):
        return g1 * tf.ones_like(z)

    @staticmethod
    def s2_photon_detection_eff(z, *, g1_gas=0.092103545):
        return g1_gas * tf.ones_like(z)

    @staticmethod
    def get_elife(event_time):
        t0 = pd.to_datetime('2021-10-01T00:00:00')
        t0 = t0.tz_localize(tz='America/Denver').value

        time_diff = event_time - t0
        days_since_t0 = time_diff / (24. * 3600 * 1e9)

        elife = np.piecewise(days_since_t0, [days_since_t0 <= 104.,
                                             (days_since_t0 > 104.) & (days_since_t0 <= 174.41597),
                                             days_since_t0 > 174.41597],
                             [lambda days_since_t0: (5526.52 - np.exp(27.0832 - 0.254022 * days_since_t0)) * 1000.,
                              lambda days_since_t0: (8271.97 - np.exp(9.5676 - 0.0160078 * days_since_t0)) * 1000.,
                              lambda days_since_t0: (7941.37 - np.exp(34.3876 - 0.148244 * days_since_t0)) * 1000.])

        return elife

    def electron_detection_eff(self, drift_time, electron_lifetime):
        return self.extraction_eff * tf.exp(-drift_time / electron_lifetime)

    @staticmethod
    def s1_posDependence(s1_pos_corr_latest):
        return s1_pos_corr_latest

    @staticmethod
    def s2_posDependence(s2_pos_corr_latest):
        return s2_pos_corr_latest

    def s1_acceptance(self, s1, cs1, cs1_acc_curve):

        acceptance = tf.where((s1 >= self.spe_thr) &
                              (cs1 >= self.cS1_min) & (cs1 <= self.cS1_max),
                              tf.ones_like(s1, dtype=fd.float_type()),  # if condition non-zero
                              tf.zeros_like(s1, dtype=fd.float_type()))  # if false

        # multiplying by efficiency curve
        acceptance *= cs1_acc_curve

        return acceptance

    def s2_acceptance(self, s2, cs2, cs2_acc_curve,
                      fv_acceptance, resistor_acceptance, timestamp_acceptance):

        acceptance = tf.where((s2 >= self.s2_thr) &
                              (s2 >= self.S2_min) & (cs2 <= self.cS2_max),
                              tf.ones_like(s2, dtype=fd.float_type()),  # if condition non-zero
                              tf.zeros_like(s2, dtype=fd.float_type()))  # if false

        # multiplying by efficiency curve
        acceptance *= cs2_acc_curve

        # We will insert the FV acceptance here
        acceptance *= fv_acceptance
        # We will insert the resistor acceptance here
        acceptance *= resistor_acceptance
        # We will insert the timestamp acceptance here
        acceptance *= timestamp_acceptance

        return acceptance

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if (self.s1_map_LZAP is not None) and (self.s2_map_LZAP is not None) and \
                (self.s1_map_latest is not None) and (self.s2_map_latest is not None):
            d['s1_pos_corr_LZAP'] = self.s1_map_LZAP(
                np.transpose([d['x'].values,
                              d['y'].values,
                              d['drift_time'].values * 1e-9 / 1e-6]))
            d['s2_pos_corr_LZAP'] = self.s2_map_LZAP(
                np.transpose([d['x'].values,
                              d['y'].values]))
            d['s1_pos_corr_latest'] = self.s1_map_latest(
                np.transpose([d['x'].values,
                              d['y'].values,
                              d['drift_time'].values * 1e-9 / 1e-6]))
            d['s2_pos_corr_latest'] = self.s2_map_latest(
                np.transpose([d['x'].values,
                              d['y'].values]))
        else:
            d['s1_pos_corr_LZAP'] = np.ones_like(d['x'].values)
            d['s2_pos_corr_LZAP'] = np.ones_like(d['x'].values)
            d['s1_pos_corr_latest'] = np.ones_like(d['x'].values)
            d['s2_pos_corr_latest'] = np.ones_like(d['x'].values)

        if 'event_time' in d.columns and 'electron_lifetime' not in d.columns:
            d['electron_lifetime'] = self.get_elife(d['event_time'].values)

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr_LZAP']
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_pos_corr_LZAP']
                * np.exp(d['drift_time'] / d['electron_lifetime']))

        if 'cs1' in d.columns and 'cs2' in d.columns and 'ces_er_equivalent' not in d.columns:
            d['ces_er_equivalent'] = (d['cs1'] / self.g1 + d['cs2'] / self.g2) * self.Wq_keV

        if 'cs1' in d.columns and 'cs1_acc_curve' not in d.columns:
            if self.cs1_acc_domain is not None:
                d['cs1_acc_curve'] = interpolate_acceptance(
                    d['cs1'].values,
                    self.cs1_acc_domain,
                    self.cs1_acc_curve)
            else:
                d['cs1_acc_curve'] = np.ones_like(d['cs1'].values)
        if 'cs2' in d.columns and 'cs2_acc_curve' not in d.columns:
            if self.log10_cs2_acc_domain is not None:
                d['cs2_acc_curve'] = interpolate_acceptance(
                    np.log10(d['cs2'].values),
                    self.log10_cs2_acc_domain,
                    self.log10_cs2_acc_curve)
            else:
                d['cs2_acc_curve'] = np.ones_like(d['cs2'].values)

        if 's2' in d.columns and 'fv_acceptance' not in d.columns:
            standoffDistance_cm = 4.

            m_idealFiducialWallFit = [72.4403, 0.00933984, 5.06325e-5, 1.65361e-7,
                                      2.92605e-10, 2.53539e-13, 8.30075e-17]

            boundaryR = 0
            drift_time_us = d['drift_time'].values / 1000.
            for i in range(len(m_idealFiducialWallFit)):
                boundaryR += m_idealFiducialWallFit[i] * pow(-drift_time_us, i)

            boundaryR = np.where(drift_time_us < 200., boundaryR - 5.2, boundaryR)
            boundaryR = np.where(drift_time_us > 800., boundaryR - 5., boundaryR)
            boundaryR = np.where((drift_time_us > 200.) & (drift_time_us < 800.), boundaryR - standoffDistance_cm, boundaryR)

            radius_cm = d['r'].values

            accept_upper_drift_time = np.where(drift_time_us < 936.5, 1., 0.)
            accept_lower_drift_time = np.where(drift_time_us > 86., 1., 0.)
            accept_radial = np.where(radius_cm < boundaryR, 1., 0.)

            d['fv_acceptance'] = accept_upper_drift_time * accept_lower_drift_time * accept_radial

        if 's2' in d.columns and 'resistor_acceptance' not in d.columns:
            x = d['x'].values
            y = d['y'].values

            res1X = -71.2
            res1Y = 4.4
            res1R = 6
            res2X = -69.2
            res2Y = -14.6
            res2R = 6
            not_inside_res1 = np.where(np.sqrt( (x-res1X)*(x-res1X) + (y-res1Y)*(y-res1Y) ) < res1R, 0., 1.)
            not_inside_res2 = np.where(np.sqrt( (x-res2X)*(x-res2X) + (y-res2Y)*(y-res2Y) ) < res2R, 0., 1.)

            d['resistor_acceptance'] = not_inside_res1 * not_inside_res2

        if 's2' in d.columns and 'timestamp_acceptance' not in d.columns:
            t_start = pd.to_datetime('2021-12-23T09:37:51')
            t_start = t_start.tz_localize(tz='America/Denver').value

            days_since_start = (d['event_time'].values - t_start) / 3600. / 24. / 1e9

            not_inside_window1 = np.where((days_since_start >= 25.5) & (days_since_start <= 33.), 0., 1.)
            not_inside_window2 = np.where((days_since_start >= 90.) & (days_since_start <= 93.5), 0., 1.)

            d['timestamp_acceptance'] = not_inside_window1 * not_inside_window2


##
# Different interaction types: flat spectra
##


@export
class LZERSource(LZSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZGammaSource(LZSource, fd.nest.nestGammaSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZERGammaWeightedSource(LZSource, fd.nest.nestERGammaWeightedSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZNRSource(LZSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


##
# Calibration sources
##


@export
class LZCH3TSource(LZSource, fd.nest.CH3TSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


##
# Signal sources
##


@export
class LZWIMPSource(LZSource, fd.nest.nestWIMPSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


##
# Background sources
##


@export
# class LZPb214Source(LZSource, fd.nest.Pb214Source, fd.nest.nestSpatialRateERSource):
class LZPb214Source(LZSource, fd.nest.Pb214Source):
    def __init__(self, *args, bins=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'

        if bins is None:
            bins=(np.sqrt(np.linspace(0.**2, 67.8**2, num=51)),
                  np.linspace(86000., 936500., num=51))

        # mh = build_position_map_from_data('Pb214_spatial_map_data.pkl', ['r', 'drift_time'], bins)
        self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZDetERSource(LZSource, fd.nest.DetERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZBetaSource(LZSource, fd.nest.BetaSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZXe136Source(LZSource, fd.nest.Xe136Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
# class LZvERSource(LZSource, fd.nest.vERSource, fd.nest.nestTemporalRateOscillationERSource):
class LZvERSource(LZSource, fd.nest.vERSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'

        if amplitude is None:
            self.amplitude = 2. * 0.01671
        else:
            self.amplitude = amplitude

        if phase_ns is None:
            self.phase_ns = pd.to_datetime('2022-01-04T00:00:00').value
        else:
            self.phase_ns = phase_ns

        if period_ns is None:
            self.period_ns = 1. * 3600. * 24. * 365.25 * 1e9
        else:
            self.period_ns = period_ns

        super().__init__(*args, **kwargs)


@export
# class LZAr37Source(LZSource, fd.nest.Ar37Source, fd.nest.nestTemporalRateDecayERSource):
class LZAr37Source(LZSource, fd.nest.Ar37Source):
    def __init__(self, *args, time_constant_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'

        if time_constant_ns is None:
            self.time_constant_ns = 35.0 * 1e9 * 3600. * 24.
        else:
            self.time_constant_ns = time_constant_ns

        super().__init__(*args, **kwargs)


@export
class LZXe124Source(LZSource, fd.nest.Xe124Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZXe127Source(LZSource, fd.nest.Xe127Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZB8Source(LZSource, fd.nest.B8Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZDetNRSource(LZSource, fd.nest.nestNRSource):
    """
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'

        df_DetNR = fd.get_lz_file('DetNR_spectrum.pkl')

        self.energies = tf.convert_to_tensor(df_DetNR['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_DetNR['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


##
# Source groups
##


@export
class LZERSourceGroup(LZSource, fd.nest.nestERSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZERGammaWeightedSourceGroup(LZSource, fd.nest.nestERGammaWeightedSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZNRSourceGroup(LZSource, fd.nest.nestNRSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)
#Spatial sources:



@export
class LZDetNRSource(LZSource, fd.nest.SpatialNRSource):
    """ 
        DetNR source, with spatial variation
        Use energy spectra derived from BACARRAT simulations 
    """
    def __init__(self,FilePath,*args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        mh,value_energy,value_norm=create_spatial_hist(FilePath,is_histdd=False,**kwargs)
        self.energies = value_energy
        self.rates_vs_energy = value_norm
        super().__init__(mh=mh,*args, **kwargs)
@export
class LZSpatialFlatERSource(LZSource, fd.nest.SpatialERSource):
    """ 
        FlatER source, with spatial variation
        use default flat energy spectra 
    """
    def __init__(self,FilePath,*args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        mh,value_energy,value_norm=create_spatial_hist(FilePath,is_histdd=False,**kwargs)
        self.energies =tf.convert_to_tensor(np.arange(0.05,20.5,0.05), dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(np.full(len(self.energies),1/len(self.energies)), dtype=fd.float_type())
        super().__init__(mh=mh,*args, **kwargs)
@export
class LZSpatialXe127Source(LZSource, fd.nest.SpatialERSource):
    """ 
        Xe127 source, with spatial variation
        use default energy spectra for the two Electron-capture peaks that fall in the WSROI
    """
    def __init__(self, FilePath,*args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        mh,value_energy,value_norm=create_spatial_hist(FilePath,is_histdd=True,**kwargs)
        df_127Xe = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/127Xe_spectrum.pkl'))
        self.energies = tf.convert_to_tensor(df_127Xe['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_127Xe['spectrum_value_norm'].values, dtype=fd.float_type())
        super().__init__(mh=mh,*args, **kwargs)
