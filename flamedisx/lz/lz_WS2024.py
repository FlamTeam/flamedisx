"""lz detector implementation

"""
import numpy as np
import math as m
import tensorflow as tf
import scipy
import configparser
import os
import pandas as pd

import flamedisx as fd
from .. import nest as fd_nest
from .WS2024_cuts_and_acceptances import *
import pickle as pkl

from multihist import Histdd
from scipy import interpolate


export, __all__ = fd.exporter()
pi = tf.constant(m.pi)
GAS_CONSTANT = 8.314
N_AVAGADRO = 6.0221409e23
A_XENON = 131.293
XENON_REF_DENSITY = 2.90
BRANCH = 'JRG_WS2024_data'
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
    interpolation_kwargs = dict(kind='slinear', bounds_error=False,
                                fill_value="extrapolate")
    return interpolate.interp1d(domain, acceptances,**interpolation_kwargs)(arg)

def build_position_map_from_data(map_file, axis_names, bins):
    """
    """
    map_df= fd.get_lz_file(map_file, branch=BRANCH)
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


##
# Flamedisx sources
##

##
# Common to all LZ sources
##

@export
class LZWS2024Source:

    path_s1_corr_latest = 'WS2024/s1Area_Correction_TPC_WS2024_radon_31Jan2024.json'
    path_s2_corr_latest = 'WS2024/s2Area_Correction_TPC_WS2024_radon_31Jan2024.json'

    path_s1_acc_curve = 'WS2024/cS1_tritium_acceptance_curve.json'
    path_s2_splitting_curve='WS2024/WS2024_S2splittingReconEff_mean.pkl'
    path_drift_map_dt='WS2024/drift_map_dt_WS2024.json'
    path_drift_map_x= 'WS2024/drift_map_x_WS2024.json'
    path_field_map_E= 'WS2024/WS2024_field_map.json'
    
    def __init__(self, *args, 
                 ignore_field_map=False, 
                 ignore_LCE_maps=False, ignore_acc_maps=False,
                 ignore_all_cuts=False, ignore_drift_map=False, 
                 cap_upper_cs1=False, 
                 S2_splitting_max_OOB=False,
                 **kwargs):
        #how to interpret splitting efficiency
        self.S2_splitting_max_OOB = S2_splitting_max_OOB
        #set up start and end time
        self.t_start = pd.to_datetime('2023-03-28 09:00:00')
        self.t_stop = pd.to_datetime('2024-03-31 23:00:00')
        
        super().__init__(*args, **kwargs)
        
        
        self.cap_upper_cs1 = cap_upper_cs1
        self.ignore_all_cuts = ignore_all_cuts
        assert kwargs['detector'] in ('lz_WS2024',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.cS1_min = config.getfloat('NEST', 'cS1_min_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.cS1_max = config.getfloat('NEST', 'cS1_max_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.S2_min = config.getfloat('NEST', 'S2_min_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.cS2_max = config.getfloat('NEST', 'cS2_max_config') * (1 + self.double_pe_fraction)  # phd to phe
        
        if not ignore_field_map:
            try:
                field_map=fd.get_lz_file(self.path_field_map_E)
                self.field_map_E = interpolate.LinearNDInterpolator(field_map['coordinate_system'],field_map['map'],fill_value=0)
                self.model_blocks[1].has_driftField=True #tell quanta splitting there's a drift field!
            except:
                print("Failed to load field map")
                self.field_map_E = None

        if not ignore_drift_map:
            try:
                drift_map=fd.get_lz_file(self.path_drift_map_dt, branch=BRANCH)
                self.drift_map_dt = interpolate.LinearNDInterpolator(drift_map['coordinate_system'],drift_map['map'],fill_value=0)
                drift_map_xy=fd.get_lz_file(self.path_drift_map_x, branch=BRANCH)
                self.drift_map_x = interpolate.LinearNDInterpolator(drift_map_xy['coordinate_system'],drift_map_xy['map'],fill_value=0)
            except:
                self.drift_map_dt = None
                self.drift_map_x = None

                print('Could not load drift maps \n !Using default NEST Calculation!')
        else:
            print("Ignoring drift map")
        
        
        
        self.ignore_acceptances_maps=False
        if ignore_acc_maps:
            print("ignoring acceptances")
            self.ignore_acceptances_maps = True

            self.cs1_acc_domain = None
            self.cS2_drift_acceptance_hist = None
        else:
            try:
                df_S1_acc = fd.get_lz_file(self.path_s1_acc_curve, branch=BRANCH)
                self.cs1_acc_domain = np.array(df_S1_acc['cS1_phd']) * (1 + self.double_pe_fraction)  # phd to phe
                self.cs1_acc_curve = np.array(df_S1_acc['cS1_acceptance'])
                #TO-DO: Adapt to json for get_lz_file
                self.cS2_drift_acceptance_hist= fd.get_lz_file(self.path_s2_splitting_curve, branch=BRANCH)
            except Exception:
                print("Could not load acceptance curves; setting to 1")

                self.cs1_acc_domain = None
                self.cS2_drift_acceptance_hist = None

        if ignore_LCE_maps:
            print("ingoring LCE maps")
            self.s1_map_latest = None
            self.s2_map_latest = None
        else:
            try:
                self.s1_map_latest = fd.InterpolatingMap(fd.get_lz_file(self.path_s1_corr_latest, branch=BRANCH))
                self.s2_map_latest = fd.InterpolatingMap(fd.get_lz_file(self.path_s2_corr_latest, branch=BRANCH))
            except Exception:
                print("Could not load maps; setting position corrections to 1")
                self.s1_map_latest = None
                self.s2_map_latest = None
       
    @staticmethod
    def photon_detection_eff(drift_time, *, g1=0.1122):
        """
            g1_gas: Floatable (defined in function argument)
            ensure this default is correct
        """
        return g1 * tf.ones_like(drift_time)

    @staticmethod
    def s2_photon_detection_eff(drift_time, *, g1_gas=0.076404):
        """
            g1_gas: Floatable (defined in function argument)
            ensure this default is correct
        """
        return g1_gas * tf.ones_like(drift_time)

    @staticmethod
    def get_elife(event_time):
        """
            Static method so just need to know
            Rob: Any reason I can't have self?
        """
        return 9e6*np.ones_like(event_time)

    def electron_detection_eff(self, drift_time, electron_lifetime):
        return self.extraction_eff * tf.exp(-drift_time / electron_lifetime)

    @staticmethod
    def s1_posDependence(s1_pos_corr_latest):
        return s1_pos_corr_latest

    @staticmethod
    def s2_posDependence(s2_pos_corr_latest):
        return s2_pos_corr_latest

    def s1_acceptance(self, s1, cs1, cs1_acc_curve):
        if self.ignore_all_cuts:
            return tf.ones_like(s1, dtype=fd.float_type())
        acceptance = tf.where((s1 >= self.spe_thr) &
                              (cs1 >= self.cS1_min),
                              tf.ones_like(s1, dtype=fd.float_type()),  # if condition non-zero
                              tf.zeros_like(s1, dtype=fd.float_type()))  # if false
        if self.cap_upper_cs1:
            acceptance *= tf.where(cs1 <= self.cS1_max,
                                  tf.ones_like(s1, dtype=fd.float_type()),  # if condition non-zero
                                  tf.zeros_like(s1, dtype=fd.float_type()))  # if false

        # multiplying by efficiency curve
        if not self.ignore_acceptances_maps:
            acceptance *= cs1_acc_curve
        
        return acceptance

    def s2_acceptance(self, s2, cs2, cs2_acc_curve,
                      fv_acceptance, resistor_acceptance, timestamp_acceptance):
        if self.ignore_all_cuts:
            return tf.ones_like(s2, dtype=fd.float_type())
        acceptance = tf.where((s2 >= self.s2_thr) &
                              (s2 >= self.S2_min) & (cs2 <= self.cS2_max),
                              tf.ones_like(s2, dtype=fd.float_type()),  # if condition non-zero
                              tf.zeros_like(s2, dtype=fd.float_type()))  # if false

        # multiplying by efficiency curve
        if not self.ignore_acceptances_maps:
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
        
        
        if 'x_obs' not in d.columns:
            print("ERROR: Require observed X and Y")
            raise NotImplemented

        if 'r_obs' not in d.columns:
            d['r_obs']=np.hypot(d['x_obs'],d['y_obs'])
        if 'drift_field' not in d:
            d['drift_field'] = self.model_blocks[0].derive_drift_field(d)


        if (self.s1_map_latest is not None) and (self.s2_map_latest is not None):
            #LZLAMA uses correctedX and Y
            #I think this is meant to represent cluster (and therfore True position)
            d['s1_pos_corr_latest'] = self.s1_map_latest(
                np.transpose([d['x_obs'].values,
                              d['y_obs'].values,
                              d['drift_time'].values * 1e-9 / 1e-6]))
            d['s2_pos_corr_latest'] = self.s2_map_latest(
                np.transpose([d['x_obs'].values,
                              d['y_obs'].values]))
        else:
            d['s1_pos_corr_latest'] = np.ones_like(d['x_obs'].values)
            d['s2_pos_corr_latest'] = np.ones_like(d['x_obs'].values)
        
        if 'event_time' in d.columns and 'electron_lifetime' not in d.columns:
            d['electron_lifetime'] = self.get_elife(d['event_time'].values)

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr_latest']
            d['cs1_phd'] = d['cs1'] / (1 + self.double_pe_fraction)

        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_pos_corr_latest']
                * np.exp(d['drift_time'] / d['electron_lifetime']))
            d['log10_cs2_phd'] = np.log10(d['cs2'] / (1 + self.double_pe_fraction))

        if 'cs1' in d.columns and 's1' not in d.columns:
            d['s1'] = d['cs1'] * d['s1_pos_corr_latest']
        if 'cs2' in d.columns and 's2' not in d.columns:
            d['s2'] = (
                d['cs2']
                * d['s2_pos_corr_latest']
                / np.exp(d['drift_time'] / d['electron_lifetime']))

        if 'cs1' in d.columns and 'cs2' in d.columns and 'ces_er_equivalent' not in d.columns:
            g1 = self.photon_detection_eff(0.)
            g1_gas = self.s2_photon_detection_eff(0.)
            g2 = fd_nest.calculate_g2(self.gas_field, self.density_gas, self.gas_gap,
                                      g1_gas, self.extraction_eff)
            d['ces_er_equivalent'] = (d['cs1'] / fd.tf_to_np(g1) + d['cs2'] / fd.tf_to_np(g2)) * self.Wq_keV

        if 'cs1' in d.columns and 'cs1_acc_curve' not in d.columns:
            if self.cs1_acc_domain is not None:
                d['cs1_acc_curve'] = interpolate_acceptance(
                    d['cs1'].values,
                    self.cs1_acc_domain,
                    self.cs1_acc_curve)
            else:
                d['cs1_acc_curve'] = np.ones_like(d['cs1'].values)
        if 'cs2' in d.columns and 'cs2_acc_curve' not in d.columns:
            if self.cS2_drift_acceptance_hist is not None:
                d['cs2_acc_curve'] = WS2024_S2splitting_reconstruction_efficiency(
                                                    d['cs2'].values/(1+self.double_pe_fraction),#phe->phd
                                                    d['drift_time'].values/1e3,#ns->us
                                                    self.cS2_drift_acceptance_hist,
                                                    max_OOB = self.S2_splitting_max_OOB)
                d['cs2_acc_curve'] *=WS2024_trigger_acceptance(d['s2'].values/(1+self.double_pe_fraction))

            else:
                d['cs2_acc_curve'] = np.ones_like(d['cs2'].values)
        
        if 'fv_acceptance' not in d.columns:
            x = d['x_obs'].values
            y = d['y_obs'].values
            dt=d['drift_time'].values/1e3
            d['fv_acceptance']=WS2024_fiducial_volume_cut(x,y,dt)
            if self.ignore_all_cuts:
                d['fv_acceptance']=np.ones_like(d['fv_acceptance'],dtype=bool)

        if 'resistor_acceptance' not in d.columns:
            x = d['x_obs'].values
            y = d['y_obs'].values
            d['resistor_acceptance'] = WS2024_resistor_XY_cut(x,y)
            if self.ignore_all_cuts:
                d['resistor_acceptance']=np.ones_like(d['resistor_acceptance'],dtype=bool)
        if 'timestamp_acceptance' not in d.columns:
            d['timestamp_acceptance'] = np.ones_like(d['event_time'],dtype=bool)
        
##
# Different interaction types: flat spectra
##

@export
class LZ24ERSource(LZWS2024Source, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)
    def mean_yield_electron(self, *args):
        """
            Update the mean yields to WS2024 LZLAMA (!397)
            ERYieldParams from NEST/LZLAMA
            Constants are direct over-rides of eqn 6 in Arxiv: 2211.10726 
            Energy: energy in keV
        """
        energy=args[0]
        m1 = 12.4886
        m2 = 85.0   
        m3 = 0.6050 
        m4 = 2.14687
        m5 = 5.721   
        m6 = 0.     
        m7 = 59.651 
        m8 = 3.6869  
        m9 = 0.2872 
        m10 = 0.1121

        Nq = energy  / self.Wq_keV  #equation is in keV   

        Qy = m1 + (m2 - m1) / pow((1. + pow(energy /m3,m4)),m9) + \
            m5 + (m6 - m5) / pow((1. + pow(energy /m7, m8)), m10)

        coeff_TI = tf.cast(pow(1. / XENON_REF_DENSITY, 0.3), fd.float_type())
        coeff_Ni = tf.cast(pow(1. / XENON_REF_DENSITY, 1.4), fd.float_type())
        coeff_OL = tf.cast(pow(1. / XENON_REF_DENSITY, -1.7) /
                           fd.tf_log10(1. + coeff_TI * coeff_Ni * pow(XENON_REF_DENSITY, 1.7)), fd.float_type())

        Qy *= coeff_OL * fd.tf_log10(1. + coeff_TI * coeff_Ni * pow(tf.cast(self.density,fd.float_type()), 1.7)) * pow(tf.cast(self.density,fd.float_type()), -1.7)

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        return nel
    def fano_factor(self, *args):
        """
            Update fano factor
            ERNRWidthParams from NEST/LZLAMA
            WS2024 directly over-ride and lose functional form of Arix:2211.10726
            args: nq_mean, drift_field (if field map used)
            nq_mean: mean number of quanta (self.mean_yield_quanta)
        """
        nq_mean = args[0]
        er_free_a = 0.3
        return tf.constant(er_free_a,tf.float32)
    
    def variance(self, *args):
        """
            Update variance
            ERNRWidthParams from NEST/LZLAMA
            Variance of skew guessian. I don't understand the args thing.
        """
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        if self.detector in ['lz','lz_WS2024']:
            er_free_b = 0.04311
        else:
            er_free_b = 0.0553
        er_free_c = 0.15505
        er_free_d = 0.46894
        er_free_e = -0.26564

        elec_frac = nel_mean / nq_mean
        ampl = er_free_b
        
        ampl =  tf.cast(0.086036 + (er_free_b - 0.086036) /
                       pow((1. + pow(self.drift_field / 295.2, 251.6)), 0.0069114),
                       fd.float_type()) 
        wide = er_free_c
        cntr = er_free_d
        skew = er_free_e

        mode = cntr + 2. / (tf.sqrt(2. * pi)) * skew * wide / tf.sqrt(1. + skew * skew)
        norm = 1. / (tf.exp(-0.5 * pow(mode - cntr, 2.) / (wide * wide)) *
                     (1. + tf.math.erf(skew * (mode - cntr) / (wide * tf.sqrt(2.)))))

        omega = norm * ampl * tf.exp(-0.5 * pow(elec_frac - cntr, 2.) / (wide * wide)) * \
            (1. + tf.math.erf(skew * (elec_frac - cntr) / (wide * tf.sqrt(2.))))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         omega)

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni
    
@export
class LZ24GammaSource(LZWS2024Source, fd.nest.nestGammaSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


@export
class LZ24ERGammaWeightedSource(LZ24ERSource, fd.nest.nestERGammaWeightedSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)
        
    def mean_yield_electron(self, energy,b=1.3):
        # Weighted ER model
        # New Parameters for L-shell used here: LZLAMA/include/Modules/ModuleNest.hh#L193
        # Defined here: LZLLAMA/src/Detectors/LzTpcDetector.cc#L283
        weight_param_a = 0.48 #0.23
        weight_param_b = 0. #0.77
        weight_param_c = 0. #2.95
        weight_param_d = 0. #-1.44
        #field dependence ignored (with zeros above)
        weight_param_e = 421.15
        weight_param_f = 3.27

        gamma_weight = tf.cast(weight_param_a + weight_param_b * tf.math.erf(weight_param_c *
                          (tf.math.log(energy) + weight_param_d)) *
                          (1. - (1. / (1. + pow(self.drift_field / weight_param_e, weight_param_f)))),
                          fd.float_type())
        beta_weight = tf.cast(1. - gamma_weight, fd.float_type())


        nel_gamma = tf.cast(fd.lz.LZ24GammaSource.mean_yield_electron(self, energy), fd.float_type())
        nel_beta = tf.cast(fd.lz.LZ24ERSource.mean_yield_electron(self, energy), fd.float_type())

        nel = nel_gamma * gamma_weight + nel_beta * beta_weight

        return nel


@export
class LZ24NRSource(LZWS2024Source, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)
    def mean_yields(self, *args):
        """
            Update the mean yields to WS2024 LZLAMA (!397)
            NRYieldParams from NEST/LZLAMA
            See section C. in Arxiv: 2211.10726 
            Energy: energy in keV
        """
        energy = args[0]
        nr_nuis_alpha = 10.19
        nr_nuis_beta = 1.11
        nr_nuis_gamma = 0.0498
        nr_nuis_delta = -0.0533
        nr_nuis_epsilon = 12.46
        nr_nuis_zeta =  0.2942
        nr_nuis_eta = 1.899
        nr_nuis_theta = 0.3197
        nr_nuis_l = 2.066
        nr_nuis_p = 0.509
        nr_new_nuis_a = 0.996
        nr_new_nuis_b =  0.999
 
        TIB = nr_nuis_gamma * pow(self.drift_field, nr_nuis_delta) * pow(self.density / XENON_REF_DENSITY, 0.3)
        Qy = 1. / (TIB * pow(energy + nr_nuis_epsilon, nr_nuis_p))
        Qy *= (1. - (1. /pow(1. + pow(tf.math.divide_no_nan(energy , nr_nuis_zeta), nr_nuis_eta), nr_new_nuis_a)))

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        nq_temp = nr_nuis_alpha * pow(energy, nr_nuis_beta)

        nph_temp = (nq_temp - nel) * (1. - (1. / pow(1. + pow(tf.math.divide_no_nan(energy , nr_nuis_theta), nr_nuis_l), nr_new_nuis_b)))
        # Don't let number of photons go negative
        nph = tf.where(nph_temp < 0,
                       0 * nph_temp,
                       nph_temp)

        nq = nel + nph

        ni = (4. / TIB) * (tf.exp(nel * TIB / 4.) - 1.)

        nex = nq - ni

        ex_ratio = tf.cast(tf.math.divide_no_nan(nex , ni), fd.float_type())

        ex_ratio = tf.where(tf.logical_and(ex_ratio < self.alpha, energy > 100.),
                            self.alpha * tf.ones_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)
        ex_ratio = tf.where(tf.logical_and(ex_ratio > 1., energy < 1.),
                            tf.ones_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)
        ex_ratio = tf.where(tf.math.is_nan(ex_ratio),
                            tf.zeros_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)

        return nel, nq, ex_ratio

    def yield_fano(self, *args):
        """
            Update fano factor
            ERNRWidthParams from NEST/LZLAMA
            nq_mean: mean number of quanta (self.mean_yield_quanta)
        """
        nq_mean = args[0]
        if self.detector in ['lz','lz_WS2024']:
            nr_free_a = 0.404
            nr_free_b = 0.393
        else:
            nr_free_a = 1.
            nr_free_b = 1.

        ni_fano = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_a
        nex_fano = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_b

        return ni_fano, nex_fano

    @staticmethod
    def skewness(*args):
        """
            Update skewness 
            ERNRWidthParams from NEST/LZLAMA
            nq_mean: mean number of quanta (self.mean_yield_quanta)
        """
        nq_mean = args[0]
        nr_free_f =  2.220

        mask = tf.less(nq_mean, 1e4 * tf.ones_like(nq_mean))
        skewness = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_f
        skewness_masked = tf.multiply(skewness, tf.cast(mask, fd.float_type()))

        return skewness_masked

    def variance(self, *args):
        """
            Update Variance 
            ERNRWidthParams from NEST/LZLAMA
            I don't understand args
        """
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        if self.detector in ['lz','lz_WS2024']:
            nr_free_c = 0.0383
        else:
            nr_free_c = 0.1
        nr_free_d = 0.497
        nr_free_e =  0.1906

        elec_frac = nel_mean / nq_mean

        omega = nr_free_c * tf.exp(-0.5 * pow(elec_frac - nr_free_d, 2.) / (nr_free_e * nr_free_e))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         tf.cast(omega, dtype=fd.float_type()))

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni

##
# New sources!
# This is an architectural problem!
# can't specific changes like this, would need to make a whole new NEST!
# Maybe I can play with the inheritance
##


@export
class LZ24C14Source(LZ24ERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        m_e = 510.9989461  # e- rest mass-energy [keV]
        aa = 0.0072973525664;          # fine structure constant
        ZZ = 7.;
        V0 = 0.495;  # effective offset in T due to screening of the nucleus by electrons
        qValue = 156.
        #energy range to avoid nans
        energies = tf.linspace(0.01, qValue, 1000)

        Ee=energies+m_e
        pe=np.sqrt(np.square(Ee)-np.square(m_e))
        dNdE_phasespace=pe * Ee * (qValue - energies)**2
        Ee_screen = Ee - V0
        W_screen = (Ee_screen) / m_e
        p_screen = np.sqrt(W_screen * W_screen - 1)
        p_screen=np.where(W_screen<1,0.,p_screen)
        WW = (Ee) / m_e
        pp = np.sqrt(WW * WW - 1)
        G_screen = (Ee_screen) / (m_e)  ## Gamma, Total energy(KE+M) over M
        B_screen = np.sqrt((G_screen * G_screen - 1)*(G_screen * G_screen))  # v/c of electron. Ratio of
        B_screen=np.where(G_screen<1,0.,B_screen)
        x_screen = (2 * pi * ZZ * aa) / B_screen
        F_nr_screen = W_screen * p_screen / (WW * pp) * x_screen * (1 / (1 - np.exp(-x_screen)))
        F_nr_screen=np.where(p_screen<=0. ,0. ,F_nr_screen)
        F_bb_screen =F_nr_screen *np.power(W_screen * W_screen * (1 + 4 * (aa * ZZ) * (aa * ZZ)) - 1,np.sqrt(1 - aa * aa * ZZ * ZZ) - 1)
        spectrum = dNdE_phasespace * F_bb_screen
        spectrum=spectrum/np.sum(spectrum)
        energies = tf.cast(energies, fd.float_type())
        rates_vs_energy = tf.cast(spectrum, fd.float_type())
        self.energies = tf.cast(energies, fd.float_type())
        self.rates_vs_energy = tf.cast(spectrum, fd.float_type())
        super().__init__(*args, **kwargs)

@export
class LZ24RnBetaSource(LZ24ERSource):
    """Radon Beta background source combining 212Pb and 218Po.
    Reads in energy spectra from .pkl files. Normalise such that the sum of
    rates_vs_energy is 1.
    ToDo: 
        - Add in 218Po spectra (10% of counts to RnBetas), for now ignore
    Arguments:
        - weights: tuple (length 2) of weights to apply to the spectra,
        in the order given above. 212Pb relative to 1 uBq / kg.
    """

    def __init__(self, *args, weights=(1, 1,), **kwargs):
        assert len(weights) == 2, "Weights must be a tuple of length 3"
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        df_212Pb = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../nest/background_spectra/212Pb_spectrum.pkl'))
        # df_85Kr = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'background_spectra/85Kr_spectrum.pkl'))

        # assert (df_212Pb['energy_keV'].values == df_85Kr['energy_keV'].values).all(), \
        #     "Energy spectrum components must have equal energies"

        # Weight the spectra according to the weights provided, then combine
        df_212Pb_values = df_212Pb['spectrum_value_norm'].values #* weights[0]
        # df_85Kr_values = df_85Kr['spectrum_value_norm'].values * weights[1]

        # combined_rates_vs_energy = df_212Pb_values + df_85Kr_values

        # Re-normalise the summed spectra
        # combined_rates_vs_energy = combined_rates_vs_energy / sum(combined_rates_vs_energy)

        self.energies = tf.convert_to_tensor(df_212Pb['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_212Pb['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)


@export
class LZ24Kr85Source(LZ24ERSource):
    """Technically in the paper this is Kr85 + Ar39 + DetER/Rock Gammas
        The other two make 6% of the counts and are functionally degenerate so omit for now
        TODO:
             -Ar39 and Flat-ER contributions
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        df_85Kr = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../nest/background_spectra/85Kr_spectrum.pkl'))

        self.energies = tf.convert_to_tensor(df_85Kr['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_85Kr['spectrum_value_norm'].values, dtype=fd.float_type())

        super().__init__(*args, **kwargs)



##
# Calibration sources
##


@export
class LZ24DDSource(LZ24NRSource, fd.nest.DDSource):
    # t_start = pd.to_datetime('2022-04-19T00:00:00')
    # t_start = t_start.tz_localize(tz='America/Denver')

    # t_stop = pd.to_datetime('2022-04-19T00:00:00')
    # t_stop = t_stop.tz_localize(tz='America/Denver')

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


##
# Signal sources
##


@export
class LZ24WIMPSource(LZ24NRSource, fd.nest.nestWIMPSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)
@export
class LZ24WIMPSource(LZ24NRSource, fd.nest.nestSpatialWIMPSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        # == Need a new DMCalc Import for this
        # t_start = pd.to_datetime('2023-06-05T09:37:51')
        # self.t_start = t_start.tz_localize(tz='America/Denver')
    
        # t_stop = pd.to_datetime('2024-05-14T07:58:01')
        # self.t_stop = t_stop.tz_localize(tz='America/Denver')
        self.spatial_hist = load_position_map_hist('WS2024/spatial_maps/Uniform_spatial_map.pkl')
        super().__init__(*args, **kwargs)
        

@export
class LZ24FermionicDMSource(LZ24ERSource, fd.nest.FermionicDMSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


##
# Background sources
##



@export
class LZ24Pb214Source(LZ24ERSource, fd.nest.Pb214Source):#, fd.nest.nestSpatialRateERSource):
    def __init__(self, *args, bins=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        # if bins is None:
        #     bins=(np.sqrt(np.linspace(0.**2, 67.8**2, num=21)),
        #           np.linspace(86000., 936500., num=21))

        # mh = build_position_map_from_data('sr1/Pb214_spatial_map_data.pkl', ['r', 'drift_time'], bins)
        # self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZ24DetERSource(LZ24ERSource, fd.nest.DetERSource):#, fd.nest.nestSpatialRateERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        # mh = fd.get_lz_file('sr1/DetER_spatial_map_hist.pkl')
        # self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZ24BetaSource(LZ24ERSource, fd.nest.BetaSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


@export
class LZ24Xe136Source(LZ24ERSource, fd.nest.Xe136Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


@export
class LZ24vERSource(LZ24ERSource, fd.nest.vERSource, fd.nest.nestTemporalRateOscillationERSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

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
class LZ24Ar37Source(LZ24ERSource, fd.nest.Ar37Source, fd.nest.nestTemporalRateDecayERSource):
    def __init__(self, *args, time_constant_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        if time_constant_ns is None:
            self.time_constant_ns = (35.0 / np.log(2)) * 1e9 * 3600. * 24.
        else:
            self.time_constant_ns = time_constant_ns

        super().__init__(*args, **kwargs)


@export
class LZ24Xe127Source(LZ24ERSource,fd.nest.Xe127Source):#, fd.nest.nestSpatialTemporalRateDecayERSource):
    def __init__(self, *args, bins=None, time_constant_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        if bins is None:
            bins=(np.sqrt(np.linspace(0.**2, 67.8**2, num=51)),
                  np.linspace(LZ24ERSource().z_bottom, LZ24ERSource().z_top, num=51))

        # mh = fd.get_lz_file('sr1/Xe127_spatial_map_hist.pkl')
        # self.spatial_hist = mh

        if time_constant_ns is None:
            self.time_constant_ns = (36.4 / np.log(2)) * 1e9 * 3600. * 24.
        else:
            self.time_constant_ns = time_constant_ns

        super().__init__(*args, **kwargs)

    def mean_yield_electron(self, energy):
        # Weighted ER model different for L-shell ECs and Xe127
        # New Parameters for L-shell used here: LZLAMA/include/Modules/ModuleNest.hh#L193
        # Defined here: LZLLAMA/src/Detectors/LzTpcDetector.cc#L283
        weight_param_a = 0.48 #0.23
        weight_param_b = 0. #0.77
        weight_param_c = 0. #2.95
        weight_param_d = 0. #-1.44
        #field dependence ignored (with zeros above)
        weight_param_e = 421.15
        weight_param_f = 3.27

        gamma_weight = tf.cast(weight_param_a + weight_param_b * tf.math.erf(weight_param_c *
                          (tf.math.log(energy) + weight_param_d)) *
                          (1. - (1. / (1. + pow(self.drift_field / weight_param_e, weight_param_f)))),
                          fd.float_type())
        beta_weight = tf.cast(1. - gamma_weight, fd.float_type())


        nel_gamma = tf.cast(fd.lz.LZ24GammaSource.mean_yield_electron(self, energy), fd.float_type())
        nel_beta = tf.cast(fd.lz.LZ24ERSource.mean_yield_electron(self, energy), fd.float_type())
        return nel_gamma * gamma_weight + nel_beta * beta_weight


@export
class LZ24Xe124Source(LZ24ERSource, fd.nest.Xe124Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)

    def mean_yield_electron(self, energy,b=1.3):
        # Weighted ER model different for L-shell ECs and Xe127
        # New Parameters for L-shell used here: LZLAMA/include/Modules/ModuleNest.hh#L193
        # Defined here: LZLLAMA/src/Detectors/LzTpcDetector.cc#L283
        weight_param_a = 0.48 #0.23
        weight_param_b = 0. #0.77
        weight_param_c = 0. #2.95
        weight_param_d = 0. #-1.44
        #field dependence ignored (with zeros above)
        weight_param_e = 421.15
        weight_param_f = 3.27

        gamma_weight = tf.cast(weight_param_a + weight_param_b * tf.math.erf(weight_param_c *
                          (tf.math.log(energy) + weight_param_d)) *
                          (1. - (1. / (1. + pow(self.drift_field / weight_param_e, weight_param_f)))),
                          fd.float_type())
        beta_weight = tf.cast(1. - gamma_weight, fd.float_type())


        nel_gamma = tf.cast(fd.lz.LZ24GammaSource.mean_yield_electron(self, energy), fd.float_type())
        nel_beta = tf.cast(fd.lz.LZ24ERSource.mean_yield_electron(self, energy), fd.float_type())

        nel_raw = nel_gamma * gamma_weight + nel_beta * beta_weight
        
        # Based on xi and b, calculate scaling factor s
        xi_L=8.064#7.52
        
        s=tf.math.log(1+b*xi_L) / tf.math.log(1+xi_L) /b
        nel_DEC = nel_raw * s
        
        nel_temp = tf.where(tf.logical_and((energy >9.7), (energy <10.1)), nel_DEC, nel_raw) # select LL shell based on energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        return nel


@export
class LZ24B8Source(LZ24NRSource, fd.nest.B8Source, fd.nest.nestTemporalRateOscillationNRSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

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
class LZ24DetNRSource(LZ24NRSource):#, fd.nest.nestSpatialRateNRSource):
    """
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        df_DetNR = fd.get_lz_file('sr1/DetNR_spectrum.pkl', branch=BRANCH)

        self.energies = tf.convert_to_tensor(df_DetNR['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_DetNR['spectrum_value_norm'].values, dtype=fd.float_type())

        # mh = fd.get_lz_file('sr1/DetNR_spatial_map_hist.pkl')
        # self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZ24CH3TSource(LZ24ERSource,fd.nest.CH3TSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'

        super().__init__(*args, **kwargs)
@export
class LZ24C14Source(LZ24ERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        m_e = 510.9989461  # e- rest mass-energy [keV]
        aa = 0.0072973525664;          # fine structure constant
        ZZ = 7.;
        V0 = 0.495;  # effective offset in T due to screening of the nucleus by electrons
        qValue = 156.
        #energy range to avoid nans
        energies = tf.linspace(0.01, qValue, 1000)

        Ee=energies+m_e
        pe=np.sqrt(np.square(Ee)-np.square(m_e))
        dNdE_phasespace=pe * Ee * (qValue - energies)**2
        Ee_screen = Ee - V0
        W_screen = (Ee_screen) / m_e
        p_screen = np.sqrt(W_screen * W_screen - 1)
        p_screen=np.where(W_screen<1,0.,p_screen)
        WW = (Ee) / m_e
        pp = np.sqrt(WW * WW - 1)
        G_screen = (Ee_screen) / (m_e)  ## Gamma, Total energy(KE+M) over M
        B_screen = np.sqrt((G_screen * G_screen - 1)*(G_screen * G_screen))  # v/c of electron. Ratio of
        B_screen=np.where(G_screen<1,0.,B_screen)
        x_screen = (2 * pi * ZZ * aa) / B_screen
        F_nr_screen = W_screen * p_screen / (WW * pp) * x_screen * (1 / (1 - np.exp(-x_screen)))
        F_nr_screen=np.where(p_screen<=0. ,0. ,F_nr_screen)
        F_bb_screen =F_nr_screen *np.power(W_screen * W_screen * (1 + 4 * (aa * ZZ) * (aa * ZZ)) - 1,np.sqrt(1 - aa * aa * ZZ * ZZ) - 1)
        spectrum = dNdE_phasespace * F_bb_screen
        spectrum=spectrum/np.sum(spectrum)
        energies = tf.cast(energies, fd.float_type())
        rates_vs_energy = tf.cast(spectrum, fd.float_type())
        self.energies = tf.cast(energies, fd.float_type())
        self.rates_vs_energy = tf.cast(spectrum, fd.float_type())
        super().__init__(*args, **kwargs)


@export
class LZ24AccidentalsSource(fd.TemplateSource):
    """
        Adapted to allow set_data to work for toys
    """
    path_s1_corr_latest = 'WS2024/s1Area_Correction_TPC_WS2024_radon_31Jan2024.json'
    path_s2_corr_latest = 'WS2024/s2Area_Correction_TPC_WS2024_radon_31Jan2024.json'

    def __init__(self, *args, simulate_safety_factor=2.,lz_source = None, **kwargs):
        hist = fd.get_lz_file('WS2024/accidentals_model.pkl')
        if lz_source is None:
            print("No source given, generating generic one")
            self.lz_source = LZ24ERSource()
        else:
            self.lz_source = lz_source
        hist_values = hist['hist_values']
        s1_edges = hist['cs1_phd_edges']
        s2_edges = hist['log10_cs2_phd_edges']

        mh = Histdd(bins=[len(s1_edges) - 1, len(s2_edges) - 1]).from_histogram(hist_values, bin_edges=[s1_edges, s2_edges])
        mh = mh / mh.n
        mh = mh / mh.bin_volumes()

        self.simulate_safety_factor = simulate_safety_factor

        try:
            self.s1_map_latest = fd.InterpolatingMap(fd.get_lz_file(self.path_s1_corr_latest))
            self.s2_map_latest = fd.InterpolatingMap(fd.get_lz_file(self.path_s2_corr_latest))
        except Exception:
            print("Could not load maps; setting position corrections to 1")
            self.s1_map_latest = None
            self.s2_map_latest = None

        super().__init__(*args, template=mh, interpolate=True,
                         axis_names=('cs1_phd', 'log10_cs2_phd'),
                         **kwargs)

    def _annotate(self,ignore_priors=False, **kwargs):
        """
            Currently quite weird, ignore priors isn't used
        """
        super()._annotate(**kwargs)

        self.data[self.column] /= (1 + self.lz_source.double_pe_fraction)
        self.data[self.column] /= (np.log(10) * self.data['cs2'].values)
        self.data[self.column] /= self.data['s1_pos_corr_latest'].values
        self.data[self.column] *= (np.exp(self.data['drift_time'].values /
                                          self.data['electron_lifetime'].values) /
                                   self.data['s2_pos_corr_latest'].values)

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        df = super().simulate(int(n_events * self.simulate_safety_factor), fix_truth=fix_truth,
                              full_annotate=full_annotate, keep_padding=keep_padding, **params)

        df_pos = pd.DataFrame(self.lz_source.model_blocks[0].draw_positions(len(df)))
        df = df.join(df_pos)
        #ISSUE WITH OVER-RIDING THE ACCIDENTALS SOURCE AAAHHHHH
        df_time = pd.DataFrame(self.lz_source.model_blocks[0].draw_time(len(df)), columns=['event_time'])
        df = df.join(df_time)

        self.lz_source.add_extra_columns(df)
        df['acceptance'] = df['fv_acceptance'].values * df['resistor_acceptance'].values * df['timestamp_acceptance'].values

        df['cs1'] = df['cs1_phd'] * (1 + self.lz_source.double_pe_fraction)
        df['cs2'] = 10**df['log10_cs2_phd'] * (1 + self.lz_source.double_pe_fraction)
        df['s1'] = df['cs1'] * df['s1_pos_corr_latest']
        df['s2'] = (
            df['cs2']
            * df['s2_pos_corr_latest']
            / np.exp(df['drift_time'] / df['electron_lifetime']))

        df['acceptance'] *= (df['s2'].values >= self.lz_source.S2_min)

        df = df[df['acceptance'] == 1.]
        df = df.reset_index(drop=True)

        try:
            df = df.head(n_events)
        except Exception:
            raise RuntimeError('Too many events lost due to spatial/temporal cuts, try increasing \
                                simulate_safety_factor')
        df = df.drop(columns=['fv_acceptance', 'resistor_acceptance', 'timestamp_acceptance',
                              'acceptance'])

        self.lz_source.add_extra_columns(df)

        return df

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if (self.s1_map_latest is not None) and (self.s2_map_latest is not None):
            d['s1_pos_corr_latest'] = self.s1_map_latest(
                np.transpose([d['x_obs'].values,
                              d['y_obs'].values,
                              d['drift_time'].values * 1e-9 / 1e-6]))
            d['s2_pos_corr_latest'] = self.s2_map_latest(
                np.transpose([d['x_obs'].values,
                              d['y_obs'].values]))
        else:
            d['s1_pos_corr_latest'] = np.ones_like(d['x_obs'].values)
            d['s2_pos_corr_latest'] = np.ones_like(d['x_obs'].values)

        self.lz_source = fd.lz.LZ24ERSource()

        if 'event_time' in d.columns and 'electron_lifetime' not in d.columns:
            d['electron_lifetime'] = self.lz_source.get_elife(d['event_time'].values)

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr_latest']
            d['cs1_phd'] = d['cs1'] / (1 + self.lz_source.double_pe_fraction)
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_pos_corr_latest']
                * np.exp(d['drift_time'] / d['electron_lifetime']))
            d['log10_cs2_phd'] = np.log10(d['cs2'] / (1 + self.lz_source.double_pe_fraction))

    def estimate_position_acceptance(self, n_trials=int(1e5)):
        
        df = pd.DataFrame(self.lz_source.model_blocks[0].draw_positions(n_trials))
        df_time = pd.DataFrame(self.lz_source.model_blocks[0].draw_time(n_trials), columns=['event_time'])
        df = df.join(df_time)

        self.lz_source.add_extra_columns(df)
        df['acceptance'] = df['fv_acceptance'].values * df['resistor_acceptance'].values * df['timestamp_acceptance'].values

        return np.sum(df['acceptance'].values) / n_trials
    def set_data(self,data=None,
                 data_is_annotated=False,
                 ignore_priors=False,
                 input_column_index=None,
                 input_data_tensor=None,
                 output_data_tensor=None,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 **params):
        """
            Anntotating Templates incredibly fast, best just let it create a new diff rates
            column. 
        """
        data_is_annotated = False
        super().set_data(data=data,
                 data_is_annotated=data_is_annotated,
                 ignore_priors=ignore_priors,
                 input_column_index=input_column_index,
                 input_data_tensor=input_data_tensor,
                 output_data_tensor=output_data_tensor,
                 _skip_tf_init=_skip_tf_init,
                 _skip_bounds_computation=_skip_bounds_computation,
                 **params)


##
# Source groups
##


@export
class LZ24ERSourceGroup(LZWS2024Source, fd.nest.nestERSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


@export
class LZ24ERGammaWeightedSourceGroup(LZWS2024Source, fd.nest.nestERGammaWeightedSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


@export
class LZ24NRSourceGroup(LZWS2024Source, fd.nest.nestNRSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)


@export            
class LZ24ModifiedERSourceGroup(LZ24ERSource,fd.nest.nestModifiedERSourceGroup):
     def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)
@export
class LZ24ModifiedNRSourceGroup(LZ24NRSource,fd.nest.nestModifiedNRSourceGroup):
     def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_WS2024'
        super().__init__(*args, **kwargs)
