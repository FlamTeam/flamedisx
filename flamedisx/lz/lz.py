"""lz detector implementation

"""
import numpy as np
import tensorflow as tf

import configparser
import os
import pandas as pd

import flamedisx as fd
from .. import nest as fd_nest

import pickle as pkl

from multihist import Histdd

export, __all__ = fd.exporter()


##
# Useful functions
##

from scipy.interpolate import RegularGridInterpolator
## Interpolate S2 splitting efficiency 

def SR3S2splittingReconEff(S2c, driftTime_us, hist,weird=False):    
    ## Make values more python friendly
    weights = np.reshape(hist[0], hist[0].size)
    zmax =  np.max(weights)
    zmin = np.min(weights[weights>0.])
    ## Read into interpolator
    xentries_vals = np.array(hist[1][:-1])
    yentries_vals = np.array(hist[2][:-1])
    Interp = RegularGridInterpolator((xentries_vals,yentries_vals), hist[0])
    
    ## Convert S2c to mean N_e
    mean_SE_area = 44.5 # phd/e-
    mean_Ne = S2c/mean_SE_area

    ## Initialize acceptance values
    acceptance = np.ones_like(mean_Ne)
    ## curves not defined for above 100 e- 
    ## Assume 100% eff. (probably ok...)
    acceptance[mean_Ne>np.max(xentries_vals)] = 1.
    ## Also not defined for < 10e-
    ## ok with 15e- ROI threshold
    acceptance[mean_Ne<np.min(xentries_vals)] = 0.
    
    temp_drift = driftTime_us
    temp_drift[temp_drift<np.min(yentries_vals)] = np.min(yentries_vals)
    temp_drift[temp_drift>np.max(yentries_vals)] = np.max(yentries_vals)
    
    mask = (mean_Ne>=np.min(xentries_vals)) & (mean_Ne<=np.max(xentries_vals))
    ## Acceptances are provided in percent - divide by 100.
    acceptance[mask] = Interp(np.vstack([mean_Ne[mask],temp_drift[mask]]).T)/100.
    
    
    return acceptance
##Build S2raw acceptance
def SR3TriggerAcceptance(S2raw, which='mean'):
    
    # 50% threshold = 106.89 +/- 0.43 phd
    # 95% threshold = 160.7 +/- 3.5 phd
    # k = 0.0547 +/- 0.0035 phd-1
    
    x0 = 160.7
    k = 0.0547
    
    if which == 'plus1sigma' or which == 'p1sig':
        x0 -=  3.5
        k +=  0.0035
    if which == 'minus1sigma' or which == 'm1sig':
        x0 +=  3.5
        k -=  0.0035
        
    
    accValues =  ( 1./( 1 + np.exp( -k*(S2raw-x0) ) ) )
    
    ## make sure acceptances can't go above 1. or below 0.
    accValues[accValues>1.] = 1. 
    accValues[accValues<0.] = 0.
    
    return accValues


# Define polynomial coefficients associated with each azimuthal wall position;
# these start at the 7 o'clock position and progress anti-clockwise
phi_coeffs = [[-1.78880746e-13, 4.91268301e-10, -4.96134607e-07, 2.26430932e-04, -4.71792008e-02, 7.33811298e+01],
              [-1.72264463e-13, 4.59149636e-10, -4.59325165e-07, 2.14612376e-04, -4.85599108e-02, 7.35290867e+01],
              [-3.17099156e-14, 7.26336129e-11, -6.99495385e-08, 3.85531008e-05, -1.33386004e-02, 7.18002889e+01],
              [-6.12280314e-14, 1.67968911e-10, -1.83625538e-07, 1.00457608e-04, -2.86728022e-02, 7.22754350e+01],
              [-1.89897962e-14, 1.52777215e-11, -2.79681508e-09, 1.25689887e-05, -1.33093804e-02, 7.17662251e+01],
              [-2.32118621e-14, 7.30043322e-11, -9.40606298e-08, 6.29728588e-05, -2.28150175e-02, 7.22661091e+01],
              [-8.29749194e-14, 2.31096069e-10, -2.47867121e-07, 1.27576029e-04, -3.24702414e-02, 7.26357609e+01],
              [-2.00718008e-13, 5.44135757e-10, -5.59484466e-07, 2.73028553e-04, -6.46879791e-02, 7.45264998e+01],
              [-7.77420021e-14, 1.97357045e-10, -1.90016273e-07, 8.99659454e-05, -2.30169916e-02, 7.25038258e+01],
              [-5.27296334e-14, 1.49415580e-10, -1.58205132e-07, 8.00275441e-05, -2.13559394e-02, 7.23995451e+01],
              [-6.00198219e-14, 1.55333004e-10, -1.60367908e-07, 7.97754165e-05, -1.94435594e-02, 7.22714399e+01],
              [-8.89919309e-14, 2.40830027e-10, -2.57060475e-07, 1.33002951e-04, -3.32969110e-02, 7.28696020e+01]]


# Use the above set of coefficients to define azimuthal wall position contours
phi_walls = [np.poly1d(phi_coeffs[i]) for i in range(len(phi_coeffs))]


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


##
# Flamedisx sources
##

##
# Common to all LZ sources
##


class LZSource:
    path_s1_corr_LZAP = 'new_data/s1Area_Correction_TPC_SR3_radon_31Jan2024.json'#'new_data/s1Area_Correction_TPC_SR3_06Apr23.json'
    path_s2_corr_LZAP = 'new_data/s2Area_Correction_TPC_SR3_radon_31Jan2024.json'#'new_data/s2Area_Correction_TPC_SR3_06Apr23.json'
    path_s1_corr_latest = 'new_data/s1Area_Correction_TPC_SR3_radon_31Jan2024.json'
    path_s2_corr_latest = 'new_data/s2Area_Correction_TPC_SR3_radon_31Jan2024.json'

    path_s1_acc_curve = 'new_data/cS1_acceptance_curve.json'
    # path_s2_acc_curve = 'sr1/cS2_acceptance_curve.pkl' Not used
    path_s2_splitting_curve='lz_private_data/new_data/WS2024_S2splittingReconEff_mean.pickle'
    def __init__(self, *args, ignore_maps=False, ignore_acc_maps=False, cap_upper_cs1=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.cap_upper_cs1 = cap_upper_cs1

        assert kwargs['detector'] in ('lz_SR3',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.cS1_min = config.getfloat('NEST', 'cS1_min_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.cS1_max = config.getfloat('NEST', 'cS1_max_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.S2_min = config.getfloat('NEST', 'S2_min_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.cS2_max = config.getfloat('NEST', 'cS2_max_config') * (1 + self.double_pe_fraction)  # phd to phe
        self.ignore_acceptances_maps=False
        if ignore_acc_maps:
            print("ignoring acceptances")
            self.ignore_acceptances_maps = True

            self.cs1_acc_domain = None
            self.cS2_drift_acceptance_hist = None
            
        else:
            try:
                df_S1_acc = fd.get_lz_file(self.path_s1_acc_curve)
                # df_S2_acc =fd.get_lz_file(self.path_s2_acc_curve)
                self.cs1_acc_domain = np.array(df_S1_acc['cS1_phd']) * (1 + self.double_pe_fraction)  # phd to phe
                self.cs1_acc_curve = np.array(df_S1_acc['cS1_acceptance'])
                input_curve=pkl.load(open(self.path_s2_splitting_curve,'rb'))
                self.cS2_drift_acceptance_hist= (input_curve[0],
                                                 input_curve[1],
                                                 input_curve[2])
                
                # self.log10_cs2_acc_domain = df_S2_acc['log10_cS2_phd'].values + \
                #     np.log10(1 + self.double_pe_fraction)  # log_10(phd) to log_10(phe)
                # self.log10_cs2_acc_curve = df_S2_acc['cS2_acceptance'].values
            except Exception:
                print("Could not load acceptance curves; setting to 1")

                self.cs1_acc_domain = None
                self.cS2_drift_acceptance_hist = None

        if ignore_maps:
            print("ingoring LCE maps")
            self.s1_map_LZAP = None
            self.s2_map_LZAP = None
            self.s1_map_latest = None
            self.s2_map_latest = None
        else:
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

       
    @staticmethod
    def photon_detection_eff(z, *, g1=0.1122):
        return g1 * tf.ones_like(z)

    @staticmethod
    def s2_photon_detection_eff(z, *, g1_gas=0.076404):
        return g1_gas * tf.ones_like(z)

    @staticmethod
    def get_elife(event_time):
        t0 = pd.to_datetime('2021-09-30T08:00:00').value

        time_diff = event_time - t0
        days_since_t0 = time_diff / (24. * 3600 * 1e9)

        elife = np.piecewise(days_since_t0, [days_since_t0 <= 104.,
                                             (days_since_t0 > 104.) & (days_since_t0 <= 174.4167),
                                             days_since_t0 > 174.4167],
                             [lambda days_since_t0: (5526.52 - np.exp(27.0832 - 0.254022 * days_since_t0)) * 1000.,
                              lambda days_since_t0: (8315.35 - np.exp(9.5533 - 0.0157167 * days_since_t0)) * 1000.,
                              lambda days_since_t0: (7935.03 - np.exp(35.0987 - 0.15228 * days_since_t0)) * 1000.])

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
        if not self.ignore_acceptances_maps:
            acceptance *= cs1_acc_curve

        return acceptance

    def s2_acceptance(self, s2, cs2, cs2_acc_curve,
                      fv_acceptance, resistor_acceptance, timestamp_acceptance):

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
            d['electron_lifetime'] = self.elife#self.get_elife(d['event_time'].values)

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr_LZAP']
            d['cs1_phd'] = d['cs1'] / (1 + self.double_pe_fraction)
            if self.cap_upper_cs1 == True:
                d['cs1'] = np.where(d['cs1'].values <= self.cS1_max, d['cs1'].values, self.cS1_max)
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_pos_corr_LZAP']
                * np.exp(d['drift_time'] / d['electron_lifetime']))
            d['log10_cs2_phd'] = np.log10(d['cs2'] / (1 + self.double_pe_fraction))

        if 'cs1' in d.columns and 's1' not in d.columns:
            d['s1'] = d['cs1'] * d['s1_pos_corr_LZAP']
        if 'cs2' in d.columns and 's2' not in d.columns:
            d['s2'] = (
                d['cs2']
                * d['s2_pos_corr_LZAP']
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
                d['cs2_acc_curve'] = SR3S2splittingReconEff(d['cs2'].values/(1+self.double_pe_fraction),
                                                            d['drift_time'].values/1e3,#us
                                                            self.cS2_drift_acceptance_hist)
                #if cs2 exists s2 must (handled above)
                d['cs2_acc_curve'] *=SR3TriggerAcceptance(d['s2'].values/(1+self.double_pe_fraction))
                # interpolate_acceptance(
                #     np.log10(d['cs2'].values),
                #     self.log10_cs2_acc_domain,
                #     self.log10_cs2_acc_curve)
            else:
                d['cs2_acc_curve'] = np.ones_like(d['cs2'].values)
        
            

        if 'fv_acceptance' not in d.columns:
            
            x = d['x'].values
            y = d['y'].values
            dt=d['drift_time'].values/1e3
            # Define radial contour for N_tot = 0.01 expected counts (drift-∆R_phi space)
            FV_poly = np.poly1d([-4.44147071e-14,  1.43684777e-10, -1.82739476e-07,
                                 1.02160174e-04, -2.31617857e-02, -2.05932471e+00])
            contour=FV_poly(dt)
            #===CALCULATE DR_DPHI
            # Define azimuthal slices
            n_phi_slices = 12
            phi_slices = np.linspace(-np.pi, np.pi, n_phi_slices + 1) + np.pi/4
            phi_slices[phi_slices > np.pi] -= 2*np.pi

            # Calculate event radii and angles, then mask them according to each slice
            R = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            phi_cuts = [((phi >= phi_slices[i]) & (phi < phi_slices[i + 1])) if not (phi_slices[i] > phi_slices[i + 1]) else \
                       ~((phi <= phi_slices[i]) & (phi > phi_slices[i + 1])) for i in range(n_phi_slices)]

            # Calculate dR_phi by replacing relevant points in loops over phi slices, then return
            dR_phi = np.zeros(len(R))
            for i, p in enumerate(phi_cuts):
                dR_phi[p] = R[p] - phi_walls[i](dt[p])

            # Segment events by whether or not they're in the expandable part
            expandable = (dt > 71) & (dt < 900)

            # Get the radial cut as a mask between two parts
            expansion = 0
            mask = ((dR_phi < (contour + expansion)) & expandable) | ((dR_phi < contour) & ~expandable)
            
            #cut the drift time 
            dt_cut = (dt > 71) & (dt < 1030)
                
            d['fv_acceptance'] =dt_cut&mask&(dR_phi<=0)

        if 'resistor_acceptance' not in d.columns:
            x = d['x'].values
            y = d['y'].values

            res1X = -69.8
            res1Y = 3.5
            res1R = 6
            res2X = -67.5
            res2Y = -14.3
            res2R = 6

            not_inside_res1 = np.where(np.sqrt( (x-res1X)*(x-res1X) + (y-res1Y)*(y-res1Y) ) < res1R, 0., 1.)
            not_inside_res2 = np.where(np.sqrt( (x-res2X)*(x-res2X) + (y-res2Y)*(y-res2Y) ) < res2R, 0., 1.)

            d['resistor_acceptance'] =not_inside_res1 * not_inside_res2
        if 'timestamp_acceptance' not in d.columns:
            d['timestamp_acceptance'] = np.ones_like(d['event_time'],dtype=bool)
        
##
# Different interaction types: flat spectra
##

GAS_CONSTANT = 8.314
N_AVAGADRO = 6.0221409e23
A_XENON = 131.293
XENON_REF_DENSITY = 2.90

@export
class LZERSource(LZSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)
    

@export
class LZGammaSource(LZSource, fd.nest.nestGammaSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZERGammaWeightedSource(LZSource, fd.nest.nestERGammaWeightedSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZNRSource(LZSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


##
# Calibration sources
##


@export
class LZCH3TSource(LZSource, fd.nest.CH3TSource):
    t_start = pd.to_datetime('2022-04-19T00:00:00')
    t_start = t_start.tz_localize(tz='America/Denver')

    t_stop = pd.to_datetime('2022-04-19T00:00:00')
    t_stop = t_stop.tz_localize(tz='America/Denver')

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_elife(event_time):
        return 6600000.


@export
class LZDDSource(LZSource, fd.nest.DDSource):
    t_start = pd.to_datetime('2022-04-19T00:00:00')
    t_start = t_start.tz_localize(tz='America/Denver')

    t_stop = pd.to_datetime('2022-04-19T00:00:00')
    t_stop = t_stop.tz_localize(tz='America/Denver')

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_elife(event_time):
        return 6600000.


##
# Signal sources
##


@export
class LZWIMPSource(LZSource, fd.nest.nestWIMPSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZFermionicDMSource(LZSource, fd.nest.FermionicDMSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


##
# Background sources
##


@export
class LZPb214Source(LZSource, fd.nest.Pb214Source, fd.nest.nestSpatialRateERSource):
    def __init__(self, *args, bins=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

        if bins is None:
            bins=(np.sqrt(np.linspace(0.**2, 67.8**2, num=21)),
                  np.linspace(86000., 936500., num=21))

        mh = build_position_map_from_data('Pb214_spatial_map_data.pkl', ['r', 'drift_time'], bins)
        self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZDetERSource(LZSource, fd.nest.DetERSource, fd.nest.nestSpatialRateERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

        mh = fd.get_lz_file('DetER_spatial_map_hist.pkl')
        self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZBetaSource(LZSource, fd.nest.BetaSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZXe136Source(LZSource, fd.nest.Xe136Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZvERSource(LZSource, fd.nest.vERSource, fd.nest.nestTemporalRateOscillationERSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

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
class LZAr37Source(LZSource, fd.nest.Ar37Source, fd.nest.nestTemporalRateDecayERSource):
    def __init__(self, *args, time_constant_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

        if time_constant_ns is None:
            self.time_constant_ns = (35.0 / np.log(2)) * 1e9 * 3600. * 24.
        else:
            self.time_constant_ns = time_constant_ns

        super().__init__(*args, **kwargs)


@export
class LZXe124Source(LZSource, fd.nest.Xe124Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZXe127Source(LZSource, fd.nest.Xe127Source, fd.nest.nestSpatialTemporalRateDecayERSource):
    def __init__(self, *args, bins=None, time_constant_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

        if bins is None:
            bins=(np.sqrt(np.linspace(0.**2, 67.8**2, num=51)),
                  np.linspace(fd.lz.LZERSource().z_bottom, fd.lz.LZERSource().z_top, num=51))

        mh = fd.get_lz_file('Xe127_spatial_map_hist.pkl')
        self.spatial_hist = mh

        if time_constant_ns is None:
            self.time_constant_ns = (36.4 / np.log(2)) * 1e9 * 3600. * 24.
        else:
            self.time_constant_ns = time_constant_ns

        super().__init__(*args, **kwargs)


@export
class LZB8Source(LZSource, fd.nest.B8Source, fd.nest.nestTemporalRateOscillationNRSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

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
class LZDetNRSource(LZSource, fd.nest.nestSpatialRateNRSource):
    """
    """

    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'

        df_DetNR = fd.get_lz_file('DetNR_spectrum.pkl')

        self.energies = tf.convert_to_tensor(df_DetNR['energy_keV'].values, dtype=fd.float_type())
        self.rates_vs_energy = tf.convert_to_tensor(df_DetNR['spectrum_value_norm'].values, dtype=fd.float_type())

        mh = fd.get_lz_file('DetNR_spatial_map_hist.pkl')
        self.spatial_hist = mh

        super().__init__(*args, **kwargs)


@export
class LZAccidentalsSource(fd.TemplateSource):
    path_s1_corr_LZAP = 's1_map_22Apr22.json'
    path_s2_corr_LZAP = 's2_map_30Mar22.json'

    def __init__(self, *args, simulate_safety_factor=2., **kwargs):
        hist = fd.get_lz_file('Accidentals.npz')

        hist_values = hist['hist_values']
        s1_edges = hist['s1_edges']
        s2_edges = hist['s2_edges']

        mh = Histdd(bins=[len(s1_edges) - 1, len(s2_edges) - 1]).from_histogram(hist_values, bin_edges=[s1_edges, s2_edges])
        mh = mh / mh.n
        mh = mh / mh.bin_volumes()

        self.simulate_safety_factor = simulate_safety_factor

        try:
            self.s1_map_LZAP = fd.InterpolatingMap(fd.get_lz_file(self.path_s1_corr_LZAP))
            self.s2_map_LZAP = fd.InterpolatingMap(fd.get_lz_file(self.path_s2_corr_LZAP))
        except Exception:
            print("Could not load maps; setting position corrections to 1")
            self.s1_map_LZAP = None
            self.s2_map_LZAP = None

        super().__init__(*args, template=mh, interp_2d=True,
                         axis_names=('cs1_phd', 'log10_cs2_phd'),
                         **kwargs)

    def _annotate(self, **kwargs):
        super()._annotate(**kwargs)

        lz_source = LZERSource()
        self.data[self.column] /= (1 + lz_source.double_pe_fraction)
        self.data[self.column] /= (np.log(10) * self.data['cs2'].values)
        self.data[self.column] /= self.data['s1_pos_corr_LZAP'].values
        self.data[self.column] *= (np.exp(self.data['drift_time'].values /
                                          self.data['electron_lifetime'].values) /
                                   self.data['s2_pos_corr_LZAP'].values)

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        df = super().simulate(int(n_events * self.simulate_safety_factor), fix_truth=fix_truth,
                              full_annotate=full_annotate, keep_padding=keep_padding, **params)

        lz_source = LZERSource()
        df_pos = pd.DataFrame(lz_source.model_blocks[0].draw_positions(len(df)))
        df = df.join(df_pos)

        df_time = pd.DataFrame(lz_source.model_blocks[0].draw_time(len(df)), columns=['event_time'])
        df = df.join(df_time)

        lz_source.add_extra_columns(df)
        df['acceptance'] = df['fv_acceptance'].values * df['resistor_acceptance'].values * df['timestamp_acceptance'].values

        df['cs1'] = df['cs1_phd'] * (1 + lz_source.double_pe_fraction)
        df['cs2'] = 10**df['log10_cs2_phd'] * (1 + lz_source.double_pe_fraction)
        df['s1'] = df['cs1'] * df['s1_pos_corr_LZAP']
        df['s2'] = (
            df['cs2']
            * df['s2_pos_corr_LZAP']
            / np.exp(df['drift_time'] / df['electron_lifetime']))

        df['acceptance'] *= (df['s2'].values >= lz_source.S2_min)

        df = df[df['acceptance'] == 1.]
        df = df.reset_index(drop=True)

        try:
            df = df.head(n_events)
        except Exception:
            raise RuntimeError('Too many events lost due to spatial/temporal cuts, try increasing \
                                simulate_safety_factor')
        df = df.drop(columns=['fv_acceptance', 'resistor_acceptance', 'timestamp_acceptance',
                              'acceptance'])

        lz_source.add_extra_columns(df)

        return df

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if (self.s1_map_LZAP is not None) and (self.s2_map_LZAP is not None):
            d['s1_pos_corr_LZAP'] = self.s1_map_LZAP(
                np.transpose([d['x'].values,
                              d['y'].values,
                              d['drift_time'].values * 1e-9 / 1e-6]))
            d['s2_pos_corr_LZAP'] = self.s2_map_LZAP(
                np.transpose([d['x'].values,
                              d['y'].values]))
        else:
            d['s1_pos_corr_LZAP'] = np.ones_like(d['x'].values)
            d['s2_pos_corr_LZAP'] = np.ones_like(d['x'].values)

        lz_source = LZERSource()

        if 'event_time' in d.columns and 'electron_lifetime' not in d.columns:
            d['electron_lifetime'] =self.elife # lz_source.get_elife(d['event_time'].values)

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr_LZAP']
            d['cs1_phd'] = d['cs1'] / (1 + lz_source.double_pe_fraction)
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_pos_corr_LZAP']
                * np.exp(d['drift_time'] / d['electron_lifetime']))
            d['log10_cs2_phd'] = np.log10(d['cs2'] / (1 + lz_source.double_pe_fraction))

    def estimate_position_acceptance(self, n_trials=int(1e5)):
        lz_source = LZERSource()
        df = pd.DataFrame(lz_source.model_blocks[0].draw_positions(n_trials))
        df_time = pd.DataFrame(lz_source.model_blocks[0].draw_time(n_trials), columns=['event_time'])
        df = df.join(df_time)

        lz_source.add_extra_columns(df)
        df['acceptance'] = df['fv_acceptance'].values * df['resistor_acceptance'].values * df['timestamp_acceptance'].values

        return np.sum(df['acceptance'].values) / n_trials


##
# Source groups
##


@export
class LZERSourceGroup(LZSource, fd.nest.nestERSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZERGammaWeightedSourceGroup(LZSource, fd.nest.nestERGammaWeightedSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)


@export
class LZNRSourceGroup(LZSource, fd.nest.nestNRSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz_SR3'
        super().__init__(*args, **kwargs)
