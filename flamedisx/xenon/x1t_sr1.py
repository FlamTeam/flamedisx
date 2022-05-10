"""XENON1T SR1 implementation"""
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
from multihist import Histdd

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis


##
# Parameters
##
DEFAULT_G1 = 0.142
DEFAULT_G2 = 11.4  # g2 bottom

DEFAULT_AREA_FRACTION_TOP = 0.63  # fraction of light from top array
DEFAULT_P_DPE = 0.219
DEFAULT_EXTRACTION_EFFICIENCY = 0.96

DEFAULT_ELECTRON_LIFETIME = 641e3
DEFAULT_DRIFT_VELOCITY = 1.34 * 1e-4   # cm/ns, from analysis paper II

DEFAULT_DRIFT_FIELD = 81.

DEFAULT_G2_TOTAL = DEFAULT_G2 / (1.-DEFAULT_AREA_FRACTION_TOP)
DEFAULT_SINGLE_ELECTRON_GAIN = DEFAULT_G2_TOTAL / DEFAULT_EXTRACTION_EFFICIENCY
DEFAULT_SINGLE_ELECTRON_WIDTH = 0.25 * DEFAULT_SINGLE_ELECTRON_GAIN

# Official numbers from BBF
DEFAULT_S1_RECONSTRUCTION_BIAS_PIVOT = 0.5948841302444277
DEFAULT_S2_RECONSTRUCTION_BIAS_PIVOT = 0.49198507921078005
DEFAULT_S1_RECONSTRUCTION_EFFICIENCY_PIVOT = -0.31816407029454036


def read_maps_tf(path_bag, is_bbf=False):
    """ Function to read reconstruction bias/combined cut acceptances/dummy maps.
    Note that this implementation fundamentally assumes upper and lower bounds
    have exactly the same domain definition.
    :param path_bag: List with filenames of acceptance maps
    :param is_bbf: True if reading file from BBF folder.
    :return: List of acceptance maps and their domain definitions
    """
    data_bag = []
    yy_ref_bag = []
    for loc_path in path_bag:
        if is_bbf:
            tmp = fd.get_bbf_file(loc_path)
        else:
            tmp = fd.get_nt_file(loc_path)
        yy_ref_bag.append(tf.convert_to_tensor(tmp['map'], dtype=fd.float_type()))
        data_bag.append(tmp)
    domain_def = tmp['coordinate_system'][0][1]
    return yy_ref_bag, domain_def

def interpolate_tf(sig_tf, fmap, domain):
    """ Function to interpolate values from map given S1, S2 values
    :param sig: S1 or S2 values as tf tensor of type float
    :param fmap: specific acceptance map to be interpolated from returned by read_maps_tf
    :param domain: domain returned by read_maps_tf
    :return: Tensor of interpolated map values (same shape as x)
    """
    return tfp.math.interp_regular_1d_grid(x=sig_tf,
            x_ref_min=domain[0], x_ref_max=domain[1],
            y_ref=fmap, fill_value='constant_extension')

def calculate_reconstruction_bias(sig, fmap, domain_def, pivot_pt):
    """ Computes the reconstruction bias mean given the pivot point.

    The pax reconstruction bias mean is a function of the S1 or S2 size and is
    defined as:
    bias = (reconstructed_area - true_area)/ true_area
    reconstructed_area = (bias+1)*true_area

    It has a lower bound and upper bound because we do not know exactly how this
    bias varies as a function of the actual waveform. The bias interpolated
    linearly between the lower and upper bound according to a single scalar, the
    pivot point:
    bias = (upper_bound - lower_bound)*pivot + lower_bound

    :param sig: S1 or S2 values
    :param fmap: map returned by read_maps_tf
    :param domain_def: domain returned by read_maps_tf
    :param pivot_pt: Pivot point value (scalar)
    :return: Tensor of bias values (same shape as sig)
    """
    sig_tf = tf.convert_to_tensor(sig, dtype=fd.float_type())
    bias_low = interpolate_tf(sig_tf, fmap[0], domain_def)
    bias_high = interpolate_tf(sig_tf, fmap[1], domain_def)

    bias = (bias_high - bias_low) * pivot_pt + bias_low
    bias_out = bias + tf.ones_like(bias)

    return bias_out

def calculate_reconstruction_efficiency(sig, fmap, domain_def, pivot_pt):
    """ Computes the reconstruction efficiency given the pivot point
    :param sig: photon detected
    :param fmap: map returned by read_maps_tf
    :param domain_def: domain returned by read_maps_tf
    :param pivot_pt: Pivot point value (scalar)
    :return: Tensor of bias values (same shape as sig)
    """
    sig_tf = tf.convert_to_tensor(sig, dtype=fd.float_type())
    bias_median = interpolate_tf(sig_tf, fmap[1], domain_def)

    bias_diff = tf.cond(
        pivot_pt < 0,
        lambda: bias_median - interpolate_tf(sig_tf, fmap[0], domain_def),
        lambda: interpolate_tf(sig_tf, fmap[2], domain_def) - bias_median)
    return bias_median + pivot_pt * bias_diff

## 
# Utility for the spatial template construction 
##
def construct_exponential_r_spatial_hist(n = 2e6, max_r = 42.8387,
                                         exp_const=1.36 ):
  """ Utility function to construct a spatial template for sources
  :param n: number of samples in the template
  :param max_r: maximum radius for the exponential r template
  :param exp_const: exponential constant for the exponential function in r
  :return: multihist.Histdd 3D normalised histogram in the format needed 
           for the spatial_hist method of fd.SpatialRateERSource
  """
  assert max_r < 50, "max_r should be < 50cm."
  theta_arr= np.random.uniform(0, 2 * np.pi, size = n)
  r = np.zeros(n)
  z = np.zeros(n)
  theta_edges = np.linspace(0,2 * np.pi, 361)
  z_edges = np.linspace(-100, 0, 101)
  r_edges = np.sqrt(np.linspace(0, 50**2,51))
  # For SR1 FV
  for i in range(len(r)):
    rr = max_r - stats.expon.rvs(loc = 0, scale = exp_const, size = 1)
    zr = np.random.uniform(-94., -8.)
    while  ((-94 > zr) | (zr > -8) | (rr > max_r) | \
                (zr > -2.63725 - 0.00946597 * rr * rr) | \
                (zr < -158.173 + 0.0456094 * rr * rr)):
      rr = max_r - stats.expon.rvs(loc = 0, scale = exp_const, size = 1)
      zr = np.random.uniform(-94., -8.)
    r[i] = rr 
    z[i] = zr
  hist, edges = np.histogramdd([r,theta_arr,z],bins=(r_edges, theta_edges,
                                                 z_edges))
  exp_spatial_rate = Histdd.from_histogram(hist, bin_edges = edges,
                                          axis_names = ['r','theta','z'])
  return exp_spatial_rate / np.mean(exp_spatial_rate.histogram / \
                                    exp_spatial_rate.bin_volumes())


##
# Flamedisx sources
##
class SR1Source:
    model_attributes = ('s2_area_fraction_top',
                        'path_cut_accept_s1',
                        'path_cut_accept_s2',
                        'path_s1_rly',
                        'path_s2_rly',
                        'path_reconstruction_bias_mean_s1',
                        'path_reconstruction_bias_mean_s2',
                        'path_reconstruction_efficiencies_s1',
                        'variable_elife',
                        'default_elife',
                        'path_electron_lifetimes',
                        'variable_drift_field',
                        'default_drift_field',
                        'path_drift_field',
                        'path_drift_field_distortion',
                        'path_drift_field_distortion_correction',
                        )

    s2_area_fraction_top = DEFAULT_AREA_FRACTION_TOP
    drift_velocity = DEFAULT_DRIFT_VELOCITY
    default_elife = DEFAULT_ELECTRON_LIFETIME
    default_drift_field = DEFAULT_DRIFT_FIELD

    # Light yield maps
    path_s1_rly = '1t_maps/XENON1T_s1_xyz_ly_kr83m-SR1_pax-664_fdc-adcorrtpf.json'
    path_s2_rly = '1t_maps/XENON1T_s2_xy_ly_SR1_v2.2.json'

    # Combined cuts acceptances
    path_cut_accept_s1 = ('cut_acceptance/XENON1T/S1AcceptanceSR1_v7_Median.json',)
    path_cut_accept_s2 = ('cut_acceptance/XENON1T/S2AcceptanceSR1_v7_Median.json',)

    # Pax reconstruction bias maps
    path_reconstruction_bias_mean_s1 = (
        'reconstruction_bias/XENON1T/ReconstructionS1BiasMeanLowers_SR1_v2.json',
        'reconstruction_bias/XENON1T/ReconstructionS1BiasMeanUppers_SR1_v2.json')
    path_reconstruction_bias_mean_s2 = (
        'reconstruction_bias/XENON1T/ReconstructionS2BiasMeanLowers_SR1_v2.json',
        'reconstruction_bias/XENON1T/ReconstructionS2BiasMeanUppers_SR1_v2.json')

    # Pax reconstruction efficiency maps (do not reorder: Lowers, Medians, Uppers)
    path_reconstruction_efficiencies_s1 = (
        'reconstruction_efficiency/XENON1T/RecEfficiencyLowers_SR1_70phd_v1.json',
        'reconstruction_efficiency/XENON1T/RecEfficiencyMedians_SR1_70phd_v1.json',
        'reconstruction_efficiency/XENON1T/RecEfficiencyUppers_SR1_70phd_v1.json')

    # Elife maps
    variable_elife = True
    path_electron_lifetimes = ('1t_maps/electron_lifetimes_sr1.json',)

    # Comsol map
    variable_drift_field = False
    path_drift_field = 'nt_maps/fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p.json'

    # Field distortion map
    path_drift_field_distortion = 'nt_maps/init_to_final_position_mapping_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p.json'

    # FDC map
    path_drift_field_distortion_correction = 'nt_maps/XnT_3D_FDC_xyt_MLP_v0.2_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p.json'

    def set_defaults(self, *args, **kwargs):
        super().set_defaults(*args, **kwargs)

        # Yield maps
        self.s1_map = fd.InterpolatingMap(fd.get_nt_file(self.path_s1_rly))
        self.s2_map = fd.InterpolatingMap(fd.get_nt_file(self.path_s2_rly))

        # Loading combined cut acceptances
        self.cut_accept_map_s1, self.cut_accept_domain_s1 = \
            read_maps_tf(self.path_cut_accept_s1, is_bbf=True)
        self.cut_accept_map_s2, self.cut_accept_domain_s2 = \
            read_maps_tf(self.path_cut_accept_s2, is_bbf=True)

        # Loading reconstruction efficiencies map
        self.recon_eff_map_s1, self.domain_def_ph = \
            read_maps_tf(self.path_reconstruction_efficiencies_s1, is_bbf=True)

        # Loading reconstruction bias map
        self.recon_map_s1_tf, self.domain_def_s1 = \
            read_maps_tf(self.path_reconstruction_bias_mean_s1, is_bbf=True)
        self.recon_map_s2_tf, self.domain_def_s2 = \
            read_maps_tf(self.path_reconstruction_bias_mean_s2, is_bbf=True)

        # Loading electron lifetime map
        self.elife_tf, self.domain_def_elife = \
            read_maps_tf(self.path_electron_lifetimes, is_bbf=False)

        # Field maps
        self.field_map = fd.InterpolatingMap(fd.get_nt_file(self.path_drift_field))

        # Field distortion maps
        # cheap hack
        aa = fd.get_nt_file(self.path_drift_field_distortion) 
        aa['map'] = aa['r_distortion_map']
        self.drift_field_distortion_map = fd.InterpolatingMap(aa, method='RectBivariateSpline')
        del aa

        # FDC maps
        self.fdc_map = fd.InterpolatingMap(fd.get_nt_file(self.path_drift_field_distortion_correction))


    def reconstruction_bias_s1(self,
                               s1,
                               s1_reconstruction_bias_pivot=\
                                   DEFAULT_S1_RECONSTRUCTION_BIAS_PIVOT):
        return calculate_reconstruction_bias(
            s1,
            self.recon_map_s1_tf,
            self.domain_def_s1,
            pivot_pt=s1_reconstruction_bias_pivot)

    def reconstruction_bias_s2(self,
                               s2,
                               s2_reconstruction_bias_pivot=\
                                   DEFAULT_S2_RECONSTRUCTION_BIAS_PIVOT):
        return calculate_reconstruction_bias(
            s2,
            self.recon_map_s2_tf,
            self.domain_def_s2,
            pivot_pt=s2_reconstruction_bias_pivot)

    def random_truth(self, n_events, fix_truth=None, **params):
        d = super().random_truth(n_events, fix_truth=fix_truth, **params)

        # Add extra needed columns
        # x, y, z are the true positions of the event
        #
        # x_observed, y_observed are the reconstructed positions with 
        # field distortion and with posrec smearing but without 
        # field distortion correction
        #
        # x_fdc, y_fdc, z_fdc are the fdc-corrected positions
        
        # going from true position to position distorted by field
        d['r_observed'] = self.drift_field_distortion_map(
            np.transpose([d['r'].values,
                          d['z'].values])) 
        d['x_observed'] = d['r_observed'] * np.cos(d['theta'])
        d['y_observed'] = d['r_observed'] * np.sin(d['theta'])

        # leave z intact, might want to correct this with drift velocity later
        d['z_observed'] = d['z']

        # Adding some smear according to posrec resolution
        d['x_observed'] = np.random.normal(d['x_observed'].values, scale=0.4) # 4 mm resolution)
        d['y_observed'] = np.random.normal(d['y_observed'].values, scale=0.4) # 4 mm resolution)
        
        # applying fdc
        delta_r = self.fdc_map(
            np.transpose([d['x_observed'].values,
                          d['y_observed'].values,
                          d['z_observed'].values,]))
                              
        # apply radial correction
        with np.errstate(invalid='ignore', divide='ignore'):
            d['r_fdc'] = d['r_observed'] + delta_r
            scale = d['r_fdc'] / d['r_observed']

        d['x_fdc'] = d['x_observed'] * scale
        d['y_fdc'] = d['y_observed'] * scale
        d['z_fdc'] = d['z_observed']
        
        return d

    def add_extra_columns(self, d):
        super().add_extra_columns(d)
        d['s2_relative_ly'] = self.s2_map(
             np.transpose([d['x_observed'].values,
                          d['y_observed'].values]))
        d['s1_relative_ly'] = self.s1_map(
            np.transpose([d['x_fdc'].values,
                          d['y_fdc'].values,
                          d['z_fdc'].values]))

        # Not too good. patchy. event_time should be int since event_time in actual
        # data is int64 in ns. But need this to be float32 to interpolate.
        if 'elife' not in d.columns:
            if self.variable_elife:
                d['event_time'] = d['event_time'].astype('float32')
                d['elife'] = interpolate_tf(d['event_time'], self.elife_tf[0],
                                        self.domain_def_elife)
            else:
                d['elife'] = self.default_elife

        if self.variable_drift_field:
            d['drift_field'] = self.field_map(
                np.transpose([d['r'].values,
                              d['z'].values]))
        else:
            d['drift_field'] = self.default_drift_field

        # Add cS1 and cS2 following XENON conventions.
        # Skip this if s1/s2 are not known, since we're simulating
        # TODO: This is a kludge...
        if 's1' in d.columns:
            d['cs1'] = d['s1'] / d['s1_relative_ly']
        if 's2' in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_relative_ly']
                * np.exp(d['drift_time'] / d['elife']))
    
    @staticmethod
    def electron_detection_eff(drift_time,
                               elife,
                               *,
                               extraction_eff=DEFAULT_EXTRACTION_EFFICIENCY):
        return extraction_eff * tf.exp(-drift_time / elife)

    @staticmethod
    def electron_gain_mean(s2_relative_ly,
                           *,
                           single_electron_gain=DEFAULT_SINGLE_ELECTRON_GAIN):
        return single_electron_gain * s2_relative_ly

    @staticmethod
    def electron_gain_std(s2_relative_ly,
                          *,
                          single_electron_width=DEFAULT_SINGLE_ELECTRON_WIDTH):
        # 0 * light yield is to fix the shape
        return single_electron_width + 0. * s2_relative_ly

    @staticmethod
    def double_pe_fraction(z, *, dpe=DEFAULT_P_DPE):
        # Ties the double_pe_fraction model function to the dpe
        # parameter in the sources
        return dpe + 0. * z

    @staticmethod
    def photon_detection_eff(s1_relative_ly, g1=DEFAULT_G1, dpe=DEFAULT_P_DPE):
        mean_eff = g1 / (1. + dpe)
        return mean_eff * s1_relative_ly

    def photon_acceptance(self,
                          photons_detected,
                          s1_reconstruction_efficiency_pivot=\
                              DEFAULT_S1_RECONSTRUCTION_EFFICIENCY_PIVOT):
        return calculate_reconstruction_efficiency(
            photons_detected,
            self.recon_eff_map_s1,
            self.domain_def_ph,
            s1_reconstruction_efficiency_pivot)

    def s1_acceptance(self,
                      s1,
                      cs1,
                      # Only used here, DEFAULT_.. would be super verbose
                      cs1_min=3.,
                      cs1_max=70.):
        acceptance = tf.where((cs1 > cs1_min) & (cs1 < cs1_max),
                              tf.ones_like(s1, dtype=fd.float_type()),
                              tf.zeros_like(s1, dtype=fd.float_type()))

        # multiplying by combined cut acceptance
        acceptance *= interpolate_tf(s1,
                                     self.cut_accept_map_s1[0],
                                     self.cut_accept_domain_s1)
        return acceptance

    def s2_acceptance(self,
                      s2,
                      cs2,
                      s2_min=200.,
                      # Needed for future sources i.e. wall
                      cs2b_min=50.1,
                      cs2b_max=7940.):
        cs2b = cs2*(1-self.s2_area_fraction_top)
        
        acceptance = tf.where((cs2b > cs2b_min) & (cs2b < cs2b_max) 
                                                & (s2 > s2_min),
                              tf.ones_like(s2, dtype=fd.float_type()),
                              tf.zeros_like(s2, dtype=fd.float_type()))

        # multiplying by combined cut acceptance
        acceptance *= interpolate_tf(s2,
                                     self.cut_accept_map_s2[0],
                                     self.cut_accept_domain_s2)
        return acceptance


# ER Source for SR1
@export
class SR1ERSource(SR1Source, fd.ERSource):

    @staticmethod
    def p_electron(nq, drift_field, *, W=13.7e-3, mean_nexni=0.15,  q0=1.13, q1=0.47,
                   gamma_er=0.031 , omega_er=31., delta_er=0.24):
        # gamma_er from paper 0.124/4
        #F = tf.constant(self.default_drift_field, dtype=fd.float_type())

        if tf.is_tensor(nq):
            # in _compute, n_events = batch_size
            # drift_field is originally a (n_events) tensor, nq a (n_events, n_nq) tensor
            # Insert empty axis in drift_field for broadcasting for tf to broadcast over nq dimension

            drift_field = drift_field[:, None]

        e_kev = nq * W
        fi = 1. / (1. + mean_nexni)
        ni, nex = nq * fi, nq * (1. - fi)
        wiggle_er = gamma_er * tf.exp(-e_kev / omega_er) * drift_field ** (-delta_er)

        # delta_er and gamma_er are highly correlated
        # F **(-delta_er) set to constant
        r_er = 1. - tf.math.log(1. + ni * wiggle_er) / (ni * wiggle_er)
        r_er /= (1. + tf.exp(-(e_kev - q0) / q1))
        p_el = ni * (1. - r_er) / nq
        return fd.safe_p(p_el)

    @staticmethod
    def p_electron_fluctuation(nq, q2=0.034, q3_nq=124.):
        # From SR0, BBF model, right?
        # q3 = 1.7 keV ~= 123 quanta
        # For SR1:
        return tf.clip_by_value(
            q2 * (tf.constant(1., dtype=fd.float_type()) - tf.exp(-nq / q3_nq)),
            tf.constant(1e-4, dtype=fd.float_type()),
            float('inf'))


@export
class SR1NRSource(SR1Source, fd.NRSource):
    # TODO: Define the proper nr spectrum
    # TODO: Modify the SR1NRSource to fit AmBe data better

    def p_electron(self, nq, drift_field, *,
                   alpha=1.280, zeta=0.045, beta=273 * .9e-4,
                   gamma=0.0141, delta=0.062):
        """Fraction of detectable NR quanta that become electrons,
        slightly adjusted from Lenardo et al.'s global fit
        (https://arxiv.org/abs/1412.4417).

        Penning quenching is accounted in the photon detection efficiency.
        """
        
        # in _compute, n_events = batch_size
        # drift_field is originally a (n_events) tensor, nq a (n_events, n_nq) tensor
        # Insert empty axis in drift_field for broadcasting for tf to broadcast over nq dimension
        if tf.is_tensor(nq):
            drift_field = drift_field[:, None]

        # prevent /0  # TODO can do better than this
        nq = nq + 1e-9

        # Note: final term depends on nq now, not energy
        # this means beta is different from lenardo et al
        nexni = alpha * drift_field ** -zeta * (1 - tf.exp(-beta * nq))
        ni = nq * 1 / (1 + nexni)

        # Fraction of ions NOT participating in recombination
        squiggle = gamma * drift_field ** -delta
        fnotr = tf.math.log(1 + ni * squiggle) / (ni * squiggle)

        # Finally, number of electrons produced..
        n_el = ni * fnotr
        return fd.safe_p(n_el / nq)

@export
class SR1WallSource(fd.SpatialRateERSource, SR1ERSource):
      
      # Should set FV here 
      #fv_radius = R 
      #fv_high = z_max
      #fv_low = z_min
      
      # Should set the spatial histogram here
      #spatial_hist = normalised_Histdd_wall_spatial_hist
      
      # TODO: The parameters here will need to be polished. 
      # They are fitted parameters and give a reasonable result.
      @staticmethod
      def p_electron(nq, *, w_er_pel_a = -123. , w_er_pel_b = -47.7, 
                    w_er_pel_c = 68., w_er_pel_e0 = 9.95):
      
          """Fraction of ER quanta that become electrons
          Simplified form from Jelle's thesis
          """
          # The original model depended on energy, but in flamedisx
          # it has to be a direct function of nq.
          e_kev_sortof = nq * 13.7e-3
          eps = fd.tf_log10(e_kev_sortof / w_er_pel_e0 + 1e-9)
          qy = (
              w_er_pel_a * eps ** 2
              + w_er_pel_b * eps
              + w_er_pel_c)
          return fd.safe_p(qy * 13.7e-3)

      @staticmethod
      def electron_detection_eff(drift_time,
                                 elife,
                                 *,
                                 w_extraction_eff=0.0169):
          return w_extraction_eff * tf.exp(-drift_time / elife)
          
      @staticmethod
      def p_electron_fluctuation(nq, w_q2 = 0.0237, w_q3_nq = 123.): 
        return tf.clip_by_value(
            w_q2 * (tf.constant(1., dtype=fd.float_type()) - \
            tf.exp(-nq / w_q3_nq)),
            tf.constant(1e-4, dtype=fd.float_type()),
            float('inf'))
      # It is preferred to have higher energy spectrum for the wall
      energies = tf.cast(tf.linspace(0., 25. , 1000),
                       dtype=fd.float_type())

@export
class SR1WIMPSource(SR1NRSource, fd.WIMPSource):
    pass
