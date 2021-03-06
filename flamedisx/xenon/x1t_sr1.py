"""XENON1T SR1 implementation
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
import json

export, __all__ = fd.exporter()

o = tf.newaxis

##
# Yield maps
##
s1_map, s2_map = [
    fd.InterpolatingMap(fd.get_resource(fd.pax_file(x)))
    for x in ('XENON1T_s1_xyz_ly_kr83m-SR1_pax-664_fdc-adcorrtpf.json',
              'XENON1T_s2_xy_ly_SR1_v2.2.json')]

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

# Official numbers from BBF, do not question them ;-)
DEFAULT_S1_RECONSTRUCTION_BIAS_PIVOT = 0.5948841302444277
DEFAULT_S2_RECONSTRUCTION_BIAS_PIVOT = 0.49198507921078005

##
# Loading Pax reconstruction bias
##
path_reconstruction_bias_mean_s1 = ['ReconstructionS1BiasMeanLowers_SR1_v2.json',
        'ReconstructionS1BiasMeanUppers_SR1_v2.json']
path_reconstruction_bias_mean_s2 = ['ReconstructionS2BiasMeanLowers_SR1_v2.json',
        'ReconstructionS2BiasMeanUppers_SR1_v2.json']


def read_maps_tf(path_bag, is_bbf=False):
    """ Function to read reconstruction bias/combined cut acceptances/dummy maps.
    Note that this implementation fundamentally assumes upper and lower bounds
    have exactly the same domain definition.

    """
    data_bag = []
    yy_ref_bag = []
    for loc_path in path_bag:
        if is_bbf:
            tmp = fd.get_bbf_file(loc_path)
        else:
            with open(loc_path) as json_file:
                tmp = json.load(json_file)
        yy_ref_bag.append(tf.convert_to_tensor(tmp['map'], dtype=fd.float_type()))
        data_bag.append(tmp)
    domain_def = tmp['coordinate_system'][0][1]

    return yy_ref_bag, domain_def


def cal_bias_tf(sig, fmap, domain_def, pivot_pt):
    """ Computes the reconstruction bias mean given the pivot point
    :param sig: S1 or S2 values
    :param fmap: map returned by read_maps_tf
    :param domain_def: domain returned by read_maps_tf
    :param pivot_pt: Pivot point value (scalar)
    :return: Tensor of bias values (same shape as sig)
    """
    tmp = tf.convert_to_tensor(sig, dtype=fd.float_type())
    bias_low = tfp.math.interp_regular_1d_grid(x=tmp,
            x_ref_min=domain_def[0], x_ref_max=domain_def[1], y_ref=fmap[0],
            fill_value='constant_extension')
    bias_high = tfp.math.interp_regular_1d_grid(x=tmp,
            x_ref_min=domain_def[0], x_ref_max=domain_def[1], y_ref=fmap[1],
            fill_value='constant_extension')

    tmp = tf.math.subtract(bias_high, bias_low)
    tmp1 = tf.math.scalar_mul(pivot_pt, tmp)
    bias_out = tf.math.add(tmp1, bias_low)
    bias_out = tf.math.add(bias_out, tf.ones_like(bias_out))

    return bias_out


##
# Loading combined cuts acceptances
##
path_cut_accept_s1 = ['S1AcceptanceSR1_v7_Median.json']
path_cut_accept_s2 = ['S2AcceptanceSR1_v7_Median.json']


def itp_cut_accept_tf(sig, fmap, domain_def):
    accept_out = tf.squeeze(tfp.math.interp_regular_1d_grid(x=sig,
            x_ref_min=domain_def[0], x_ref_max=domain_def[1], y_ref=fmap,
            fill_value='constant_extension'))
    return accept_out


##
# Flamedisx sources
##
class SR1Source:
    drift_velocity = DEFAULT_DRIFT_VELOCITY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Loading combined cut acceptances
        self.cut_accept_map_s1, self.cut_accept_domain_s1 = \
            read_maps_tf(path_cut_accept_s1, is_bbf=True)
        self.cut_accept_map_s2, self.cut_accept_domain_s2 = \
            read_maps_tf(path_cut_accept_s2, is_bbf=True)

        # Loading reconstruction bias map
        self.recon_map_s1_tf, self.domain_def_s1 = \
            read_maps_tf(path_reconstruction_bias_mean_s1, is_bbf=True)
        self.recon_map_s2_tf, self.domain_def_s2 = \
            read_maps_tf(path_reconstruction_bias_mean_s2, is_bbf=True)

    def reconstruction_bias_s1(self,
                               sig,
                               bias_pivot_pt1=DEFAULT_S1_RECONSTRUCTION_BIAS_PIVOT):
        reconstruction_bias = cal_bias_tf(sig,
                                          self.recon_map_s1_tf,
                                          self.domain_def_s1,
                                          pivot_pt=bias_pivot_pt1)
        return reconstruction_bias

    def reconstruction_bias_s2(self,
                               sig,
                               # Need to change the name; the pivot points
                               # for S2 and S2 are independent
                               bias_pivot_pt2=DEFAULT_S2_RECONSTRUCTION_BIAS_PIVOT):
        reconstruction_bias = cal_bias_tf(sig,
                                          self.recon_map_s2_tf,
                                          self.domain_def_s2,
                                          pivot_pt=bias_pivot_pt2)
        return reconstruction_bias

    def random_truth(self, n_events, fix_truth=None, **params):
        d = super().random_truth(n_events, fix_truth=fix_truth, **params)

        # Add extra needed columns
        # TODO: Add FDC maps instead of posrec resolution
        d['x_observed'] = np.random.normal(d['x'].values,
                                           scale=2)  # 2cm resolution)
        d['y_observed'] = np.random.normal(d['y'].values,
                                           scale=2)  # 2cm resolution)
        return d

    def add_extra_columns(self, d):
        super().add_extra_columns(d)
        d['s2_relative_ly'] = s2_map(
             np.transpose([d['x_observed'].values,
                          d['y_observed'].values]))
        d['s1_relative_ly'] = s1_map(
            np.transpose([d['x'].values,
                          d['y'].values,
                          d['z'].values]))

        # Add cS1 and cS2 following XENON conventions.
        # Skip this if s1/s2 are not known, since we're simulating
        # TODO: This is a kludge...
        if 's1' in d.columns:
            d['cs1'] = d['s1'] / d['s1_relative_ly']
        if 's2' in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_relative_ly']
                * np.exp(d['drift_time'] / self.defaults['elife']))

    @staticmethod
    def electron_detection_eff(drift_time,
                               *,
                               elife=DEFAULT_ELECTRON_LIFETIME,
                               extraction_eff=DEFAULT_EXTRACTION_EFFICIENCY):
        #TODO: include function for elife time dependency
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

    #TODO: implement better the double_pe_fraction or photon_detection_efficiency as parameter
    @staticmethod
    def photon_detection_eff(s1_relative_ly, g1=DEFAULT_G1):
        mean_eff= g1 / (1. + DEFAULT_P_DPE)
        return mean_eff * s1_relative_ly

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
        acceptance *= itp_cut_accept_tf(s1,
                                        self.cut_accept_map_s1,
                                        self.cut_accept_domain_s1)
        return acceptance

    def s2_acceptance(self,
                      s2,
                      cs2,
                      cs2b_min=50.1,
                      cs2b_max=7940.):
        acceptance = tf.where((cs2 > cs2b_min) & (cs2 < cs2b_max),
                              tf.ones_like(s2, dtype=fd.float_type()),
                              tf.zeros_like(s2, dtype=fd.float_type()))

        # multiplying by combined cut acceptance
        acceptance *= itp_cut_accept_tf(s2,
                                        self.cut_accept_map_s2,
                                        self.cut_accept_domain_s2)

        return acceptance


# ER Source for SR1
@export
class SR1ERSource(SR1Source,fd.ERSource):

    @staticmethod
    def p_electron(nq, W=13.8e-3, mean_nexni=0.15,  q0=1.13, q1=0.47,
                   gamma_er=0.031 , omega_er=31.):
        # gamma_er from paper 0.124/4
        F = tf.constant(DEFAULT_DRIFT_FIELD, dtype=fd.float_type())

        e_kev = nq * W
        fi = 1. / (1. + mean_nexni)
        ni, nex = nq * fi, nq * (1. - fi)
        wiggle_er = gamma_er * tf.exp(-e_kev / omega_er) * F ** (-0.24)
        # delta_er and gamma_er are highly correlated
        # F **(-delta_er) set to constant
        r_er = 1. - tf.math.log(1. + ni * wiggle_er) / (ni * wiggle_er)
        r_er /= (1. + tf.exp(-(e_kev - q0) / q1))
        p_el = ni * (1. - r_er) / nq

        return fd.safe_p(p_el)

    @staticmethod
    def p_electron_fluctuation(nq, q2=0.034, q3_nq=123.):
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

    def p_electron(self, nq, *,
                   alpha=1.280, zeta=0.045, beta=273 * .9e-4,
                   gamma=0.0141, delta=0.062,
                   drift_field=DEFAULT_DRIFT_FIELD):
        """Fraction of detectable NR quanta that become electrons,
        slightly adjusted from Lenardo et al.'s global fit
        (https://arxiv.org/abs/1412.4417).

        Penning quenching is accounted in the photon detection efficiency.
        """
        # TODO: so to make field pos-dependent, override this entire f?
        # could be made easier...

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
class SR1WIMPSource(SR1NRSource, fd.WIMPSource):
    pass
