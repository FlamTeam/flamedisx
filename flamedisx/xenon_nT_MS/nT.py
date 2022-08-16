"""Matthew Syzdagis' mock-up of the XENONnT detector implementation

"""
import numpy as np
import pandas as pd
import tensorflow as tf

import configparser
import os

import flamedisx as fd
from .. import nest as fd_nest
from .. import xenon_nT_MS as fd_nT

import math as m
pi = tf.constant(m.pi)

export, __all__ = fd.exporter()
o = tf.newaxis


##
# Flamedisx sources
##


class XENONnTSource:
    path_s1_rly = 'nt_maps/XnT_S1_xyz_MLP_v0.3_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_v0d677.json'
    path_s2_rly = 'nt_maps/XENONnT_s2_xy_map_v4_210503_mlp_3_in_1_iterated.json'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('XENONnT',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.drift_velocity = config.getfloat('NEST', 'drift_velocity_config')

        self.z_top = config.getfloat('NEST', 'z_top_config')
        self.z_bottom = config.getfloat('NEST', 'z_bottom_config')

        self.extraction_eff = config.getfloat('NEST', 'extraction_eff_config')
        self.elYield = config.getfloat('NEST', 'elYield_config')

        self.cS1_min = config.getfloat('NEST', 'cS1_min_config')
        self.cS1_max = config.getfloat('NEST', 'cS1_max_config')
        self.cS2_min = config.getfloat('NEST', 'cS2_min_config')
        self.cS2_max = config.getfloat('NEST', 'cS2_max_config')

        try:
            self.s1_map = fd.InterpolatingMap(fd.get_nt_file(self.path_s1_rly))
            self.s2_map = fd.InterpolatingMap(fd.get_nt_file(self.path_s2_rly))
        except Exception:
            print("Could not load maps; setting position corrections to 1")
            self.s1_map = None
            self.s2_map = None

    def electron_gain(self, s2_relative_ly):
        return self.elYield * s2_relative_ly

    @staticmethod
    def photon_detection_eff(z, *, g1=0.126):
        return g1 * tf.ones_like(z)

    @staticmethod
    def s2_photon_detection_eff(z, *, g1_gas=0.851):
        return g1_gas * tf.ones_like(z)

    @staticmethod
    def s1_posDependence(s1_relative_ly):
        return s1_relative_ly

    @staticmethod
    def s2_posDependence(r):
        return tf.ones_like(r)

    def s1_acceptance(self, s1, cs1):
        return tf.where((s1 < self.S1_min) | (s1 < self.spe_thr) | (s1 > self.S1_max) |
                        (cs1 < self.cS1_min) | (cs1 > self.cS1_max),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    def s2_acceptance(self, s2, cs2):
        return tf.where((s2 < self.S2_min) | (s2 > self.S2_max) |
                        (cs2 < self.cS2_min) | (cs2 > self.cS2_max),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if (self.s1_map is not None) and (self.s2_map is not None):
            d['s1_relative_ly'] = self.s1_map(
                np.transpose([d['x'].values,
                              d['y'].values,
                              d['z'].values]))
            d['s2_relative_ly'] = self.s2_map(
                np.transpose([d['x'].values,
                              d['y'].values]))
        else:
            d['s1_relative_ly'] = np.ones_like(d['x'].values)
            d['s2_relative_ly'] = np.ones_like(d['x'].values)

        if 's1' in d.columns:
            d['cs1'] = d['s1'] / d['s1_relative_ly']
        if 's2' in d.columns:
            d['cs2'] = (
                d['s2']
                / d['s2_relative_ly']
                * np.exp(d['drift_time'] / self.elife))


@export
class XENONnTERSource(XENONnTSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'XENONnT'
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd_nest.FixedShapeEnergySpectrumER,
        fd_nest.MakePhotonsElectronER,
        fd_nest.DetectPhotons,
        fd_nest.MakeS1Photoelectrons,
        fd_nest.DetectS1Photoelectrons,
        fd_nest.MakeS1,
        fd_nT.DetectElectronsMod,
        fd_nest.DetectS2Photons,
        fd_nest.MakeS2Photoelectrons,
        fd_nest.MakeS2)

    def variance(self, *args):
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        er_free_b = 0.05
        er_free_c = 0.205
        er_free_d = 0.45
        er_free_e = -0.2

        elec_frac = nel_mean / nq_mean
        ampl = tf.cast(0.086036 + (er_free_b - 0.086036) /
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
class XENONnTNRSource(XENONnTSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'XENONnT'
        super().__init__(*args, **kwargs)

    @staticmethod
    def yield_fano(nq_mean):
        nr_free_a = 0.4
        nr_free_b = 0.4

        ni_fano = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_a
        nex_fano = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_b

        return ni_fano, nex_fano

    @staticmethod
    def variance(*args):
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        nr_free_c = 0.04
        nr_free_d = 0.5
        nr_free_e = 0.19

        elec_frac = nel_mean / nq_mean

        omega = nr_free_c * tf.exp(-0.5 * pow(elec_frac - nr_free_d, 2.) / (nr_free_e * nr_free_e))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         omega)

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni
