"""Matthew Syzdagis' mock-up of the XENONnT detector implementation

"""
import tensorflow as tf

import configparser
import os

import flamedisx as fd
from .. import nest as fd_nest

import math as m
pi = tf.constant(m.pi)

export, __all__ = fd.exporter()
o = tf.newaxis


##
# Flamedisx sources
##


class XENONnTSource:
    def __init__(self, *args, **kwargs):
        assert kwargs['detector'] in ('XENONnT',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.z_topDrift = config.getfloat('NEST', 'z_topDrift_config')
        self.dt_cntr = config.getfloat('NEST', 'dt_cntr_config')

        self.density = fd_nest.calculate_density(
            config.getfloat('NEST', 'temperature_config'),
            config.getfloat('NEST', 'pressure_config')).item()
        self.drift_velocity = fd_nest.calculate_drift_velocity(
         config.getfloat('NEST', 'drift_field_config'),
         self.density,
         config.getfloat('NEST', 'temperature_config')).item()

        super().__init__(*args, **kwargs)

        self.extraction_eff = 0.52

    @staticmethod
    def electron_gain_mean():
        elYield = 31.2
        return tf.cast(elYield, fd.float_type())[o]

    def electron_gain_std(self):
        elYield = 31.2
        return tf.sqrt(self.s2Fano * elYield)[o]

    @staticmethod
    def photon_detection_eff(z, *, g1=0.126):
        return g1 * tf.ones_like(z)

    @staticmethod
    def s2_photon_detection_eff(z, *, g1_gas=0.851):
        return g1_gas * tf.ones_like(z)


@export
class XENONnTERSource(XENONnTSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'XENONnT'
        super().__init__(*args, **kwargs)

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
