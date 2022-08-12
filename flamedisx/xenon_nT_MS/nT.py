"""Matthew Syzdagis' mock-up of the XENONnT detector implementation

"""
import tensorflow as tf

import configparser
import os

import flamedisx as fd
from .. import nest as fd_nest

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

    def electron_gain_mean(self):
        elYield = 31.2

        return tf.cast(elYield, fd.float_type())[o]

    def electron_gain_std(self):
        elYield = 31.2

        return tf.sqrt(self.s2Fano * elYield)[o]


@export
class XENONnTERSource(XENONnTSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'XENONnT'
        super().__init__(*args, **kwargs)


@export
class XENONnTNRSource(XENONnTSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'XENONnT'
        super().__init__(*args, **kwargs)
