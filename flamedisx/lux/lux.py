"""lux detector implementation

"""
import tensorflow as tf

import configparser
import os

import flamedisx as fd
from .. import nest as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##


class LUXSource:
    def __init__(self, *args, detector='default', **kwargs):
        assert detector in ('default',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', detector + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 detector + '.ini'))

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

    def s1_posDependence(self, r, z):
        """
        Returns position-dependent S1 scale factor.
        Requires r/z to be in cm, and in the FV.
        """
        r_mm = r*10
        z_mm = z*10

        amplitude = 307.9 - 0.3071 * z_mm + 0.0002257 * pow(z_mm, 2)
        shape = 1.1525e-7 * tf.sqrt(abs(z_mm - 318.84))
        finalCorr = (-shape * pow(r_mm, 3) + amplitude) / 307.9

        # We want to normalise to the detector centre
        z_cntr = self.z_topDrift - self.drift_velocity * self.dt_cntr
        z_cntr_mm = z_cntr * 10

        amplitude_0 = 307.9 - 0.3071 * z_cntr_mm + 0.0002257 * pow(z_cntr_mm, 2)
        finalCorr_0 = amplitude_0 / 307.9

        return finalCorr / finalCorr_0

    def s2_posDependence(self, r):
        """
        Returns position-dependent S2 scale factor.
        Requires r to be in cm, and in the FV.
        """
        r_mm = r*10

        finalCorr = 9156.3 + 6.22750 * pow(r_mm, 1) + 0.38126 * pow(r_mm, 2) \
            - 0.017144 * pow(r_mm, 3) + 0.0002474 * pow(r_mm, 4) \
            - 1.6953e-6 * pow(r_mm, 5) + 5.6513e-9 * pow(r_mm, 6) \
            - 7.3989e-12 * pow(r_mm, 7)

        return finalCorr / 9156.3


@export
class LUXERSource(LUXSource, fd.nest.nestERSource):
    def __init__(self, *args, detector='default', **kwargs):
        super().__init__(*args, **kwargs)


@export
class LUXNRSource(LUXSource, fd.nest.nestNRSource):
    def __init__(self, *args, detector='default', **kwargs):
        super().__init__(*args, **kwargs)
