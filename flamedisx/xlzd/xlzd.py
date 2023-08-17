"""Toy XLZD detector implementation

"""
import numpy as np
import tensorflow as tf

import configparser
import os
import pandas as pd

import flamedisx as fd
from .. import nest as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##


class XLZDSource:
    def __init__(self, *args,
                 drift_field_V_cm=100., gas_field_kV_cm=8., elife_ns=10000e3, g1=0.27,
                 ignore_maps_acc=False, **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('xlzd',)
        assert kwargs['configuration'] in ('80t', '60t', '40t')

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        if ignore_maps_acc:
            self.ignore_acceptances = True

        self.radius = config.getfloat(kwargs['configuration'], 'radius_config')
        self.z_topDrift = config.getfloat(kwargs['configuration'], 'z_topDrift_config')
        self.z_top = config.getfloat(kwargs['configuration'], 'z_top_config')
        self.z_bottom = config.getfloat(kwargs['configuration'], 'z_bottom_config')

        self.configuration = kwargs['configuration']

        self.drift_field = drift_field_V_cm
        self.gas_field = gas_field_kV_cm
        self.elife = elife_ns
        self.g1 = g1 # this represents PMT QE

        self.drift_velocity = fd_nest.calculate_drift_velocity(
            self.drift_field, self.density, self.temperature, self.detector)
        self.extraction_eff = fd_nest.calculate_extraction_eff(self.gas_field, self.temperature)
        self.g2 = fd_nest.calculate_g2(self.gas_field, self.density_gas, self.gas_gap,
                                       self.g1_gas, self.extraction_eff)

    def s1_posDependence(self, z):
        """
        Returns LCE. PMT QE then handled by the g1 value.
        Coefficients come from fit to LCE curve obtained by Theresa Fruth via
        BACCARAT.
        Requires z to be in cm, and in the FV.
        """
        if self.configuration == '80t':
            a = 4.93526997e-01
            b = 5.58488459e-04
            c = -1.06051830e-07
            d = -1.02545243e-08
            e = -1.30897496e-11
        elif self.configuration == '60t':
            a = 5.00828879e-01
            b = 4.25478465e-04
            c = 2.02171646e-07
            d = -1.31075129e-08
            e = -2.29858516e-11
        elif self.configuration == '40t':
            a = 5.68299226e-01
            b = 7.29787915e-05
            c = -4.37731127e-06
            d = -6.39383595e-08
            e = -1.87387400e-10

        LCE = a + b * z + c * z**2 + d * z**3 + e * z**4

        return LCE

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if self.configuration == '80t':
            LCE_average = 0.471
        elif self.configuration == '60t':
            LCE_average = 0.493
        elif self.configuration == '40t':
            LCE_average = 0.570
        d['s1_pos_corr'] = self.s1_posDependence(d['z'].values) / LCE_average # normalise to volume-averaged LCE

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr']
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = d['s2'] * np.exp(d['drift_time'] / self.elife)

        if 'cs1' in d.columns and 'cs2' in d.columns and 'ces_er_equivalent' not in d.columns:
             d['ces_er_equivalent'] = (d['cs1'] / (self.g1 * LCE_average) + d['cs2'] / self.g2) * self.Wq_keV / (1. + self.double_pe_fraction)


@export
class XLZDERSource(XLZDSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDNRSource(XLZDSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDERSourceGroup(XLZDSource, fd.nest.nestERSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDNRSourceGroup(XLZDSource, fd.nest.nestNRSourceGroup):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


##
# Signal sources
##


@export
class XLZDWIMPSource(XLZDSource, fd.nest.nestWIMPSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDSolarAxionSource(XLZDSource, fd.nest.nestSolarAxionSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


##
# Background sources
##


@export
class XLZDXe136Source(XLZDSource, fd.nest.Xe136Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDPb214Source(XLZDSource, fd.nest.Pb214Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDKr85Source(XLZDSource, fd.nest.Kr85Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDvERSource(XLZDSource, fd.nest.vERSource, fd.nest.nestTemporalRateOscillationERSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'

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
class XLZDvNRSolarSource(XLZDSource, fd.nest.vNRSolarSource, fd.nest.nestTemporalRateOscillationNRSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'

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
class XLZDvNROtherSource(XLZDSource, fd.nest.vNROtherSource, fd.nest.nestTemporalRateOscillationNRSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'

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
