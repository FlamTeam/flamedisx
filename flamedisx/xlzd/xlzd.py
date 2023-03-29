"""Toy XLZD detector implementation

"""
import numpy as np
import tensorflow as tf

import configparser
import os

import flamedisx as fd
from .. import nest as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##


class XLZDSource:
    def __init__(self, *args,
                 drift_field_V_cm=100., gas_field_kV_cm=8., elife_ns=13000e3, g1=0.27,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('xlzd',)
        assert kwargs['configuration'] in ('80t', '60t', '40t', '20t', '60t_AR1', '40t_wide',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.radius = config.getfloat(kwargs['configuration'], 'radius_config')
        self.z_topDrift = config.getfloat(kwargs['configuration'], 'z_topDrift_config')
        self.z_top = config.getfloat(kwargs['configuration'], 'z_top_config')
        self.z_bottom = config.getfloat(kwargs['configuration'], 'z_bottom_config')
        self.anode_gate = config.getfloat(kwargs['configuration'], 'anode_gate_config')

        self.S1_max = config.getfloat(kwargs['configuration'], 'S1_max_config')

        self.configuration = kwargs['configuration']

        self.anode_gate_80t = config.getfloat('NEST', 'anode_gate_80t_config')

        self.drift_field = drift_field_V_cm / self.anode_gate * self.anode_gate_80t
        self.gas_field = gas_field_kV_cm
        self.elife = elife_ns
        self.g1 = g1 # this represents PMT QE

        self.drift_velocity = fd_nest.calculate_drift_velocity(
            self.drift_field, self.density, self.temperature)
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
            a = 3.82211839e-01
            b = 1.14254580e-03
            c = 2.24850367e-06
            d = -9.77272624e-10
        elif self.configuration == '60t':
            a = 4.16053591e-01
            b = 9.99867174e-04
            c = 2.83162470e-06
            d = -1.50717789e-09
        elif self.configuration == '40t':
            a = 4.71252343e-01
            b = 7.83829990e-04
            c = 4.42536234e-06
            d = 2.09610330e-10
        elif self.configuration == '20t':
            a = 5.56369703e-01
            b = 3.81596554e-04
            c = 7.92471387e-06
            d = 2.44831514e-08
        elif self.configuration == '60t_AR1':
            a = 4.62941158e-01
            b = 8.02019214e-04
            c = 3.54689661e-06
            d = -1.27689610e-09
        elif self.configuration == '40t_wide':
            a = 5.19626499e-01
            b = 5.61037171e-04
            c = 5.06533421e-06
            d = 2.51777266e-09

        LCE = a + b * z + c * z**2 + d * z**3

        return LCE

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        # Numbers from https://docs.google.com/presentation/d/13qbvvxCj4eE79Hdw-IpaF56i2h85Tc-Kh_ZfQIjZHj8/edit#slide=id.g21b562c0a80_0_40
        if self.configuration == '80t':
            LCE_average = 0.317666
        elif self.configuration == '60t':
            LCE_average = 0.383039
        elif self.configuration == '40t':
            LCE_average = 0.465242
        elif self.configuration == '20t':
            LCE_average = 0.55959
        elif self.configuration == '60t_AR1':
            LCE_average = 0.453847
        elif self.configuration == '40t_wide':
            LCE_average = 0.522645
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
class XLZDvERSource(XLZDSource, fd.nest.vERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDvNRSolarSource(XLZDSource, fd.nest.vNRSolarSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDvNROtherSource(XLZDSource, fd.nest.vNROtherSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)
