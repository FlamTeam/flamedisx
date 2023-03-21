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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('xlzd',)
        assert kwargs['configuration'] in ('TEST',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.radius = config.getfloat(kwargs['configuration'], 'radius_config')
        self.z_top = config.getfloat(kwargs['configuration'], 'z_top_config')
        self.z_bottom = config.getfloat(kwargs['configuration'], 'z_bottom_config')

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1']
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = d['s2'] * np.exp(d['drift_time'] / self.elife)

        if 'cs1' in d.columns and 'cs2' in d.columns:
             d['ces_er_equivalent'] = (d['cs1'] / self.g1 + d['cs2'] / self.g2) * self.Wq_keV


@export
class XLZDERSource(XLZDSource, fd.nest.nestERSource):
    def __init__(self, *args, detector='xlzd', configuration='TEST', **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)


@export
class XLZDNRSource(XLZDSource, fd.nest.nestNRSource):
    def __init__(self, *args, detector='xlzd', configuration='TEST', **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)


@export
class XLZDERSourceGroup(XLZDSource, fd.nest.nestERSourceGroup):
    def __init__(self, *args, detector='xlzd', configuration='TEST', **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)


@export
class XLZDNRSourceGroup(XLZDSource, fd.nest.nestNRSourceGroup):
    def __init__(self, *args, detector='xlzd', configuration='TEST', **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
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
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)


##
# Background sources
##


@export
class XLZDvERSource(XLZDSource, fd.nest.vERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)


@export
class XLZDvNRSolarSource(XLZDSource, fd.nest.vNRSolarSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)


@export
class XLZDvNROtherSource(XLZDSource, fd.nest.vNROtherSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = 'TEST'
        super().__init__(*args, **kwargs)
