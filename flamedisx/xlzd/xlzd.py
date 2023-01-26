"""Toy XLZD detector implementation

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
