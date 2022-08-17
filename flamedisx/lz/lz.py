"""lz detector implementation

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


class LZSource:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('lz',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.drift_velocity = self.drift_velocity * 0.96 / 0.95


@export
class LZERSource(LZSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)


@export
class LZNRSource(LZSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)
