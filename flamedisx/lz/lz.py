"""lz detector implementation

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


class LZSource:
    path_s1_corr = '/Users/Robert/s1_map_22Apr22.json'
    path_s2_corr = '/Users/Robert/s2_map_30Mar22.json'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert kwargs['detector'] in ('lz',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.drift_velocity = self.drift_velocity * 0.96 / 0.95

        try:
            self.s1_map = fd.InterpolatingMap(fd.get_resource(self.path_s1_corr))
            self.s2_map = fd.InterpolatingMap(fd.get_resource(self.path_s2_corr))
        except Exception:
            print("Could not load maps; setting position corrections to 1")
            self.s1_map = None
            self.s2_map = None

    @staticmethod
    def s1_posDependence(s1_pos_corr):
        return s1_pos_corr

    @staticmethod
    def s2_posDependence(s2_pos_corr):
        return s2_pos_corr

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if (self.s1_map is not None) and (self.s2_map is not None):
            d['s1_pos_corr'] = self.s1_map(
                np.transpose([d['x'].values,
                              d['y'].values,
                              d['drift_time'].values]))
            d['s2_pos_corr'] = self.s2_map(
                np.transpose([d['x'].values,
                              d['y'].values]))
        else:
            d['s1_pos_corr'] = np.ones_like(d['x'].values)
            d['s2_pos_corr'] = np.ones_like(d['x'].values)


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


@export
class LZCH3TSource(LZSource, fd.nest.CH3TSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'lz'
        super().__init__(*args, **kwargs)
