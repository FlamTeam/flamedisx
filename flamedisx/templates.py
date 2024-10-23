import random
import string
import typing as ty

import numpy as np
from multihist import Histdd
import pandas as pd
import scipy.interpolate
import tensorflow as tf
import tensorflow_probability as tfp

from flamedisx.tfbspline import bspline

import flamedisx as fd

from copy import deepcopy


export, __all__ = fd.exporter()

@export
class TemplateWrapper:
    """Wrapper around a template (probably a histogram from a simulation)

    Arguments:
        - template: numpy array, multhist.Histdd, or (hist/boost_histogram).
            containing the differential rate.
        - bin_edges: None, or a list of numpy arrays with bin edges.
            If None, get this info from template.
        - axis_names: None, or a sequence of axis names.
            If None, get this info from template.
        - events_per_bin: set to True if template specifies expected events per
            bin, rather than differential rate.
        - interpolate: if True, differential rates are interpolated linearly
            between the bin centers.
    """

    #: Total expected events
    mu: float

    #: Names of template axes = names of final dimensions
    axis_names: str

    def __init__(
            self,
            template,
            bin_edges=None,
            axis_names=None,
            events_per_bin=False,
            interpolate=False):
        # Get template, bin_edges, and axis_names
        if bin_edges is None:
            # Hopefully we got some kind of histogram container
            if isinstance(template, tuple) and len(template) == 2:
                # (hist, bin_edges) tuple, e.g. from np.histdd
                template, bin_edges = template
            elif hasattr(template, "to_numpy"):
                # boost_histogram / hist
                if not axis_names:
                    axis_names = [ax.name for ax in template.axes]
                template, bin_edges = template.to_numpy()
            elif hasattr(template, "bin_edges"):
                # multihist
                if not axis_names:
                    axis_names = template.axis_names
                template, bin_edges = template.histogram, template.bin_edges
            else:
                raise ValueError("Need histogram, bin_edges, and axis_names")

        if not axis_names or len(axis_names) != len(template.shape):
            raise ValueError("Axis names missing or mismatched")
        self.axis_names = axis_names

        # Build a diff rate and events/bin multihist from the template
        _mh = Histdd.from_histogram(template, bin_edges=bin_edges)
        if events_per_bin:
            self._mh_events_per_bin = _mh
            self._mh_diff_rate = _mh / _mh.bin_volumes()
        else:
            self._mh_events_per_bin = _mh * _mh.bin_volumes()
            self._mh_diff_rate = _mh

        self.mu = fd.np_to_tf(self._mh_events_per_bin.n)

        if interpolate:
            # Build an interpolator for the differential rate
            bin_centers = [
                0.5 * (edges[1:] + edges[:-1])
                for edges in bin_edges]
            self._interpolator = scipy.interpolate.RegularGridInterpolator(
                points=tuple(bin_centers),
                values=self._mh_diff_rate.histogram,
                method='linear',
                fill_value=None,
                bounds_error=False)
        else:
            self._interpolator = None

    def differential_rates_numpy(self, data):
        data = np.stack([
            data[dim].values
            for dim in self.axis_names])

        if self._interpolator:
            # transpose since RegularGridInterpolator expects (n_points, n_dims)
            interp_diff_rates = self._interpolator(data.T)
            lookup_diff_rates = self._mh_diff_rate.lookup(*data)
            return np.where(interp_diff_rates <= 0., lookup_diff_rates, interp_diff_rates)
        else:
            return self._mh_diff_rate.lookup(*data)

    def simulate(self, n_events):
        return pd.DataFrame(dict(zip(
            self.axis_names,
            self._mh_events_per_bin.get_random(n_events).T)))
    
    
@export
class TemplateSource(fd.ColumnSource):
    """Source that looks up precomputed differential rates in a template
    (probably a histogram from a simulation).

    Arguments:
        - template: numpy array, multhist.Histdd, or (hist/boost_histogram).
            containing the differential rate.
        - bin_edges: None, or a list of numpy arrays with bin edges.
            If None, get this info from template.
        - axis_names: None, or a sequence of axis names.
            If None, get this info from template.
        - events_per_bin: set to True if template specifies expected events per
            bin, rather than differential rate.
        - interpolate: if True, differential rates are interpolated linearly
            between the bin centers.

    For other arguments, see flamedisx.source.Source
    """

    def __init__(
            self,
            template,
            bin_edges=None,
            axis_names=None,
            events_per_bin=False,
            interpolate=False,
            *args,
            **kwargs):
        self._template = TemplateWrapper(
            template, bin_edges, axis_names, events_per_bin, interpolate)

        self.final_dimensions = self._template.axis_names
        self.mu = self._template.mu

        # Generate a random column name to use to store the diff rates
        # of observed events
        self.column = (
            'template_diff_rate_'
            + ''.join(random.choices(string.ascii_lowercase, k=8)))

        super().__init__(*args, **kwargs)

    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        self.data[self.column] = self._template.differential_rates_numpy(self.data)

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        """Simulate n events.
        """
        if fix_truth:
            raise NotImplementedError("TemplateSource does not yet support fix_truth")
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        # TODO: all other arguments are ignored, they make no sense
        # for this source. Should we warn about this? Remove them from def?

        return self._template.simulate(n_events)