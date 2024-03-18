import random
import string
import typing as ty

import numpy as np
from multihist import Histdd
import pandas as pd
import scipy.interpolate
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd

from copy import deepcopy

export, __all__ = fd.exporter()


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
            return self._interpolator(data.T)
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


@export
class MultiTemplateSource(fd.Source):
    """Source that interpolates linearly between multiple templates,
    each representing the expected differential rates at a single set of
    parameters.

    Arguments:
        - params_and_templates: 2-tuples of
                (dict of parameter names and values, template histogram).
            The parameter names must be the same for all templates.
            For allowed types of template histogram, see TemplateSource.
        - bin_edges: None, or a list of numpy arrays with bin edges.
            If None, get this info from template.
        - axis_names: None, or a sequence of axis names.
            If None, get this info from template.
        - events_per_bin: set to True if templates specify expected events per
            bin, rather than differential rate.
        - interpolate: if True, differential rates are interpolated linearly
            between the bin centers in each template,
            in addition to the interpolation between templates that happens
            regardless.
    """

    def __init__(
            self,
            params_and_templates: ty.Tuple[ty.Dict[str, float], ty.Any],
            bin_edges=None,
            axis_names=None,
            events_per_bin=False,
            interpolate=False,
            *args,
            **kwargs):

        self._templates = [
            TemplateWrapper(
                template, bin_edges, axis_names, events_per_bin, interpolate)
            for _, template in params_and_templates]

        # We assume that mu does not change as we interpolate
        self.mu = self._templates[0].mu

        # Grab parameter names. Promote first set of values to defaults.
        self.n_templates = n_templates = len(self._templates)
        assert n_templates > 0
        defaults = params_and_templates[0][0]
        for params, _ in params_and_templates:
            assert tuple(params.keys()) == tuple(defaults.keys())

        # Build an interpolator that produces the _weights_ of each template
        # at a given parameter space point, according to linear interpolation.
        #
        # This interpolator maps an (n_templates = n_params,) array to
        # an (n_templates,) array.
        #
        # When evaluated at the exact location of a template, the result has 1
        # in the corresponding template's position, and zeros elsewhere.

        _template_weights = scipy.interpolate.interp1d(
            x=np.asarray([list(params.values())[0] for params, _ in params_and_templates]),
            y=np.eye(n_templates))

        # Unfortunately TensorFlow has no equivalent of LinearNDInterpolator,
        # only interpolators that work on rectilinear grids. Thus, instead of
        # calling something like the above interpolator directly, we have to
        # evaluate it on a rectilinear grid first. :-(

        # Get the sorted unique values for each parameter, then use those
        # to build a rectilinear grid. Tuple of differently-shaped arrays.
        _grid_coordinates = tuple([
            np.asarray(sorted(set(params[param]
                                  for params, _ in params_and_templates)))
            for param in defaults])
        _full_grid_coordinates = np.meshgrid(*_grid_coordinates, indexing='ij')
        n_grid_points = np.prod([len(x) for x in _grid_coordinates])

        # Evaluate our irregular-grid scipy-interpolator on the grid.
        # This gives an array of shape (n_templates, ngrid_dim0, ngrid_dim1, ...)
        # for use in tensorflow interpolation.
        _grid_weights = _template_weights(*_full_grid_coordinates)

        # Generate a random column name to use to store the diff rates
        # of observed events under every template
        self.column = (
            'template_diff_rate_'
            + ''.join(random.choices(string.ascii_lowercase, k=8)))

        # ... this column will hold an array, with one entry per template
        self.array_columns = ((self.column, n_templates),)

        # This source has parameters but no model functions, so we can't do the
        # usual Source.scan_model_functions.
        self.f_dims = dict()
        self.f_params = dict()
        self.defaults = {k: tf.cast(v, fd.float_type()) for k, v in defaults.items()}

        # This is needed in tensorflow, so convert it now
        self._grid_coordinates = tuple([fd.np_to_tf(np.asarray(g)) for g in _grid_coordinates])
        self._grid_weights = fd.np_to_tf(_grid_weights)

        super().__init__(*args, **kwargs)

    def scan_model_functions(self):
        # Don't do anything here, already set defaults etc. in __init__ above
        pass

    def extra_needed_columns(self):
        return super().extra_needed_columns() + [self.column]

    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        # Get array of differential rates for each template.
        # Outer list() is to placate pandas, which does not like array columns..
        self.data[self.column] = list(np.asarray([
            template.differential_rates_numpy(self.data)
            for template in self._templates]).T)

    def mu_before_efficiencies(self, **params):
        return self.mu

    def estimate_mu(self, n_trials=None, **params):
        return self.mu

    def _differential_rate(self, data_tensor, ptensor):
        # Compute template weights at this parameter point
        # (n_templates,) tensor
        # (The axis order is weird here. It seems to work...)
        template_weights = tfp.math.batch_interp_regular_1d_grid(
            x=ptensor[None, :],
            x_ref_min=self._grid_coordinates[0][0],
            x_ref_max=self._grid_coordinates[0][-1],
            y_ref=self._grid_weights,
        )[:, 0]
        # Ensure template weights sum to one.
        template_weights /= tf.reduce_sum(template_weights)

        # Fetch precomputed diff rates for each template.
        # (n_events, n_templates) tensor
        template_diffrates = self._fetch(self.column, data_tensor)

        # Compute weighted average of diff rates
        # (n_events,) tensor
        return tf.reduce_sum(
            template_diffrates * template_weights[None, :],
            axis=1)

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

        assert len(self.defaults) == 1

        template_weights = tfp.math.batch_interp_regular_1d_grid(
            x=params[next(iter(self.defaults))],
            x_ref_min=self._grid_coordinates[0][0],
            x_ref_max=self._grid_coordinates[0][-1],
            y_ref=self._grid_weights,
        )

        template_weights /= tf.reduce_sum(template_weights)

        template_epb = [template._mh_events_per_bin for template in self._templates]
        template_epb_combine = deepcopy(template_epb[0])
        template_epb_combine.histogram = np.sum([template.histogram * weight for template, weight in
                                                 zip(template_epb, template_weights)], axis=0)

        return pd.DataFrame(dict(zip(
            self._templates[0].axis_names,
            template_epb_combine.get_random(n_events).T)))
