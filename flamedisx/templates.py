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
    _method = 'linear'

    def __init__(
            self,
            params_and_templates: ty.Tuple[ty.Dict[str, float], ty.Any],
            params_and_normalisations: ty.Tuple[ty.Dict[str, float], float] = None,
            bin_edges=None,
            axis_names=None,
            events_per_bin=False,
            interpolate=False,
            _skip_tf_init=False,
            method='BSpline',
            *args,
            **kwargs):
        """
            Initialize the MultiTemplateSource that allows for template morphinh
            Args:
                - params_and_templates: 2-tuples of
                        (dict of parameter names and values, template histogram).
                        The parameter names must be the same for all templates.
                - params_and_normalisations: 2-tuples of
                        (dict of parameter names and values, normalisation factor).
                        The parameter names must be the same for all normalisations.
            Kwargs:
                - bin_edges: None, or a list of numpy arrays with bin edges.
                - axis_names: None, or a sequence of axis names.
                - events_per_bin: set to True if templates specify expected events per bin,
                                  rather than differential rate.
                - interpolate: if True, differential rates are to be interpolated linearly
                - _skip_tf_init: if True, skip tensorflow initialization (for subclassing).
                - method: interpolation method between templates, either 'linear' or 'BSpline'.
                          normalisations are always linaerly interpolated.
        """
        self._templates = [
            TemplateWrapper(
                template, bin_edges, axis_names, events_per_bin, interpolate)
            for _, template in params_and_templates]
        assert method in ('linear','BSpline'), "Only 'linear' and 'BSpline' methods are supported"
        self._method = method
        if self._method == 'BSpline':
            assert len(params_and_templates[0][0]) == 1, "BSpline only supports moprhing of 1 parameter"
        self.param_name = list(params_and_templates[0][0].keys())[0]
        # Grab parameter names. Promote first set of values to defaults.
        self.n_templates = n_templates = len(self._templates)
        assert n_templates > 0, "Need at least one template to morph"
        # We will include mu variation separately
        self.mu = self._templates[0].mu
        if params_and_normalisations is not None:
            assert self.mu ==1, "If providing normalisations, template mu must be 1"
        defaults = params_and_templates[0][0]
        for params, _ in params_and_templates:
            assert tuple(params.keys()) == tuple(defaults.keys())
        # Where BSpline code diverges
        if self._method == 'BSpline':
            return self._init_BSpline(
                            params_and_templates, params_and_normalisations, bin_edges,
                            axis_names, events_per_bin, interpolate, _skip_tf_init,
                            n_templates,
                            defaults,
                            *args, **kwargs)
        if self._method == 'linear':
            return self._init_linear(
                params_and_templates, params_and_normalisations, bin_edges,
                axis_names, events_per_bin, interpolate, _skip_tf_init,
                n_templates,
                defaults,
                *args, **kwargs)
        raise NotImplementedError("Only 'linear' and 'BSpline' methods are supported, how did you get here?")

    def _init_linear(self, params_and_templates, params_and_normalisations, bin_edges,
                    axis_names, events_per_bin, interpolate, _skip_tf_init,
                    n_templates,
                    defaults,
                    *args, **kwargs):
        """
            Initialize the original linear interpolation method. Works for many parameters, not C2 continuous.
            TODO: implement params_and_normalisations support.
            Args & Kwargs:
                Most of the parameters are the same as for __init__.
                Common parameters derived in __init__ are line seperate for clarity.
        """
        assert params_and_normalisations is None, "Normalisations not yet supported for linear interpolation"
        # Build an interpolator that produces the _weights_ of each template
        # at a given parameter space point, according to linear interpolation.
        #
        # This interpolator maps an (n_templates = n_params,) array to
        # an (n_templates,) array.
        #
        # When evaluated at the exact location of a template, the result has 1
        # in the corresponding template's position, and zeros elsewhere.
        _template_weights = scipy.interpolate.LinearNDInterpolator(
            points=np.asarray([list(params.values()) for params, _ in params_and_templates]),
            values=np.eye(n_templates))

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

        # The expected number of events must also be interpolated.
        # For consistency, it must be done in the same way (first interpolate
        # to a regular grid, then linearly from there).
        # (n_templates,) array
        _template_mus = np.asarray([
            template.mu for template in self._templates])
        self._grid_mus = np.average(
            # numpy won't let us get away with a size-1 axis here, we have to
            # actually repeat the values. (If we had jax we could just vmap...)
            np.repeat(_template_mus[:, None], n_grid_points, axis=1),
            axis=0,
            weights=_grid_weights.reshape(n_templates, n_grid_points))
        assert self._grid_mus.shape == (n_templates,)
        self._mu_interpolator = scipy.interpolate.RegularGridInterpolator(
            points=_grid_coordinates,
            values=self._grid_mus.reshape(_full_grid_coordinates[0].shape),
            method='linear')

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
        self.defaults = defaults

        # This is needed in tensorflow, so convert it now
        self._grid_coordinates = tuple([fd.np_to_tf(np.asarray(g)) for g in _grid_coordinates])
        self._grid_weights = fd.np_to_tf(_grid_weights)
        super().__init__(*args, **kwargs)
        self.defaults = {**defaults,**{k: tf.cast(v, fd.float_type()) for k, v in defaults.items()}}
        self.parameter_index = fd.index_lookup_dict(self.defaults.keys())
        if not _skip_tf_init:
            self.trace_differential_rate()

    def _init_BSpline(self, params_and_templates, params_and_normalisations, bin_edges,
                        axis_names, events_per_bin, interpolate, _skip_tf_init,
                        n_templates,
                        defaults,
                        *args, **kwargs):
        """
            Initiliaztion the BSpline method, which works for 1 parameter only, but is C2 continuous.
            Args & Kwargs:
                Most of the parameters are the same as for __init__.
                Common parameters derived in __init__ are line seperate for clarity.
        """

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

        # This is needed in tensorflow, so convert it now
        self._grid_coordinates = tuple([fd.np_to_tf(np.asarray(g)) for g in _grid_coordinates])
        self._grid_weights = fd.np_to_tf(_grid_weights)

        param_vals = np.asarray([list(params.values())[0] for params, _ in params_and_templates])
        self.pmin = tf.constant(min(param_vals), fd.float_type())
        self.pmax = tf.constant(max(param_vals), fd.float_type())
        pvals = tf.convert_to_tensor(param_vals, fd.float_type())

        normalisations = np.array([norm.mu.numpy() for norm in self._templates])
        if params_and_normalisations is not None:
            normalisations = np.array([norm for _, norm in params_and_normalisations])
        self.normalisations = tf.convert_to_tensor(normalisations / normalisations[0],
                                                   fd.float_type())

        # Assume equi-spacing!
        self.dstep = pvals[1] - pvals[0]
        # Need to pad domain.. four might be excessive. ToDo: what is this exception?
        try:
            self.pvals = list(np.arange(pvals[0] - 4. * self.dstep, pvals[-1] + 4. * self.dstep, self.dstep))
            assert len(self.pvals) == len(pvals) + 8, "Something went wrong with the padding!"
        except:
            self.pvals = list(np.arange(pvals[0] - 4. * self.dstep, pvals[-1] + 5. * self.dstep, self.dstep))
            assert len(self.pvals) == len(pvals) + 8, "Something went wrong with the padding!"

        self.array_columns = ((self.column, n_templates+8),)

        super().__init__(*args, **kwargs)
        self.defaults = {**defaults,**{k: tf.cast(v, fd.float_type()) for k, v in defaults.items()}}
        self.parameter_index = fd.index_lookup_dict(self.defaults.keys())
        if not _skip_tf_init:
            self.trace_differential_rate()

    def extra_needed_columns(self):
        return super().extra_needed_columns() + [self.column]

    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        if self._method == 'linear':
            # Get array of differential rates for each template.
            # Outer list() is to placate pandas, which does not like array columns..
            self.data[self.column] = list(np.asarray([
                template.differential_rates_numpy(self.data)
                for template in self._templates]).T)
            return 0
        elif self._method != 'BSpline':
            raise NotImplementedError("Only 'linear' and 'BSpline' methods are supported for annotation")
        #construct tensor of knots
        #requires a tensor of elements
        #data is stored as [[d_evt1^h1,d_evt1^h2..],[d_evt2^h1,d_evt2^h2..]]
        # so just need to construct and x-values object and let data column handle y-values
        #with some padding for the domain!
        Nk=len(self.pvals)
        knot_range=self.pvals[-1]-self.pvals[0]
        linear_shift=2*self.dstep/knot_range
        start=min(self.pvals)
        end=max(self.pvals)
        self.original_range=tf.constant(end-start,dtype=fd.float_type())
        self.max_pos=tf.constant(Nk- 2,dtype=fd.float_type())

        self.start=tf.constant(start,dtype=fd.float_type())
        self.linear_shift=tf.constant(linear_shift,dtype=fd.float_type())
        self.linear_shift_shift=tf.constant(knot_range/2,dtype=fd.float_type())

        self.data[self.column] = list(np.asarray([
            template.differential_rates_numpy(self.data)
            for template in self._templates]).T)

        linear_interp_padded_diff_rates=[]
        for diff_rate_per_hist in self.data[self.column]:

            if np.sum(diff_rate_per_hist[:2])>0:
                left_edge=scipy.interpolate.interp1d(
                    self.pvals[4:6],diff_rate_per_hist[:2],
                    kind='linear',fill_value="extrapolate",
                    bounds_error=False)(self.pvals[:4])
            else:
                left_edge=list(np.repeat(diff_rate_per_hist[0],4))

            if np.sum(diff_rate_per_hist[-2:])>0:
                right_edge=scipy.interpolate.interp1d(
                    self.pvals[-6:-4],diff_rate_per_hist[-2:],
                    kind='linear',fill_value="extrapolate",
                    bounds_error=False)(self.pvals[-4:])
            else:
                right_edge=list(np.repeat(diff_rate_per_hist[-1],4))

            linear_interp_padded_diff_rates.append(np.concatenate([left_edge,diff_rate_per_hist,right_edge]))

        self.data[self.column]=linear_interp_padded_diff_rates
        self.tensor_xvals=tf.convert_to_tensor([self.pvals for _ in range(self.batch_size)],dtype=fd.float_type())

    def mu_before_efficiencies(self, **params):
        return self.mu

    def estimate_mu(self, n_trials=None, **params):
        if self._method == 'linear':
            return self._mu_interpolator([
                            params.get(param, default)
                            for param, default in self.defaults.items()])
        elif self._method != 'BSpline':
            raise NotImplementedError("Only 'linear' and 'BSpline' methods are supported for mu estimation")
        norm = tfp.math.batch_interp_regular_1d_grid(
                    x=params[self.param_name],
                    x_ref_min=self.pmin,
                    x_ref_max=self.pmax,
                    y_ref=self.normalisations,
                    )

        return tf.reshape(norm, shape=[]) * self.mu

    def bspline_interpolate_per_bin(self, param,knots):
        def interp(knots_for_event):
            #second order non-cyclical b-spline with varying knots
            #returns [x,y] so ignore x
            #hackiest shit ever
            shift = self.linear_shift*(param-self.linear_shift_shift)
            knot_coord = self.max_pos*(param-self.start)/self.original_range+shift
            return tf.reduce_sum(bspline.interpolate(knots_for_event,
                                                     knot_coord,
                                                        2, False) \
                                * tf.constant([0,1],dtype=fd.float_type()))
        #vectorized map over all events
        y=tf.vectorized_map(interp,elems=knots)
        return y

    def _differential_rate_BSpline(self, data_tensor, ptensor):
        norm = tfp.math.batch_interp_regular_1d_grid(
                x=self._fetch_param(self.param_name, ptensor),
                x_ref_min=self.pmin,
                x_ref_max=self.pmax,
                y_ref=self.normalisations,
                )
        
        knots_per_event=tf.convert_to_tensor([self.tensor_xvals, self._fetch(self.column, data_tensor)],dtype=fd.float_type())
        bspline_diff_rates=self.bspline_interpolate_per_bin(self._fetch_param(self.param_name, ptensor), 
                                                            tf.transpose(knots_per_event,perm=[1,0,2]))
        dr=tf.squeeze(norm)*bspline_diff_rates

        return dr
    
    def _differential_rate_linear(self, data_tensor, ptensor):
        # Compute template weights at this parameter point
        # (n_templates,) tensor
        # (The axis order is weird here. It seems to work...)
        permutation = (
            [self._grid_weights.ndim - 1]
            + list(range(0, self._grid_weights.ndim - 1)))
        template_weights = tfp.math.batch_interp_rectilinear_nd_grid(
            x=ptensor[None, :],
            x_grid_points=self._grid_coordinates,
            y_ref=tf.transpose(self._grid_weights, permutation),
            axis=1,
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
    
    def _differential_rate(self, data_tensor, ptensor):
        if self._method == 'linear':
            return self._differential_rate_linear(data_tensor, ptensor)
        elif self._method == 'BSpline':
            return self._differential_rate_BSpline(data_tensor, ptensor)
        else:
            raise NotImplementedError("Only 'linear' and 'BSpline' methods are supported for differential rate estimation")

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