from copy import copy
from contextlib import contextmanager
import inspect

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis


@export
class Source:
    n_batches = None
    n_padding = None
    trace_difrate = True

    model_functions = tuple()
    special_model_functions = tuple()
    inner_dimensions = tuple()

    frozen_model_functions = tuple()
    array_columns = tuple()

    # Final observable dimensions; for use in domain / cross-domain
    final_dimensions = tuple()

    # List all columns that are manually _fetch ed here
    # These will be added to the data_tensor even when the model function
    # inspection will not find them.
    def extra_needed_columns(self):
        return []

    data = None

    ##
    # Initialization and helpers
    ##

    def scan_model_functions(self):
        """Discover which functions need which arguments / dimensions
        Discover possible parameters.
        Returns f_dims, f_params and defaults.
        """
        self.f_dims = f_dims = {x: [] for x in self.model_functions}
        self.f_params = f_params = {x: [] for x in self.model_functions}
        self.defaults = defaults = dict()

        for fname in self.model_functions:
            f = getattr(self, fname)
            if not callable(f):
                # Constant
                continue
            seen_special = False
            for pname, p in inspect.signature(f).parameters.items():
                if pname == 'self':
                    continue
                if p.default is inspect.Parameter.empty:
                    if fname in self.special_model_functions and not seen_special:
                        seen_special = True
                    else:
                        # It's an observable dimension
                        f_dims[fname].append(pname)
                else:
                    # It's a parameter that can be fitted
                    f_params[fname].append(pname)
                    if pname in defaults and p.default != defaults[pname]:
                        raise ValueError(f"Inconsistent defaults for {pname}")
                    defaults[pname] = tf.convert_to_tensor(
                        p.default, dtype=fd.float_type())

    def __init__(self,
                 data=None,
                 batch_size=10,
                 max_sigma=3,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 fit_params=None,
                 progress=False,
                 **params):
        """Initialize a flamedisx source

        :param data: Dataframe with events to use in the inference
        :param batch_size: Number of events / tensorflow batch
        :param max_sigma: Hint for hidden variable bounds computation
        :param data_is_annotated: If True, skip annotation
        :param _skip_tf_init: If True, skip tensorflow cache initialization
        :param _skip_bounds_computation: If True, skip bounds compuation
        :param fit_params: List of parameters to fit
        :param progress: whether to show progress bars for mu estimation
          (if data is not None)
        :param params: New defaults to use
        """
        self.max_sigma = max_sigma

        # Check for duplicated model functions
        for attrname in ['model_functions', 'special_model_functions']:
            l_ = getattr(self, attrname)
            if len(set(l_)) != len(l_):
                raise ValueError(f"{attrname} contains duplicates: {l_}")

        # Discover which functions need which arguments / dimensions
        # Discover possible parameters.
        self.scan_model_functions()

        # Change from (column, length) tuple to dict
        self.array_columns = dict(self.array_columns)

        # Which columns are needed from data?
        ctc = list(set(sum(self.f_dims.values(), [])))      # Used in model functions.
        ctc += list(self.final_dimensions)                  # Final observables (e.g. S1, S2)
        ctc += self.extra_needed_columns()                  # Manually fetched columns
        ctc += self.frozen_model_functions                     # Frozen methods (e.g. not tf-compatible)
        ctc += [x + '_min' for x in self.inner_dimensions]  # Left bounds of domains
        ctc = list(set(ctc))

        self.column_index = fd.index_lookup_dict(ctc,
                                                 column_widths=self.array_columns)
        self.n_columns_in_data_tensor = (
                len(self.column_index) + sum(self.array_columns.values())
                - len(self.array_columns))

        self.set_defaults(**params)

        if fit_params is None:
            fit_params = list(self.defaults.keys())
        # Filter out parameters the source does not use
        self.fit_params = [x for x in fit_params if x in self.defaults]

        self.parameter_index = fd.index_lookup_dict(self.defaults.keys())
        # Indices of params we actually want to fit; we have to differentiate wrt these
        self.fit_param_indices = tuple([
            self.parameter_index[param_name]
            for param_name in self.fit_params])

        if data is None:
            # We're calling the source without data. Set the batch_size here
            # since we can't pass it to set_data later
            self.batch_size = batch_size
        else:
            self.batch_size = min(batch_size, len(data))
            self.set_data(data,
                          progress=progress,
                          data_is_annotated=data_is_annotated,
                          _skip_tf_init=_skip_tf_init,
                          _skip_bounds_computation=_skip_bounds_computation)

        if not _skip_tf_init:
            self.trace_differential_rate()

    def set_defaults(self, **params):
        for k, v in params.items():
            if k in self.defaults:
                self.defaults[k] = tf.convert_to_tensor(
                    v, dtype=fd.float_type())

    def set_data(self,
                 data=None,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 **params):
        self.set_defaults(**params)

        if data is None:
            self.data = self.n_batches = self.n_padding = None
            return
        self.data = data
        del data

        # Annotate requests n_events, currently no padding
        self.n_padding = 0
        self.n_events = len(self.data)
        self.n_batches = np.ceil(
            self.n_events / self.batch_size).astype(np.int)

        if not _skip_tf_init:
            # Extend dataframe with events to nearest batch_size multiple
            # We're using actual events for padding, since using zeros or
            # nans caused problems with gradient calculation
            # padded events are clipped when summing likelihood terms
            self.n_padding = self.n_batches * self.batch_size - len(self.data)
            if self.n_padding > 0:
                # Repeat first event n_padding times and concat to rest of data
                df_pad = self.data.iloc[np.zeros(self.n_padding)]
                self.data = pd.concat([self.data, df_pad], ignore_index=True)

        if not data_is_annotated:
            self.add_extra_columns(self.data)
            if not _skip_bounds_computation:
                self._annotate()

        if not _skip_tf_init:
            self._check_data()
            self._populate_tensor_cache()
            self._calculate_dimsizes()

    def _check_data(self):
        """Do any final checks on the self.data dataframe,
        before passing it on to the tensorflow layer.
        """
        for column in self.column_index:
            if (column not in self.data.columns
                    and column not in self.frozen_model_functions):
                raise ValueError(f"Data lacks required column {column}; "
                                 f"did annotation happen correctly?")

    def _populate_tensor_cache(self):
        """Set self.data_tensor to a big tensor of shape:
          (n_batches, events_per_batch, n_columns_in_data_tensor)
        """
        # First, build a list of (n_events, 1 or column_width) tensors
        result = []
        for column in self.column_index:

            if column in self.frozen_model_functions:
                # Calculate the column
                y = self.gimme(column)
            else:
                # Just fetch it from the dataframe
                y = self._fetch(column)
            y = tf.cast(y, dtype=fd.float_type())

            # For non-array columns, add size-1 axis for concatenation
            if len(y.shape) == 1:
                assert column not in self.array_columns
                y = tf.reshape(y, (len(y), 1))

            result.append(y)

        # Concat these and shape them to the batch size
        result = tf.concat(result, axis=1)
        self.data_tensor = tf.reshape(
            result,
            [self.n_batches, -1, self.n_columns_in_data_tensor])

    def _calculate_dimsizes(self):
        self.dimsizes = dict()
        for dim in self.inner_dimensions:
            ma = self._fetch(dim + '_max')
            mi = self._fetch(dim + '_min')
            self.dimsizes[dim] = int(tf.reduce_max(ma - mi + 1).numpy())
        for dim in self.final_dimensions:
            self.dimsizes[dim] = 1

    @contextmanager
    def _set_temporarily(self, data, **kwargs):
        """Set data and/or defaults temporarily, without affecting the
        data tensor state"""
        if data is None:
            raise ValueError("No point in setting data = None temporarily")
        old_defaults = copy(self.defaults)
        if data is None:
            self.set_defaults(**kwargs)
        else:
            if self.data is None:
                old_data = None
            else:
                old_data = self.data[:self.n_events]  # Remove padding
            self.set_data(data, **kwargs, _skip_tf_init=True)
        try:
            yield
        finally:
            self.defaults = old_defaults
            if old_data is not None:
                self.set_data(
                    old_data,
                    data_is_annotated=True,
                    _skip_tf_init=True)

    def annotate_data(self, data, **params):
        """Add columns to data with inference information"""
        with self._set_temporarily(data, **params):
            self._annotate()
            return self.data

    ##
    # Data fetching / calculation
    ##

    def _fetch(self, x, data_tensor=None):
        """Return a tensor column from the original dataframe (self.data)
        :param x: column name
        :param data_tensor: Data tensor, columns as in self.column_index
        """
        if data_tensor is None:
            # We're inside annotate, just return the column
            x = self.data[x].values
            if x.dtype == np.object:
                # This will only work on homogeneous array fields
                x = np.stack(x)
            return fd.np_to_tf(x)

        return data_tensor[:, self.column_index[x]]

    def _fetch_param(self, param, ptensor):
        if ptensor is None:
            return self.defaults[param]
        idx = tf.dtypes.cast(self.parameter_index[param],
                             dtype=fd.int_type())
        return ptensor[idx]

    def gimme(self, fname,
              *,
              data_tensor=None, ptensor=None, bonus_arg=None, numpy_out=False):
        """Evaluate the model function fname with all required arguments

        :param fname: Name of the model function to compute
        :param bonus_arg: If fname takes a bonus argument, the data for it
        :param numpy_out: If True, return (tuple of) numpy arrays,
        otherwise (tuple of) tensors.
        :param data_tensor: Data tensor, columns as self.column_index
        If not given, use self.data (used in annotate)
        :param ptensor: Parameter tensor, columns as self.param_id
        If not give, use defaults dictionary (used in annotate)
        Before using gimme, you must use set_data to
        populate the internal caches.
        """
        assert (bonus_arg is not None) == (fname in self.special_model_functions)
        assert isinstance(fname, str), \
            f"gimme needs fname to be a string, not {type(fname)}"

        if data_tensor is None:
            # We're in an annotate
            assert hasattr(self, 'data'), "You must set data first"
        else:
            # We're computing
            if not hasattr(self, 'data_tensor'):
                raise ValueError(
                    "You must set_data first (and populate the tensor cache)")

        f = getattr(self, fname)

        # Frozen data methods should not be called again,
        # just fetch them from the data tensor (if we have one)
        if fname in self.frozen_model_functions:
            if data_tensor is not None:
                return self._fetch(fname, data_tensor)

        if callable(f):
            args = [self._fetch(x, data_tensor) for x in self.f_dims[fname]]
            if bonus_arg is not None:
                args = [bonus_arg] + args
            kwargs = {pname: self._fetch_param(pname, ptensor)
                      for pname in self.f_params[fname]}
            res = f(*args, **kwargs)

        else:
            if bonus_arg is None:
                n = len(self.data) if data_tensor is None else data_tensor.shape[0]
                x = tf.ones(n, dtype=fd.float_type())
            else:
                x = tf.ones_like(bonus_arg, dtype=fd.float_type())
            res = f * x

        if numpy_out:
            return fd.tf_to_np(res)
        return fd.np_to_tf(res)

    def gimme_numpy(self, fname, bonus_arg=None):
        """Gimme for use in simulation / annotate"""
        return self.gimme(fname=fname,
                          data_tensor=None, ptensor=None,
                          bonus_arg=bonus_arg,
                          numpy_out=True)

    gimme_numpy.__doc__ = gimme.__doc__

    ##
    # Differential rate computation
    ##

    def batched_differential_rate(self, progress=True, **params):
        """Return numpy array with differential rate for all events.
        """
        progress = (lambda x: x) if not progress else tqdm
        y = []
        for i_batch in progress(range(self.n_batches)):
            q = self.data_tensor[i_batch]
            y.append(fd.tf_to_np(self.differential_rate(data_tensor=q,
                                                        **params)))

        return np.concatenate(y)[:self.n_events]

    def _batch_data_tensor_shape(self):
        return [self.batch_size, self.n_columns_in_data_tensor]

    def trace_differential_rate(self):
        input_signature = (
            tf.TensorSpec(shape=self._batch_data_tensor_shape(),
                          dtype=fd.float_type()),
            tf.TensorSpec(shape=[len(self.parameter_index)],
                          dtype=fd.float_type()))
        self._differential_rate_tf = tf.function(
            self._differential_rate,
            input_signature=input_signature)

    def differential_rate(self, data_tensor=None, autograph=True, **kwargs):
        ptensor = self.ptensor_from_kwargs(**kwargs)
        if autograph and self.trace_difrate:
            return self._differential_rate_tf(
                data_tensor=data_tensor, ptensor=ptensor)
        else:
            return self._differential_rate(
                data_tensor=data_tensor, ptensor=ptensor)

    def ptensor_from_kwargs(self, **kwargs):
        return tf.convert_to_tensor([kwargs.get(k, self.defaults[k])
                                     for k in self.defaults])

    ##
    # Helpers for response model implementation
    ##

    def domain(self, x, data_tensor=None):
        """Return (n_events, |possible x values|) matrix containing all
        possible integer values of x for each event.

        If x is a final dimension (e.g. s1, s2), we return an (n_events, 1)
        tensor with observed values -- NOT a (n_events,) array!
        """
        if x in self.final_dimensions:
            return self._fetch(x, data_tensor=data_tensor)[:, o]

        x_range = tf.cast(tf.range(self.dimsizes[x]),
                          dtype=fd.float_type())[o, :]
        left_bound = self._fetch(x + '_min', data_tensor=data_tensor)[:, o]
        return left_bound + x_range

    def cross_domains(self, x, y, data_tensor):
        """Return (x, y) two-tuple of (n_events, |x|, |y|) tensors
        containing possible integer values of x and y, respectively.
        """
        # TODO: somehow mask unnecessary elements and save computation time
        x_domain = self.domain(x, data_tensor)
        y_domain = self.domain(y, data_tensor)
        result_x = tf.repeat(x_domain[:, :, o], y_domain.shape[1], axis=2)
        result_y = tf.repeat(y_domain[:, o, :], x_domain.shape[1], axis=1)
        return result_x, result_y

    ##
    # Simulation methods and helpers
    ##

    def simulate(self, n_events, fix_truth=None, full_annotate=False, **params):
        """Simulate n events.

        Will omit events lost due to selection/detection efficiencies
        """
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        # Draw random "deep truth" variables (energy, position)
        # Pass on a copy of the dict or DataFrame
        fix_truth = self.validate_fix_truth(fix_truth.copy()
                                            if fix_truth is not None
                                            else None)
        sim_data = self.random_truth(n_events, fix_truth=fix_truth, **params)
        assert isinstance(sim_data, pd.DataFrame)

        with self._set_temporarily(sim_data, _skip_bounds_computation=True,
                                   **params):
            # Do the forward simulation of the detector response
            d = self._simulate_response()
            if full_annotate:
                # Now that we have s1 and s2 values, we can populate
                # columns like e_vis, photon_produced_mle, etc.
                # This is optional since it can be expensive (e.g. for
                # the WIMPsource, where it includes the full energy spectrum!)
                return self.annotate_data(d)
            return d

    def validate_fix_truth(self, fix_truth):
        """Return checked fix truth, with extra derived variables if needed"""
        return fix_truth

    @staticmethod
    def _overwrite_fixed_truths(data, fix_truth, n_events):
        """Replaces all columns in data with fix_truth.

        Careful: ensure mutual constraints are accounted for first!
        (e.g. fixing energy for a modulating WIMP has consequences for the
         time distribution.)
        """
        if fix_truth is not None:
            for k, v in fix_truth.items():
                data[k] = np.ones(n_events, dtype=np.float) * v

    ##
    # Mu estimation
    ##

    def mu_function(self,
                    interpolation_method='star',
                    n_trials=int(1e5),
                    progress=True,
                    **param_specs):
        """Return interpolator for number of expected events
        Parameters must be specified as kwarg=(start, stop, n_anchors)
        """
        if interpolation_method != 'star':
            raise NotImplementedError(
                f"mu interpolation method {interpolation_method} "
                f"not implemented")

        # Estimate mu under the current defaults
        base_mu = tf.constant(self.estimate_mu(n_trials=n_trials),
                              dtype=fd.float_type())

        # Estimate mus under the specified variations
        pspaces = dict()    # parameter -> tf.linspace of anchors
        mus = dict()        # parameter -> tensor of mus
        _iter = param_specs.items()
        if progress:
            _iter = tqdm(_iter, desc="Estimating mus")
        for pname, (start, stop, n) in _iter:
            if pname not in self.defaults:
                # We don't take this parameter. Consistent with __init__,
                # don't complain and just discard it silently.
                continue
            # Parameters are floats, but users might input ints as anchors
            # accidentally, triggering a confusing tensorflow device placement
            # message
            start, stop = float(start), float(stop)
            pspaces[pname] = tf.linspace(start, stop, n)
            mus[pname] = tf.convert_to_tensor(
                 [self.estimate_mu(**{pname: x}, n_trials=n_trials)
                  for x in np.linspace(start, stop, n)],
                 dtype=fd.float_type())

        def mu_itp(**kwargs):
            mu = base_mu
            for pname, v in kwargs.items():
                mu *= tfp.math.interp_regular_1d_grid(
                    x=v,
                    x_ref_min=param_specs[pname][0],
                    x_ref_max=param_specs[pname][1],
                    y_ref=mus[pname]) / base_mu
            return mu

        return mu_itp

    def estimate_mu(self, n_trials=int(1e5), **params):
        """Return estimate of total expected number of events
        :param n_trials: Number of events to simulate for estimate
        """
        d_simulated = self.simulate(n_trials, **params)
        return (self.mu_before_efficiencies(**params)
                * len(d_simulated) / n_trials)

    ##
    # Functions you have to override
    ##

    def _differential_rate(self, data_tensor, ptensor):
        raise NotImplementedError

    def mu_before_efficiencies(self, **params):
        """Return mean expected number of events BEFORE efficiencies/response
        using data for the evaluation of the energy spectra
        """
        raise NotImplementedError

    ##
    # Functions you probably should override
    ##

    def _annotate(self):
        """Add columns needed in inference to self.data
        """

    def add_extra_columns(self, data):
        """Add additional columns to data

        :param data: pandas DataFrame
        """

    def random_truth(self, n_events, fix_truth=None, **params):
        """Draw random "deep truth" variables (energy, position) """
        raise NotImplementedError

    def _simulate_response(self):
        """Do a forward simulation of the detector response, using self.data"""
        return self.data


@export
class ColumnSource(Source):
    """Source that expects precomputed differential rate in a column,
    and precomputed mu in an attribute
    """
    column = 'rename_me!'
    mu = 42.

    def extra_needed_columns(self):
        return super().extra_needed_columns() + [self.column]

    def estimate_mu(self, n_trials=None, **params):
        return self.mu

    def mu_before_efficiencies(self, **params):
        return self.mu

    def _differential_rate(self, data_tensor, ptensor):
        return self._fetch(self.column, data_tensor)

    def random_truth(self, n_events, fix_truth=None, **params):
        print(f"{self.__class__.__name__} cannot generate events, skipping")
        return pd.DataFrame()
