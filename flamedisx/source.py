from copy import copy
from contextlib import contextmanager
import inspect
import typing as ty
import warnings

import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from scipy import stats

from tqdm import tqdm

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis


@export
class Source:
    #: Number of event batches to use in differential rate computations
    n_batches = None

    #: Number of fake events that were padded to the final batch
    #: to make it match the batch size
    n_padding = None

    #: Whether to trace (compile into a tensorflow graph) the differential
    #: rate computation
    trace_difrate = True

    #: Names of model functions
    model_functions: ty.Tuple[str] = tuple()

    #: Names of model functions that take an additional first argument
    #: ('bonus arg'). This must be a subset of model_functions.
    special_model_functions: ty.Tuple[str] = tuple()

    #: Model functions whose results should be evaluated once per event,
    #: then stored with the data. For example, non-tensorflow functions.
    #: Note these cannot have any fittable parameters.
    frozen_model_functions: ty.Tuple[str] = tuple()

    #: Columns that we don't want to include in the tensor of data columns
    exclude_data_tensor: ty.Tuple[str] = tuple()

    #: Array-valued data columns: tuple of (name, length)
    array_columns: ty.Tuple[ty.Tuple[str, int]] = tuple()

    #: Any additional source attributes that should be configurable.
    model_attributes = tuple()

    # List all columns that are manually _fetch ed here
    # These will be added to the data_tensor even when the model function
    # inspection will not find them.
    def extra_needed_columns(self):
        return []

    #: The fully annotated event data
    data: pd.DataFrame = None

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
                if pname in self.model_functions:
                    raise AttributeError(
                        f"{pname} is used both as a model function and "
                        f"as a parameter of a model function, {fname}")
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

    def print_config(self,
                     format='table',
                     column_widths=(40, 20),
                     omit=tuple()):
        """Print the defaults of all parameters (from Source.defaults), and of
        model functions that have been set to constants (from Source.f_dims)

        :param format: 'table' to print a fixed-width table, 'config' to print
            as a configuration file
        :param column_widths: 2-tuple of column widths to use for table format
        :param omit: settings to omit from printout. Useful for format='config',
            since some things (like arrays and tensors) cannot be evaluated
            from their string representation.
        """

        def print_row(*cols, header=False):
            cols = [str(x).replace('\n', '') for x in cols]
            if format == 'table':
                print(''.join([
                    col.ljust(w) if len(col) < w else col[:w - 3] + '...'
                    for col, w in zip(cols, column_widths)]))
            else:
                result = '# ' if header else ''
                result += ' = '.join(cols)
                print(result)

        format_value = str if format == 'table' else repr

        def print_line(marker='-'):
            if format == 'table':
                print(marker * sum(column_widths))
            else:
                print()

        print_row('Parameter', 'Default', header=True)
        print_line()
        for pname, default in sorted(self.defaults.items()):
            if pname in omit:
                continue
            print_row(pname, format_value(default.numpy()))
        print()

        print_row("Constant (could be made a function)", 'Default', header=True)
        print_line()
        for fname in sorted(self.model_functions):
            if fname in omit:
                continue
            f = getattr(self, fname)
            if not callable(f):
                print_row(fname, format_value(f))
        print()

        print_row("Other attribute", 'Default', header=True)
        print_line()
        for fname in sorted(self.model_attributes):
            if fname in omit:
                continue
            print_row(fname, format_value(getattr(self, fname)))
        print()

    def __init__(self,
                 data=None,
                 batch_size=10,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 fit_params=None,
                 progress=False,
                 **params):
        """Initialize a flamedisx source

        :param data: Dataframe with events to use in the inference
        :param batch_size: Number of events / tensorflow batch
        :param data_is_annotated: If True, skip annotation
        :param _skip_tf_init: If True, skip tensorflow cache initialization
        :param _skip_bounds_computation: If True, skip bounds compuation
        :param fit_params: List of parameters to fit
        :param progress: whether to show progress bars for mu estimation
            (if data is not None)
        :param params: New defaults to for parameters, and new values for
        constant-valued model functions.
        """
        # Check for duplicated model functions
        for attrname in ['model_functions', 'special_model_functions']:
            l_ = getattr(self, attrname)
            if len(set(l_)) != len(l_):
                raise ValueError(f"{attrname} contains duplicates: {l_}")
        # Check all special model functions are actually model functions
        for fname in self.special_model_functions:
            if fname not in self.model_functions:
                raise ValueError(
                    f"{attrname} is listed as a special model function, "
                    f"but not as a model function")

        # Discover which functions need which arguments / dimensions
        # Discover possible parameters.
        self.scan_model_functions()

        # Change from (column, length) tuple to dict
        self.array_columns = dict(self.array_columns)

        # Which columns are needed from data?
        ctc = list(set(sum(self.f_dims.values(), [])))      # Used in model functions
        ctc += self.extra_needed_columns()                  # Manually fetched columns
        ctc += self.frozen_model_functions                  # Frozen methods (e.g. not tf-compatible)
        self.ctc = list(set(ctc) - set([x for x in self.exclude_data_tensor]))  # We want to ignore these

        self.column_index = fd.index_lookup_dict(self.ctc,
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

    def set_defaults(self, *, config=None, **params):
        # Load new params from configuration files
        params = {**fd.load_config(config), **params}

        # Apply new defaults
        unused = dict()
        for k, v in params.items():
            if k in self.defaults:
                # Change a default
                self.defaults[k] = tf.convert_to_tensor(
                    v, dtype=fd.float_type())
            elif k in self.model_functions:
                # Change a model function, only allowed if it is a constant
                # (otherwise, just subclass the source)
                f = getattr(self, k)
                if callable(f):
                    raise AttributeError(
                        f"Use source subclassing to override the non-constant "
                        f"model function {k}")
                setattr(self, k, v)
            elif k in self.model_attributes:
                # Change a generic model attribute
                setattr(self, k, v)
            else:
                unused[k] = v
        if unused:
            warnings.warn(f"Defaults for unused settings ignored: {unused}")

    def set_data(self,
                 data=None,
                 data_is_annotated=False,
                 input_column_index=None,
                 input_data_tensor=None,
                 output_data_tensor=None,
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
            self.n_events / self.batch_size).astype(int)

        if not _skip_tf_init:
            # Extend dataframe with events to nearest batch_size multiple
            # We're using actual events for padding, since using zeros or
            # NaNs caused problems with gradient calculation.
            # Padded events are clipped when summing likelihood terms.
            self.n_padding = self.n_batches * self.batch_size - len(self.data)
            if self.n_padding:
                # Repeat first event n_padding times and concat to rest of data
                df_pad = self.data.iloc[np.zeros(self.n_padding)]
                self.data = pd.concat([self.data, df_pad], ignore_index=True)
            self.data = self.data.reset_index(drop=True)

        if input_data_tensor is not None:
            self.column_index = pkl.load(open(input_column_index, 'rb'))
            self._populate_tensor_cache(input_data_tensor=input_data_tensor)
            return

        if not data_is_annotated:
            self.add_extra_columns(self.data)
            if not _skip_bounds_computation:
                self._annotate()
                self._calculate_dimsizes()

        if not _skip_tf_init:
            self._check_data()
            self._populate_tensor_cache(output_data_tensor=output_data_tensor)

    def _check_data(self):
        """Do any final checks on the self.data dataframe,
        before passing it on to the tensorflow layer.
        """
        for column in self.column_index:
            if (column not in self.data.columns
                    and column not in self.frozen_model_functions):
                raise ValueError(f"Data lacks required column {column}; "
                                 f"did annotation happen correctly?")

    def _populate_tensor_cache(self, input_data_tensor=None, output_data_tensor=None):
        """Set self.data_tensor to a big tensor of shape:
          (n_batches, events_per_batch, n_columns_in_data_tensor)
        """
        if input_data_tensor is not None:
            read_in = \
                tf.data.TFRecordDataset(input_data_tensor).map(lambda x:
                                                               tf.io.parse_tensor(x,
                                                                                  out_type=fd.float_type()))
            for data_tensor in read_in:
                self.data_tensor = data_tensor
            return

        shape = [self.n_batches, self.batch_size, self.n_columns_in_data_tensor]
        if not self.column_index:
            # We want no columns from the data, so
            self.data_tensor = tf.zeros(shape, dtype=fd.float_type())
            return

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
        self.data_tensor = tf.reshape(result, shape)

        if output_data_tensor is not None:
            write_out = tf.io.serialize_tensor(self.data_tensor)
            with tf.io.TFRecordWriter(output_data_tensor) as writer:
                writer.write(write_out.numpy())

    def _calculate_dimsizes(self):
        # Overriden in IntegratingSource
        pass

    @contextmanager
    def _set_temporarily(self, data, keep_padding=False, **kwargs):
        """Set data and/or defaults temporarily, without affecting the
        data tensor state. Choose whether or not we keep padding from the currently
        set data"""
        if data is None:
            raise ValueError("No point in setting data = None temporarily")
        old_defaults = copy(self.defaults)
        if data is None:
            self.set_defaults(**kwargs)
        else:
            if self.data is None:
                old_data = None
            else:
                if keep_padding:
                    old_data = self.data
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
            if x.dtype == object:
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
            If not given, use defaults dictionary (used in annotate)
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
                if isinstance(bonus_arg, (list, tuple)):
                    args = list(bonus_arg) + args
                else:
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
        """Compile the differential rate computation to a tensorflow graph"""
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
    # Simulation methods and helpers
    ##

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
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
                                   keep_padding=keep_padding, **params):
            # Do the forward simulation of the detector response
            d = self._simulate_response()
            if 'p_accepted' in d.columns:
                # Draw which events are accepted
                d = d.iloc[np.random.rand(len(d)) < d['p_accepted'].values].copy()
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
                data[k] = np.ones(n_events, dtype=float) * v

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
        print(f"{self.__class__.__name__} cannot generate events, skipping")
        return pd.DataFrame()

    def _simulate_response(self) -> pd.DataFrame:
        """Return a dataframe with simulated observed events
        from simulating the detector response, using self.data.
        Note self.data is already set to something random_truth provides.

        You may include a p_accepted column with probabilities
        that an event survives cuts.

        Do not call or duplicate annotate_data: other functions
        will call this when needed.
        """
        return self.data


@export
class ColumnSource(Source):
    """Source that expects precomputed differential rate in a column,
    and precomputed mu in an attribute
    """

    #: Name of the data column containing the precomputed differential rate
    column = 'rename_me!'

    #: Expected events for this source
    mu = 42.

    def extra_needed_columns(self):
        return super().extra_needed_columns() + [self.column]

    def estimate_mu(self, n_trials=None, **params):
        return self.mu

    def mu_before_efficiencies(self, **params):
        return self.mu

    def _differential_rate(self, data_tensor, ptensor):
        return self._fetch(self.column, data_tensor)


@export
class IntegratingSource(Source):
    """Source that sums or integrates its differential rate over one or more
    unobservable dimensions / hidden variables"""

    #: Names of final observable dimensions (e.g. s1, s2)
    #: for use in domain / cross-domain
    final_dimensions: ty.Tuple[str] = tuple()

    #: Names of dimensions of hidden variables (e.g. produced electrons)
    #: for which domain computations and dimsize calculations are to be done
    inner_dimensions: ty.Tuple[str] = tuple()

    #: inner_dimensions excluded from variable stepping logic, i.e.
    #: for which the domain is always a single interval of integers
    no_step_dimensions: ty.Tuple[str] = tuple()

    #: inner_dimensions which take non-integer values; integral will be
    #: approximated as a sum, with resolution controlled by max_dim_sizes
    #: if defined, otherwise by default_max_dim_size.
    non_integer_dimensions: ty.Tuple[str] = tuple()

    #: Names of dimensions of hidden variables for which
    #: dimsize calculations are NOT done here (but in user-defined code)
    #: but for which we DO track _min and _dimsizes
    bonus_dimensions: ty.Tuple[str] = tuple()

    # Dimensions for which we want to calculate priors in bounds computation.
    prior_dimensions: ty.Tuple[ty.Tuple[ty.Tuple[str], ty.Tuple[str]]] = tuple()

    #: Hints for hidden variable bound computation
    default_max_sigma = 3
    default_max_sigma_outer = 3
    default_max_dim_size = 70

    def __init__(self, *args, max_sigma=None, max_sigma_outer=None, **kwargs):
        """Create an integrating source

        :param max_sigma: Hint for hidden variable bounds computation
            If omitted, set to default_max_sigma
        :param max_sigma_outer: Hint for hidden variable bounds computation for outer blocks
            If omitted, set to default_max_sigma_outer

        All other arguments are passed to Source.__init__
        """
        if max_sigma is None:
            max_sigma = self.default_max_sigma
        if max_sigma_outer is None:
            max_sigma_outer = self.default_max_sigma_outer
        self.bounds_prob = stats.norm.cdf(-max_sigma)
        self.bounds_prob_outer = stats.norm.cdf(-max_sigma_outer)
        self.max_sigma = max_sigma
        assert self.bounds_prob > 0., \
            "max_sigma too high!"
        assert self.bounds_prob_outer > 0., \
            "max_sigma_outer too high!"

        # Capping the domain size for hidden variable dimensions. Any which aren't
        # set will default to default_max_dim_size
        if not hasattr(self, 'max_dim_sizes'):
            self.max_dim_sizes = dict()

        # A source may choose to fill these in for improved bounds computation.
        # See bounds.py for details
        self.mc_reservoir = pd.DataFrame()
        self.prior_PDFs_LB = tuple(dict())
        self.prior_PDFs_UB = tuple(dict())

        super().__init__(*args, **kwargs)

    def domain(self, x, data_tensor=None):
        """Return (n_events, n_x) matrix containing all
        possible integer values of x for each event.

        If x is a final dimension (e.g. s1, s2), we return an (n_events, 1)
        tensor with observed values -- NOT a (n_events,) array!
        """
        if x in self.final_dimensions:
            return self._fetch(x, data_tensor=data_tensor)[:, o]

        # Cover the bounds range in integer steps not necessarily of 1
        left_bound = self._fetch(x + '_min', data_tensor=data_tensor)[:, o]
        steps = self._fetch(x + '_steps', data_tensor=data_tensor)[:, o]
        x_range = tf.range(tf.reduce_max(self._fetch(x + '_dimsizes', data_tensor=data_tensor))) * steps
        return left_bound + x_range

    def cross_domains(self, x, y, data_tensor):
        """Return (x, y) two-tuple of (n_events, n_x, n_y) tensors
        containing possible integer values of x and y, respectively.
        """
        x_domain = self.domain(x, data_tensor)
        y_domain = self.domain(y, data_tensor)
        result_x = tf.repeat(x_domain[:, :, o], tf.shape(y_domain)[1], axis=2)
        result_y = tf.repeat(y_domain[:, o, :], tf.shape(x_domain)[1], axis=1)
        return result_x, result_y

    def extra_needed_columns(self):
        cols = []
        for dim in (self.inner_dimensions + self.bonus_dimensions):
            # Track domain bounds and stepping information
            cols += [dim + '_' + x for x in ('min', 'steps', 'dimsizes')]
        for dim in self.final_dimensions:
            # Need the dimension sizes (though they are always 1...)
            # and the observed values themselves
            cols += [dim, dim + '_dimsizes']
        return cols + super().extra_needed_columns()

    def _calculate_dimsizes(self):
        self.dimsizes = dict()
        d = self.data

        for dim in self.inner_dimensions:
            # Minimum and maximum value to compute. This is a per-event quantity,
            # so these are arrays
            ma = d[dim + '_max'].to_numpy()
            mi = d[dim + '_min'].to_numpy()
            # Number of integer steps in the dimension, including endpoints
            n = ma - mi + 1

            if dim in self.no_step_dimensions:
                cap = np.inf
            else:
                cap = self.max_dim_sizes.get(dim, self.default_max_dim_size)

            if dim in self.non_integer_dimensions:
                # Step size is non-integer
                self.dimsizes[dim] = self.max_dim_sizes[dim]
                step_size = (ma - mi) / (cap - 1)
            else:
                # Step size is an integer multiple of 1
                # Step size = (original n.of intervals) / (new n.of intervals)
                # E.g. [1,2,3,4,5] => [1,3,5] means step size (5-1)/(3-1) = 2
                self.dimsizes[dim] = n.clip(None, cap)
                step_size = np.ceil((n - 1) / (self.dimsizes[dim] - 1))

            # Store the step size in the dataframe
            d[dim + '_steps'] = step_size

        for dim in self.final_dimensions:
            self.dimsizes[dim] = np.ones(len(d))

        # Calculate all custom dimsizes
        self.calculate_dimsizes_special()

        # Store the dimsizes in the dataframe
        for dim in (self.inner_dimensions + self.bonus_dimensions + self.final_dimensions):
            d[dim + "_dimsizes"] = self.dimsizes[dim]

    def calculate_dimsizes_special(self):
        pass

    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        # Obtain any desired hidden variable priors, in case we want
        # to improve the bounds estimation for any hidden variables.
        #
        # Requires populating source.mc_reservoir during the source's annotate().
        # The source should also set prior_dimensions, which is a list of
        # [(prior_dims), (filter_dims)]. For each batch of events, the extremal
        # bounds values from annotate() for (filter_dims) are used to filter the
        # MC reservoir to obtain bounds on (prior_dims).
        if self.mc_reservoir.empty:
            return

        for prior_dims, filter_dims in self.prior_dimensions:
            prior_data_columns = [
                self.mc_reservoir.columns.get_loc(dim)
                for dim in prior_dims
            ]
            filter_data_columns = [
                self.mc_reservoir.columns.get_loc(dim)
                for dim in filter_dims
            ]

            for batch in range(self.n_batches):
                start, stop = batch * self.batch_size, (batch + 1) * self.batch_size
                df_batch = self.data[start:stop]

                filter_dims_min = [
                    min(df_batch[dim + '_min'])
                    for dim in filter_dims
                ]
                filter_dims_max = [
                    max(df_batch[dim + '_max'])
                    for dim in filter_dims
                ]

                fd.bounds.get_priors(self, self.mc_reservoir.values, prior_dims,
                                     prior_data_columns, filter_data_columns,
                                     filter_dims_min, filter_dims_max)
