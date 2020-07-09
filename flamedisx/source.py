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

    data_methods = tuple()
    special_data_methods = tuple()
    inner_dimensions = tuple()

    # List all columns that are manually _fetch ed here
    # These will be added to the data_tensor even when the model function
    # inspection will not find them.
    def extra_needed_columns(self):
        return []

    # List all columns for which sneaky hacks are used to intercept _fetch here
    # These will not be added to the data tensor even when the model function
    # inspection does find them.
    def ignore_columns(self):
        return []

    data = None

    ##
    # Initialization and helpers
    ##

    @classmethod
    def find_defaults(cls):
        """Discover which functions need which arguments / dimensions
        Discover possible parameters.
        Returns f_dims, f_params and defaults.
        """
        f_dims = {x: [] for x in cls.data_methods}
        f_params = {x: [] for x in cls.data_methods}
        defaults = dict()
        for fname in cls.data_methods:
            f = getattr(cls, fname)
            if not callable(f):
                # Constant
                continue
            seen_special = False
            for pname, p in inspect.signature(f).parameters.items():
                if pname == 'self':
                    continue
                if p.default is inspect.Parameter.empty:
                    if fname in cls.special_data_methods and not seen_special:
                        seen_special = True
                    else:
                        # It's an observable dimension
                        f_dims[fname].append(pname)
                else:
                    # It's a parameter that can be fitted
                    f_params[fname].append(pname)
                    if (pname in defaults and p.default != defaults[pname]):
                        raise ValueError(f"Inconsistent defaults for {pname}")
                    defaults[pname] = tf.convert_to_tensor(
                        p.default, dtype=fd.float_type())
        return f_dims, f_params, defaults

    def __init__(self,
                 data=None,
                 batch_size=10,
                 max_sigma=3,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 fit_params=None,
                 **params):
        """Initialize a flamedisx source

        :param data: Dataframe with events to use in the inference
        :param batch_size: Number of events / tensorflow batch
        :param max_sigma: Hint for hidden variable bounds computation
        :param data_is_annotated: If True, skip annotation
        :param _skip_tf_init: If True, skip tensorflow cache initialization
        :param _skip_bounds_computation: If True, skip bounds compuation
        :param fit_params: List of parameters to fit
        :param params: New defaults to use
        """
        self.max_sigma = max_sigma

        # Discover which functions need which arguments / dimensions
        # Discover possible parameters.
        self.f_dims, self.f_params, self.defaults = self.find_defaults()

        # Which columns are needed from data?
        ctc = list(set(sum(self.f_dims.values(), [])))
        ctc += self.extra_needed_columns()
        ctc += [x + '_min' for x in self.inner_dimensions]  # Needed in domain
        ctc = [x for x in ctc if x not in self.ignore_columns()]
        self.cols_to_cache = ctc

        # Check for duplicate columns and give error
        _seen = set()
        for x in self.cols_to_cache:
            if x in _seen:
                raise RuntimeError(
                    f"Column {x} requested twice for data_tensor!")
            _seen.add(x)

        self.name_id = fd.index_lookup_dict(ctc)

        self.set_defaults(**params)

        if fit_params is None:
            fit_params = list(self.defaults.keys())
        self.fit_params = [x for x in fit_params
                           if x in self.defaults]

        self.param_id = fd.index_lookup_dict(self.defaults.keys())
        # Indices of params we actually want to fit; we have to differentiate wrt these
        self.fit_param_indices = tuple([
            self.param_id[param_name]
            for param_name in self.fit_params])

        if data is None:
            # We're calling the source without data. Set the batch_size here
            # since we can't pass it to set_data later
            self.batch_size = batch_size
        else:
            self.batch_size = min(batch_size, len(data))
            self.set_data(data,
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
            self._annotate(_skip_bounds_computation=_skip_bounds_computation)

        if not _skip_tf_init:
            self._check_data()
            self._populate_tensor_cache()
            self._calculate_dimsizes()

    def _check_data(self):
        """Do any final checks on the self.data dataframe,
        before passing it on to the tensorflow layer.
        """
        for column in self.cols_to_cache:
            if column not in self.data.columns:
                raise ValueError(f"Data lacks required column {column}; "
                                 f"did annotation happen correctly?")

    def _populate_tensor_cache(self):

        # Create one big data tensor (n_batches, events_per_batch, n_cols)
        # TODO: make a list
        ctc = self.cols_to_cache
        self.data_tensor = tf.constant(self.data[ctc].values,
                                       dtype=fd.float_type())
        self.data_tensor = tf.reshape(self.data_tensor, [self.n_batches,
                                                         -1,
                                                         len(ctc)])

    def _calculate_dimsizes(self):
        self.dimsizes = dict()
        for var in self.inner_dimensions:
            ma = self._fetch(var + '_max')
            mi = self._fetch(var + '_min')
            self.dimsizes[var] = int(tf.reduce_max(ma - mi + 1).numpy())

    @contextmanager
    def _set_temporarily(self, data, **kwargs):
        """Set data and/or defaults temporarily, without affecting the
        data tensor state"""
        if data is None:
            raise ValueError("No point in setting data = None temporarily")
        old_defaults = self.defaults
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

    def annotate_data(self, data, _skip_bounds_computation=False, **params):
        """Add columns to data with inference information"""
        with self._set_temporarily(data, **params):
            self._annotate(_skip_bounds_computation=_skip_bounds_computation)
            return self.data

    ##
    # Data fetching / calculation
    ##

    def _fetch(self, x, data_tensor=None):
        """Return a tensor column from the original dataframe (self.data)
        :param x: column name
        :param data_tensor: Data tensor, columns as in self.name_id
        """
        if x in self.ignore_columns():
            raise RuntimeError(
                "Attempt to fetch %s, which is in ignore_columns" % x)
        if data_tensor is None:
            # We're inside annotate, just return the column
            return fd.np_to_tf(self.data[x].values)

        return data_tensor[:, self.name_id[x]]

    def _fetch_param(self, param, ptensor):
        if ptensor is None:
            return self.defaults[param]
        id = tf.dtypes.cast(self.param_id[param],
                            dtype=fd.int_type())
        return ptensor[id]

    # TODO: make data_tensor and ptensor keyword-only arguments
    # after https://github.com/tensorflow/tensorflow/issues/28725
    def gimme(self, fname, data_tensor=None, ptensor=None, bonus_arg=None, numpy_out=False):
        """Evaluate the model function fname with all required arguments

        :param fname: Name of the model function to compute
        :param bonus_arg: If fname takes a bonus argument, the data for it
        :param numpy_out: If True, return (tuple of) numpy arrays,
        otherwise (tuple of) tensors.
        :param data_tensor: Data tensor, columns as self.name_id
        If not given, use self.data (used in annotate)
        :param ptensor: Parameter tensor, columns as self.param_id
        If not give, use defaults dictionary (used in annotate)
        Before using gimme, you must use set_data to
        populate the internal caches.
        """
        assert (bonus_arg is not None) == (fname in self.special_data_methods)

        if data_tensor is None:
            # We're in an annotate
            assert hasattr(self, 'data'), "You must set data first"
        else:
            # We're computing
            if not hasattr(self, 'name_id'):
                raise ValueError(
                    "You must set_data first (and populate the tensor cache)")

        f = getattr(self, fname)

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

    ##
    # Differential rate computation
    ##

    # TODO: remove duplication for batch loop? Also in inference
    def batched_differential_rate(self, progress=True, **params):
        progress = (lambda x: x) if not progress else tqdm
        y = []
        for i_batch in progress(range(self.n_batches)):
            q = self.data_tensor[i_batch]
            y.append(fd.tf_to_np(self.differential_rate(data_tensor=q,
                                                        **params)))

        return np.concatenate(y)[:self.n_events]

    def _batch_data_tensor_shape(self):
        return [self.batch_size, len(self.name_id)]

    def trace_differential_rate(self):
        input_signature = (
            tf.TensorSpec(shape=self._batch_data_tensor_shape(),
                          dtype=fd.float_type()),
            tf.TensorSpec(shape=[len(self.param_id)],
                          dtype=fd.float_type()))
        self._differential_rate_tf = tf.function(
            self._differential_rate,
            input_signature=input_signature)

    # TODO: remove duplication?
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
        """Return (n_events, |possible x values|) matrix containing all possible integer
        values of x for each event"""
        result1 = tf.cast(tf.range(self.dimsizes[x]),
                          dtype=fd.float_type())[o, :]
        result2 = self._fetch(x + '_min', data_tensor=data_tensor)[:, o]
        return result1 + result2

    def cross_domains(self, x, y, data_tensor):
        """Return (x, y) two-tuple of (n_events, |x|, |y|) tensors
        containing possible integer values of x and y, respectively.
        """
        # TODO: somehow mask unnecessary elements and save computation time
        x_size = self.dimsizes[x]
        y_size = self.dimsizes[y]
        # Change to tf.repeat once it's in the api
        result_x = fd.repeat(self.domain(x, data_tensor)[:, :, o], y_size, axis=2)
        result_y = fd.repeat(self.domain(y, data_tensor)[:, o, :], x_size, axis=1)
        return result_x, result_y


    ##
    # Simulation methods and helpers
    ##

    def simulate(self, n_events, fix_truth=None, **params):
        """Simulate n events.

        Will not return events lost due to selection/detection efficiencies
        """
        # Draw random "deep truth" variables (energy, position)
        sim_data = self.random_truth(n_events, fix_truth=fix_truth, **params)

        with self._set_temporarily(sim_data, _skip_bounds_computation=True,
                                   **params):
            # Do the forward simulation of the detector response
            d = self._simulate_response()
            # Now that we have s1 and s2 values, we can do the full annotate,
            # populating columns like e_vis, photon_produced_mle, etc.
            # Set the data, annotate, compute bounds, skip TF
            self.set_data(d, _skip_tf_init=True)
            return self.data

    ##
    # Mu estimation
    ##

    def mu_function(self,
                    interpolation_method='star',
                    n_trials=int(1e5),
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
        for pname, (start, stop, n) in tqdm(param_specs.items(),
                                       desc="Estimating mus"):
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

    def _annotate(self, _skip_bounds_computation=False):
        """Add columns needed in inference to self.data
        :param _skip_bounds_computation: Do not compute min/max bounds
        TODO: explain why useful, see simulator
        """
        pass

    def add_extra_columns(self, data):
        """Add additional columns to data

        :param data: pandas DataFrame
        """
        pass

    def random_truth(self, n_events, fix_truth=None, **params):
        """Draw random "deep truth" variables (energy, position) """
        assert isinstance(n_events, int), \
            f"n_events must be an int, not {type(n_events)}"
        return pd.DataFrame({'energy': np.ones(n_events)})

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
