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
class SourceBase:
    """Base class of Source"""

    n_batches = None
    n_padding = None
    trace_difrate = True

    @classmethod
    def find_defaults(cls):
        """Discover which functions need which arguments / dimensions
        Discover possible parameters.
        Returns f_dims, f_params and defaults.

        Overwritten by Source, SourceBase has no default f_dims,
        f_params or defaults.
        """
        return dict(), dict(), dict()

    def simulate(self, n_events, fix_truth=None, **params):
        """Simulate n events.

        Sources should override this method with their own implentation.
        This method simply returns an empty DataFrame essentially skipping
        the simulation of this source.
        This ensures LogLikelihood can call simulate on all sources.
        See issue #47
        """
        print(f"{self.__class__.__name__} has no simulate method, skipping")
        return pd.DataFrame()

    def _init_padding(self, batch_size, _skip_tf_init):
        # Annotate requests n_events, currently no padding
        self.n_padding = 0
        self.n_events = len(self.data)

        if hasattr(self, 'batch_size'):
            # We have self.batch_size so the Source has been initialized before
            # meaning we cannot change the batch_size anymore, check that we
            # are not trying to change it by accident
            assert batch_size is None
        else:
            if batch_size is None or batch_size > self.n_events or _skip_tf_init:
                batch_size = self.n_events
            self.batch_size = max(1, batch_size)

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

        # Add i_batch column to data for use with precomputed model functions
        self.data['i_batch'] = np.repeat(np.arange(self.n_batches),
                                         self.batch_size)[:len(self.data)]


@export
class ColumnSource(SourceBase):
    """Source with a fixed mu (specified as self.mu)
     and differential rate specified by a column in the data (self.column)
    """
    column = "Rename_me!"
    mu = 42.

    trace_difrate = False

    def __init__(self,
                 data=None,
                 batch_size=10,
                 max_sigma=3,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 fit_params=None,
                 **params):
        """

        :param data:
        :param batch_size: used
        :param max_sigma:
        :param data_is_annotated:
        :param _skip_tf_init:
        :param _skip_bounds_computation:
        :param fit_params: List of parameters to fit
        :param params: New defaults
        """
        self.defaults = dict()
        self.data = data
        if data is None:
            # We're calling the source without data. Set the batch_size here
            # since we can't pass it to set_data later
            self.batch_size = batch_size
        else:
            self._init_padding(batch_size, _skip_tf_init)
            self.data_tensor = fd.np_to_tf(self.data[self.column])
            self.data_tensor = tf.reshape(self.data_tensor,
                                          (-1, self.batch_size, 1))

    def differential_rate(self, data_tensor, **params):
        return data_tensor[:, 0]

    @classmethod
    def mu_function(cls,
                    interpolation_method='star',
                    n_trials=int(1e5),
                    **params):
        """Return function that maps params -> expected number of events
        Parameters must be specified as kwarg=(start, stop, n_anchors)
        """
        return lambda **kwargs: cls.mu


@export
class Source(SourceBase):
    data_methods = tuple()
    special_data_methods = tuple()
    inner_dimensions = tuple()
    extra_needed_columns = tuple()

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
                if pname == 'i_batch':
                    # This function uses precomputed data
                    f_dims[fname].append(pname)
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
        # Discover which functions need which arguments / dimensions
        # Discover possible parameters.
        self.f_dims, self.f_params, self.defaults = self.find_defaults()

        self.max_sigma = max_sigma

        self.set_defaults(**params)

        if fit_params is None:
            fit_params = list(self.defaults.keys())
        self.fit_params = [tf.constant(x) for x in fit_params
                           if x in self.defaults]

        if len(self.defaults):
            self.param_id = tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(tf.constant(list(self.defaults.keys())),
                                                    tf.range(len(self.defaults),
                                                             dtype=tf.dtypes.int64)),
                num_oov_buckets=1,
                lookup_key_dtype=tf.dtypes.string)
            # Indices of params we actually want to fit; we have to differentiate wrt these
            self.fit_param_indices = tuple([
                self.param_id.lookup(param_name)
                for param_name in self.fit_params])

        if data is None:
            # We're calling the source without data. Set the batch_size here
            # since we can't pass it to set_data later
            self.batch_size = batch_size
        else:
            self.set_data(data,
                          batch_size=batch_size,
                          data_is_annotated=data_is_annotated,
                          _skip_tf_init=_skip_tf_init,
                          _skip_bounds_computation=_skip_bounds_computation)

    def set_defaults(self, **params):
        for k, v in params.items():
            if k in self.defaults:
                self.defaults[k] = tf.convert_to_tensor(
                    v, dtype=fd.float_type())

    def set_data(self,
                 data,
                 batch_size=None,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 **params):
        self.data = data
        del data

        self.set_defaults(**params)

        self._init_padding(batch_size, _skip_tf_init)

        if not data_is_annotated:
            self.add_extra_columns(self.data)
            self._annotate(_skip_bounds_computation=_skip_bounds_computation)

        if not _skip_tf_init:
            self._populate_tensor_cache()
            self._calculate_dimsizes()

            self.trace_differential_rate()

    def _populate_tensor_cache(self):
        # Cache only float and int cols
        cols_to_cache = [x for x in self.data.columns
                         if fd.is_numpy_number(self.data[x])]

        self.name_id = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(cols_to_cache),
                                                tf.range(len(cols_to_cache),
                                                         dtype=tf.dtypes.int64)),
            num_oov_buckets=1,
            lookup_key_dtype=tf.dtypes.string)

        # Create one big data tensor (n_batches, events_per_batch, n_cols)
        # TODO: make a list
        self.data_tensor = tf.constant(self.data[cols_to_cache].values,
                                       dtype=fd.float_type())
        self.data_tensor = tf.reshape(self.data_tensor, [self.n_batches,
                                                         -1,
                                                         len(cols_to_cache)])

    def _calculate_dimsizes(self):
        self.dimsizes = dict()
        for var in self.inner_dimensions:
            ma = self._fetch(var + '_max')
            mi = self._fetch(var + '_min')
            self.dimsizes[var] = int(tf.reduce_max(ma - mi + 1).numpy())

    @contextmanager
    def _set_temporarily(self, data=None, **kwargs):
        """Set data and/or defaults temporarily"""
        old_defaults = self.defaults
        if data is None:
            self.set_defaults(**kwargs)
        else:
            old_data = self.data
            self.set_data(data, **kwargs, _skip_tf_init=True)
        try:
            yield
        finally:
            self.defaults = old_defaults
            if data is not None:
                self.data = old_data

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
        if data_tensor is None:
            # We're inside annotate, just return the column
            return fd.np_to_tf(self.data[x].values)

        col_id = tf.dtypes.cast(self.name_id.lookup(tf.constant(x)),
                                fd.int_type())
        # if i_batch is None:
        #     return tf.reshape(self.data_tensor[:,:,col_id], [-1])
        # else:
        return data_tensor[:, col_id]

    def _fetch_param(self, param, ptensor):
        if ptensor is None:
            return self.defaults[param]
        id = tf.dtypes.cast(self.param_id.lookup(tf.constant(param)),
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
        # TODO: make a clean way to keep track of i_batch or have it as input
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
        y = np.concatenate([
            fd.tf_to_np(self.differential_rate(
                data_tensor=self.data_tensor[i_batch],
                **params))
            for i_batch in progress(range(self.n_batches))])
        return y[:self.n_events]

    def trace_differential_rate(self):
        input_signature = (
            tf.TensorSpec(shape=self.data_tensor.shape[1:],
                          dtype=fd.float_type()),
            tf.TensorSpec(shape=[len(self.defaults)],
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
        for pname, pspace_spec in tqdm(param_specs.items(),
                                       desc="Estimating mus"):
            pspaces[pname] = tf.linspace(*pspace_spec)
            mus[pname] = tf.convert_to_tensor(
                 [self.estimate_mu(**{pname: x}, n_trials=n_trials)
                  for x in np.linspace(*pspace_spec)],
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

    def mu_before_efficiencies(self, **params):
        """Return mean expected number of events BEFORE efficiencies/response
        using data for the evaluation of the energy spectra
        """
        raise NotImplementedError

    def estimate_mu(self, n_trials=int(1e5), **params):
        """Return estimate of total expected number of events
        :param n_trials: Number of events to simulate for estimate
        """
        d_simulated = self.simulate(n_trials, **params)
        return (self.mu_before_efficiencies(**params)
                * len(d_simulated) / n_trials)

    ##
    # Functions probably want to override
    ##
    def energy_spectrum(self, **params):
        raise NotImplementedError

    def _differential_rate(self, data_tensor, ptensor):
        raise NotImplementedError

    def _annotate(self, _skip_bounds_computation=False):
        """Add columns needed in inference to self.data
        :param _skip_bounds_computation: Do not compute min/max bounds
        TODO: explain why useful, see simulator
        """
        pass

    def add_extra_columns(self, data):
        """Add additional columns to data

        You must add any columns from data you use here to
        extra_needed.columns.

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
        raise NotImplementedError
