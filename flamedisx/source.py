import inspect

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm

import flamedisx as fd
export, __all__ = fd.exporter()

quanta_types = 'photon', 'electron'
signal_name = dict(photon='s1', electron='s2')

# Data methods that take an additional positional argument
special_data_methods = [
    'p_electron',
    'p_electron_fluctuation',
    'electron_acceptance',
    'photon_acceptance',
    's1_acceptance',
    's2_acceptance',
    'penning_quenching_eff'
]

data_methods = (
    special_data_methods
    + ['energy_spectrum', 'work', 'double_pe_fraction'])
hidden_vars_per_quanta = 'detection_eff gain_mean gain_std'.split()
for _qn in quanta_types:
    data_methods += [_qn + '_' + x for x in hidden_vars_per_quanta]

o = tf.newaxis


@export
class SourceBase:
    """Base class of Source"""

    def _init_padding(self, batch_size, _skip_tf_init):
        # Annotate requests n_events, currently no padding
        self.n_padding = 0
        self.n_events = len(self.data)

        if batch_size is None or batch_size > self.n_events or _skip_tf_init:
            batch_size = self.n_events

        self.batch_size = batch_size
        self.n_batches = np.ceil(
            self.n_events / self.batch_size).astype(np.int)

        if not _skip_tf_init:
            # Extend dataframe with events to nearest batch_size multiple
            # We're using actual events for padding, since using zeros or
            # nans caused problems with gradient calculation
            # padded events are clipped when summing likelihood terms
            self.n_padding = self.n_batches * batch_size - len(self.data)
            if self.n_padding > 0:
                df_pad = self.data.iloc[:self.n_padding, :]
                self.data = pd.concat([self.data, df_pad], ignore_index=True)


@export
class ColumnSource(SourceBase):
    column = "Rename_me!"
    mu = 42.

    def __init__(self,
                 data,
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
        self.data = data
        self.batch_size = batch_size

        self._init_padding(batch_size, _skip_tf_init)

        self.data_tensor = fd.np_to_tf(self.data[self.column])
        self.data_tensor = tf.reshape(self.data_tensor, (self.batch_size, -1, 1))

    def differential_rate(self, data_tensor, **params):
        return data_tensor[:,0]

    @classmethod
    def mu_interpolator(cls,
                        data,
                        interpolation_method='star',
                        n_trials=int(1e5),
                        **params):
        """Return interpolator for number of expected events
        Parameters must be specified as kwarg=(start, stop, n_anchors)
        """
        return lambda *args, **kwargs: cls.mu

@export
class Source(SourceBase):
    data_methods = tuple(data_methods)
    special_data_methods = tuple(special_data_methods)

    # Whether or not to simulate overdispersion in electron/photon split
    # (e.g. due to non-binomial recombination fluctuation)
    do_pel_fluct = True

    # tuple with columns needed from data to run add_extra_columns
    # I guess we don't really need x y z by default, but they are just so nice
    # we should keep them around regardless.
    extra_needed_columns = tuple(['x', 'y', 'z', 'r', 'theta'])

    data: pd.DataFrame

    def __init__(self,
                 data,
                 batch_size=10,
                 max_sigma=3,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 fit_params=None,
                 **params):
        """

        :param data:
        :param batch_size:
        :param max_sigma:
        :param data_is_annotated:
        :param _skip_tf_init:
        :param _skip_bounds_computation:
        :param fit_params: List of parameters to fit
        :param params: New defaults
        """
        self.max_sigma = max_sigma
        self.data = data
        del data

        # Discover which functions need which arguments / dimensions
        # Discover possible parameters
        self.f_dims = {x: [] for x in self.data_methods}
        self.f_params = {x: [] for x in self.data_methods}
        self.defaults = dict()
        for fname in self.data_methods:
            f = getattr(self, fname)
            if not callable(f):
                # Constant
                continue
            for i, (pname, p) in enumerate(
                    inspect.signature(f).parameters.items()):
                if p.default is inspect.Parameter.empty:
                    if not (fname in self.special_data_methods and i == 0):
                        # It's an observable dimension
                        self.f_dims[fname].append(pname)
                else:
                    # It's a parameter that can be fitted
                    self.f_params[fname].append(pname)
                    if (pname in self.defaults
                            and p.default != self.defaults[pname]):
                        raise ValueError(f"Inconsistent defaults for {pname}")
                    self.defaults[pname] = tf.convert_to_tensor(
                        p.default, dtype=fd.float_type())
        for k, v in params.items():
            if k in self.defaults:
                self.defaults[k] = tf.convert_to_tensor(
                    v, dtype=fd.float_type())
            else:
                raise ValueError(f"Key {k} not in defaults")

        if fit_params is None:
            fit_params = list(self.defaults.keys())
        self.fit_params = [tf.constant(x) for x in fit_params
                           if x in self.defaults]

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

        self._init_padding(batch_size, _skip_tf_init)

        if not data_is_annotated:
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
                                                         dtype=tf.dtypes.int64)
                                                ),
            num_oov_buckets=1,
            lookup_key_dtype=tf.dtypes.string,
        )

        # Create one big data tensor (n_batches, events_per_batch, n_cols)
        # TODO: make a list
        self.data_tensor = tf.constant(self.data[cols_to_cache].values,
                                       dtype=fd.float_type())
        self.data_tensor = tf.reshape(self.data_tensor, [self.n_batches,
                                                         -1,
                                                         len(cols_to_cache)])

    def _calculate_dimsizes(self):
        self.dimsizes = dict()
        for var in ['nq',
                  'photon_detected',
                  'electron_detected',
                  'photon_produced',
                  'electron_produced']:
            ma = self._fetch(var + '_max')
            mi = self._fetch(var + '_min')
            self.dimsizes[var] = int(tf.reduce_max(ma - mi + 1).numpy())

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
    def gimme(self, fname, data_tensor, ptensor, bonus_arg=None, numpy_out=False):
        """Evaluate the model function fname with all required arguments

        :param fname: Name of the model function to compute
        :param bonus_arg: If fname takes a bonus argument, the data for it
        :param numpy_out: If True, return (tuple of) numpy arrays,
        otherwise (tuple of) tensors.
        :param data_tensor: Data tensor, columns as self.name_id
        :param ptensor: Parameter tensor, columns as self.param_id

        Before using gimme, you must use set_data to
        populate the internal caches.
        """
        # TODO: make a clean way to keep track of i_batch or have it as input
        assert (bonus_arg is not None) == (fname in self.special_data_methods)

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

    def _q_det_clip_range(self, qn):
        return (self.min_s1_photons_detected if qn == 'photon'
                else self.min_s2_electrons_detected,
                None)

    @classmethod
    def annotate_data(cls, data, **params):
        """Add columns to data with inference information"""
        return cls(data, _skip_tf_init=True, **params)

    def _annotate(self, _skip_bounds_computation=False):
        """Annotate self.data with columns needed for inference.
        """
        d = self.data
        self.add_extra_columns(d)

        # Annotate data with eff, mean, sigma
        # according to the nominal model
        for qn in quanta_types:
            for parname in hidden_vars_per_quanta:
                fname = qn + '_' + parname
                try:
                    d[fname] = self.gimme(fname, data_tensor=None, ptensor=None, numpy_out=True)
                except Exception:
                    print(fname)
                    raise
        d['double_pe_fraction'] = self.gimme('double_pe_fraction',
                                             data_tensor=None, ptensor=None,
                                             numpy_out=True)

        if _skip_bounds_computation:
            return

        # Find likely number of detected quanta
        # Don't round them yet, we'll do that after estimating quantities
        # derived from this
        obs = dict(photon=d['s1'], electron=d['s2'])
        for qn in quanta_types:
            n_det_mle = (obs[qn] / d[qn + '_gain_mean'])
            if qn == 'photon':
                n_det_mle /= (1 + d['double_pe_fraction'])
            d[qn + '_detected_mle'] = \
                n_det_mle.clip(*self._q_det_clip_range(qn))

        # The Penning quenching depends on the number of produced
        # photons.... But we don't have that yet.
        # Thus, "rewrite" penning eff vs detected photons
        # using interpolation
        # TODO: this will fail when someone gives penning quenching some
        # data-dependent args
        _nprod_temp = np.logspace(-1., 8., 1000)
        peff = self.gimme('penning_quenching_eff',
                          data_tensor=None, ptensor=None,
                          bonus_arg=_nprod_temp,
                          numpy_out=True)
        d['penning_quenching_eff_mle'] = np.interp(
            d['photon_detected_mle'] / d['photon_detection_eff'],
            _nprod_temp * peff,
            peff)

        # Approximate energy reconstruction (visible energy only)
        # TODO: how to do CES estimate if someone wants a variable W?
        work = self.gimme('work',
                          data_tensor=None, ptensor=None,
                          numpy_out=True)
        d['e_charge_vis'] = work * (
            d['electron_detected_mle'] / d['electron_detection_eff'])
        d['e_light_vis'] = work * (
            d['photon_detected_mle'] / (
                d['photon_detection_eff'] / d['penning_quenching_eff_mle']))
        d['e_vis'] = d['e_charge_vis'] + d['e_light_vis']
        d['nq_vis_mle'] = d['e_vis'] / work
        d['fel_mle'] = self.gimme('p_electron',
                                  data_tensor=None, ptensor=None,
                                  bonus_arg=d['nq_vis_mle'].values,
                                  numpy_out=True)

        # .round().astype(np.int).clip(
        #     *self._q_det_clip_range(qn))

        # .round().astype(np.int).clip(
        #     *self._q_det_clip_range(qn))


        # Find plausble ranges for detected and observed quanta
        # based on the observed S1 and S2 sizes
        # (we could also derive ranges assuming the CES reconstruction,
        #  but these won't work well for outliers along one of the dimensions)
        # TODO: Meh, think about this, considering also computation cost
        # / space width
        for qn in quanta_types:
            # We need the copy, otherwise the in-place modification below
            # will have the side effect of messing up the dataframe column!
            eff = d[qn + '_detection_eff'].values.copy()
            if qn == 'photon':
                eff *= d['penning_quenching_eff_mle'].values

            n_prod_mle = d[qn + '_produced_mle'] = (
                    d[qn + '_detected_mle'] / eff).astype(np.int)

            # Prepare for bounds computation
            n = d[qn + '_detected_mle'].values
            m = d[qn + '_gain_mean'].values
            s = d[qn + '_gain_std'].values
            if qn == 'photon':
                _, scale = self.dpe_mean_std(n, d['double_pe_fraction'],
                                             m, s)
                scale = scale.values
            else:
                scale = n ** 0.5 * s / m

            for bound, sign in (('min', -1), ('max', +1)):
                # For detected quanta the MLE is quite accurate
                # (since fluctuations are tiny)
                # so let's just use the relative error on the MLE
                d[qn + '_detected_' + bound] = stats.norm.ppf(
                    stats.norm.cdf(sign * self.max_sigma),
                    loc=n,
                    scale=scale,
                ).round().clip(*self._q_det_clip_range(qn)).astype(np.int)

                # For produced quanta, it is trickier, since the number
                # of detected quanta is also uncertain.
                # TODO: where did this derivation come from again?
                # TODO: maybe do a second bound based on CES
                q = 1 / eff
                d[qn + '_produced_' + bound] = stats.norm.ppf(
                    stats.norm.cdf(sign * self.max_sigma),
                    loc=n_prod_mle,
                    scale=(q + (q**2 + 4 * n_prod_mle * q)**0.5)/2
                ).round().clip(*self._q_det_clip_range(qn)).astype(np.int)

            # Finally, round the detected MLEs
            d[qn + '_detected_mle'] = \
                d[qn + '_detected_mle'].values.round().astype(np.int)

        # Bounds on total visible quanta
        d['nq_min'] = d['photon_produced_min'] + d['electron_produced_min']
        d['nq_max'] = d['photon_produced_max'] + d['electron_produced_max']

    @staticmethod
    def add_extra_columns(data):
        """Add additional columns to data

        You must add any columns from data you use here to
        extra_needed.columns.

        :param data: pandas DataFrame
        """
        pass

    # TODO: remove duplication for batch loop? Also in inference
    def batched_differential_rate(self, progress=True, **params):
        progress = (lambda x: x) if not progress else tqdm
        y = np.concatenate([
            fd.tf_to_np(self.differential_rate(data_tensor=self.data_tensor[i_batch],
                                               **params))
            for i_batch in progress(range(self.n_batches))])
        return y[:self.n_events]

    def trace_differential_rate(self):

        input_signature=(tf.TensorSpec(shape=self.data_tensor.shape[1:], dtype=fd.float_type()),
                         tf.TensorSpec(shape=[len(self.defaults)], dtype=fd.float_type()),)
        self._differential_rate_tf = tf.function(self._differential_rate,
                                                 input_signature=input_signature)

    # TODO: remove duplication?
    def differential_rate(self, data_tensor=None, autograph=True, **kwargs):
        ptensor = self.ptensor_from_kwargs(**kwargs)

        if autograph:
            return self._differential_rate_tf(data_tensor=data_tensor, ptensor=ptensor)
        else:
            return self._differential_rate(data_tensor=data_tensor, ptensor=ptensor)

    def ptensor_from_kwargs(self, **kwargs):
        return tf.convert_to_tensor([kwargs.get(k, self.defaults[k])
                                     for k in self.defaults])

    def _differential_rate(self, data_tensor, ptensor):
        # (n_events, |photons_produced|, |electrons_produced|)
        y = self.rate_nphnel(data_tensor, ptensor)

        p_ph = self.detection_p('photon', data_tensor, ptensor)
        p_el = self.detection_p('electron', data_tensor, ptensor)
        d_ph = self.detector_response('photon', data_tensor, ptensor)
        d_el = self.detector_response('electron', data_tensor, ptensor)

        # Rearrange dimensions so we can do a single matrix mult
        p_el = tf.transpose(p_el, (0, 2, 1))
        d_ph = d_ph[:, o, :]
        d_el = d_el[:, :, o]
        y = d_ph @ p_ph @ y @ p_el @ d_el
        return tf.reshape(y, [-1])

    def rate_nq(self, nq_1d, data_tensor, ptensor):
        """Return differential rate at given number of produced quanta
        differs for ER and NR"""
        # TODO: this implementation echoes that for NR, but I feel there
        # must be a less clunky way...

        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum',
                                data_tensor=data_tensor, ptensor=ptensor)
        q_produced = tf.cast(
            tf.floor(es / self.gimme('work',
                                     data_tensor=data_tensor, ptensor=ptensor)[:, o]),
            dtype=fd.float_type())

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tf.cast(tf.equal(nq_1d[:, :, o], q_produced[:, o, :]),
                         dtype=fd.float_type())

        q = tf.reduce_sum(p_nq_e * rate_e[:, o, :], axis=2)
        return q

    def rate_nphnel(self, data_tensor, ptensor):
        """Return differential rate tensor
        (n_events, |photons_produced|, |electrons_produced|)
        """
        # Get differential rate and electron probability vs n_quanta
        # these four are (n_events, |nq|) tensors
        _nq_1d = self.domain('nq', data_tensor)
        rate_nq = self.rate_nq(_nq_1d,
                               data_tensor=data_tensor, ptensor=ptensor)
        pel = self.gimme('p_electron', bonus_arg=_nq_1d,
                         data_tensor=data_tensor, ptensor=ptensor)
        pel_fluct = self.gimme('p_electron_fluctuation', bonus_arg=_nq_1d,
                               data_tensor=data_tensor, ptensor=ptensor)

        # Create tensors with the dimensions of our fin al result
        # i.e. (n_events, |photons_produced|, |electrons_produced|),
        # containing:
        # ... numbers of photons and electrons produced:
        nph, nel = self.cross_domains('photon_produced', 'electron_produced', data_tensor)
        # ... numbers of total quanta produced
        nq = nel + nph
        # ... indices in nq arrays
        _nq_ind = nq - self._fetch('nq_min', data_tensor=data_tensor)[:, o, o]
        # ... differential rate
        rate_nq = fd.lookup_axis1(rate_nq, _nq_ind)
        # ... probability of a quantum to become an electron
        pel = fd.lookup_axis1(pel, _nq_ind)
        # ... probability fluctuation
        pel_fluct = fd.lookup_axis1(pel_fluct, _nq_ind)
        # Finally, the main computation is simple:
        pel_num = tf.where(tf.math.is_nan(pel),
                           tf.zeros_like(pel, dtype=fd.float_type()),
                           pel)
        pel_clip = tf.clip_by_value(pel_num, 1e-6, 1. - 1e-6)
        pel_fluct_clip = tf.clip_by_value(pel_fluct, 1e-6, 1.)
        if self.do_pel_fluct:
            return rate_nq * fd.beta_binom_pmf(
                nel,
                n=nq,
                p_mean=pel_clip,
                p_sigma=pel_fluct_clip)
        else:
            return rate_nq * tfp.distributions.Binomial(
                total_count=nq, probs=pel_clip).prob(nel)

    def detection_p(self, quanta_type, data_tensor, ptensor):
        """Return (n_events, |detected|, |produced|) te nsor
        encoding P(n_detected | n_produced)
        """
        n_det, n_prod = self.cross_domains(quanta_type + '_detected',
                                           quanta_type + '_produced',
                                           data_tensor)

        p = self.gimme(quanta_type + '_detection_eff',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        if quanta_type == 'photon':
            # Note *= doesn't work, p will get reshaped
            p = p * self.gimme('penning_quenching_eff', bonus_arg=n_prod,
                               data_tensor=data_tensor, ptensor=ptensor)

        result = tfp.distributions.Binomial(
                total_count=n_prod,
                probs=tf.cast(p, dtype=fd.float_type())
            ).prob(n_det)
        return result * self.gimme(quanta_type + '_acceptance', bonus_arg=n_det,
                                   data_tensor=data_tensor, ptensor=ptensor)

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

    def detector_response(self, quanta_type, data_tensor, ptensor):
        """Return (n_events, |n_detected|) probability of observing the S[1|2]
        for different number of detected quanta.
        """
        ndet = self.domain(quanta_type + '_detected', data_tensor)

        observed = self._fetch(
            signal_name[quanta_type], data_tensor=data_tensor)[:, o]

        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme(quanta_type + '_gain_mean',
                                data_tensor=data_tensor, ptensor=ptensor)[:, o]
        std_per_q = self.gimme(quanta_type + '_gain_std',
                               data_tensor=data_tensor, ptensor=ptensor)[:, o]

        if quanta_type == 'photon':
            mean, std = self.dpe_mean_std(
                ndet=ndet,
                p_dpe=self.gimme('double_pe_fraction',
                                 data_tensor=data_tensor, ptensor=ptensor)[:, o],
                mean_per_q=mean_per_q,
                std_per_q=std_per_q)
        else:
            mean = ndet * mean_per_q
            std = ndet**0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfp.distributions.Normal(
                loc=mean, scale=std + 1e-10
            ).prob(observed)

        # Add detection/selection efficiency
        result *= self.gimme(signal_name[quanta_type] + '_acceptance',
                             bonus_arg=observed,
                             data_tensor=data_tensor, ptensor=ptensor)
        return result

    @staticmethod
    def dpe_mean_std(ndet, p_dpe, mean_per_q, std_per_q):
        """Return (mean, std) of S1 signals
        :param ndet: photons detected
        :param p_dpe: double pe emission probability
        :param mean_per_q: gain mean per PE
        :param std_per_q: gain std per PE
        """
        npe_mean = ndet * (1 + p_dpe)
        mean = npe_mean * mean_per_q

        # Variance due to PMT resolution
        var = (npe_mean ** 0.5 * std_per_q)**2
        # Variance due to binomial variation in double-PE emission
        var += ndet * p_dpe * (1 - p_dpe)

        return mean, var**0.5

    ##
    # Simulation
    #
    # These are class methods. We could have implemented them as instance
    # methods, but then we'd need a set_data, keep track of state, etc.
    ##

    @classmethod
    def simulate_aux(cls, n_events):
        raise NotImplementedError

    @classmethod
    def simulate(cls, energies, data=None, **params):
        """Simulate events at energies.

        If data is given, we will draw auxiliary observables (e.g. positions)
        from it. Otherwise we will call _simulate_aux to do this.

        Will not return | energies | events lost due to
        selection/detection efficiencies
        """
        if isinstance(energies, (float, int)):
            n_to_sim = int(energies)
        else:
            n_to_sim = len(energies)

        create_aux = data is None

        if create_aux:
            data = cls.simulate_aux(n_to_sim)
            # Add fake s1, s2 necessary for set_data to succeed
            data['s1'] = 1
            data['s2'] = 100
        else:
            data = data.copy()  # In case someone passes in a slice
            # Annoying, f_dims isn't a class property...
            s = cls(data=data, _skip_tf_init=True, **params)
            # Drop dimensions we do not need / like
            data = data[list(set(sum(
                s.f_dims.values(),
                list(s.extra_needed_columns))))].copy()

        # simulate_es cannot be a class method; the energy-spectrum might
        # be position/time/other dependent.
        s = cls(data=data,
                _skip_tf_init=True, _skip_bounds_computation=True,
                **params)
        if isinstance(energies, (float, int)):
            energies = s.simulate_es(n_to_sim)

        # Create and set new dataset, with just the dimensions we need
        # (note we should NOT include s1 and s2 here, we're going to simulate
        # them)
        # Use replace if someone gave us data (e.g. a small Kr file to draw many
        # ER Bg events from), otherwise we already simulated exactly enough aux
        if not create_aux:
            d = data.sample(n=len(energies), replace=True)
        else:
            d = data
        s = cls(data=d,
                _skip_tf_init=True,
                _skip_bounds_computation=True,
                **params)
        assert 'e_vis' not in d.columns
        assert len(s.data) == len(d)

        def gimme(fname, bonus_arg=None):
            return s.gimme(fname, bonus_arg=bonus_arg, data_tensor=None, ptensor=None, numpy_out=True)

        d['energy'] = energies
        d['nq'] = s._simulate_nq(energies)
        d['p_el_mean'] = gimme('p_electron', d['nq'].values)

        if s.do_pel_fluct:
            d['p_el_fluct'] = gimme('p_electron_fluctuation', d['nq'].values)
            d['p_el_actual'] = stats.beta.rvs(
                *fd.beta_params(d['p_el_mean'], d['p_el_fluct']))
        else:
            d['p_el_fluct'] = 0.
            d['p_el_actual'] = d['p_el_mean']
        d['p_el_actual'] = np.nan_to_num(d['p_el_actual']).clip(0, 1)
        d['electron_produced'] = stats.binom.rvs(
            n=d['nq'],
            p=d['p_el_actual'])
        d['photon_produced'] = d['nq'] - d['electron_produced']
        d['electron_detected'] = stats.binom.rvs(
            n=d['electron_produced'],
            p=gimme('electron_detection_eff'))
        d['photon_detected'] = stats.binom.rvs(
            n=d['photon_produced'],
            p=(gimme('photon_detection_eff')
               * gimme('penning_quenching_eff', d['photon_produced'].values)))

        d['s2'] = stats.norm.rvs(
            loc=d['electron_detected'] * gimme('electron_gain_mean'),
            scale=d['electron_detected']**0.5 * gimme('electron_gain_std'))

        d['s1'] = stats.norm.rvs(*s.dpe_mean_std(
            ndet=d['photon_detected'],
            p_dpe=gimme('double_pe_fraction'),
            mean_per_q=gimme('photon_gain_mean'),
            std_per_q=gimme('photon_gain_std')))

        acceptance = np.ones(len(d))
        for q in quanta_types:
            acceptance *= gimme(q + '_acceptance', d[q + '_detected'].values)
            sn = signal_name[q]
            acceptance *= gimme(sn + '_acceptance', d[sn].values)
        d = d.iloc[np.random.rand(len(d)) < acceptance].copy()

        # Now that we have s1 and s2 values, we can do the full annotate,
        # populating columns like e_vis, photon_produced_mle, etc.
        cls.annotate_data(d, **params)
        assert 'e_vis' in d.columns
        return d

    def _simulate_nq(self, energies):
        raise NotImplementedError

    @classmethod
    def mu_interpolator(cls,
                        data,
                        interpolation_method='star',
                        n_trials=int(1e5),
                        **params):
        """Return interpolator for number of expected events
        Parameters must be specified as kwarg=(start, stop, n_anchors)
        """
        # TODO: is the mu also to be batched?
        if interpolation_method != 'star':
            raise NotImplementedError(
                f"mu interpolation method {interpolation_method} "
                f"not implemented")

        base_mu = tf.constant(cls.estimate_mu(data, n_trials=n_trials),
                              dtype=fd.float_type())
        pspaces = dict()    # parameter -> tf.linspace of anchors
        mus = dict()        # parameter -> tensor of mus
        for pname, pspace_spec in tqdm(params.items(),
                                       desc="Estimating mus"):
            pspaces[pname] = tf.linspace(*pspace_spec)
            mus[pname] = tf.convert_to_tensor(
                 [cls.estimate_mu(data, **{pname: x}, n_trials=n_trials)
                 for x in np.linspace(*pspace_spec)],
                dtype=fd.float_type())

        def mu_itp(**kwargs):
            mu = base_mu
            for pname, v in kwargs.items():
                mu *= tfp.math.interp_regular_1d_grid(
                    x=v,
                    x_ref_min=params[pname][0],
                    x_ref_max=params[pname][1],
                    y_ref=mus[pname]) / base_mu
            return mu

        return mu_itp

    @classmethod
    def estimate_mu(cls, data=None, n_trials=int(1e5), **params):
        """Return estimate of total expected number of events
        :param data: Data used for drawing auxiliary observables
        (e.g. position and time)
        :param n_trials: Number of events to simulate for efficiency estimate
        """
        # TODO what if e_spectrum is pos/time dependent?
        # TODO: eh, not even looking at defaults now???
        _, spectra = cls(data, _skip_tf_init=True).gimme(
            'energy_spectrum',
            # TODO: BAD!
            data_tensor=None, ptensor=None,
            numpy_out=True)
        mean_rate = spectra.sum(axis=1).mean(axis=0)

        d_simulated = cls.simulate(n_trials, data=data, **params)
        return mean_rate * len(d_simulated) / n_trials
