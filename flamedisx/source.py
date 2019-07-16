import inspect
import typing

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
class Source:
    data_methods = tuple(data_methods)
    special_data_methods = tuple(special_data_methods)

    # Whether or not to simulate overdispersion in electron/photon split
    # (e.g. due to non-binomial recombination fluctuation)
    do_pel_fluct = True

    # tuple with columns needed from data to run add_extra_columns
    extra_needed_columns = tuple()

    _params: dict = None
    data: pd.DataFrame

    _tensor_cache: typing.Dict[str, tf.Tensor]
    _tensor_cache_list: typing.List[typing.Dict[str, tf.Tensor]]

    def __init__(self,
                 data,
                 batch_size=10,
                 max_sigma=3,
                 data_is_annotated=False,
                 _skip_tf_init=False,
                 _skip_bounds_computation=False,
                 **params):
        self.max_sigma = max_sigma
        self._params = params
        self.data = data

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
                if p.default == inspect.Parameter.empty:
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

        if batch_size is None:
            batch_size = len(data)
        self.batch_size = batch_size
        self.n_batches = np.ceil(
            self.n_events() / self.batch_size).astype(np.int)

        if not data_is_annotated:
            self._annotate(_skip_bounds_computation=_skip_bounds_computation)
        if not _skip_tf_init:
            self._populate_tensor_cache()

    def _populate_tensor_cache(self):
        self._tensor_cache = {
            x: fd.np_to_tf(self.data[x].values)
            for x in self.data.columns
            if (np.issubdtype(self.data[x].dtype, np.integer)
                or np.issubdtype(self.data[x].dtype, np.floating))}

        self.dimsizes = dict()
        for var in ['nq',
                  'photon_detected',
                  'electron_detected',
                  'photon_produced',
                  'electron_produced']:
            ma = self._fetch(var + '_max')
            mi = self._fetch(var + '_min')
            self.dimsizes[var] = int(tf.reduce_max(ma - mi + 1).numpy())

        # Split up tensors for batched evaluation
        start = np.arange(self.n_batches) * self.batch_size
        stop = np.concatenate([start[1:], [self.n_events()]])
        self.tensor_cache_list = [
            {k: v[start[i]:stop[i]]
             for k, v in self._tensor_cache.items()}
            for i in range(self.n_batches)]

    def n_events(self, i_batch=None):
        if i_batch is None:
            return len(self.data)
        if i_batch == self.n_batches - 1:
            return self.n_events() - self.batch_size * (self.n_batches - 1)
        return self.batch_size

    def _fetch(self, x, i_batch=None):
        """Return a tensor column from the original dataframe (self.data)
        :param x: column name
        :param i_batch: Batch index. If None, return results for the entire
        dataset.
        """
        if not hasattr(self, '_tensor_cache'):
            # We're inside annotate, just return the column
            return fd.np_to_tf(self.data[x].values)
        if i_batch is None:
            return self._tensor_cache.get(
                x,
                fd.np_to_tf(self.data[x].values))
        else:
            return self.tensor_cache_list[i_batch][x]

    def gimme(self, fname, bonus_arg=None, i_batch=None, numpy_out=False):
        """Evaluate the model function fname with all required arguments

        :param fname: Name of the model function to compute
        :param bonus_arg: If fname takes a bonus argument, the data for it
        :param numpy_out: If True, return (tuple of) numpy arrays,
        otherwise (tuple of) tensors.
        :param i_batch: Batch index. If None, return results for the entire
        dataset.

        Before using gimme, you must use set_data to
        populate the internal caches.
        """
        # TODO: make a clean way to keep track of i_batch or have it as input
        assert (bonus_arg is not None) == (fname in self.special_data_methods)

        f = getattr(self, fname)

        if callable(f):
            args = [self._fetch(x, i_batch) for x in self.f_dims[fname]]
            if bonus_arg is not None:
                args = [bonus_arg] + args
            kwargs = {pname: self._params.get(pname, self.defaults[pname])
                      for pname in self.f_params[fname]}
            res = f(*args, **kwargs)

        else:
            if bonus_arg is None:
                x = tf.ones(self.n_events(i_batch),
                            dtype=fd.float_type())
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
                    d[fname] = self.gimme(fname, numpy_out=True)
                except Exception:
                    print(fname)
                    raise
        d['double_pe_fraction'] = self.gimme('double_pe_fraction',
                                             numpy_out=True)

        if _skip_bounds_computation:
            return

        # Find likely number of detected quanta
        obs = dict(photon=d['s1'], electron=d['s2'])
        for qn in quanta_types:
            n_det_mle = (obs[qn] / d[qn + '_gain_mean'])
            if qn == 'photon':
                n_det_mle /= (1 + d['double_pe_fraction'])
            d[qn + '_detected_mle'] = n_det_mle.round().astype(np.int).clip(
                *self._q_det_clip_range(qn))

        # The Penning quenching depends on the number of produced
        # photons.... But we don't have that yet.
        # Thus, "rewrite" penning eff vs detected photons
        # using interpolation
        # TODO: this will fail when someone gives penning quenching some
        # data-dependent args
        _nprod_temp = np.logspace(-1., 8., 1000)
        peff = self.gimme('penning_quenching_eff', _nprod_temp, numpy_out=True)
        d['penning_quenching_eff_mle'] = np.interp(
            d['photon_detected_mle'] / d['photon_detection_eff'],
            _nprod_temp * peff,
            peff)

        # Approximate energy reconstruction (visible energy only)
        # TODO: how to do CES estimate if someone wants a variable W?
        d['nq_vis_mle'] = (
            d['electron_detected_mle'] / d['electron_detection_eff']
            + (d['photon_detected_mle'] / d['photon_detection_eff']
               / d['penning_quenching_eff_mle']))
        d['e_vis'] = self.gimme('work', numpy_out=True) * d['nq_vis_mle']
        d['fel_mle'] = self.gimme('p_electron', d['nq_vis_mle'].values,
                                  numpy_out=True)

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

    def batched_differential_rate(self, progress=True, **params):
        progress = (lambda x: x) if not progress else tqdm
        return np.concatenate([
            fd.tf_to_np(self.differential_rate(i_batch=i_batch, **params))
            for i_batch in progress(range(self.n_batches))])

    @tf.function
    def differential_rate(self, i_batch=None, **params):
        return self._differential_rate(i_batch=None, **params)

    def _differential_rate(self, i_batch=None, **params):
        self._params = params
        # (n_events, |photons_produced|, |electrons_produced|)
        y = self.rate_nphnel(i_batch)

        p_ph = self.detection_p('photon', i_batch)
        p_el = self.detection_p('electron', i_batch)
        d_ph = self.detector_response('photon', i_batch)
        d_el = self.detector_response('electron', i_batch)

        # Rearrange dimensions so we can do a single matrix mult
        p_el = tf.transpose(p_el, (0, 2, 1))
        d_ph = d_ph[:, o, :]
        d_el = d_el[:, :, o]
        y = d_ph @ p_ph @ y @ p_el @ d_el
        return tf.reshape(y, [-1])

    def rate_nq(self, nq_1d, i_batch=None):
        """Return differential rate at given number of produced quanta
        differs for ER and NR"""
        # TODO: this implementation echoes that for NR, but I feel there
        # must be a less clunky way...

        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum', i_batch=i_batch)
        q_produced = tf.cast(
            tf.floor(es / self.gimme('work', i_batch=i_batch)[:, o]),
            dtype=fd.float_type())

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tf.cast(tf.equal(nq_1d[:, :, o], q_produced[:, o, :]),
                         dtype=fd.float_type())

        q = tf.reduce_sum(p_nq_e * rate_e[:, o, :], axis=2)
        return q

    def rate_nphnel(self, i_batch=None):
        """Return differential rate tensor
        (n_events, |photons_produced|, |electrons_produced|)
        """
        # Get differential rate and electron probability vs n_quanta
        # these four are (n_events, |nq|) tensors
        _nq_1d = self.domain('nq', i_batch)
        rate_nq = self.rate_nq(_nq_1d, i_batch=i_batch)
        pel = self.gimme('p_electron', _nq_1d, i_batch=i_batch)
        pel_fluct = self.gimme('p_electron_fluctuation', _nq_1d, i_batch=i_batch)

        # Create tensors with the dimensions of our fin al result
        # i.e. (n_events, |photons_produced|, |electrons_produced|),
        # containing:
        # ... numbers of photons and electrons produced:
        nph, nel = self.cross_domains('photon_produced', 'electron_produced', i_batch)
        # ... numbers of total quanta produced
        nq = nel + nph
        # ... indices in nq arrays
        _nq_ind = nq - self._fetch('nq_min', i_batch=i_batch)[:, o, o]
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

    def detection_p(self, quanta_type, i_batch=None):
        """Return (n_events, |detected|, |produced|) te nsor
        encoding P(n_detected | n_produced)
        """
        n_det, n_prod = self.cross_domains(quanta_type + '_detected',
                                           quanta_type + '_produced',
                                           i_batch)

        p = self.gimme(quanta_type + '_detection_eff',
                        i_batch=i_batch)[:, o, o]
        if quanta_type == 'photon':
            # Note *= doesn't work, p will get reshaped
            p = p * self.gimme('penning_quenching_eff', n_prod,
                                i_batch=i_batch)

        result = tfp.distributions.Binomial(
                total_count=n_prod,
                probs=tf.cast(p, dtype=fd.float_type())
            ).prob(n_det)
        return result * self.gimme(quanta_type + '_acceptance', n_det,
                                    i_batch = i_batch)

    def domain(self, x, i_batch=None):
        """Return (n_events, |possible x values|) matrix containing all possible integer
        values of x for each event"""
        result1 = tf.cast(tf.range(self.dimsizes[x]),
                          dtype=fd.float_type())[o, :]
        result2 = self._fetch(x + '_min', i_batch=i_batch)[:, o]
        return result1 + result2

    def cross_domains(self, x, y, i_batch=None):
        """Return (x, y) two-tuple of (n_events, |x|, |y|) tensors
        containing possible integer values of x and y, respectively.
        """
        # TODO: somehow mask unnecessary elements and save computation time
        x_size = self.dimsizes[x]
        y_size = self.dimsizes[y]
        # Change to tf.repeat once it's in the api
        result_x = fd.repeat(self.domain(x, i_batch)[:, :, o], y_size, axis=2)
        result_y = fd.repeat(self.domain(y, i_batch)[:, o, :], x_size, axis=1)
        return result_x, result_y

    def detector_response(self, quanta_type, i_batch=None):
        """Return (n_events, |n_detected|) probability of observing the S[1|2]
        for different number of detected quanta.
        """
        ndet = self.domain(quanta_type + '_detected', i_batch)

        observed = self._fetch(
            signal_name[quanta_type], i_batch=i_batch)[:, o]

        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme(quanta_type + '_gain_mean', i_batch=i_batch)[:, o]
        std_per_q = self.gimme(quanta_type + '_gain_std',i_batch=i_batch)[:, o]

        if quanta_type == 'photon':
            mean, std = self.dpe_mean_std(
                ndet=ndet,
                p_dpe=self.gimme('double_pe_fraction',i_batch=i_batch)[:, o],
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
                             observed, i_batch=i_batch)
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
    def simulate(cls, energies, data=None, **params):
        """Simulate events at energies,
        drawing values of additional observables (e.g. positions)
        from data.

        Will not return | energies | events lost due to
        selection/detection efficiencies
        """
        # simulate_es cannot be a class method; the energy-spectrum might
        # be position/time/other dependent.
        s = cls(data=data, _skip_tf_init=True, **params)
        if isinstance(energies, (float, int)):
            energies = s.simulate_es(int(energies))

        # Create and set new dataset, with just the dimensions we need
        # (note we should NOT include s1 and s2 here, we're going to simulate
        # them)
        d = data[list(set(sum(
            s.f_dims.values(),
            list(s.extra_needed_columns))))]
        d = d.sample(n=len(energies), replace=True)
        s = cls(data=d,
                _skip_tf_init=True,
                _skip_bounds_computation=True,
                **params)
        assert 'e_vis' not in d.columns

        def gimme(*args):
            return s.gimme(*args, numpy_out=True)

        d['energy'] = energies
        s._simulate_nq()

        d['p_el_mean'] = gimme('p_electron', d['nq'].values)
        d['p_el_fluct'] = gimme('p_electron_fluctuation', d['nq'].values)

        d['p_el_actual'] = stats.beta.rvs(
            *fd.beta_params(d['p_el_mean'], d['p_el_fluct']))
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

    def _simulate_nq(self):
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
        _, spectra = cls(data, _skip_tf_init=True).gimme(
            'energy_spectrum', numpy_out=True)
        mean_rate = spectra.sum(axis=1).mean(axis=0)

        d_simulated = cls.simulate(n_trials, data=data, **params)
        return mean_rate * len(d_simulated) / n_trials
