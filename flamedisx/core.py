import inspect

from multihist import Hist1d
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
# Remove once tf.repeat is available in the tf api
from tensorflow.python.ops.ragged.ragged_util import repeat
from tqdm import tqdm

import flamedisx as fd
export, __all__ = fd.exporter()

tfd = tfp.distributions

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
class ERSource:
    data_methods = tuple(data_methods)
    special_data_methods = tuple(special_data_methods)

    # Whether or not to simulate overdispersion in electron/photon split
    # (e.g. due to non-binomial recombination fluctuation)
    do_pel_fluct = True

    # tuple with columns needed from data to run add_extra_columns
    extra_needed_columns = tuple()

    ##
    # Model functions
    ##

    def energy_spectrum(self, drift_time):
        """Return (energies in keV, rate at these energies),
        both (n_events, n_energies) tensors.
        """
        # TODO: doesn't depend on drift_time...
        n_evts = len(drift_time)
        return (repeat(tf.cast(tf.linspace(0., 10., 1000)[o, :],
                               dtype=fd.float_type()),
                       n_evts, axis=0),
                repeat(tf.ones(1000, dtype=fd.float_type())[o, :],
                       n_evts, axis=0))

    def energy_spectrum_hist(self):
        # TODO: fails if e is pos/time dependent
        es, rs = self.gimme('energy_spectrum', numpy_out=True)
        return Hist1d.from_histogram(rs[0, :-1], es[0, :])

    def simulate_es(self, n):
        return self.energy_spectrum_hist().get_random(n)

    work = 13.7e-3

    @staticmethod
    def p_electron(nq):
        return 0.5 * tf.ones_like(nq, dtype=fd.float_type())

    @staticmethod
    def p_electron_fluctuation(nq):
        return 0.01 * tf.ones_like(nq, dtype=fd.float_type())

    @staticmethod
    def penning_quenching_eff(nph):
        return tf.ones_like(nph, dtype=fd.float_type())

    # Detection efficiencies

    @staticmethod
    def electron_detection_eff(drift_time, *, elife=600e3, extraction_eff=1.):
        return extraction_eff * tf.exp(-drift_time / elife)

    photon_detection_eff = 0.1

    # Acceptance of selection/detection on photons/electrons detected
    # The min_xxx attributes are also used in the bound computations
    min_s1_photons_detected = 3.
    min_s2_electrons_detected = 3.

    def electron_acceptance(self, electrons_detected):
        return tf.where(
            electrons_detected < self.min_s2_electrons_detected,
            tf.zeros_like(electrons_detected, dtype=fd.float_type()),
            tf.ones_like(electrons_detected, dtype=fd.float_type()))

    def photon_acceptance(self, photons_detected):
        return tf.where(
            photons_detected < self.min_s1_photons_detected,
            tf.zeros_like(photons_detected, dtype=fd.float_type()),
            tf.ones_like(photons_detected, dtype=fd.float_type()))

    # Acceptance of selections on S1/S2 directly

    @staticmethod
    def s1_acceptance(s1):
        return tf.where(s1 < 2,
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def s2_acceptance(s2):
        return tf.where(s2 < 200,
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    electron_gain_mean = 20.
    electron_gain_std = 5.

    photon_gain_mean = 1.
    photon_gain_std = 0.5
    double_pe_fraction = 0.219

    ##
    # State attributes, set later
    ##
    data: pd.DataFrame = None
    params: dict = None

    ##
    # Main code body
    ##

    def __init__(self, data=None, **params):
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

        if data is not None:
            self.set_data(data)
        self._params = params

        # Dictionary that maps
        # observable dimension -> tensor
        self._tensor_cache = dict()

    @property
    def n_evts(self):
        return len(self.data)

    def gimme(self, fname, bonus_arg=None, numpy_out=False):
        """Evaluate the model function fname with all required arguments

        :param fname: Name of the model function to compute
        :param bonus_arg: If fname takes a bonus argument, the data for it
        :param numpy_out: If True, return (tuple of) numpy arrays,
        otherwise (tuple of) tensors.

        Before using gimme, you must use set_data to
        populate the internal caches.
        """
        assert (bonus_arg is not None) == (fname in self.special_data_methods)

        f = getattr(self, fname)

        if callable(f):
            args = [self._tensor_cache.get(
                        x,
                        fd.np_to_tf(self.data[x].values))
                    for x in self.f_dims[fname]]
            if bonus_arg is not None:
                args = [bonus_arg] + args

            kwargs = {pname: self._params.get(pname, self.defaults[pname])
                      for pname in self.f_params[fname]}

            res = f(*args, **kwargs)
        else:
            if bonus_arg is None:
                x = tf.ones(len(self.data), dtype=fd.float_type())
            else:
                x = tf.ones_like(bonus_arg, dtype=fd.float_type())
            res = f * x

        if numpy_out:
            return fd.tf_to_np(res)
        return fd.np_to_tf(res)

    def _clip_range(self, qn):
        return (self.min_s1_photons_detected if qn == 'photon'
                else self.min_s2_electrons_detected,
                None)

    def annotate_data(self, data, max_sigma=3, restore_prev=True, **params):
        """Annotate data with columns needed for inference.
        :param data: data to set
        :param max_sigma: Maximum sigma level to consider for bound estimation
        :param restore_prev: Restore previous state (default is true)
        (data, params, tensor cache)

        Other kwargs are interpreted as model parameters, used in the bound
        estimation.
        """
        # Store ref to old state, in case we have to restore it
        old_data = self.data
        old_params = self._params
        old_tensor_cache = self._tensor_cache

        # Set new data
        self._tensor_cache = dict()
        self._params = params
        self.data = d = data

        self.add_extra_columns(d)

        # TODO precompute energy spectra for each event?

        # Annotate data with eff, mean, sigma
        # according to the nominal model
        # These can still change during the inference!
        # TODO: so maybe you shouldn't store them in df...
        for qn in quanta_types:
            for parname in hidden_vars_per_quanta:
                fname = qn + '_' + parname
                d[fname] = self.gimme(fname, numpy_out=True)
        d['double_pe_fraction'] = self.gimme('double_pe_fraction',
                                             numpy_out=True)

        # Find likely number of detected quanta
        obs = dict(photon=d['s1'], electron=d['s2'])
        for qn in quanta_types:
            n_det_mle = (obs[qn] / d[qn + '_gain_mean'])
            if qn == 'photon':
                n_det_mle /= (1 + d['double_pe_fraction'])
            d[qn + '_detected_mle'] = n_det_mle.round().astype(np.int).clip(
                *self._clip_range(qn))

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
                    stats.norm.cdf(sign * max_sigma),
                    loc=n,
                    scale=scale,
                ).round().clip(*self._clip_range(qn)).astype(np.int)

                # d[qn + '_produced_' + bound] = fd.binom_n_bound(
                #     n_detected=d[qn + '_detected_' + bound].values,
                #     p=eff,
                #     sigma=sign * max_sigma,
                # ).round().clip(*clip_range).astype(np.int)

                # For produced quanta, I have to think harder..
                # TODO: where did this derivation come from again?
                q = 1 / eff
                d[qn + '_produced_' + bound] = stats.norm.ppf(
                    stats.norm.cdf(sign * max_sigma),
                    loc=n_prod_mle,
                    scale=(q + (q**2 + 4 * n_prod_mle * q)**0.5)/2
                ).round().clip(*self._clip_range(qn)).astype(np.int)

        # Bounds on total visible quanta
        d['nq_min'] = d['photon_produced_min'] + d['electron_produced_min']
        d['nq_max'] = d['photon_produced_max'] + d['electron_produced_max']

        if restore_prev:
            # Restore state
            self._tensor_cache = old_tensor_cache
            self.data = old_data
            self._params = old_params

    @staticmethod
    def add_extra_columns(data):
        """Add additional columns to data

        You must add any columns from data you use here to
        extra_needed.columns.

        :param data: pandas DataFrame
        """
        pass

    def set_data(self, data, max_sigma=3, annotated=False, **params):
        """Set new data to be used for inference

        :param data: data to set
        :param max_sigma: Maximum sigma level to consider for bound estimation
        :param annotated: Whether data has already been annotated. If False
        (default), will call annotate on it first.

        Other kwargs are interpreted as model parameters, used in the bound
        estimation.
        """
        # Set new data and params
        # tensor cache can only be set after annotation
        self._tensor_cache = dict()
        self.data = data
        self._params = params

        if not annotated:
            self.annotate_data(data,
                               max_sigma=max_sigma,
                               restore_prev=False,
                               **params)

        for x in set(sum(self.f_dims.values(), ['s1', 's2'])):
            self._tensor_cache[x] = fd.np_to_tf(data[x].values)

    def batched_likelihood(self, batch_size=50,
                           data=None, max_sigma=3, progress=True,
                           **params):
        if data is not None:
            self.set_data(data, max_sigma, **params)
        n_batches = np.ceil(len(self.data) / batch_size).astype(np.int)
        progress = tqdm if progress else lambda x: x
        orig_data = self.data

        result = []
        for i in progress(list(range(n_batches))):
            d = orig_data[i * batch_size:(i + 1) * batch_size].copy()
            self.set_data(d, annotated=True, **params)
            result.append(self._likelihood(**params).numpy())

        self.set_data(orig_data, annotated=True)
        return np.concatenate(result)[:len(orig_data)]

    def mu_interpolator(self, interpolation_method='star',
                        n_trials=int(1e5),
                        **params):
        """Return interpolator for number of expected events
        Parameters must be specified as kwarg=(start, stop, n_anchors)
        """
        if interpolation_method != 'star':
            raise NotImplementedError(
                f"mu interpolation method {interpolation_method} "
                f"not implemented")

        base_mu = tf.constant(self.estimate_mu(n_trials=n_trials),
                              dtype=fd.float_type())
        pspaces = dict()    # parameter -> tf.linspace of anchors
        mus = dict()        # parameter -> tensor of mus
        for pname, pspace_spec in tqdm(params.items(),
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
                    x_ref_min=params[pname][0],
                    x_ref_max=params[pname][1],
                    y_ref=mus[pname]) / base_mu
            return mu

        return mu_itp

    def estimate_mu(self, data=None, n_trials=int(1e5), **params):
        """Return estimate of total expected number of events
        :param data: Data used for drawing auxiliary observables
        (e.g. position and time)
        :param n_trials: Number of events to simulate for efficiency estimate
        """
        if data is None:
            data = self.data

        _, spectra = self.gimme('energy_spectrum', numpy_out=True)
        mean_rate = spectra.sum(axis=1).mean(axis=0)

        eff = len(self.simulate(n_trials, data=data, **params)) / n_trials

        return eff * mean_rate

    def likelihood(self, **params):
        return self._likelihood(**params)

    def _likelihood(self, **params):
        self._params = params
        # (n_events, |photons_produced|, |electrons_produced|)
        y = self.rate_nphnel()

        p_ph = self.detection_p('photon')
        p_el = self.detection_p('electron')
        d_ph = self.detector_response('photon')
        d_el = self.detector_response('electron')

        # Rearrange dimensions so we can do a single matrix mult
        p_el = tf.transpose(p_el, (0, 2, 1))
        d_ph = d_ph[:, o, :]
        d_el = d_el[:, :, o]
        y = d_ph @ p_ph @ y @ p_el @ d_el
        return tf.reshape(y, [-1])

    def _dimsize(self, var):
        ma = self._tensor_cache.get(
            var + '_max',
            fd.np_to_tf(self.data[var + '_max'].values))
        mi = self._tensor_cache.get(
            var + '_min',
            fd.np_to_tf(self.data[var + '_min'].values))
        return tf.cast(tf.reduce_max(ma - mi), tf.int32)
        # return len(self._tensor_cache.get(x, self.data[x].values))
        # return int((self.data[var + '_max']
        #             - self.data[var + '_min']).max())

    def rate_nq(self, nq_1d):
        """Return differential rate at given number of produced quanta
        differs for ER and NR"""
        # TODO: this implementation echoes that for NR, but I feel there
        # must be a less clunky way...

        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum')
        q_produced = tf.cast(tf.floor(es / self.gimme('work')[:, o]),
                             dtype=fd.float_type())

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tf.cast(tf.equal(nq_1d[:, :, o], q_produced[:, o, :]),
                         dtype=fd.float_type())

        return tf.reduce_sum(p_nq_e * rate_e[:, o, :], axis=2)

    def rate_nphnel(self):
        """Return differential rate tensor
        (n_events, |photons_produced|, |electrons_produced|)
        """
        # Get differential rate and electron probability vs n_quanta
        # these four are (n_events, |nq|) tensors
        _nq_1d = self.domain('nq')
        rate_nq = self.rate_nq(_nq_1d)
        pel = self.gimme('p_electron', _nq_1d)
        pel_fluct = self.gimme('p_electron_fluctuation', _nq_1d)

        # Create tensors with the dimensions of our final result
        # i.e. (n_events, |photons_produced|, |electrons_produced|),
        # containing:
        # ... numbers of photons and electrons produced:
        nph, nel = self.cross_domains('photon_produced', 'electron_produced')
        # ... numbers of total quanta produced
        nq = nel + nph
        # ... indices in nq arrays
        _nq_ind = nq - self.data['nq_min'].values[:, o, o]
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
            return rate_nq * beta_binom_pmf(nel,
                                            n=nq,
                                            p_mean=pel_clip,
                                            p_sigma=pel_fluct_clip)
        else:
            return rate_nq * tfd.Binomial(total_count=nq,
                                          probs=pel_clip).prob(nel)

    def detection_p(self, quanta_type):
        """Return (n_events, |detected|, |produced|) tensor
        encoding P(n_detected | n_produced)
        """
        n_det, n_prod = self.cross_domains(quanta_type + '_detected',
                                           quanta_type + '_produced')

        p = self.gimme(quanta_type + '_detection_eff')[:, o, o]
        if quanta_type == 'photon':
            # Note *= doesn't work, p will get reshaped
            p = p * self.gimme('penning_quenching_eff', n_prod)

        result = tfd.Binomial(total_count=n_prod,
                              probs=tf.cast(p, dtype=fd.float_type()),
                              ).prob(n_det)
        return result * self.gimme(quanta_type + '_acceptance', n_det)

    def domain(self, x):
        """Return (n_events, |x|) matrix containing all possible integer
        values of x for each event"""
        n = self._dimsize(x)
        res = tf.range(n)[o, :] + self.data[x + '_min'][:, o]
        return tf.cast(res, dtype=fd.float_type())

    def cross_domains(self, x, y):
        """Return (x, y) two-tuple of (n_events, |x|, |y|) tensors
        containing possible integer values of x and y, respectively.
        """
        # TODO: somehow mask unnecessary elements and save computation time
        x_size = self._dimsize(x)
        y_size = self._dimsize(y)
        # Change to tf.repeat once its in the api
        result_x = repeat(self.domain(x)[:, :, o], y_size, axis=2)
        result_y = repeat(self.domain(y)[:, o, :], x_size, axis=1)
        return result_x, result_y

    def detector_response(self, quanta_type):
        """Return (n_events, |n_detected|) probability of observing the S[1|2]
        for different number of detected quanta.
        """
        ndet = self.domain(quanta_type + '_detected')

        observed = self._tensor_cache[signal_name[quanta_type]][:, o]

        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme(quanta_type + '_gain_mean')[:, o]
        std_per_q = self.gimme(quanta_type + '_gain_std')[:, o]

        if quanta_type == 'photon':
            mean, std = self.dpe_mean_std(
                ndet=ndet,
                p_dpe=self.gimme('double_pe_fraction')[:, o],
                mean_per_q=mean_per_q,
                std_per_q=std_per_q)
        else:
            mean = ndet * mean_per_q
            std = ndet**0.5 * std_per_q

        # add offset to std to avoid NaNs from norm.pdf if std = 0
        result = tfd.Normal(loc=mean, scale=std + 1e-10).prob(observed)

        # Add detection/selection efficiency
        result *= self.gimme(signal_name[quanta_type] + '_acceptance',
                             observed)
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

    def simulate(self, energies, data=None, **params):
        """Simulate events at energies,
        drawing values of additional observables (e.g. positions)
        from data.

        Will not return | energies | events lost due to
        selection/detection efficiencies
        """
        if not len(params):
            params = self._params
        if data is None:
            data = self.data
        orig_data = self.data
        orig_params = self._params

        # This is necessary if the energy spectrum is position dependent
        self.set_data(data.copy(), **params)

        if isinstance(energies, (float, int)):
            energies = self.simulate_es(int(energies))

        # Create and set  new dataset, with just the dimensions we need
        d = data[list(set(sum(
            self.f_dims.values(),
            ['s1', 's2'] + list(self.extra_needed_columns))))]
        d = d.sample(n=len(energies), replace=True)
        self.set_data(d, **params)

        def gimme(*args):
            return self.gimme(*args, numpy_out=True)

        d['energy'] = energies
        self.simulate_nq(data=d)

        d['p_el_mean'] = gimme('p_electron', d['nq'].values)
        d['p_el_fluct'] = gimme('p_electron_fluctuation', d['nq'].values)

        d['p_el_actual'] = stats.beta.rvs(
            *beta_params(d['p_el_mean'], d['p_el_fluct']))
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

        d['s1'] = stats.norm.rvs(*self.dpe_mean_std(
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

        # This is useful, so we already have inference bounds on the
        # returned data.
        self.set_data(d, **params)

        # Restore original data
        self.set_data(orig_data, **orig_params)
        return d

    def simulate_nq(self, data):
        work = self.gimme('work', numpy_out=True)
        data['nq'] = np.floor(data['energy'].values / work).astype(np.int)


def beta_params(mean, sigma):
    """Convert (p_mean, p_sigma) to (alpha, beta) params of beta distribution
    """
    # From Wikipedia:
    # variance = 1/(4 * (2 * beta + 1)) = 1/(8 * beta + 4)
    # mean = 1/(1+beta/alpha)
    # =>
    # beta = (1/variance - 4) / 8
    # alpha
    b = (1. / (8. * sigma ** 2) - 0.5)
    a = b * mean / (1. - mean)
    return a, b


def beta_binom_pmf(x, n, p_mean, p_sigma):
    """Return probability mass function of beta-binomial distribution.

    That is, give the probability of obtaining x successes in n trials,
    if the success probability p is drawn from a beta distribution
    with mean p_mean and standard deviation p_sigma.

    Implemented using Dirichlet Multinomial distribution which is
    identically the Beta-Binomial distribution when len(beta_pars) == 2
    """
    # TODO: check if the number of successes wasn't reversed in the original
    # code. Should we have [x, n-x] or [n-x, x]?

    beta_pars = tf.stack(beta_params(p_mean, p_sigma), axis=-1)

    # DirichletMultinomial only gives correct output on float64 tensors!
    # Cast inputs to float64 explicitly!
    beta_pars = tf.cast(beta_pars, dtype=tf.float64)
    x = tf.cast(x, dtype=tf.float64)
    n = tf.cast(n, dtype=tf.float64)

    counts = tf.stack([x, n-x], axis=-1)
    res = tfd.DirichletMultinomial(n,
                                   beta_pars,
                                   # validate_args=True,
                                   # allow_nan_stats=False
                                   ).prob(counts)
    res = tf.cast(res, dtype=fd.float_type())
    return tf.where(tf.math.is_finite(res),
                    res,
                    tf.zeros_like(res, dtype=fd.float_type()))


@export
class NRSource(ERSource):
    do_pel_fluct = False
    data_methods = tuple(data_methods + ['lindhard_l'])
    special_data_methods = tuple(special_data_methods + ['lindhard_l'])

    @staticmethod
    def lindhard_l(e, lindhard_k=0.138):
        """Return Lindhard quenching factor at energy e in keV"""
        eps = 11.5 * e * 54**(-7/3)             # Xenon: Z = 54
        g = 3. * eps**0.15 + 0.7 * eps**0.6 + eps
        res = lindhard_k * g/(1. + lindhard_k * g)
        return res

    def energy_spectrum(self, drift_time):
        """Return (energies in keV, events at these energies),
        both (n_events, n_energies) tensors.
        """
        e = repeat(tf.cast(tf.linspace(0.7, 150., 100)[o, :],
                           fd.float_type()),
                   len(drift_time), axis=0)
        return e, tf.ones_like(e, dtype=fd.float_type())

    def rate_nq(self, nq_1d):
        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum')
        mean_q_produced = (
                es
                * self.gimme('lindhard_l', es)
                / self.gimme('work')[:, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tfd.Poisson(mean_q_produced[:, o, :]).prob(nq_1d[:, :, o])

        return tf.reduce_sum(p_nq_e * rate_e[:, o, :], axis=2)

    @staticmethod
    def penning_quenching_eff(nph, eta=8.2e-5 * 3.3, labda=0.8 * 1.15):
        return 1. / (1. + eta * nph ** labda)

    def simulate_nq(self, data,):
        work = self.gimme('work', numpy_out=True)
        lindhard_l = self.gimme('lindhard_l', data['energy'].values,
                                numpy_out=True)
        data['nq'] = stats.poisson.rvs(
            data['energy'].values * lindhard_l / work)
