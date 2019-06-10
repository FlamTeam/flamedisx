import inspect

import tensorflow as tf
import tensorflow_probability as tfp
# Remove once tf.repeat is available in the tf api
from tensorflow.python.ops.ragged.ragged_util import repeat
import numpy as np
from scipy import stats
from scipy.special import gammaln
import pandas as pd
from multihist import Hist1d

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

def _lookup_axis1(x, indices, fill_value=0):
    """Return values of x at indices along axis 1,
       returning fill_value for out-of-range indices.
    """

    mask = indices < x.shape[1]
    a, b = x.shape
    x = tf.reshape(x, [-1])
    indices = tf.dtypes.cast(indices, dtype=tf.int32)
    indices = indices + b * tf.range(a)[:, o, o]
    result = tf.reshape(tf.gather(x,
                                  tf.reshape(indices, shape=(-1,))),
                        shape=indices.shape)
    return tf.cast(tf.where(mask, result, tf.zeros_like(result) + fill_value),
                   dtype=tf.float32)


class ERSource:
    data_methods = tuple(data_methods)
    special_data_methods = tuple(special_data_methods)

    # Whether or not to simulate overdispersion in electron/photon split
    # (e.g. due to non-binomial recombination fluctuation)
    do_pel_fluct = True

    ##
    # Model functions
    ##

    def energy_spectrum(self, drift_time):
        """Return (energies in keV, rate at these energies),
        both (n_events, n_energies) tensors.
        """
        # TODO: doesn't depend on drift_time...
        n_evts = len(drift_time)
        return (
            np.linspace(0, 10, 1000)[o, :].repeat(n_evts, axis=0),
            np.ones(1000)[o, :].repeat(n_evts, axis=0))

    def energy_spectrum_hist(self):
        # TODO: fails if e is pos/time dependent
        es, rs = self.gimme('energy_spectrum')
        return Hist1d.from_histogram(rs[0, :-1], es[0, :])

    def simulate_es(self, n):
        return self.energy_spectrum_hist().get_random(n)

    work = 13.7e-3

    @staticmethod
    def p_electron(nq):
        return 0.5 * np.ones_like(nq)

    @staticmethod
    def p_electron_fluctuation(nq):
        return 0.01 * np.ones_like(nq)

    @staticmethod
    def penning_quenching_eff(nph):
        return np.ones_like(nph)

    # Detection efficiencies

    @staticmethod
    def electron_detection_eff(drift_time, *, elife=600e3, extraction_eff=1):
        return extraction_eff * np.exp(-drift_time / elife)

    photon_detection_eff = 0.1

    # Acceptance of selection/detection on photons/electrons detected

    electron_acceptance = 1

    @staticmethod
    def photon_acceptance(photons_detected):
        return np.where(photons_detected < 3, 0, 1)

    # Acceptance of selections on S1/S2 directly

    @staticmethod
    def s1_acceptance(s1):
        return np.where(s1 < 2, 0, 1)

    @staticmethod
    def s2_acceptance(s2):
        return np.where(s2 < 200, 0, 1)

    electron_gain_mean = 20
    electron_gain_std = 5

    photon_gain_mean = 1
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
        self.defaults = {}
        for fname in self.data_methods:
            f = getattr(self, fname)
            if callable(f):
                for i, (pname, p) in enumerate(
                        inspect.signature(f).parameters.items()):
                    if p.default == inspect.Parameter.empty:
                        if fname in self.special_data_methods and i == 0:
                            continue
                        self.f_dims[fname].append(pname)
                    else:
                        self.f_params[fname].append(pname)

        if data is not None:
            self.set_data(data)
        self._params = params
        self.tensor_data = dict()
        self.batch_slice = slice(None)

    @property
    def n_evts(self):
        return len(self.data)

    def gimme(self, fname, bonus_arg=None, data=None, params=None):
        if fname in self.special_data_methods:
            assert bonus_arg is not None
        else:
            assert bonus_arg is None

        if data is None:
            data = self.data
        if params is None:
            params = self._params

        f = getattr(self, fname)

        if callable(f):
            if fname in self.tensor_data.keys():
                args = [v[self.batch_slice] for v in self.tensor_data[fname]]
            else:
                # args = list(data[self.f_dims[fname]].values.T[..., self.batch_slice])
                args = [data[x].values[self.batch_slice] for x in self.f_dims[fname]]
            if bonus_arg is not None:
                args = [bonus_arg] + args

            kwargs = {k: v for k, v in params.items()
                      if k in self.f_params[fname]}

            return f(*args, **kwargs)

        else:
            if bonus_arg is None:
                return f * np.ones(len(data))[self.batch_slice]
            return f * np.ones_like(bonus_arg)

    def annotate_data(self, data, max_sigma=3, **params):
        """Annotate data with columns needed or inference,
        using params for maximum likelihood estimates"""
        old_data = self.data
        old_params = self._params
        try:
            self._params = params
            self.set_data(data, max_sigma=max_sigma)
        except Exception:
            raise
        finally:
            self.data = old_data
            self._params = old_params

    def set_data(self, data, max_sigma=3):
        # remove any previously computed tensors
        self.tensor_data = dict()
        self.batch_slice = slice(None)
        # Set new data
        self.data = d = data

        # TODO precompute energy spectra for each event?

        # Annotate data with eff, mean, sigma
        # according to the nominal model
        # These can still change during the inference!
        # TODO: so maybe you shouldn't store them in df...
        for qn in quanta_types:
            for parname in hidden_vars_per_quanta:
                fname = qn + '_' + parname
                d[fname] = self.gimme(fname)
        d['double_pe_fraction'] = self.gimme('double_pe_fraction')

        # Find likely number of detected quanta
        obs = dict(photon=d['s1'], electron=d['s2'])
        for qn in quanta_types:
            n_det_mle = (obs[qn] / d[qn + '_gain_mean'])
            if qn == 'photon':
                n_det_mle /= (1 + d['double_pe_fraction'])
            d[qn + '_detected_mle'] = n_det_mle.round().astype(np.int)

        # The Penning quenching depends on the number of produced
        # photons.... But we don't have that yet.
        # Thus, "rewrite" penning eff vs detected photons
        # using interpolation
        # TODO: this will fail when someone gives penning quenching some
        # data-dependent args
        _nprod_temp = np.logspace(-1, 8, 1000)
        peff = self.gimme('penning_quenching_eff', _nprod_temp)
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
        d['e_vis'] = self.gimme('work') * d['nq_vis_mle']
        d['fel_mle'] = self.gimme('p_electron', d['nq_vis_mle'])

        # Find plausble ranges for detected and observed quanta
        # based on the observed S1 and S2 sizes
        # (we could also derive ranges assuming the CES reconstruction,
        #  but these won't work well for outliers along one of the dimensions)
        # TODO: Meh, think about this, considering also computation cost
        # / space width
        for qn in quanta_types:
            p_q = d['fel_mle'] if qn == 'electron' else 1 - d['fel_mle']

            # If p_q gets very close to 0 or 1, the bounds blow up
            # this helps restore some sanity
            # TODO: think more about this
            p_q = p_q.clip(0.01, 0.99)

            if qn == 'photon':
                # Don't use *=, it will modify in place !
                eff = (d[qn + '_detection_eff']
                       * d['penning_quenching_eff_mle'])
            else:
                eff = d[qn + '_detection_eff']

            n_prod_mle = d[qn + '_produced_mle'] = (
                    d[qn + '_detected_mle'] / eff).astype(np.int)

            # Note this is a different estimate of nq than the CES one!
            nq_mle = (n_prod_mle / p_q).astype(np.int)

            clip_range = (0, None)

            for bound, sign in (('min', -1), ('max', +1)):

                # For detected quanta the MLE is quite accurate
                # (since fluctuations are tiny)
                # so let's just use the relative error on the MLE
                n = d[qn + '_detected_mle']
                m = d[qn + '_gain_mean']
                s = d[qn + '_gain_std']
                if qn == 'photon':
                    _, scale = self.dpe_mean_std(n, d['double_pe_fraction'],
                                                 m, s)
                else:
                    scale = n ** 0.5 * s / m

                d[qn + '_detected_' + bound] = stats.norm.ppf(
                    stats.norm.cdf(sign * max_sigma),
                    loc=d[qn + '_detected_mle'],
                    scale=scale,
                ).round().clip(*clip_range).astype(np.int)

                # TODO: For produced quanta I have to think harder!
                # How to set good bounds?
                # The formula below is just a fudge with a manually tuned
                # bonus sigma

                q = 1 - eff
                d[qn + '_produced_' + bound] = stats.norm.ppf(
                    stats.norm.cdf(sign * (max_sigma + 3)),
                    loc=n_prod_mle,
                    scale=(q + (q**2 + 4 * n_prod_mle * q)**0.5)/2
                ).round().clip(*clip_range).astype(np.int)

        # Bounds on total visible quanta
        d['nq_min'] = d['photon_produced_min'] + d['electron_produced_min']
        d['nq_max'] = d['photon_produced_max'] + d['electron_produced_max']

        # Precompute tensors for use in gimme
        for fname, v in self.f_dims.items():
            self.tensor_data[fname] = [tf.convert_to_tensor(d[x], dtype=tf.float32) for x in v]
        for fname in ['s1', 's2']:
            self.tensor_data[fname] = tf.convert_to_tensor(d[fname], dtype=tf.float32)

    def likelihood(self, data=None, max_sigma=3, batch_size=10,
                   progress=lambda x: x, **params):
        self._params = params
        if data is not None:
            self.set_data(data, max_sigma)
        del data   # Just so we don't reference it by accident

        # Evaluate in batches to save memory
        n_batches = np.ceil(len(self.data[self.batch_slice]) / batch_size).astype(np.int)
        if n_batches > 1:
            result = []
            for i in progress(list(range(n_batches))):
                self.batch_slice = slice(i * batch_size,
                                         (i + 1) * batch_size)
                result.append(self.likelihood(**params))
            return np.concatenate(result)

        # (n_events, |photons_produced|, |electrons_produced|)
        y = self.rate_nphnel()
        p_ph = self.detection_p('photon')
        p_el = self.detection_p('electron')
        d_ph = self.detector_response('photon')
        d_el = self.detector_response('electron')

        # Rearrange dimensions so we can do a single matrix mult
        # Alternatively, you could do
        #         return np.einsum('ij,ijk,ikl,iml,im->i',
        #                          d_ph, p_ph, y, p_el, d_el)
        # but that's about 10x slower!
        p_el = tf.transpose(p_el, (0, 2, 1))
        d_ph = d_ph[:, o, :]
        d_el = d_el[:, :, o]
        y = d_ph @ p_ph @ y @ p_el @ d_el
        return tf.reshape(y, [-1]).numpy()

    def _dimsize(self, var):
        return int((self.data[var + '_max'][self.batch_slice]
                    - self.data[var + '_min'][self.batch_slice]).max())

    def rate_nq(self, nq_1d):
        """Return differential rate at given number of produced quanta
        differs for ER and NR"""
        # TODO: this implementation echoes that for NR, but I feel there
        # must be a less clunky way...

        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum')
        q_produced = tf.cast(tf.floor(es / self.gimme('work')[:, o]),
                             dtype=tf.float32)

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tf.cast(tf.equal(nq_1d[:, :, o], q_produced[:, o, :]),
                         dtype=tf.float32)

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
        _nq_ind = nq - self.data['nq_min'][self.batch_slice].values[:, o, o]
        # ... differential rate
        rate_nq = _lookup_axis1(rate_nq, _nq_ind)
        # ... probability of a quantum to become an electron
        pel = _lookup_axis1(pel, _nq_ind)
        # ... probability fluctuation
        pel_fluct = _lookup_axis1(pel_fluct, _nq_ind)

        # Finally, the main computation is simple:
        if self.do_pel_fluct:
            return rate_nq * beta_binom_pmf(nel,
                                            n=nq,
                                            p_mean=pel,
                                            p_sigma=pel_fluct)
        else:
            pel_num = tf.where(tf.math.is_nan(pel), tf.zeros_like(pel), pel)
            pel_clip = tf.clip_by_value(pel_num, 0., 1.)
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
                              probs=tf.cast(p, dtype=tf.float32),
                              ).prob(n_det)
        return result * self.gimme(quanta_type + '_acceptance', n_det)

    def domain(self, x):
        """Return (n_events, |x|) matrix containing all possible integer
        values of x for each event"""
        n = self._dimsize(x)
        res = tf.range(n)[o, :] + self.data[x + '_min'][self.batch_slice][:, o]
        return tf.cast(res, dtype=tf.float32)

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

        observed = self.tensor_data[signal_name[quanta_type]][self.batch_slice, o]

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
        """Return mean, std of S1 signal for
        ndet: photons detected
        p_dpe: double pe emission probability
        mean_per_q: gain mean per PE
        std_per_q: gain std per PE
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

        Will not return | energies | events due to
        selection/detection efficiencies

        This is not used in the likelihood computation itself,
        but it's a useful check.
        """
        if not len(params):
            params = self._params
        if data is None:
            data = self.data

        if isinstance(energies, (float, int)):
            energies = self.simulate_es(int(energies))

        # Keep only dims we need
        d = data[list(set(sum(self.f_dims.values(), [])))]
        d = d.sample(n=len(energies), replace=True)

        def gimme(*args):
            return self.gimme(*args, data=d, params=params)

        d['energy'] = energies
        self.simulate_nq(data=d, params=params)

        d['p_el_mean'] = gimme('p_electron', d['nq'])
        d['p_el_fluct'] = gimme('p_electron_fluctuation', d['nq'])

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
               * gimme('penning_quenching_eff', d['photon_produced'])))

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
            acceptance *= gimme(q + '_acceptance', d[q + '_detected'])
            sn = signal_name[q]
            acceptance *= gimme(sn + '_acceptance', d[sn])
        d = d.iloc[np.random.rand(len(d)) < acceptance]
        return d

    def simulate_nq(self, data, params):
        work = self.gimme('work', data=data, params=params)
        data['nq'] = np.floor(data['energy'].values / work).astype(np.int)


def beta_params(mean, sigma):
    # Convert (p_mean, p_sigma) to (alpha, beta) params of beta distribution
    # From Wikipedia:
    # variance = 1/(4 * (2 * beta + 1)) = 1/(8 * beta + 4)
    # mean = 1/(1+beta/alpha)
    # =>
    # beta = (1/variance - 4) / 8
    # alpha
    b = (1 / (8 * sigma ** 2) - 0.5)
    a = b * mean / (1 - mean)
    return a, b


def beta_binom_pmf(x, n, p_mean, p_sigma):
    """Return probability mass function of beta-binomial distribution.

    That is, give the probability of obtaining x successes in n trials,
    if the success probability p is drawn from a beta distribution
    with mean p_mean and standard deviation p_sigma.

    Implemented using Dirichlet Multinomial distribution which is
    identically the Beta-Binomial distribution when len(beta_pars) == 2

    TODO: check if the number of successes wasn't reversed in the original
    code. Should we have [x, n-x] or [n-x, x]?
    """
    beta_pars = tf.stack(beta_params(p_mean, p_sigma), axis=-1)
    counts = tf.stack([x, n-x], axis=-1)
    res = tfd.DirichletMultinomial(n,
                                   beta_pars,
                                   allow_nan_stats=False).prob(counts)
    return tf.where(tf.math.is_finite(res), res, tf.zeros_like(res))


class NRSource(ERSource):
    do_pel_fluct = False
    data_methods = tuple(data_methods + ['lindhard_l'])
    special_data_methods = tuple(special_data_methods + ['lindhard_l'])

    @staticmethod
    def lindhard_l(e, lindhard_k=0.138):
        """Return Lindhard quenching factor at energy e in keV"""
        eps = 11.5 * e * 54**(-7/3)             # Xenon: Z = 54
        g = 3 * eps**0.15 + 0.7 * eps**0.6 + eps
        return lindhard_k * g/(1 + lindhard_k * g)

    def energy_spectrum(self, drift_time):
        """Return (energies in keV, events at these energies),
        both (n_events, n_energies) tensors.
        """
        e = np.linspace(0.7, 150, 100)[o, :].repeat(len(drift_time), axis=0)
        return e, np.ones_like(e)

    def rate_nq(self, nq_1d):
        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum')
        mean_q_produced = (
                es
                * self.gimme('lindhard_l', es)
                / self.gimme('work')[:, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = stats.poisson.pmf(nq_1d[:, :, o],
                                   mean_q_produced[:, o, :])

        return (p_nq_e * rate_e[:, o, :]).sum(axis=2)

    @staticmethod
    def penning_quenching_eff(nph, eta=8.2e-5 * 3.3, labda=0.8 * 1.15):
        return 1 / (1 + eta * nph ** labda)

    def simulate_nq(self, data, params):
        work = self.gimme('work',
                          data=data, params=params)
        lindhard_l = self.gimme('lindhard_l', data['energy'],
                                data=data, params=params)
        data['nq'] = stats.poisson.rvs(
            data['energy'].values * lindhard_l / work)
