import inspect

import numpy as np
from scipy import stats
from scipy.special import gammaln
import pandas as pd

quanta_types = 'photon', 'electron'
signal_name = dict(photon='s1', electron='s2')

# Data methods that take an additional positional argument
special_data_methods = [
    'energy_spectrum',
    'p_electron',
    'p_electron_fluctuation',
    'electron_acceptance',
    'photon_acceptance',
    's1_acceptance',
    's2_acceptance'
]

data_methods = special_data_methods + ['work']
hidden_vars_per_quanta = 'detection_eff gain_mean gain_std'.split()
for _qn in quanta_types:
    data_methods += [_qn + '_' + x for x in hidden_vars_per_quanta]


def _lookup_axis1(x, indices, fill_value=0):
    """Return values of x at indices along axis 1,
    returning fill_value for out-of-range indices"""
    d = indices
    imax = x.shape[1]
    mask = d >= imax
    d[mask] = 0
    result = np.take_along_axis(
            x,
            d.reshape(len(d), -1), axis=1
        ).reshape(d.shape)
    result[mask] = fill_value
    return result


class XenonSource:

    ##
    # Model functions
    ##
    @staticmethod
    def energy_spectrum(e):
        return np.ones_like(e)

    work = 13.7e-3

    @staticmethod
    def p_electron(nq):
        return 0.5 * np.ones_like(nq)

    @staticmethod
    def p_electron_fluctuation(nq):
        return 0.01 * np.ones_like(nq)

    # Detection efficiencies

    @staticmethod
    def electron_detection_eff(drift_time,
                               *, elife=600, extraction_eff=1):
        return extraction_eff * np.exp(-drift_time / elife)

    photon_detection_eff = 0.1

    # Acceptance of selection/detection on photons/electrons detected

    electron_acceptance = 1

    @staticmethod
    def photon_acceptance(photons_detected):
        result = np.ones_like(photons_detected)
        result[photons_detected < 3] = 0
        return result

    # Acceptance of selections on S1/S2 directly

    @staticmethod
    def s1_acceptance(s1):
        result = np.ones_like(s1)
        result[s1 < 2] = 0
        return result

    @staticmethod
    def s2_acceptance(s2):
        result = np.ones_like(s2)
        result[s2 < 200] = 0
        return result

    electron_gain_mean = 20
    electron_gain_std = 5

    photon_gain_mean = 1
    photon_gain_std = 0.5

    ##
    # State attributes, set later
    ##
    data: pd.DataFrame = None
    params: dict = None
    n_evts = -1

    ##
    # Main code body
    ##

    def __init__(self, data=None, **params):
        # Discover which functions need which arguments / dimensions
        # Discover possible parameters
        self.f_dims = {x: [] for x in data_methods}
        self.f_params = {x: [] for x in data_methods}
        self.defaults = {}
        for fname in data_methods:
            f = getattr(self, fname)
            if callable(f):
                for i, (pname, p) in enumerate(
                        inspect.signature(f).parameters.items()):
                    if p.default == inspect.Parameter.empty:
                        if fname in special_data_methods and i == 0:
                            continue
                        self.f_dims[fname].append(pname)
                    else:
                        self.f_params[fname].append(pname)

        if data is not None:
            self.set_data(data)
        self._params = params

    def gimme(self, fname, bonus_arg=None, data=None, params=None):
        if fname in special_data_methods:
            assert bonus_arg is not None
        else:
            assert bonus_arg is None

        if data is None:
            data = self.data
        if params is None:
            params = self._params

        f = getattr(self, fname)

        if callable(f):
            args = [data[x].values for x in self.f_dims[fname]]
            if bonus_arg is not None:
                args = [bonus_arg] + args

            kwargs = {k: v for k, v in params.items()
                      if k in self.f_params[fname]}

            return f(*args, **kwargs)

        else:
            if bonus_arg is None:
                return f * np.ones(len(data))
            return f * np.ones_like(bonus_arg)

    def set_data(self, data, max_sigma=5):
        self.data = d = data
        self.n_evts = len(data)

        # Annotate data with eff, mean, sigma
        # according to the nominal model
        # These can still change during the inference!
        # TODO: so maybe you shouldn't store them in df...
        for qn in quanta_types:
            for parname in hidden_vars_per_quanta:
                fname = qn + '_' + parname
                d[fname] = self.gimme(fname)

        # Approximate energy reconstruction (CES)
        # TODO: Fix for NR!
        # TODO: how to do CES estimate with variable W?
        obs = dict(photon=d['s1'], electron=d['s2'])
        d['eces_mle'] = sum([
            obs[qn] / (d[qn + '_gain_mean'] * d[qn + '_detection_eff'])
            for qn in quanta_types
        ]) * self.gimme('work')

        d['nq_mle'] = d['eces_mle'].values / self.gimme('work')
        d['fel_mle'] = self.gimme('p_electron', d['nq_mle'])

        # Find plausble ranges for detected and observed quanta
        # based on the observed S1 and S2 sizes
        # (we could also derive ranges assuming the CES reconstruction,
        #  but these won't work well for outliers along one of the dimensions)
        # TODO: Meh, think about this, considering also computation cost
        # / space width
        for qn in quanta_types:
            p_q = d['fel_mle'] if qn == 'electron' else 1 - d['fel_mle']

            n_det_mle = d[qn + '_detected_mle'] = (
                    obs[qn] / d[qn + '_gain_mean']).astype(np.int)
            n_prod_mle = d[qn + '_produced_mle'] = (
                    n_det_mle / d[qn + '_detection_eff']).astype(np.int)
            # Note this is a different estimate of nq than the CES one!
            nq_mle = (n_prod_mle / p_q).astype(np.int)

            clip_range = (0, None)

            for bound, sign in (('min', -1), ('max', +1)):
                level = stats.norm.cdf(sign * max_sigma)

                d[qn + '_produced_' + bound] = stats.binom.ppf(
                    level, nq_mle, p_q
                ).clip(*clip_range)

                d[qn + '_detected_' + bound] = stats.binom.ppf(
                    level, n_prod_mle, d[qn + '_detection_eff']
                ).clip(*clip_range)

        d['nq_min'] = d['photon_produced_min'] + d['electron_produced_min']
        d['nq_max'] = d['photon_produced_max'] + d['electron_produced_max']

    def likelihood(self, data=None, max_sigma=5, batch_size=10,
                   progress=lambda x: x, **params):
        if data is not None:
            self.set_data(data, max_sigma)
        del data   # Just so we don't reference it by accident

        # Evaluate in batches to save memory
        n_batches = np.ceil(len(self.data) / batch_size).astype(np.int)
        if n_batches > 1:
            orig_data = self.data
            result = []
            for i in progress(list(range(n_batches))):
                self.data = orig_data[
                            i * batch_size:(i + 1) * batch_size].copy()
                result.append(self.likelihood(**params))
            return np.concatenate(result)

        self._params = params
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
        p_el = p_el.transpose(0, 2, 1)
        d_ph = d_ph[:, np.newaxis, :]
        d_el = d_el[:, :, np.newaxis]
        y = d_ph @ p_ph @ y @ p_el @ d_el
        return y.reshape(-1)

    def _dimsize(self, var):
        return int((self.data[var + '_max']
                    - self.data[var + '_min']).max())

    def rate_nphnel(self):
        """Return differential rate tensor
        (n_events, |photons_produced|, |electrons_produced|)
        """
        # Get differential rate and electron probability vs n_quanta
        _nq_1d = self.domain('nq')
        rate_nq = self.gimme('energy_spectrum',
                             _nq_1d * self.gimme('work')[:, np.newaxis])
        pel = self.gimme('p_electron', _nq_1d)
        pel_fluct = self.gimme('p_electron_fluctuation', _nq_1d)

        # Create tensors with the dimensions of our final result, containing:
        # ... numbers of photons and electrons produced:
        nph, nel = self.cross_domains('photon_produced', 'electron_produced')
        # ... numbers of total quanta produced
        nq = nel + nph
        # ... indices in nq arrays
        _nq_ind = nq - self.data['nq_min'].values.astype(np.int)[:, np.newaxis, np.newaxis]
        # ... differential rate
        rate_nq = _lookup_axis1(rate_nq, _nq_ind)
        # ... probability of a quantum to become an electron
        pel = _lookup_axis1(pel, _nq_ind)
        # ... probability fluctuation
        pel_fluct = _lookup_axis1(pel_fluct, _nq_ind)

        # Finally, the main computation is simple:
        return rate_nq * beta_binom_pmf(
            nel.astype(np.int),
            n=nq.astype(np.int),
            p_mean=pel,
            p_sigma=pel_fluct)

    def detection_p(self, quanta_type):
        """Return (n_events, |detected|, |produced|) tensor
        encoding P(n_detected | n_produced)
        """
        n_det, n_prod = self.cross_domains(quanta_type + '_detected',
                                           quanta_type + '_produced')
        p = self.gimme(quanta_type + '_detection_eff')
        p = p.reshape(-1, 1, 1)
        result = stats.binom.pmf(n_det, n=n_prod, p=p)
        return result * self.gimme(quanta_type + '_acceptance', n_det)

    def domain(self, x):
        """Return (n_events, |x|) matrix containing all possible integer
        values of x for each event"""
        o = np.newaxis
        n = self._dimsize(x)
        return np.arange(n)[o, :] + self.data[x + '_min'].astype(np.int)[:, o]

    def cross_domains(self, x, y):
        """Return (x, y) two-tuple of (n_events, |x|, |y|) tensors
        containing possible integer values of x and y, respectively.
        """
        # TODO: somehow mask unnecessary elements and save computation time
        x_size = self._dimsize(x)
        y_size = self._dimsize(y)
        o = np.newaxis
        result_x = self.domain(x)[:, :, o].repeat(y_size, axis=2)
        result_y = self.domain(y)[:, o, :].repeat(x_size, axis=1)
        return result_x, result_y

    def detector_response(self, quanta_type):
        """Return (n_events, |n_detected|) probability of observing the S[1|2]
        for different number of detected quanta.
        """
        ndet = self.domain(quanta_type + '_detected')

        o = np.newaxis
        observed = self.data[signal_name[quanta_type]].values[:, o]

        # Lookup signal gain mean and std per detected quanta
        mean_per_q = self.gimme(quanta_type + '_gain_mean')[:, o]
        std_per_q = self.gimme(quanta_type + '_gain_std')[:, o]

        mean = ndet * mean_per_q
        # add offset to std to avoid NaNs from norm.pdf if std = 0
        std = ndet ** 0.5 * std_per_q + 1e-10
        result = stats.norm.pdf(observed, loc=mean, scale=std)

        # Add detection/selection efficiency
        result *= self.gimme(signal_name[quanta_type] + '_acceptance',
                             observed)
        return result

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

        # Keep only dims we need
        d = data[list(set(sum(self.f_dims.values(), [])))]

        d = d.sample(n=len(energies), replace=True)

        def gimme(*args):
            return self.gimme(*args, data=d, params=params)

        d['energy'] = energies
        d['nq'] = np.floor(energies / gimme('work')).astype(np.int)
        d['p_el_mean'] = gimme('p_electron', d['nq'])
        d['p_el_fluct'] = gimme('p_electron_fluctuation', d['nq'])

        d['p_el_actual'] = stats.beta.rvs(
            *beta_params(d['p_el_mean'], d['p_el_fluct']))
        d['electron_produced'] = stats.binom.rvs(
            n=d['nq'],
            p=d['p_el_actual'])
        d['photon_produced'] = d['nq'] - d['electron_produced']

        for q in quanta_types:
            d[q + '_detected'] = stats.binom.rvs(
                n=d[q + '_produced'],
                p=gimme(q + '_detection_eff'))
            d[signal_name[q]] = stats.norm.rvs(
                loc=d[q + '_detected'] * gimme(q + '_gain_mean'),
                scale=d[q + '_detected']**0.5 * gimme(q + '_gain_std'))

        acceptance = np.ones(len(d))
        for q in quanta_types:
            acceptance *= gimme(q + '_acceptance', d[q + '_detected'])
            sn = signal_name[q]
            acceptance *= gimme(sn + '_acceptance', d[sn])
        d = d.iloc[np.random.rand(len(d)) < acceptance]

        return d


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
    """
    a, b = beta_params(p_mean, p_sigma)
    return np.exp(
        gammaln(n+1) + gammaln(x+a) + gammaln(n-x+b) + gammaln(a+b) -
        (gammaln(x+1) + gammaln(n-x+1) +
         gammaln(a) + gammaln(b) + gammaln(n+a+b)))
