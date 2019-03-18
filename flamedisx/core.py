import inspect
import types

import numpy as np
from scipy import stats
from scipy.special import gammaln
import pandas as pd

quanta_types = 'photon', 'electron'
signal_name = dict(photon='s1', electron='s2')

special_data_methods = [
    'energy_spectrum',
    'p_electron',
    'p_electron_fluctuation']

data_methods = special_data_methods + ['work']
hidden_vars_per_quanta = 'detection_eff gain_mean gain_std'.split()
for _qn in quanta_types:
    data_methods += [_qn + '_' + x for x in hidden_vars_per_quanta]


def _lookup_axis1(x, indices):
    """Return values of x at indices along axis 1"""
    d = indices
    return np.take_along_axis(
        x,
        d.reshape(len(d), -1), axis=1
    ).reshape(d.shape)


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

    @staticmethod
    def electron_detection_eff(drift_time,
                               *, elife=600, extraction_eff=1):
        return extraction_eff * np.exp(-drift_time / elife)

    photon_detection_eff = 0.1

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
            else:
                def const_maker(slf, x=f):
                    return x * np.ones(slf.n_evts)
                setattr(self,
                        fname,
                        types.MethodType(const_maker, self))

        if data is not None:
            self.set_data(data)
        self._params = params

    def gimme(self, fname, bonus_arg=None):
        if fname in special_data_methods:
            assert bonus_arg is not None
        else:
            assert bonus_arg is None

        args = [self.data[x].values for x in self.f_dims[fname]]
        if bonus_arg is not None:
            args = [bonus_arg] + args

        kwargs = {k: v for k, v in self._params
                  if k in self.f_params[fname]}

        return getattr(self, fname)(*args, **kwargs)

    def set_data(self, data, max_sigma=5):
        self.data = d = data
        self.n_evts = len(data)

        # Annotate data with eff, mean, sigma
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
        # TODO: set tighter bounds here
        # Maybe not needed? Just a 1d computation
        # If we do change it, have to change rate_nphnel computation
        # indexing to account for nq_min
        d['nq_min'] = 0
        d['nq_max'] = 2 * d['nq_mle'] + 5
        d['fel_mle'] = self.gimme('p_electron', d['nq_mle'])

        # Find plausble ranges for detected and observed quanta
        # based on the observed S1 and S2 sizes
        # (we could also derive ranges assuming the CES reconstruction,
        #  but these won't work well for outliers along one of the dimensions)
        # TODO: Meh, think about this, considering also computation cost
        # / space width
        for qi, qn in enumerate(quanta_types):
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

    def likelihood(self, **params):
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
                             _nq_1d / self.gimme('work')[:, np.newaxis])
        pel = self.gimme('p_electron', _nq_1d)
        pel_fluct = self.gimme('p_electron_fluctuation', _nq_1d)

        # Create tensors with the dimensions of our final result, containing:
        # ... numbers of photons and electrons produced:
        nph, nel = self.cross_domains('photon_produced', 'electron_produced')
        # ... numbers of total quanta produced
        nq = nel + nph
        # ... differential rate
        rate_nq = _lookup_axis1(rate_nq, nq)
        # ... probability of a quantum to become an electron
        pel = _lookup_axis1(pel, nq)
        # ... probability fluctuation
        pel_fluct = _lookup_axis1(pel_fluct, nq)

        # Finally, the main computation is simple:
        return rate_nq * beta_binom_pmf(
            nph.astype(np.int),
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
        return stats.binom.pmf(n_det, n=n_prod, p=p)

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
        return stats.norm.pdf(observed, loc=mean, scale=std)


def beta_binom_pmf(x, n, p_mean, p_sigma):
    """Return probability mass function of beta-binomial distribution.

    That is, give the probability of obtaining x successes in n trials,
    if the success probability p is drawn from a beta distribution
    with mean p_mean and standard deviation p_sigma.
    """
    # Convert (p_mean, p_sigma) to (alpha, beta) params of beta distribution
    # From Wikipedia:
    # variance = 1/(4 * (2 * beta + 1)) = 1/(8 * beta + 4)
    # mean = 1/(1+beta/alpha)
    b = (1/(8 * p_sigma**2) - 0.5)
    a = b * p_mean/(1 - p_mean)
    return np.exp(
        gammaln(n+1) + gammaln(x+a) + gammaln(n-x+b) + gammaln(a+b) -
        (gammaln(x+1) + gammaln(n-x+1) +
         gammaln(a) + gammaln(b) + gammaln(n+a+b)))
