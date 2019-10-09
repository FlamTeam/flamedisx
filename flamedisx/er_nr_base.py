"""Flamedisx implementation of the liquid xenon emission model

LXeSource: common parts of ER and NR response
ERSource: ER-specific model components and defaults
NRSource: NR-specific model components and defaults
"""
from functools import partial

from multihist import Hist1d, Histdd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import wimprates as wr
from scipy import stats

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis

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


@export
class LXeSource(fd.Source):
    data_methods = tuple(data_methods)
    special_data_methods = tuple(special_data_methods)
    inner_dimensions = (
        'nq',
        'photon_detected',
        'electron_detected',
        'photon_produced',
        'electron_produced')

    # tuple with columns needed from data
    # I guess we don't really need x y z by default, but they are just so nice
    # we should keep them around regardless.
    extra_needed_columns = tuple(['x', 'y', 'z', 'r', 'theta'])

    # Whether or not to simulate overdispersion in electron/photon split
    # (e.g. due to non-binomial recombination fluctuation)
    do_pel_fluct: bool

    tpc_radius = 47.9   # cm
    tpc_length = 97.6   # cm
    drift_velocity = 1.335 * 1e-4   # cm/ns

    # Uniform timestamps between 2016-09 and 2017-09
    t_start = pd.to_datetime('2016-09-13T12:00:00')
    t_stop = pd.to_datetime('2017-09-13T12:00:00')

    # Spatial rate multiplier histogram
    # Multihist Histdd object to lookup space dependent rate multipliers
    # The histogram must have 'axis_names' set to either
    # ['r', 'theta', 'z'] or ['x', 'y', 'z']
    # The histogram must be normalized to the histogram mean
    spatial_rate_hist = None

    def __init__(self, *args, **kwargs):
        # Check validity of spatial rate hist
        if self.spatial_rate_hist is not None:
            assert isinstance(self.spatial_rate_hist, Histdd), \
                "spatial_rate_hist needs to be a multihist Histdd object"
            # The histogram mean needs to be normalized to one, meaning the
            # histogram contains nbins/dim**ndims counts. This ensures we can
            # use tf.ones is no hist is defined.
            h = self.spatial_rate_hist.histogram
            assert np.allclose(h.mean() * h.size, self.spatial_rate_hist.n), \
                "spatial_rate_hist needs to be normalized to histogram mean"
            # Check histogram dimensions
            axes = self.spatial_rate_hist.axis_names
            assert axes == ['r', 'theta', 'z'] or axes == ['x', 'y', 'z'], \
                ("axis_names of spatial_rate_hist must be either "
                 "or ['r', 'theta', 'z'] or ['x', 'y', 'z']")
            self.spatial_rate_hist_dims = axes
        # Init rest of Source, this must be done after any checks on
        # spatial_rate_hist since it calls _populate_tensor_cache as well
        super().__init__(*args, **kwargs)

    def _populate_tensor_cache(self):
        super()._populate_tensor_cache()
        if self.spatial_rate_hist is not None:
            # Setup tensor of histogram for lookup
            positions = self.data[self.spatial_rate_hist_dims].values.T
            v = self.spatial_rate_hist.lookup(*positions)

            spatial_rate_tensor = tf.convert_to_tensor(v,
                                                       dtype=fd.float_type())
            self.spatial_rate_tensor = tf.reshape(spatial_rate_tensor,
                                                  [self.n_batches, -1])
        else:
            # If no hist defined, set uniform response
            self.spatial_rate_tensor = tf.ones([self.n_batches,
                                                self.batch_size],
                                               dtype=fd.float_type())

    ##
    # Model functions (data_methods)
    ##

    # Single constant energy spectrum
    def energy_spectrum(self, i_batch):
        # Lookup the energy spectrum
        es, rs = self._single_spectrum()
        # Lookup the spatial rate multiplier and scale the spectrum with it
        sr = self.spatial_rate_mult(i_batch)
        n = i_batch.shape[0]
        return (fd.repeat(es[o, :], n, axis=0),
                fd.repeat(rs[o, :], n, axis=0) * sr[:, o])

    work = 13.7e-3

    # Detection efficiencies
    @staticmethod
    def electron_detection_eff(drift_time, *, elife=452e3, extraction_eff=0.96):
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
        return tf.where((s1 < 2) | (s1 > 70),
                        tf.zeros_like(s1, dtype=fd.float_type()),
                        tf.ones_like(s1, dtype=fd.float_type()))

    @staticmethod
    def s2_acceptance(s2):
        return tf.where((s2 < 200) | (s2 > 6000),
                        tf.zeros_like(s2, dtype=fd.float_type()),
                        tf.ones_like(s2, dtype=fd.float_type()))

    @staticmethod
    def electron_gain_mean(z, *, g2=20):
        return g2 * tf.ones_like(z)

    electron_gain_std = 5.

    photon_gain_mean = 1.
    photon_gain_std = 0.5
    double_pe_fraction = 0.219

    ##
    # Simulation
    ##

    def random_truth(self, n_events, fix_truth=None, **params):
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        data = self.random_truth_observables(n_events)
        data = self._add_random_energies(data, n_events)

        if fix_truth is not None:
            # Override any keys with fixed values defined in fix_truth
            fix_truth = self.validate_fix_truth(fix_truth)
            for k, v in fix_truth.items():
                data[k] = v

        return pd.DataFrame(data)

    def _add_random_energies(self, data, n_events):
        """Draw n_events random energies from the energy spectrum
        and add them to the data dict.
        """
        es, rs = self._single_spectrum()
        energies = Hist1d.from_histogram(rs[:-1], es).get_random(n_events)
        data['energy'] = energies
        return data

    def validate_fix_truth(self, d):
        """Clean fix_truth, ensure all needed variables are present
           Compute derived variables.
        """
        if isinstance(d, pd.DataFrame):
            # TODO: Should we still support this case? User has no control
            # over which cols to set, why not only use dicts here?

            # When passing in an event as DataFrame we select and set
            # only these columns:
            cols = ['x', 'y', 'z', 'r', 'theta', 'event_time', 'drift_time']
            # Assume fix_truth is a one-line dataframe with at least
            # cols columns
            return d[cols].iloc[0].to_dict()
        else:
            assert isinstance(d, dict), \
                "fix_truth needs to be a DataFrame or dict"

        if all(v in d for v in ['x', 'y', 'z']):
            # fixed position specified in cartesian coords
            # Add cylindrical and drift_time
            d['r'], d['theta'] = fd.cart_to_pol(d['x'], d['y'])
            d['drift_time'] = - d['z']/ self.drift_velocity
        elif all(v in d for v in ['r', 'theta', 'z']):
            # fixed position specified in cylindrical coords
            # Add cartesian and drift_time
            d['x'], d['y'] = fd.pol_to_cart(d['r'], d['theta'])
            d['drift_time'] = - d['z']/ self.drift_velocity
        elif not any(v in d for v in ['event_time', 'energy']):
            # Neither cartesian nor polar coords given and no time or energy
            raise ValueError(f"Dict should contain at least ['x', 'y', 'z'] "
                             "and/or ['r', 'theta', 'z'] and/or 'event_time' "
                             "and/or 'energy', but it contains: {d.keys()}")
        return d

    def random_truth_observables(self, n_events):
        """Return dictionary with x, y, z, r, theta, drift_time
        and event_time randomly drawn.
        S1 and S2 placeholder values are added for set_data.
        Takes into account spatial rate multiplier of the source.
        """
        data = dict()
        # Add fake s1, s2 necessary for set_data to succeed
        # TODO: check if we still need this...
        data['s1'] = 1
        data['s2'] = 100

        if self.spatial_rate_hist is None:
            # Draw uniform position
            data['r'] = (np.random.rand(n_events) * self.tpc_radius**2)**0.5
            data['theta'] = np.random.uniform(0, 2*np.pi, size=n_events)
            data['z'] = - np.random.rand(n_events) * self.tpc_length
            data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])
        elif self.spatial_rate_hist_dims == ['r', 'theta', 'z']:
            # Spatial response in cylindrical coords
            positions = self.spatial_rate_hist.get_random(size=n_events)
            for idx, col in enumerate(self.spatial_rate_hist_dims):
                data[col] = positions[:, idx]
            data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])
        else:
            # Spatial response in cartesian coords
            positions = self.spatial_rate_hist.get_random(size=n_events)
            for idx, col in enumerate(self.spatial_rate_hist_dims):
                data[col] = positions[:, idx]
            data['r'], data['theta'] = fd.cart_to_pol(data['x'], data['y'])

        data['drift_time'] = - data['z']/ self.drift_velocity

        # Draw uniform time
        data['event_time'] = np.random.uniform(
            pd.Timestamp(self.t_start).value,
            pd.Timestamp(self.t_stop).value,
            size=n_events).astype('float32')
        return data

    ##
    # Emission model implementation
    ##

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
        # Finally, the main computation is simple:
        pel = tf.where(tf.math.is_nan(pel),
                       tf.zeros_like(pel, dtype=fd.float_type()),
                       pel)
        pel = tf.clip_by_value(pel, 1e-6, 1. - 1e-6)

        if self.do_pel_fluct:
            pel_fluct = self.gimme('p_electron_fluctuation', bonus_arg=_nq_1d,
                                   data_tensor=data_tensor, ptensor=ptensor)
            pel_fluct = fd.lookup_axis1(pel_fluct, _nq_ind)
            pel_fluct = tf.clip_by_value(pel_fluct, 1e-6, 1.)
            return rate_nq * fd.beta_binom_pmf(
                nel,
                n=nq,
                p_mean=pel,
                p_sigma=pel_fluct)

        else:
            return rate_nq * tfp.distributions.Binomial(
                total_count=nq, probs=pel).prob(nel)

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
    # Hidden variable bounds estimation
    ##

    def _q_det_clip_range(self, qn):
        return (self.min_s1_photons_detected if qn == 'photon'
                else self.min_s2_electrons_detected,
                None)

    def _annotate(self, _skip_bounds_computation=False):
        d = self.data

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

    ##
    # Simulation
    ##

    def _simulate_response(self):
        def gimme(f, bonus_arg=None):
            return self.gimme(f, bonus_arg=bonus_arg, numpy_out=True)
        d = self.data

        # If you forget the .values here, you may get a Python core dump...
        d['nq'] = self._simulate_nq(d['energy'].values)

        d['p_el_mean'] = gimme('p_electron', d['nq'].values)

        if self.do_pel_fluct:
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
            scale=d['electron_detected'] ** 0.5 * gimme('electron_gain_std'))

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
        return d.iloc[np.random.rand(len(d)) < acceptance].copy()

    def mu_before_efficiencies(self, **params):
        _, rs = self._single_spectrum()
        return np.sum(rs)

    def _simulate_nq(self, energies):
        raise NotImplementedError

    def _single_spectrum(self):
        raise NotImplementedError

    def spatial_rate_mult(self, i_batch):
        batch = tf.dtypes.cast(i_batch[0], dtype=fd.int_type())
        return self.spatial_rate_tensor[batch]

@export
class ERSource(LXeSource):
    do_pel_fluct = True

    ##
    # ER-specific model defaults
    ##

    def _single_spectrum(self):
        """Return (energies in keV, rate at these energies),
        """
        return (tf.cast(tf.linspace(0., 10., 1000),
                             dtype=fd.float_type()),
                tf.ones(1000, dtype=fd.float_type()))

    @staticmethod
    def p_electron(nq, *, er_pel_a=15, er_pel_b=-27.7, er_pel_c=32.5,
                   er_pel_e0=5.):
        """Fraction of ER quanta that become electrons
        Simplified form from Jelle's thesis
        """
        # The original model depended on energy, but in flamedisx
        # it has to be a direct function of nq.
        e_kev_sortof = nq * 13.7e-3
        eps = fd.tf_log10(e_kev_sortof / er_pel_e0 + 1e-9)
        qy = (
            er_pel_a * eps ** 2
            + er_pel_b * eps
            + er_pel_c)
        return fd.safe_p(qy * 13.7e-3)

    @staticmethod
    def p_electron_fluctuation(nq):
        # From SR0, BBF model, right?
        # q3 = 1.7 keV ~= 123 quanta
        return tf.clip_by_value(0.041 * (1. - tf.exp(-nq / 123.)),
                                1e-4,
                                float('inf'))

    @staticmethod
    def penning_quenching_eff(nph):
        return 1. + 0. * nph

    ##
    # ER Energy to quanta conversion
    ##

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

    def _simulate_nq(self, energies):
        # OK to use None, simulator has set defaults
        work = self.gimme('work', numpy_out=True, data_tensor=None, ptensor=None)
        return np.floor(energies / work).astype(np.int)


@export
class NRSource(LXeSource):
    do_pel_fluct = False
    data_methods = tuple(
        [x for x in data_methods if x != 'p_electron_fluctuation']
        + ['lindhard_l'])
    special_data_methods = tuple(special_data_methods + ['lindhard_l'])

    ##
    # NR-specific model defaults
    ##

    def _single_spectrum(self):
        """Return (energies in keV, events at these energies),
        both (n_events, n_energies) tensors.
        """
        e = tf.cast(tf.linspace(0.7, 150., 100),
                    fd.float_type())
        return e, tf.ones_like(e, dtype=fd.float_type())

    @staticmethod
    def lindhard_l(e, lindhard_k=tf.constant(0.138, dtype=fd.float_type())):
        """Return Lindhard quenching factor at energy e in keV"""
        eps = e * tf.constant(11.5 * 54.**(-7./3.), dtype=fd.float_type())  # Xenon: Z = 54

        n0 = tf.constant(3., dtype=fd.float_type())
        n1 = tf.constant(0.7, dtype=fd.float_type())
        n2 = tf.constant(1.0, dtype=fd.float_type())
        p0 = tf.constant(0.15, dtype=fd.float_type())
        p1 = tf.constant(0.6, dtype=fd.float_type())

        g = n0 * tf.pow(eps, p0) + n1 * tf.pow(eps, p1) + eps
        res = lindhard_k * g/(n2 + lindhard_k * g)
        return res

    def p_electron(self, nq, *,
            alpha=1.280, zeta=0.045, beta=273 * .9e-4,
            gamma=0.0141, delta=0.062,
            drift_field=120):
        """Fraction of detectable NR quanta that become electrons,
        slightly adjusted from Lenardo et al.'s global fit
        (https://arxiv.org/abs/1412.4417).

        Penning quenching is accounted in the photon detection efficiency.
        """
        # TODO: so to make field pos-dependent, override this entire f?
        # could be made easier...

        # prevent /0  # TODO can do better than this
        nq = nq + 1e-9

        # Note: final term depends on nq now, not energy
        # this means beta is different from lenardo et al
        nexni = alpha * drift_field ** -zeta * (1 - tf.exp(-beta * nq))
        ni = nq * 1 / (1 + nexni)

        # Fraction of ions NOT participating in recombination
        squiggle = gamma * drift_field ** -delta
        fnotr = tf.math.log(1 + ni * squiggle) / (ni * squiggle)

        # Finally, number of electrons produced..
        n_el = ni * fnotr

        return fd.safe_p(n_el / nq)

    @staticmethod
    def penning_quenching_eff(nph, eta=8.2e-5 * 3.3, labda=0.8 * 1.15):
        return 1. / (1. + eta * nph ** labda)

    ##
    # NR Energy to quanta conversion
    ##

    def rate_nq(self, nq_1d, data_tensor, ptensor):
        # (n_events, |ne|) tensors
        es, rate_e = self.gimme('energy_spectrum', data_tensor=data_tensor, ptensor=ptensor)
        mean_q_produced = (
                es
                * self.gimme('lindhard_l', bonus_arg=es,
                             data_tensor=data_tensor, ptensor=ptensor)
                / self.gimme('work',
                             data_tensor=data_tensor, ptensor=ptensor)[:, o])

        # (n_events, |nq|, |ne|) tensor giving p(nq | e)
        p_nq_e = tfp.distributions.Poisson(
            mean_q_produced[:, o, :]).prob(nq_1d[:, :, o])

        return tf.reduce_sum(p_nq_e * rate_e[:, o, :], axis=2)

    def _simulate_nq(self, energies):
        # OK to use None, simulator has set defaults
        work = self.gimme('work', data_tensor=None, ptensor=None, numpy_out=True)
        lindhard_l = self.gimme('lindhard_l',
                                bonus_arg=energies,
                                data_tensor=None, ptensor=None,
                                numpy_out=True)
        return stats.poisson.rvs(energies * lindhard_l / work)


@export
class WIMPSource(NRSource):
    """NRSource with time dependent energy spectra from
    wimprates.
    """
    extra_needed_columns = tuple(
        list(NRSource.extra_needed_columns)
        + ['t', 'event_time'])
    # Recoil energies and Wimprates settings
    es = np.geomspace(0.7, 50, 100)  # [keV]
    mw = 1e3  # GeV
    sigma_nucleon = 1e-45  # cm^2

    # Interpolator settings
    t_start = pd.to_datetime('2016-09-13T12:00:00')
    t_stop = pd.to_datetime('2017-09-13T12:00:00')
    n_in = 10  # Number of time bin edges (wimprates function evaluations + 1)

    def __init__(self, *args, wimp_kwargs=None, **kwargs):
        # Compute the energy spectrum in a given time range
        # Times used by wimprates are J2000 timestamps
        assert self.n_in > 1, \
            f"Number of time bin edges needs to be at least 2"
        times = np.linspace(wr.j2000(date=self.t_start),
                            wr.j2000(date=self.t_stop), self.n_in)
        time_centers = self.bin_centers(times)

        if wimp_kwargs is None:
            # No arguments given at all;
            # use default mass, xsec and energy range
            wimp_kwargs = dict(mw=self.mw,
                               sigma_nucleon=self.sigma_nucleon,
                               es=self.es)
        else:
            assert 'mw' in wimp_kwargs and 'sigma_nucleon' in wimp_kwargs, \
                "Pass at least 'mw' and 'sigma_nucleon' in wimp_kwargs"
            if 'es' not in wimp_kwargs:
                # Energies not given, use default energy bin edges
                wimp_kwargs['es'] = self.es

        es = wimp_kwargs['es']
        es_centers = self.bin_centers(es)
        del wimp_kwargs['es']  # To avoid confusion centers / edges

        # Transform wimp_kwargs to arguments that can be passed to wimprates
        # which means transforming es from edges to centers
        spectra = np.array([wr.rate_wimp_std(t=t, es=es_centers, **wimp_kwargs)
                            * np.diff(es)
                            for t in time_centers])
        assert spectra.shape == (len(time_centers), len(es_centers))

        self.energy_hist = Histdd.from_histogram(spectra,
                                                 bin_edges=(times, es))
        # Initialize the rest of the source, needs to be after energy_hist is
        # computed because of _populate_tensor_cache
        super().__init__(*args, **kwargs)

    def mu_before_efficiencies(self, **params):
        return self.energy_hist.n / (self.n_in - 1)

    @staticmethod
    def bin_centers(x):
        return 0.5 * (x[1:] + x[:-1])

    def to_event_time(self, jtimes):
        j_start = wr.j2000(date=self.t_start)
        j_stop = wr.j2000(date=self.t_stop)
        assert j_start < j_stop

        ev_time_start = pd.Timestamp(self.t_start).value
        ev_time_stop = pd.Timestamp(self.t_stop).value
        assert ev_time_start < ev_time_stop

        jfrac = (jtimes - j_start)/(j_stop - j_start)
        return jfrac * (ev_time_stop - ev_time_start) + ev_time_start

    def _populate_tensor_cache(self):
        super()._populate_tensor_cache()
        # Get energy bin centers
        e_bin_centers = self.energy_hist.bin_centers(axis=1)
        # Construct the energy spectra at event times
        e = np.array([self.energy_hist.slicesum(t).histogram
                      for t in self.data['t']])
        energy_tensor = tf.convert_to_tensor(e, dtype=fd.float_type())
        assert energy_tensor.shape == [len(self.data), len(e_bin_centers)], \
            f"{energy_tensor.shape} != {len(self.data)}, {len(e_bin_centers)}"
        self.energy_tensor = tf.reshape(energy_tensor,
                                        [self.n_batches, self.batch_size, -1])

        es_centers = tf.convert_to_tensor(e_bin_centers,
                                          dtype=fd.float_type())
        self.all_es_centers = fd.repeat(es_centers[o, :],
                                        repeats=self.batch_size,
                                        axis=0)

    def add_extra_columns(self, d):
        super().add_extra_columns(d)
        # Add J2000 timestamps to data for use with wimprates
        if 't' not in d:
            d['t'] = [wr.j2000(date=t)
                      for t in pd.to_datetime(d['event_time'])]

    def energy_spectrum(self, i_batch):
        """Return (energies in keV, events at these energies)
        """
        batch = tf.dtypes.cast(i_batch[0], dtype=fd.int_type())
        return (self.all_es_centers, self.energy_tensor[batch, :, :])

    def _add_random_energies(self, data, n_events):
        """Draw n_events random energies and times from the energy/
        time spectrum and add them to the data dict.
        """
        events = self.energy_hist.get_random(n_events)
        data['t'] = j2000_times = events[:, 0]
        data['energy'] = events[:, 1]
        data['event_time'] = self.to_event_time(j2000_times)
        return data
