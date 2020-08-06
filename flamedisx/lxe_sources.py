import numpy as np
from multihist import Histdd
import pandas as pd
import tensorflow as tf
import wimprates as wr

import flamedisx as fd
export, __all__ = fd.exporter()


@export
class ERSource(fd.BlockModelSource):
    model_blocks = (
        fd.FixedShapeEnergySpectrum,
        fd.MakeERQuanta,
        fd.MakePhotonsElectronsBetaBinomial,
        fd.DetectPhotons,
        fd.MakeS1Photoelectrons,
        fd.MakeS1,
        fd.DetectElectrons,
        fd.MakeS2)

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

    final_dimensions = ('s1', 's2')


@export
class NRSource(fd.BlockModelSource):
    model_blocks = (
        fd.FixedShapeEnergySpectrum,
        fd.MakeNRQuanta,
        fd.MakePhotonsElectronsBinomial,
        fd.DetectPhotons,
        fd.MakeS1Photoelectrons,
        fd.MakeS1,
        fd.DetectElectrons,
        fd.MakeS2)

    final_dimensions = ('s1', 's2')

    # Use a larger default energy range, since most energy is lost
    # to heat.
    energies = tf.cast(tf.linspace(0.7, 150., 100),
                       fd.float_type())
    rates_vs_energy = tf.ones(100, fd.float_type())

    @staticmethod
    def p_electron(nq, *,
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


@export
class WIMPSource(NRSource):
    """Spin-independent dark matter source.

    This includes spectrum time-dependence (annual modulation) effects,
    though in most cases the effect of this should be extremely minor.
    """
    # Energy spectrum is variable (depends on event_time)
    # and frozen (no inference parameters, but we can implement it in numpy)
    model_blocks = tuple(
        [fd.VariableEnergySpectrum]
        + list(NRSource.model_blocks[1:]))
    frozen_model_functions = ('energy_spectrum',)

    # If set to True, the energy spectrum at each time will be set to its
    # average over the data taking period.
    pretend_wimps_dont_modulate = False

    mw = 1e3  # GeV
    sigma_nucleon = 1e-45  # cm^2
    n_time_bins = 24

    # Do not use energies, it has a special meaning (see energy_spectrum.py)
    energy_edges = np.geomspace(0.7, 50, 100)

    def __init__(self, *args, wimp_kwargs=None, **kwargs):
        # Set defaults for wimp_kwargs
        if wimp_kwargs is None:
            # No arguments given at all;
            # use default mass, xsec and energy range
            wimp_kwargs = dict(mw=self.mw,
                               sigma_nucleon=self.sigma_nucleon,
                               energy_edges=self.energy_edges)
        else:
            # 'es' was renamed to energy_edges
            if 'es' in wimp_kwargs:
                wimp_kwargs['energy_edges'] = wimp_kwargs['es']
                del wimp_kwargs['es']
            assert 'mw' in wimp_kwargs and 'sigma_nucleon' in wimp_kwargs, \
                "Pass at least 'mw' and 'sigma_nucleon' in wimp_kwargs"
            if 'energy_edges' not in wimp_kwargs:
                # Energies not given, use default energy bin edges
                wimp_kwargs['energy_edges'] = self.energy_edges

        e_centers = self.bin_centers(wimp_kwargs['energy_edges'])
        self.energies = fd.np_to_tf(e_centers)

        self.array_columns = (('energy_spectrum', len(self.energy_edges) - 1),)

        # Now that array_columns and energies are set, we can build the source
        # properly
        self.build_source_from_blocks()

        times = np.linspace(wr.j2000(self.t_start.value),
                            wr.j2000(self.t_stop.value),
                            self.n_time_bins + 1)
        time_centers = self.bin_centers(times)

        # Transform wimp_kwargs to arguments that can be passed to wimprates
        # which means transforming es from edges to centers
        del wimp_kwargs['energy_edges']
        spectra = np.array([wr.rate_wimp_std(t=t,
                                             es=e_centers,
                                             **wimp_kwargs)
                            * np.diff(self.energy_edges)
                            for t in time_centers])
        assert spectra.shape == (len(time_centers), len(e_centers))

        self.energy_hist = Histdd.from_histogram(
            spectra,
            bin_edges=(times, self.energy_edges))

        if self.pretend_wimps_dont_modulate:
            self.energy_hist.histogram = (
                np.ones_like(self.energy_hist.histogram)
                * self.energy_hist.sum(axis=0).histogram.reshape(1, -1)
                / self.n_time_bins)

        super().__init__(*args, **kwargs)

    def energy_spectrum(self, event_time):
        t_j2000 = wr.j2000(fd.tf_to_np(event_time))
        result = np.stack([self.energy_hist.slicesum(t).histogram
                           for t in t_j2000])
        return fd.np_to_tf(result)

    def mu_before_efficiencies(self, **params):
        return self.energy_hist.n / self.n_time_bins

    def random_truth(self, n_events, fix_truth=None, **params):
        """Draw n_events random energies and times from the energy/
        time spectrum and add them to the data dict.
        """
        data = self.draw_positions(n_events)

        if 'event_time' in fix_truth:
            # Time is fixed, so the energy spectrum differs.
            # (if energy is also fixed, it will just be overridden later
            #  and we're doing a bit of unnecessary work here)
            t = wr.j2000(fix_truth['event_time'])
            t_edges = self.energy_hist.bin_edges[0]
            assert t_edges[0] <= t < t_edges[-1], "fix_truth time out of bounds"
            data['energy'] = \
                self.energy_hist \
                    .slicesum(t, axis=0) \
                    .get_random(n_events)

        elif 'energy' in fix_truth:
            # Energy is fixed, so the time distribution differs.
            e_edges = self.energy_hist.bin_edges[1]
            assert e_edges[0] <= fix_truth['energy'] < e_edges[-1], \
                "fix_truth energy out of bounds"
            times = \
                self.energy_hist \
                    .slicesum(fix_truth['energy'], axis=1) \
                    .get_random(n_events)
            data['event_time'] = fd.j2000_to_event_time(times)

        else:
            times, data['energy'] = self.energy_hist.get_random(n_events).T
            data['event_time'] = fd.j2000_to_event_time(times)

        # Overwrite all fixed truth keys
        self._overwrite_fixed_truths(data, fix_truth, n_events)

        return pd.DataFrame(data)

    @staticmethod
    def bin_centers(x):
        return 0.5 * (x[1:] + x[:-1])


@export
class SpatialRateHistogramSource(fd.BlockModelSource):
    """Source whose rate multiplier is specified by a spatial histogram.

    The histogram can be in polar (r, theta, z) or Cartesian (x, y, z)
    coordinates. The coordinates are reconstructed positions.
    """

    frozen_model_functions = ('energy_spectrum_rate_multiplier',)

    spatial_rate_hist: Histdd
    spatial_rate_bin_volumes: Histdd

    final_dimensions = ('s1', 's2')

    def __init__(self, *args, **kwargs):
        self.build_source_from_blocks()

        # Check we actually have the histograms
        for attribute, kind in [('spatial_rate_hist', Histdd),
                                ('spatial_rate_bin_volumes', np.ndarray)]:
            assert hasattr(self, attribute), f"{attribute} missing"
            assert isinstance(getattr(self, attribute), kind), \
                f"{attribute} should be a {kind}"

        # Are we Cartesian, polar, or in trouble?
        axes = tuple(self.spatial_rate_hist.axis_names)
        self.polar = (axes == ('r', 'theta', 'z'))
        if not self.polar:
            assert axes == ('x', 'y', 'z'), \
                ("axis_names of spatial_rate_hist must be either "
                 "or ['r', 'theta', 'z'] or ['x', 'y', 'z']")

        # Correctly scale the events/bin histogram E to make the pdf R
        # histogram, taking into account (non uniform) bin volumes. This
        # ensures we don't need to modify mu_before_efficiencies.
        # R = E / bv
        # R_norm = (E / sum E) / (bv / sum bv)
        # R_norm = (E / bv) * (sum bv / sum E)
        bv = self.spatial_rate_bin_volumes
        E = self.spatial_rate_hist.histogram
        R_norm = (E / bv) * (bv.sum() / E.sum())

        self.spatial_rate_pdf = self.spatial_rate_hist.similar_blank_hist()
        self.spatial_rate_pdf.histogram = R_norm

        # Check the spatial rate histogram was properly normalized.
        # It's easy to mis-normalize by accident, don't cause any surprises:
        assert 0.9999 < np.mean(E / bv) < 1.0001, \
            "Spatial rate histogram is not normalized correctly"

        super().__init__(*args, **kwargs)

    def energy_spectrum_rate_multiplier(self, x, y, z):
        if self.polar:
            positions = list(fd.cart_to_pol(x, y)) + [z]
        else:
            positions = [x, y, z]
        return self.spatial_rate_pdf.lookup(*positions)

    def draw_positions(self, n_events):
        data = dict()
        positions = self.spatial_rate_hist.get_random(size=n_events)
        for idx, col in enumerate(self.spatial_rate_hist.axis_names):
            data[col] = positions[:, idx]
        if self.polar:
            data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])
        else:
            data['r'], data['theta'] = fd.cart_to_pol(data['x'], data['y'])
        return data
