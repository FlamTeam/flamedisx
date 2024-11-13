from multihist import Histdd
import numpy as np
import pandas as pd
import tensorflow as tf
import wimprates as wr

from scipy import stats

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis

@export
class EnergySpectrum(fd.FirstBlock):
    dimensions = ('energy',)
    model_attributes = (
        'energies','max_dim_size',
        'radius', 'z_top', 'z_bottom', 'z_topDrift',
        'drift_velocity',
        't_start', 't_stop',
        'drift_map_dt', 'drift_map_x')

    # The default boundaries are at points where the WIMP wind is at its
    # average speed.
    # This will then also be true at the midpoint of these times.
    t_start = pd.to_datetime('2021-12-23T09:37:51')
    t_start = t_start.tz_localize(tz='America/Denver')

    t_stop = pd.to_datetime('2022-04-18T07:58:01')
    t_stop = t_stop.tz_localize(tz='America/Denver')

    # Just a dummy 0-10 keV spectrum
    energies = tf.cast(tf.linspace(0., 10., 1000),
                       dtype=fd.float_type())
    default_size=100
    max_dim_size : dict
    def setup(self):
        """
           Setups up a managable max dim size
        """
        assert isinstance(self.max_dim_size,dict)
        self.max_dim_size={'energy':self.max_dim_size['energy']}

    drift_map_dt = None
    def derive_drift_time(self, data):
        """
            Helper function for getting drift time from true z
            data: Data DataFrame
        """
        if self.drift_map_dt is None:
            return (self.z_topDrift - data['z']) / self.drift_velocity

        return self.drift_map_dt(np.array([data['r'], data['z']]).T)

    drift_map_x = None
    def derive_observed_xy(self, data):
        """
        """
        if self.drift_map_x is None:
            return data['x'], data['y']

        cosTheta = data['x'] / data['r']
        sinTheta = data['y'] / data['r']

        x_obs = self.drift_map_x(np.array([data['r'], data['z']]).T) * cosTheta
        y_obs = self.drift_map_x(np.array([data['r'], data['z']]).T) * sinTheta

        return x_obs, y_obs

    def domain(self, data_tensor):
        assert isinstance(self.energies, tf.Tensor)  # see WIMPsource for why

        left_bound = tf.reduce_min(self.source._fetch('energy_min', data_tensor=data_tensor))
        right_bound = tf.reduce_max(self.source._fetch('energy_max', data_tensor=data_tensor))
        bool_mask = tf.logical_and(tf.greater_equal(self.energies, left_bound),
                                   tf.less_equal(self.energies, right_bound))
        # Trim the energy spectrum within the bounds
        energies_trim = tf.boolean_mask(self.energies, bool_mask)
        if 'energy' not in self.source.no_step_dimensions:
            index_step = tf.round(tf.linspace(0, tf.shape(energies_trim)[0] - 1,
                                              tf.math.minimum(tf.shape(energies_trim),
                                                              self.source.max_dim_sizes['energy'])[0]))
            # Variable stepping along the trimmed energy spectrum
            energies_trim_step = tf.gather(energies_trim, tf.cast(index_step, fd.int_type()))
        else:
            energies_trim_step = energies_trim

        return {self.dimensions[0]: tf.repeat(energies_trim_step[o, :],
                                              self.source.batch_size,
                                              axis=0)}

    def _annotate(self, d):
        # Generate an MC reservoir for obtaining energy bounds. Also use this for Bayes bounds priors
        n_events = self.source.n_events
        if self.source.mc_reservoir.empty:
            self.source.mc_reservoir = self.source.simulate(int(1e6), keep_padding=True)
        self.source.n_events = n_events
        assert not self.source.mc_reservoir.empty, \
            "MC reservoir used in energy bounds computation is empty. Are your cuts too tight?"

        energy = self.source.mc_reservoir.columns.get_loc('energy')
        electrons_produced = self.source.mc_reservoir.columns.get_loc('electrons_produced')
        photons_produced = self.source.mc_reservoir.columns.get_loc('photons_produced')
        res = self.source.mc_reservoir.values

        # Same energy bounds for all events within a batch
        for batch in range(self.source.n_batches):
            electrons_produced_min = min(d['electrons_produced_min'][
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size])
            electrons_produced_max = max(d['electrons_produced_max'][
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size])

            photons_produced_min = min(d['photons_produced_min'][
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size])
            photons_produced_max = max(d['photons_produced_max'][
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size])

            # We filter the reservoir energies by flat-prior Bayes bounds on electrons/photons produced
            energies = res[:, energy][(res[:, electrons_produced] >= electrons_produced_min)
                                      & (res[:, electrons_produced] <= electrons_produced_max)
                                      & (res[:, photons_produced] >= photons_produced_min)
                                      & (res[:, photons_produced] <= photons_produced_max)]

            if len(energies) == 0:
                self.source.data.loc[batch * self.source.batch_size:
                                     (batch + 1) * self.source.batch_size - 1, 'energy_min'] = \
                    min(res[:, energy])
                self.source.data.loc[batch * self.source.batch_size:
                                     (batch + 1) * self.source.batch_size - 1, 'energy_max'] = \
                    max(res[:, energy])
            else:
                # We use this filtered reservoir to estimate energy bounds
                self.source.data.loc[batch * self.source.batch_size:
                                     (batch + 1) * self.source.batch_size - 1, 'energy_min'] = \
                    np.quantile(energies, 0.)
                self.source.data.loc[batch * self.source.batch_size:
                                     (batch + 1) * self.source.batch_size - 1, 'energy_max'] = \
                    np.quantile(energies, 1. - self.source.bounds_prob)

    def draw_positions(self, n_events, **params):
        """Return dictionary with x, y, z, r, theta, drift_time
        randomly drawn.
        """
        data = dict()
        data['r'] = (np.random.rand(n_events) * self.radius**2)**0.5
        data['theta'] = np.random.uniform(0, 2*np.pi, size=n_events)
        data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])
        data['z'] = np.random.uniform(self.z_bottom, self.z_top,
                                      size=n_events)

        data['x_obs'], data['y_obs'] = self.derive_observed_xy(data)
        data['r_obs'], data['theta_obs'] = fd.cart_to_pol(data['x_obs'], data['y_obs'])

        data['drift_time'] = self.derive_drift_time(data)
        data.pop('z')

        return data

    def draw_time(self, n_events, **params):
        """Return n_events event_times drawn uniformaly
        between t_start and t_stop"""
        return np.random.uniform(
            self.t_start.value,
            self.t_stop.value,
            size=n_events)

    def random_truth(self, n_events, fix_truth=None, **params):
        """Return pandas dataframe with event positions and times
        randomly drawn.
        """
        data = self.draw_positions(n_events, **params)
        data['event_time'] = self.draw_time(n_events, **params)

        # The energy spectrum is represented as a 'comb' of delta functions
        # in the differential rate calculation. To get simulator-data agreement,
        # we have to do the same thing here.
        # TODO: can we fix both to be better?
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy)
        assert len(spectrum_numpy) == len(self.energies), \
            "Energies and spectrum have different length"

        data['energy'] = np.random.choice(
            fd.tf_to_np(self.energies),
            size=n_events,
            p=spectrum_numpy / spectrum_numpy.sum(),
            replace=True)
        assert np.all(data['energy'] >= 0), "Generated negative energies??"

        # For a constant-shape spectrum, fixing truth values is easy:
        # we just overwrite the simulated values.
        # Fixing one does not constrain any of the others.
        self.source._overwrite_fixed_truths(data, fix_truth, n_events)

        data = pd.DataFrame(data)
        return data

    def validate_fix_truth(self, d):
        """Clean fix_truth, ensure all needed variables are present
           Compute derived variables.
        """
        # When passing in an event as DataFrame we select and set
        # only these columns:
        cols = ['x', 'y', 'r', 'theta', 'event_time', 'drift_time']
        if d is None:
            return dict()
        elif isinstance(d, pd.DataFrame):
            # This is useful, since it allows you to fix_truth with an
            # observed event.
            # Assume fix_truth is a one-line dataframe with at least
            # cols columns
            return d[cols].iloc[0].to_dict()
        elif isinstance(d, pd.Series):
            # This is useful, since it allows you to fix_truth with an
            # observed event.
            # Assume fix_truth is a one-line series with at least
            # cols columns
            return d[cols].to_dict()
        else:
            assert isinstance(d, dict), \
                "fix_truth needs to be a DataFrame, dict, or None"
        if 'z' in d and 'drift_time' not in d:
            if 'x' in d and 'y' in d:
                d['r'], d['theta'] = fd.cart_to_pol(d['x'], d['y'])
            if 'r' not in d:
                raise ValueError("Require either (x,y,z) or (r,z) to derive drift time")
            d['drift_time']=self.derive_drift_time(d)

        if 'drift_time' in d:
            # Position is fixed. Ensure both Cartesian and polar coordinates
            # are available.
            if 'x' in d and 'y' in d:
                d['r'], d['theta'] = fd.cart_to_pol(d['x'], d['y'])
            elif 'r' in d and 'theta' in d:
                d['x'], d['y'] = fd.pol_to_cart(d['r'], d['theta'])
            if 'x' in d and 'y' in d:
                if d['r']>0:
                    d['x_obs'],d['y_obs']=self.derive_observed_xy(d)
                else:
                    d['x_obs'],d['y_obs'],d['r_obs'],d['theta_obs']=0,0,0,0
            elif 'x_obs' in d and 'y_obs' in d:
                d['r_obs'], d['theta_obs'] = fd.cart_to_pol(d['x_obs'], d['y_obs'])
            elif 'r_obs' in d and 'theta_obs' in d:
                d['x_obs'], d['y_obs'] = fd.pol_to_cart(d['r_obs'], d['theta_obs'])
            else:
                raise ValueError("When fixing position, give (x/_obs, x/_obs, z/drift_time), "
                                 "or (r/_obs, theta/_obs, z/drift_time)")
            
                
        elif 'event_time' not in d and 'energy' not in d:
            # Neither position, time, nor energy given
            raise ValueError(f"Dict should contain at least ['x', 'y', 'drift_time'] "
                             "and/or ['r', 'theta', 'drift_time'] and/or 'event_time' "
                             f"and/or 'energy', but it contains: {d.keys()}")
        return d

    def _compute(self, data_tensor, ptensor, **kwargs):
        raise NotImplementedError

    def mu_before_efficiencies(self, **params):
        raise NotImplementedError


@export
class FixedShapeEnergySpectrum(EnergySpectrum):
    """For a source whose energy spectrum has the same shape
    throughout space and time.

    If you add a rate variation with space, you must override draw_positions
     and mu_before_efficiencies.
     If you add a rate variation with time, you must override draw_times.
     If you add a rate variation depending on both space and time, you must
     override all of random_truth!

    By default, this uses a flat 0 - 10 keV spectrum, sampled at 1000 points.
    """

    model_attributes = ('rates_vs_energy',) + EnergySpectrum.model_attributes
    model_functions = ('energy_spectrum_rate_multiplier',)

    rates_vs_energy = tf.ones(1000, dtype=fd.float_type())

    energy_spectrum_rate_multiplier = 1.

    def _compute(self, data_tensor, ptensor, *, energy):
        # We want to do the same trimming/stepping treatment to the values of the energy
        # spectrum itself as we do its domain
        left_bound = tf.reduce_min(self.source._fetch('energy_min', data_tensor=data_tensor))
        right_bound = tf.reduce_max(self.source._fetch('energy_max', data_tensor=data_tensor))
        bool_mask = tf.logical_and(tf.greater_equal(self.energies, left_bound),
                                   tf.less_equal(self.energies, right_bound))
        spectrum_trim = tf.boolean_mask(self.rates_vs_energy, bool_mask)
        if 'energy' not in self.source.no_step_dimensions:
            index_step = tf.round(tf.linspace(0, tf.shape(spectrum_trim)[0] - 1,
                                              tf.math.minimum(tf.shape(spectrum_trim),
                                                              self.source.max_dim_sizes['energy'])[0]))
            spectrum_trim_step = tf.gather(spectrum_trim, tf.cast(index_step, fd.int_type()))
        else:
            spectrum_trim_step = spectrum_trim
        stepping_multiplier = tf.cast(tf.shape(spectrum_trim) / tf.shape(spectrum_trim_step), fd.float_type())

        spectrum = tf.repeat(spectrum_trim_step[o, :] * stepping_multiplier,
                             self.source.batch_size,
                             axis=0)
        rate_multiplier = self.gimme('energy_spectrum_rate_multiplier',
                                     data_tensor=data_tensor, ptensor=ptensor)
        return spectrum * rate_multiplier[:, o]

    def mu_before_efficiencies(self, **params):
        return np.sum(fd.np_to_tf(self.rates_vs_energy))

#Relics from before dimsize was an attribute-maybe this can be changed?
@export
class FixedShapeEnergySpectrumNR(FixedShapeEnergySpectrum):
    def setup(self):
        self.max_dim_size=self.max_dim_size
@export
class FixedShapeEnergySpectrumER(FixedShapeEnergySpectrum):
    def setup(self):
        self.max_dim_size=self.max_dim_size



@export
class FixedShapeEnergySpectrumFaster(FixedShapeEnergySpectrum):
    max_dim_size = {'energy': 50}


@export
class SpatialRateEnergySpectrum(FixedShapeEnergySpectrum):
    model_attributes = (('spatial_hist',)
                        + FixedShapeEnergySpectrum.model_attributes)
    frozen_model_functions = ('energy_spectrum_rate_multiplier',)

    spatial_hist: Histdd

    def setup(self):
        assert isinstance(self.spatial_hist, Histdd)

        # Are we Cartesian (x, y, dt), polar (r, theta, dt), 2D (r, dt),
        # or in trouble?
        axes = tuple(self.spatial_hist.axis_names)
        self.polar = (axes == ('r', 'theta', 'drift_time'))
        self.r_dt = (axes == ('r', 'drift_time'))

        self.bin_volumes = self.spatial_hist.bin_volumes()
        # Volume element in cylindrical coords = r * (dr dq dz)
        if self.polar:
            self.bin_volumes *= self.spatial_hist.bin_centers('r')[:, None, None]
        elif (self.r_dt):
            self.bin_volumes *= self.spatial_hist.bin_centers('r')[:, None]
        else:
            assert (axes == ('x', 'y', 'drift_time')), \
                ("axis_names of spatial_rate_hist must be "
                 "['r', 'theta', 'drift_time'], ['r', 'drift_time'] "
                 "or ['x', 'y', 'drift_time']")

        # Normalize the histogram
        self.spatial_hist.histogram = \
            self.spatial_hist.histogram.astype(float) / self.spatial_hist.n

        # Local rate multiplier = PDF / uniform PDF
        # = ((normed_hist/bin_volumes) / (1/total_volume))
        self.local_rate_multiplier = self.spatial_hist.similar_blank_hist()
        self.local_rate_multiplier.histogram = (
            (self.spatial_hist.histogram / self.bin_volumes)
            * self.bin_volumes.sum())

    def energy_spectrum_rate_multiplier(self, x, y, drift_time):
        if self.polar:
            positions = list(fd.cart_to_pol(x, y)) + [drift_time]
        elif self.r_dt:
            positions = [fd.cart_to_pol(x, y)[0]] + [drift_time]
        else:
            positions = [x, y, drift_time]
        return self.local_rate_multiplier.lookup(*positions)

    def draw_positions(self, n_events, **params):
        """Return dictionary with x, y, z, r, theta, drift_time
        drawn from the spatial rate histogram.
        """
        data = dict()
        positions = self.spatial_hist.get_random(size=n_events)
        for idx, col in enumerate(self.spatial_hist.axis_names):
            data[col] = positions[:, idx]
        if self.polar:
            data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])
        elif self.r_dt:
            theta = np.random.uniform(0, 2*np.pi, size=n_events)
            data['x'], data['y'] = fd.pol_to_cart(data['r'], theta)
        else:
            data['r'], data['theta'] = fd.cart_to_pol(data['x'], data['y'])

        return data


@export
class TemporalRateEnergySpectrumDecay(FixedShapeEnergySpectrum):
    model_attributes = (('time_constant_ns',)
                        + FixedShapeEnergySpectrum.model_attributes)
    frozen_model_functions = ('energy_spectrum_rate_multiplier',)

    def local_rate_multiplier(self, event_time):
        pdf = np.exp(-(event_time - self.t_start.value) / self.time_constant_ns)
        normalisation = 1. / (self.time_constant_ns * (1. - np.exp(-(self.t_stop.value - self.t_start.value) / self.time_constant_ns)))
        uniform_pdf = 1. / (self.t_stop.value - self.t_start.value)

        return normalisation * pdf / uniform_pdf

    def energy_spectrum_rate_multiplier(self, event_time):
        return self.local_rate_multiplier(event_time)

    def draw_time(self, n_events, **params):
        """
        """
        b = (self.t_stop.value - self.t_start.value) / self.time_constant_ns
        return stats.truncexpon.rvs(b,
                                    loc=self.t_start.value, scale=self.time_constant_ns,
                                    size=n_events)


@export
class SpatialTemporalRateEnergySpectrumDecay(SpatialRateEnergySpectrum):
    model_attributes = (('time_constant_ns',)
                        + SpatialRateEnergySpectrum.model_attributes)

    def temporal_rate_multiplier(self, event_time):
        pdf = np.exp(-(event_time - self.t_start.value) / self.time_constant_ns)
        normalisation = 1. / (self.time_constant_ns * (1. - np.exp(-(self.t_stop.value - self.t_start.value) / self.time_constant_ns)))
        uniform_pdf = 1. / (self.t_stop.value - self.t_start.value)

        return normalisation * pdf / uniform_pdf

    def energy_spectrum_rate_multiplier(self, x, y, z, event_time):
        if self.polar:
            positions = list(fd.cart_to_pol(x, y)) + [z]
        elif self.r_z:
            positions = [fd.cart_to_pol(x, y)[0]] + [z]
        elif self.r_dt:
            dt = (self.z_topDrift - z) / self.drift_velocity
            positions = [fd.cart_to_pol(x, y)[0]] + [dt]
        else:
            positions = [x, y, z]
        return self.local_rate_multiplier.lookup(*positions) * self.temporal_rate_multiplier(event_time)

    def draw_time(self, n_events, **params):
        """
        """
        b = (self.t_stop.value - self.t_start.value) / self.time_constant_ns
        return stats.truncexpon.rvs(b,
                                    loc=self.t_start.value, scale=self.time_constant_ns,
                                    size=n_events)


@export
class TemporalRateEnergySpectrumOscillation(FixedShapeEnergySpectrum):
    model_attributes = (('n_time_bins', 'period_ns', 'amplitude', 'phase_ns')
                        + FixedShapeEnergySpectrum.model_attributes)
    frozen_model_functions = ('energy_spectrum_rate_multiplier',)

    n_time_bins = 24

    def setup(self):
        times = np.linspace(self.t_start.value,
                            self.t_stop.value,
                            self.n_time_bins + 1)
        time_centers = 0.5 * (times[1:] + times[:-1])

        weights = 1. + self.amplitude * np.cos(2. * np.pi * (time_centers - self.phase_ns) / self.period_ns)

        self.temporal_hist = Histdd.from_histogram(weights, bin_edges=(times,))

    def local_rate_multiplier(self, event_time):
        pdf = 1. + self.amplitude * np.cos(2. * np.pi * (event_time - self.phase_ns) / self.period_ns)
        normalisation = 1. / ((self.t_stop.value - self.t_start.value) + self.amplitude * self.period_ns / (2. * np.pi) * \
            (np.sin(2. * np.pi * (self.t_stop.value - self.phase_ns) / self.period_ns) - \
            np.sin(2. * np.pi * (self.t_start.value - self.phase_ns) / self.period_ns)))
        uniform_pdf = 1. / (self.t_stop.value - self.t_start.value)

        return normalisation * pdf / uniform_pdf

    def energy_spectrum_rate_multiplier(self, event_time):
        return self.local_rate_multiplier(event_time)

    def draw_time(self, n_events, **params):
        """
        """
        return self.temporal_hist.get_random(size=n_events)[:, 0]


@export
class SpatialRateEnergySpectrumNR(SpatialRateEnergySpectrum):
    max_dim_size = {'energy': 150}


@export
class SpatialRateEnergySpectrumER(SpatialRateEnergySpectrum):
    max_dim_size = {'energy': 100}


@export
class TemporalRateEnergySpectrumDecayNR(TemporalRateEnergySpectrumDecay):
    max_dim_size = {'energy': 150}


@export
class TemporalRateEnergySpectrumDecayER(TemporalRateEnergySpectrumDecay):
    max_dim_size = {'energy': 100}


@export
class SpatialTemporalRateEnergySpectrumDecayNR(SpatialTemporalRateEnergySpectrumDecay):
    max_dim_size = {'energy': 150}


@export
class SpatialTemporalRateEnergySpectrumDecayER(SpatialTemporalRateEnergySpectrumDecay):
    max_dim_size = {'energy': 100}


@export
class TemporalRateEnergySpectrumOscillationNR(TemporalRateEnergySpectrumOscillation):
    max_dim_size = {'energy': 150}


@export
class TemporalRateEnergySpectrumOscillationER(TemporalRateEnergySpectrumOscillation):
    max_dim_size = {'energy': 100}



##
# Variable spectra
##

@export
class VariableEnergySpectrum(EnergySpectrum):
    """For a source for which the entire energy spectrum (not just the rate)
     depends on observables (e.g. reconstruction position or time).

    You must implement both energy_spectrum and random_truth.

    Note that you cannot draw homogeneous positions/times first, then
     energies, if your energy spectrum depends on positions/time.
    """

    model_functions = ('energy_spectrum',)

    def energy_spectrum(self, event_time):
        # Note this returns a 2d tensor!
        return tf.ones(len(event_time), len(self.energies),
                       dtype=fd.float_type())

    def _compute(self, data_tensor, ptensor, *, energy):
        # We want to do the same trimming/stepping treatment to the values of the energy
        # spectrum itself as we do its domain
        left_bound = tf.reduce_min(self.source._fetch('energy_min', data_tensor=data_tensor))
        right_bound = tf.reduce_max(self.source._fetch('energy_max', data_tensor=data_tensor))
        bool_mask = tf.logical_and(tf.greater_equal(self.energies, left_bound),
                                   tf.less_equal(self.energies, right_bound))
        spectrum_trim = tf.boolean_mask(self.gimme('energy_spectrum',
                                        data_tensor=data_tensor, ptensor=ptensor),
                                        bool_mask,
                                        axis=1)
        if 'energy' not in self.source.no_step_dimensions:
            index_step = tf.round(tf.linspace(0, tf.shape(spectrum_trim)[1] - 1,
                                              tf.math.minimum(tf.shape(spectrum_trim)[1],
                                                              self.source.max_dim_sizes['energy'])))
            spectrum_trim_step = tf.gather(spectrum_trim, tf.cast(index_step, fd.int_type()), axis=1)
        else:
            spectrum_trim_step = spectrum_trim
        stepping_multiplier = tf.cast(tf.shape(spectrum_trim)[1] / tf.shape(spectrum_trim_step)[1], fd.float_type())

        spectrum = spectrum_trim_step * stepping_multiplier[o, o]

        return spectrum

    def random_truth(self, n_events, fix_truth=None, **params):
        raise NotImplementedError

    def mu_before_efficiencies(self, **params):
        raise NotImplementedError


@export
class InvalidEventTimes(Exception):
    pass


@export
class WIMPEnergySpectrum(VariableEnergySpectrum):
    max_dim_size = {'energy': 150}

    model_attributes = ('n_time_bins',
                        'energy_hist') + VariableEnergySpectrum.model_attributes

    frozen_model_functions = ('energy_spectrum',)

    def energy_spectrum(self, event_time):
        ts = fd.tf_to_np(event_time)
        ts = wr.j2000(ts)
        ts = self.clip_j2000_times(ts)

        result = np.stack([self.energy_hist.slicesum(t).histogram
                           for t in ts])
        return fd.np_to_tf(result)

    def clip_j2000_times(self, ts):
        """Return J2000 time(s) ts, clipped to the range of the
        energy-time histogram. If times are more than one day out,
        raises InvalidEventTimes.

        :param ts: J2000 timestamp, or array of timestamps
        """
        tbins = self.energy_hist.bin_edges[0]
        if np.min(ts) < tbins[0] - 1 or np.max(ts) > tbins[-1] + 1:
            raise InvalidEventTimes(
                f"You passed J200 times in [{np.min(ts):.1f}, {np.max(ts):.1f}]"
                f"But this source expects [{tbins[0]:.1f} - {tbins[-1]:.1f}].")
        return np.clip(ts, tbins[0], tbins[-1])

    def mu_before_efficiencies(self, **params):
        return self.energy_hist.n / self.n_time_bins

    def random_truth(self, n_events, fix_truth=None, **params):
        """Draw n_events random energies and times from the energy/
        time spectrum and add them to the data dict.
        """
        data = self.draw_positions(n_events)

        if 'event_time' in fix_truth:
            # Convert to valid J200 timestamp (from whatever user specified)
            t = self.clip_j2000_times(wr.j2000(fix_truth['event_time']))

            # Time is fixed, so the energy spectrum differs.
            # (if energy is also fixed, it will just be overridden later
            #  and we're doing a bit of unnecessary work here)
            data['energy'] = \
                self.energy_hist \
                    .slicesum(t, axis=0) \
                    .get_random(n_events)
            times = t

        elif 'energy' in fix_truth:
            # Energy is fixed, so the time distribution differs.
            e_edges = self.energy_hist.bin_edges[1]
            assert e_edges[0] <= fix_truth['energy'] < e_edges[-1], \
                "fix_truth energy out of bounds"
            times = \
                self.energy_hist \
                    .slicesum(fix_truth['energy'], axis=1) \
                    .get_random(n_events)

        else:
            times, data['energy'] = self.energy_hist.get_random(n_events).T

        data['event_time'] = fd.j2000_to_event_time(times)

        # Time has already been handled, do not overwrite it again
        # (if we do, we could crash if the user specified it as a datetime
        #  object rather than a unix timestamp)
        fix_truth_notime = {k: v for k, v in fix_truth.items()
                            if k != 'event_time'}
        self.source._overwrite_fixed_truths(data, fix_truth_notime, n_events)

        return pd.DataFrame(data)

    @staticmethod
    def bin_centers(x):
        return 0.5 * (x[1:] + x[:-1])
