import numpy as np
import pandas as pd
import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class FixedShapeEnergySpectrum(fd.Block):
    """For a source whose energy spectrum has the same shape
    throughout space and time.

    If you add a rate variation with space, you must override draw_positions
     and mu_before_efficiencies.
     If you add a rate variation with time, you must override draw_times.
     If you add a rate variation depending on both space and time, you must
     override all of random_truth!

    By default, this uses a flat 0 - 10 keV spectrum, sampled at 1000 points.
    """

    model_functions = ('energy_spectrum_rate_multiplier',)
    dimensions = ('energy',)
    static_attributes = (
        'energies', 'rates_vs_energy',
        'fv_radius', 'fv_high', 'fv_low',
        'drift_velocity',
        't_start', 't_stop')

    # The fiducial volume bounds for a cylindrical volume
    # default to full (2t) XENON1T dimensions
    fv_radius = 47.9   # cm
    fv_high = 0.  # cm
    fv_low = -97.6  # cm

    drift_velocity = 1.335 * 1e-4   # cm/ns

    # The default boundaries are at points where the WIMP wind is at its
    # average speed.
    # This will then also be true at the midpoint of these times.
    t_start = pd.to_datetime('2019-09-01T08:28:00')
    t_stop = pd.to_datetime('2020-09-01T08:28:00')

    # Just a dummy 0-10 keV spectrum
    energies = tf.cast(tf.linspace(0., 10., 1000),
                       dtype=fd.float_type())
    rates_vs_energy = tf.ones(1000, dtype=fd.float_type())

    energy_spectrum_rate_multiplier = 1.

    def _compute(self, data_tensor, ptensor, *, energy):
        spectrum = fd.repeat(self.rates_vs_energy[o, :],
                             self.source.batch_size,
                             axis=0)
        rate_multiplier = self.gimme('energy_spectrum_rate_multiplier',
                                     data_tensor=data_tensor, ptensor=ptensor)
        return spectrum * rate_multiplier[:, o]

    def domain(self, data_tensor):
        assert isinstance(self.energies, tf.Tensor)  # see WIMPsource for why
        return {self.dimensions[0]: fd.repeat(fd.np_to_tf(self.energies)[o, :],
                                              self.source.batch_size,
                                              axis=0)}

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

    def draw_positions(self, n_events, **params):
        """Return dictionary with x, y, z, r, theta, drift_time
        randomly drawn.
        """
        data = dict()
        data['r'] = (np.random.rand(n_events) * self.fv_radius**2)**0.5
        data['theta'] = np.random.uniform(0, 2*np.pi, size=n_events)
        data['z'] = np.random.uniform(self.fv_low, self.fv_high,
                                      size=n_events)
        data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])

        data['drift_time'] = - data['z'] / self.drift_velocity
        return data

    def draw_time(self, n_events, **params):
        """Return n_events event_times drawn uniformaly
        between t_start and t_stop"""
        return np.random.uniform(
            self.t_start.value,
            self.t_stop.value,
            size=n_events)

    def mu_before_efficiencies(self, **params):
        return np.sum(fd.np_to_tf(self.rates_vs_energy))

    def validate_fix_truth(self, d):
        """Clean fix_truth, ensure all needed variables are present
           Compute derived variables.
        """
        if d is None:
            return dict()
        elif isinstance(d, pd.DataFrame):
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
                "fix_truth needs to be a DataFrame, dict, or None"

        if 'z' in d:
            # Position is fixed. Ensure both Cartesian and polar coordinates
            # are available, and compute drift_time from z.
            if 'x' in d and 'y' in d:
                d['r'], d['theta'] = fd.cart_to_pol(d['x'], d['y'])
            elif 'r' in d and 'theta' in d:
                d['x'], d['y'] = fd.pol_to_cart(d['r'], d['theta'])
            else:
                raise ValueError("When fixing position, give (x, y, z), "
                                 "or (r, theta, z).")
            d['drift_time'] = - d['z'] / self.drift_velocity
        elif 'event_time' not in d and 'energy' not in d:
            # Neither position, time, nor energy given
            raise ValueError(f"Dict should contain at least ['x', 'y', 'z'] "
                             "and/or ['r', 'theta', 'z'] and/or 'event_time' "
                             f"and/or 'energy', but it contains: {d.keys()}")
        return d


@export
class VariableEnergySpectrum(FixedShapeEnergySpectrum):
    """For a source for which the entire energy spectrum (not just the rate)
     depends on observables (e.g. reconstruction position or time).

    You must implement both energy_spectrum and random_truth.

    Note that you cannot draw homogeneous positions/times first, then
     energies, if your energy spectrum depends on positions/time.
    """

    model_functions = ('energy_spectrum',)
    array_columns = (('energy_spectrum', 1000),)

    def _annotate(self, d):
        d['energy_spectrum'] = fd.pandafy_twod_array(
            self.gimme_numpy('energy_spectrum'))

    def _compute(self, data_tensor, ptensor, *, energy):
        return self.gimme('energy_spectrum', data_tensor, ptensor)

    def energy_spectrum(self, event_time):
        # Note this returns a 2d tensor!
        return tf.ones(len(event_time), len(self.energies),
                       dtype=fd.float_type())

    def random_truth(self, n_events, fix_truth=None, **params):
        raise NotImplementedError
