import numpy as np
import pandas as pd
import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class UniformConstantEnergy(fd.Block):
    """For a source with a constant, homogeneous energy spectrum.

    By default, a flat 0 - 10 keV spectrum, sampled at 1000 points.
    """

    dimensions = ('deposited_energy',)
    static_attributes = (
        'energies', 'rates_vs_energy',
        'fv_radius', 'fv_high' 'fv_low' 'drift_velocity' 't_start' 't_stop')

    # The fiducial volume bounds for a cylindrical volume
    # default to full (2t) XENON1T dimensions
    fv_radius = 47.9   # cm
    fv_high = 0  # cm
    fv_low = -97.6  # cm

    drift_velocity = 1.335 * 1e-4   # cm/ns

    # The default boundaries are at points where the WIMP wind is at its
    # average speed.
    # This will then also be true at the midpoint of these times.
    t_start = pd.to_datetime('2019-09-01T08:28:00')
    t_stop = pd.to_datetime('2020-09-01T08:28:00')

    # Just a dummy 0-10 keV spectrum
    energies: tf.linspace(0., 10., 1000)
    rates_vs_energy: tf.ones(1000, dtype=fd.float_type())

    def _compute(self, data_tensor, ptensor):
        return fd.repeat(self.rates_vs_energy[o, :], self.batch_size, axis=0)

    def domain(self, data_tensor, ptensor):
        return {self.dimensions[0]:
                    fd.repeat(self.energies[o, :], self.batch_size, axis=0)}

    def random_truth(self, n_events):
        """Return dictionary with x, y, z, r, theta, drift_time
        and event_time randomly drawn.
        """
        data = self.draw_positions(n_events)

        # Draw uniform time
        data['event_time'] = np.random.uniform(
            self.t_start.value,
            self.t_stop.value,
            size=n_events)
        return data

    def draw_positions(self, n_events):
        data = dict()
        data['r'] = (np.random.rand(n_events) * self.fv_radius**2)**0.5
        data['theta'] = np.random.uniform(0, 2*np.pi, size=n_events)
        data['z'] = np.random.uniform(self.fv_low, self.fv_high,
                                      size=n_events)
        data['x'], data['y'] = fd.pol_to_cart(data['r'], data['theta'])

        data['drift_time'] = - data['z'] / self.drift_velocity
        return data


class VariableRateSpectrum(UniformConstantEnergy):
    """For a source whose rate, but not energy spectrum shape,
    depends on observables (e.g. reconstructed position or time)
    """

    model_functions = ('energy_spectrum_rate_multiplier',)

    energy_spectrum_rate_multiplier = 1

    def _compute(self, data_tensor, ptensor):
        return (
            UniformConstantEnergy._compute(
                self,
                data_tensor=data_tensor, ptensor=ptensor)
            * self.gimme('energy_spectrum_rate_multiplier',
                         data_tensor=data_tensor, ptensor=ptensor))


class VariableSpectrum(UniformConstantEnergy):
    """For a source for which the entire energy spectrum (not just the rate)
    depends on observables (e.g. reconstruction position or time)
    """

    model_functions = ('energy_spectrum',)
    array_columns = (('energy_spectrum', 1000),)

    def _annotate(self, d):
        # Note this sets an array of values!
        d['energy_spectrum'] = d.gimme_numpy('energy_spectrum')

    def energy_spectrum(self, x):
        # Note this returns a 2d tensor!
        return tf.ones(len(x), len(self.energies))

    def _compute(self, data_tensor, ptensor):
        return self.gimme('energy_spectrum', data_tensor, ptensor)
