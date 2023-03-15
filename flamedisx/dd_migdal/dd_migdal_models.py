import numpy as np
import tensorflow as tf

import os

from multihist import Hist1d

import scipy.interpolate as itp

from .. import dd_migdal as fd_dd_migdal

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class NRSource(fd.BlockModelSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstSS,
        fd_dd_migdal.MakeS1S2SS)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 1. / 0.3799912

    @staticmethod
    def signal_means(energy, a=13., b=1.06,
                     c_s2_0=2.51, c_s2_1=0.99, c_s2_2=-0.09, c_s2_3=0.03, c_s2_4=-0.02,
                     g1=0.1131, g2=47.35,
                     s1_mean_multiplier=1., s2_mean_multiplier=1.):
        P = c_s2_0 + c_s2_1 * fd.tf_log10(energy) + c_s2_2 * pow(fd.tf_log10(energy), 2) + \
            c_s2_3 * pow(fd.tf_log10(energy), 3) + c_s2_4 * pow(fd.tf_log10(energy), 4)
        s2_mean = s2_mean_multiplier * 10**P

        s1_mean = s1_mean_multiplier * (a * energy**b - s2_mean / g2) * g1
        s1_mean= tf.where(s1_mean < 0.01, 0.01 * tf.ones_like(s1_mean, dtype=fd.float_type()), s1_mean)

        return s1_mean, s2_mean

    @staticmethod
    def signal_vars(*args, d_s1=1.2, d_s2=39.):
        s1_mean = args[0]
        s2_mean = args[1]

        s1_var = d_s1 * s1_mean

        s2_var = d_s2 * s2_mean

        return s1_var, s2_var

    @staticmethod
    def signal_cov(*args, anti_corr=-0.2):
        s1_var = args[0]
        s2_var = args[1]

        s1s2_cov = anti_corr * tf.sqrt(s1_var * s2_var)

        return s1s2_cov

    final_dimensions = ('s1s2',)


@export
class NRNRSource(NRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMSU,
        fd_dd_migdal.EnergySpectrumSecondMSU,
        fd_dd_migdal.MakeS1S2MSU)

    no_step_dimensions = ('energy_second')

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 4.45e-2 / 0.6889167


@export
class Migdal2Source(NRNRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdal,
        fd_dd_migdal.EnergySpectrumSecondMigdal2,
        fd_dd_migdal.MakeS1S2Migdal)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    ER_NEST = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/ER_NEST.npz'))

    E_ER = ER_NEST['EkeVee']
    s1_mean_ER = itp.interp1d(E_ER, ER_NEST['s1mean'])
    s2_mean_ER = itp.interp1d(E_ER, ER_NEST['s2mean'])
    s1_var_ER = itp.interp1d(E_ER, ER_NEST['s1std']**2)
    s2_var_ER = itp.interp1d(E_ER, ER_NEST['s2std']**2)

    def __init__(self, *args, **kwargs):
        energies_first = self.model_blocks[0].energies_first
        energies_first = tf.where(energies_first > 49., 49. * tf.ones_like(energies_first), energies_first)
        energies_first = tf.repeat(energies_first[:, o], tf.shape(self.model_blocks[1].energies_second), axis=1)

        self.s1_mean_ER_tf, self.s2_mean_ER_tf = self.signal_means_ER(energies_first)
        self.s1_var_ER_tf, self.s2_var_ER_tf = self.signal_vars_ER(energies_first)

        super().__init__(*args, **kwargs)

    def mu_before_efficiencies(self, **params):
        return 1.60e-4 / 0.9334023

    def signal_means_ER(self, energy):
        energy_cap = np.where(energy <= 49., energy, 49.)
        s1_mean = tf.cast(self.s1_mean_ER(energy_cap), fd.float_type())
        s2_mean = tf.cast(self.s2_mean_ER(energy_cap), fd.float_type())

        return s1_mean, s2_mean

    def signal_vars_ER(self, energy):
        energy_cap = np.where(energy <= 49., energy, 49.)
        s1_var = tf.cast(self.s1_var_ER(energy_cap), fd.float_type())
        s2_var = tf.cast(self.s2_var_ER(energy_cap), fd.float_type())

        return s1_var, s2_var


@export
class Migdal3Source(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdal,
        fd_dd_migdal.EnergySpectrumSecondMigdal3,
        fd_dd_migdal.MakeS1S2Migdal)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 3.75e-3 / 0.8775706


@export
class Migdal4Source(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdal,
        fd_dd_migdal.EnergySpectrumSecondMigdal4,
        fd_dd_migdal.MakeS1S2Migdal)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 3.09e-2 / 0.7161445


@export
class IECSSource(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstIE_CS,
        fd_dd_migdal.EnergySpectrumSecondIE_CS,
        fd_dd_migdal.MakeS1S2Migdal)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 1.32e-4 / 0.5086278
