import tensorflow as tf

import configparser
import os

import flamedisx as fd
from .. import nest as fd_nest

import math as m
pi = tf.constant(m.pi)

export, __all__ = fd.exporter()
o = tf.newaxis

GAS_CONSTANT = 8.314
N_AVAGADRO = 6.0221409e23
A_XENON = 131.293
XENON_LIQUID_DIELECTRIC = 1.85
XENON_GAS_DIELECTRIC = 1.00126
XENON_REF_DENSITY = 2.90


class nestSource(fd.BlockModelSource):
    def __init__(self, *args, detector='default', **kwargs):
        assert detector in ('default',)

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), 'config/', detector + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), 'config/', detector + '.ini'))

        # common (known) parameters
        self.temperature = config.getfloat('NEST', 'temperature_config')
        self.pressure = config.getfloat('NEST', 'pressure_config')
        self.drift_field = config.getfloat('NEST', 'drift_field_config')
        self.gas_field = config.getfloat('NEST', 'gas_field_config')

        # derived (known) parameters
        self.density = fd_nest.calculate_density(
            self.temperature, self.pressure)
        # NOTE: BE CAREFUL WITH THE BELOW, ONLY VALID NEAR VAPOUR PRESSURE!!!
        self.density_gas = fd_nest.calculate_density_gas(
            self.temperature, self.pressure)
        #
        self.drift_velocity = fd_nest.calculate_drift_velocity(
            self.drift_field, self.density, self.temperature)
        self.Wq_keV = fd_nest.calculate_work(self.density)

        # energy_spectrum.py
        self.radius = config.getfloat('NEST', 'radius_config')
        self.z_topDrift = config.getfloat('NEST', 'z_topDrift_config')
        self.z_top = self.z_topDrift - self.drift_velocity * \
            config.getfloat('NEST', 'dt_min_config')
        self.z_bottom = self.z_topDrift - self.drift_velocity * \
            config.getfloat('NEST', 'dt_max_config')

        # detection.py
        self.g1 = config.getfloat('NEST', 'g1_config')
        self.min_photons = config.getint('NEST', 'min_photons_config')
        self.elife = config.getint('NEST', 'elife_config')

        # secondary_quanta_generation.py
        self.gas_gap = config.getfloat('NEST', 'gas_gap_config')
        self.g1_gas = config.getfloat('NEST', 'g1_gas_config')
        self.s2Fano = config.getfloat('NEST', 's2Fano_config')

        # double_pe.py
        self.double_pe_fraction = config.getfloat(
            'NEST', 'double_pe_fraction_config')

        # pe_detection.py
        self.spe_eff = config.getfloat('NEST', 'spe_eff_config')
        self.num_pmts = config.getfloat('NEST', 'num_pmts_config')

        # final_signals.py
        self.spe_res = config.getfloat('NEST', 'spe_res_config')
        self.S1_noise = config.getfloat('NEST', 'S1_noise_config')
        self.S2_noise = config.getfloat('NEST', 'S2_noise_config')

        self.S1_min = config.getfloat('NEST', 'S1_min_config')
        self.S1_max = config.getfloat('NEST', 'S1_max_config')
        self.S2_min = config.getfloat('NEST', 'S2_min_config')
        self.S2_max = config.getfloat('NEST', 'S2_max_config')

        super().__init__(*args, **kwargs)

    final_dimensions = ('s1', 's2')
    no_step_dimensions = ('s1_photoelectrons_produced',
                          's1_photoelectrons_detected')
    additional_bounds_dimensions = ('energy',)
    prior_dimensions = [(('electrons_produced', 'photons_produced'),
                        ('energy', 's1_photoelectrons_detected', 's2_photoelectrons_detected'))]

    # quanta_splitting.py

    @staticmethod
    def recomb_prob(*args):
        nel_mean = args[0]
        nq_mean = args[1]
        ex_ratio = args[2]

        elec_frac = nel_mean / nq_mean
        recomb_p = 1. - (ex_ratio + 1.) * elec_frac

        return tf.where(tf.logical_or(nq_mean == 0, recomb_p < 0),
                        tf.zeros_like(recomb_p, dtype=fd.float_type()),
                        recomb_p)

    @staticmethod
    def width_correction(skew):
        return tf.sqrt(1. - (2. / pi) * skew * skew / (1. + skew * skew))

    @staticmethod
    def mu_correction(*args):
        skew = tf.cast(args[0], fd.float_type())
        var = tf.cast(args[1], fd.float_type())
        width_corr = tf.cast(args[2], fd.float_type())

        return (tf.sqrt(var) / width_corr) * (skew / tf.sqrt(1. + skew * skew)) * tf.sqrt(2. / pi)

    # detection.py

    def photon_detection_eff(self, z):
        return self.g1 * tf.ones_like(z)

    def electron_detection_eff(self, drift_time):
        liquid_field_interface = self.gas_field / \
            (XENON_LIQUID_DIELECTRIC / XENON_GAS_DIELECTRIC)
        extraction_eff = -0.03754 * pow(liquid_field_interface, 2) + \
            0.52660 * liquid_field_interface - 0.84645

        return extraction_eff * tf.exp(-drift_time / self.elife)

    def s2_photon_detection_eff(self, z):
        return self.g1_gas * tf.ones_like(z)

    # secondary_quanta_generation.py

    def electron_gain_mean(self):
        elYield = (
            0.137 * self.gas_field * 1e3 -
            4.70e-18 * (N_AVAGADRO * self.density_gas / A_XENON)) \
            * self.gas_gap * 0.1

        return tf.cast(elYield, fd.float_type())[o]

    def electron_gain_std(self):
        elYield = (
            0.137 * self.gas_field * 1e3 -
            4.70e-18 * (N_AVAGADRO * self.density_gas / A_XENON)) \
            * self.gas_gap * 0.1

        return tf.sqrt(self.s2Fano * elYield)[o]

    # pe_detection.py

    def photoelectron_detection_eff(self, pe_det):
        eff = tf.where(
            self.spe_eff < 1.,
            self.spe_eff + (1. - self.spe_eff) / (2. * self.num_pmts) * pe_det,
            self.spe_eff)
        eff_trunc = tf.where(
            eff > 1.,
            1.,
            eff)

        return 1. - (1. - eff_trunc) / (1. + self.double_pe_fraction)

    # final_signals.py

    def s1_spe_smearing(self, n_pe):
        return tf.sqrt(
            self.spe_res * self.spe_res * n_pe +
            self.S1_noise * self.S1_noise * n_pe * n_pe)

    def s2_spe_smearing(self, n_pe):
        return tf.sqrt(
            self.spe_res * self.spe_res * n_pe +
            self.S2_noise * self.S2_noise * n_pe * n_pe)


@export
class nestERSource(nestSource):
    def __init__(self, *args, energy_min=0., energy_max=10., num_energies=1000, **kwargs):
        self.energies = tf.cast(tf.linspace(energy_min, energy_max, num_energies),
                                fd.float_type())
        self.rates_vs_energy = tf.ones(num_energies, fd.float_type())
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd_nest.FixedShapeEnergySpectrumER,
        fd_nest.MakePhotonsElectronER,
        fd_nest.DetectPhotons,
        fd_nest.MakeS1Photoelectrons,
        fd_nest.DetectS1Photoelectrons,
        fd_nest.MakeS1,
        fd_nest.DetectElectrons,
        fd_nest.MakeS2Photons,
        fd_nest.DetectS2Photons,
        fd_nest.MakeS2Photoelectrons,
        fd_nest.MakeS2)

    # quanta_splitting.py

    def mean_yield_electron(self, energy):
        Wq_eV = self.Wq_keV * 1e3

        QyLvllowE = tf.cast(1e3 / Wq_eV + 6.5 * (1. - 1. / (1. + pow(self.drift_field / 47.408, 1.9851))),
                            fd.float_type())
        HiFieldQy = tf.cast(1. + 0.4607 / pow(1. + pow(self.drift_field / 621.74, -2.2717), 53.502),
                            fd.float_type())
        QyLvlmedE = tf.cast(32.988 - 32.988 / (1. +
                            pow(self.drift_field / (0.026715 * tf.exp(self.density / 0.33926)), 0.6705)),
                            fd.float_type())
        QyLvlmedE *= HiFieldQy
        DokeBirks = tf.cast(1652.264 + (1.415935e10 - 1652.264) / (1. + pow(self.drift_field / 0.02673144, 1.564691)),
                            fd.float_type())
        LET_power = tf.cast(-2., fd.float_type())
        QyLvlhighE = tf.cast(28., fd.float_type())
        Qy = QyLvlmedE + (QyLvllowE - QyLvlmedE) / pow(1. + 1.304 * pow(energy, 2.1393), 0.35535) + \
            QyLvlhighE / (1. + DokeBirks * pow(energy, LET_power))

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        return nel

    def mean_yield_quanta(self, *args):
        energy = args[0]
        nel_mean = args[1]

        nq_temp = energy / self.Wq_keV

        nph_temp = nq_temp - nel_mean
        # Don't let number of photons go negative
        nph = tf.where(nph_temp < 0,
                       0 * nph_temp,
                       nph_temp)

        nq = nel_mean + nph

        return nq

    def fano_factor(self, nq_mean):
        Fano = 0.12707 - 0.029623 * self.density - 0.0057042 * pow(self.density, 2.) + 0.0015957 * pow(self.density, 3.)

        return Fano + 0.0015 * tf.sqrt(nq_mean) * pow(self.drift_field, 0.5)

    def exciton_ratio(self, energy):
        alf = 0.067366 + self.density * 0.039693

        return alf * tf.math.erf(0.05 * energy)

    def skewness(self, nq_mean):
        energy = self.Wq_keV * nq_mean

        alpha0 = tf.cast(1.39, fd.float_type())
        cc0 = tf.cast(4., fd.float_type())
        cc1 = tf.cast(22.1, fd.float_type())
        E0 = tf.cast(7.7, fd.float_type())
        E1 = tf.cast(54., fd.float_type())
        E2 = tf.cast(26.7, fd.float_type())
        E3 = tf.cast(6.4, fd.float_type())
        F0 = tf.cast(225., fd.float_type())
        F1 = tf.cast(71., fd.float_type())

        skew = 1. / (1. + tf.exp((energy - E2) / E3)) * \
            (alpha0 + cc0 * tf.exp(-1. * self.drift_field / F0) * (1. - tf.exp(-1. * energy / E0))) + \
            1. / (1. + tf.exp(-1. * (energy - E2) / E3)) * cc1 * tf.exp(-1. * energy / E1) * \
            tf.exp(-1. * tf.sqrt(self.drift_field) / tf.sqrt(F1))

        mask = tf.less(nq_mean, 10000*tf.ones_like(nq_mean))
        skewness = tf.ones_like(nq_mean, dtype=fd.float_type()) * skew
        skewness_masked = tf.multiply(skewness, tf.cast(mask, fd.float_type()))

        return skewness_masked

    def variance(self, *args):
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        elec_frac = nel_mean / nq_mean
        ampl = tf.cast(0.14 + (0.043 - 0.14) / (1. + pow(self.drift_field / 1210., 1.25)), fd.float_type())
        wide = tf.cast(0.205, fd.float_type())
        cntr = tf.cast(0.5, fd.float_type())
        skew = tf.cast(-0.2, fd.float_type())
        norm = tf.cast(0.988, fd.float_type())

        omega = norm * ampl * tf.exp(-0.5 * pow(elec_frac - cntr, 2.) / (wide * wide)) * \
            (1. + tf.math.erf(skew * (elec_frac - cntr) / (wide * tf.sqrt(2.))))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         omega)

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni


@export
class nestNRSource(nestSource):
    def __init__(self, *args, energy_min=0.7, energy_max=150., num_energies=1000, **kwargs):
        self.energies = tf.cast(tf.linspace(energy_min, energy_max, num_energies),
                                fd.float_type())
        self.rates_vs_energy = tf.ones(num_energies, fd.float_type())
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd_nest.FixedShapeEnergySpectrumNR,
        fd_nest.MakePhotonsElectronsNR,
        fd_nest.DetectPhotons,
        fd_nest.MakeS1Photoelectrons,
        fd_nest.DetectS1Photoelectrons,
        fd_nest.MakeS1,
        fd_nest.DetectElectrons,
        fd_nest.MakeS2Photons,
        fd_nest.DetectS2Photons,
        fd_nest.MakeS2Photoelectrons,
        fd_nest.MakeS2)

    # quanta_splitting.py

    def mean_yields(self, energy, *,
                    nr_nuis_a=11.,
                    nr_nuis_b=1.1,
                    nr_nuis_c=0.0480,
                    nr_nuis_d=-0.0533,
                    nr_nuis_e=12.6,
                    nr_nuis_f=0.3,
                    nr_nuis_g=2.,
                    nr_nuis_h=0.3,
                    nr_nuis_i=2,
                    nr_nuis_j=0.5,
                    nr_nuis_k=1.,
                    nr_nuis_l=1.):
        TIB = nr_nuis_c * pow(self.drift_field, nr_nuis_d) * pow(self.density / XENON_REF_DENSITY, 0.3)
        Qy = 1. / (TIB * pow(energy + nr_nuis_e, nr_nuis_j))
        Qy *= (1. - (1. / pow(1. + pow(energy / nr_nuis_f, nr_nuis_g), nr_nuis_k)))

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        nq_temp = nr_nuis_a * pow(energy, nr_nuis_b)

        nph_temp = (nq_temp - nel) * (1. - (1. / pow(1. + pow(energy / nr_nuis_h, nr_nuis_i), nr_nuis_l)))
        # Don't let number of photons go negative
        nph = tf.where(nph_temp < 0,
                       0 * nph_temp,
                       nph_temp)

        nq = nel + nph

        ni = (4. / TIB) * (tf.exp(nel * TIB / 4.) - 1.)

        nex = nq - ni

        ex_ratio = nex / ni

        alf = 0.067366 + self.density * 0.039693

        ex_ratio = tf.where(tf.logical_and(ex_ratio < alf, energy > 100.),
                            alf * tf.ones_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)
        ex_ratio = tf.where(tf.logical_and(ex_ratio > 1., energy < 1.),
                            tf.ones_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)
        ex_ratio = tf.where(tf.math.is_nan(ex_ratio),
                            tf.zeros_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)

        return nel, nq, ex_ratio

    @staticmethod
    def skewness(nq_mean, *,
                 nr_free_f=2.25):
        mask = tf.less(nq_mean, 1e4 * tf.ones_like(nq_mean))
        skewness = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_f
        skewness_masked = tf.multiply(skewness, tf.cast(mask, fd.float_type()))

        return skewness_masked

    @staticmethod
    def variance(*args,
                 nr_free_c=0.1,
                 nr_free_d=0.5,
                 nr_free_e=0.19):
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        elec_frac = nel_mean / nq_mean

        omega = nr_free_c * tf.exp(-0.5 * pow(elec_frac - nr_free_d, 2.) / (nr_free_e * nr_free_e))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         omega)

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni


@export
class nestSpatialRateERSource(nestERSource):
    model_blocks = (fd_nest.SpatialRateEnergySpectrum,) + nestERSource.model_blocks[1:]


@export
class nestSpatialRateNRSource(nestNRSource):
    model_blocks = (fd_nest.SpatialRateEnergySpectrum,) + nestNRSource.model_blocks[1:]


@export
class nestWIMPSource(nestNRSource):
    model_blocks = (fd_nest.WIMPEnergySpectrum,) + nestNRSource.model_blocks[1:]
