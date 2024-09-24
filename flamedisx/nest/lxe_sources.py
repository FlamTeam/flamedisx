import tensorflow as tf

import configparser
import os

import pickle as pkl

import flamedisx as fd
from .. import nest as fd_nest
import numpy as np
from scipy import interpolate
import math as m
pi = tf.constant(m.pi)

export, __all__ = fd.exporter()
o = tf.newaxis

GAS_CONSTANT = 8.314
N_AVAGADRO = 6.0221409e23
A_XENON = 131.293
XENON_REF_DENSITY = 2.90


class nestSource(fd.BlockModelSource):
    def __init__(self, *args, detector='default',drift_map_path=None, **kwargs):
        assert detector in ('default', 'lz','lz_SR3')
        
        if drift_map_path is not None:
            print("Loading in Drift map:\n",drift_map_path)
            drift_map=np.loadtxt(drift_map_path,delimiter=',').T
            self.drift_map=interpolate.LinearNDInterpolator(drift_map[:2].T/10,drift_map[3]*1e3)

        self.detector = detector

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
            self.drift_field, self.density, self.temperature, self.detector)
        self.Wq_keV, self.alpha = fd_nest.calculate_work(self.density)

        # energy_spectrum.py
        self.radius = config.getfloat('NEST', 'radius_config')
        self.z_topDrift = config.getfloat('NEST', 'z_topDrift_config')
        self.z_top = self.z_topDrift - self.drift_velocity * \
            config.getfloat('NEST', 'dt_min_config')
        self.z_bottom = self.z_topDrift - self.drift_velocity * \
            config.getfloat('NEST', 'dt_max_config')

        # detection.py / pe_detection.py / double_pe.py / final_signals.py
        self.g1 = config.getfloat('NEST', 'g1_config')
        self.elife = config.getfloat('NEST', 'elife_config')
        self.extraction_eff = fd_nest.calculate_extraction_eff(self.gas_field, self.temperature)
        self.spe_res = config.getfloat('NEST', 'spe_res_config')
        self.spe_thr = config.getfloat('NEST', 'spe_thr_config')
        self.spe_eff = config.getfloat('NEST', 'spe_eff_config')
        self.num_pmts = config.getfloat('NEST', 'num_pmts_config')
        self.double_pe_fraction = config.getfloat('NEST', 'double_pe_fraction_config')
        self.coin_table = fd_nest.get_coin_table(config.getint('NEST', 'coin_level_config'), self.num_pmts,
                                                 self.spe_res, self.spe_thr, self.spe_eff,
                                                 self.double_pe_fraction)

        # secondary_quanta_generation.py
        self.gas_gap = config.getfloat('NEST', 'gas_gap_config')
        self.g1_gas = config.getfloat('NEST', 'g1_gas_config')
        self.s2Fano = config.getfloat('NEST', 's2Fano_config')

        # final_signals.py
        self.s1_mean_mult = fd_nest.calculate_s1_mean_mult(self.spe_res)
        self.s2_mean_mult = 1.
        self.S1_noise = config.getfloat('NEST', 'S1_noise_config')
        self.S2_noise = config.getfloat('NEST', 'S2_noise_config')

        self.s2_thr = config.getfloat('NEST', 's2_thr_config')

        self.S1_min = config.getfloat('NEST', 'S1_min_config')
        self.S1_max = config.getfloat('NEST', 'S1_max_config')
        self.S2_min = config.getfloat('NEST', 'S2_min_config')
        self.S2_max = config.getfloat('NEST', 'S2_max_config')

        # Useful additional parameters
        self.g2 = fd_nest.calculate_g2(self.gas_field, self.density_gas, self.gas_gap,
                                       self.g1_gas, self.extraction_eff)

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
        return self.extraction_eff * tf.exp(-drift_time / self.elife)

    def s2_photon_detection_eff(self, z):
        return self.g1_gas * tf.ones_like(z)

    # secondary_quanta_generation.py

    def electron_gain_mean(self, z):
        elYield = (
            0.137 * self.gas_field * 1e3 -
            4.70e-18 * (N_AVAGADRO * self.density_gas / A_XENON)) \
            * self.gas_gap * 0.1

        return tf.cast(elYield, fd.float_type()) * tf.ones_like(z)

    def electron_gain_std(self, z):
        elYield = (
            0.137 * self.gas_field * 1e3 -
            4.70e-18 * (N_AVAGADRO * self.density_gas / A_XENON)) \
            * self.gas_gap * 0.1

        return tf.sqrt(self.s2Fano * elYield) * tf.ones_like(z)

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

    def s1_spe_mean(self, n_pe):
        return self.s1_mean_mult * n_pe

    def s2_spe_mean(self, n_pe):
        return self.s2_mean_mult * n_pe

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
    def __init__(self, *args, energy_min=0.01, energy_max=10., num_energies=1000, energy_bin_edges=None, **kwargs):
        if not hasattr(self, 'energies'):
            if energy_bin_edges is not None:
                self.energies = fd.np_to_tf(0.5 * (energy_bin_edges[1:] + energy_bin_edges[:-1]))
                self.rates_vs_energy = tf.ones(len(energy_bin_edges) - 1, fd.float_type())
            else:
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
        #Refactored to take constants from lzlama !397
        m1=12.4886
        m2=85.0
        m3=0.6050
        m4= 2.14687
        m5=25.721
        m6=-1.0
        m7=59.651
        m8=3.6869
        m9=0.2872
        m10=0.1121
        Wq_eV = self.Wq_keV * 1e3

        Nq = energy * 1e3 / Wq_eV       


        Qy = m1 + (m2 - m1) / pow((1. + pow(energy /m3,m4)),m9) + \
            m5 + (m6 - m5) / pow((1. + pow(energy /m7, m8)), m10)

        coeff_TI = tf.cast(pow(1. / XENON_REF_DENSITY, 0.3), fd.float_type())
        coeff_Ni = tf.cast(pow(1. / XENON_REF_DENSITY, 1.4), fd.float_type())
        coeff_OL = tf.cast(pow(1. / XENON_REF_DENSITY, -1.7) /
                           fd.tf_log10(1. + coeff_TI * coeff_Ni * pow(XENON_REF_DENSITY, 1.7)), fd.float_type())

        Qy *= coeff_OL * fd.tf_log10(1. + coeff_TI * coeff_Ni * pow(self.density, 1.7)) * pow(self.density, -1.7)

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        return nel
#     def mean_yield_electron(self, energy):
#         #NEED TO REFACTOR IN TERMS OF M1,M2...,M10 ignoring functional forms
#         er_m1_a=30.66
#         er_m1_b=6.1978
#         er_m1_d=73.855
#         er_m1_e=2.0318
#         er_m1_f=0.41883
#         er_m10_a=0.0508273937
#         er_m10_b=0.1166087199
#         er_m10_c= 0.0508273937
#         er_m10_d=1.39260460e+02
#         er_m10_e=-0.65763592
#         er_Qy_a=77.2931084
#         er_Qy_b=0.13946236
#         er_Qy_c=0.52561312
#         er_Qy_d=1.82217496
#         er_Qy_e=2.82528809
#         er_Qy_f=1.82217496
#         er_Qy_g=144.65029656
#         er_Qy_h=-2.80532006
#         er_Qy_i=0.3344049589
#         er_Qy_k=7.02921301
#         er_Qy_l=98.27936794 
#         er_Qy_m=7.0292130
#         er_Qy_n=256.48156448
#         er_Qy_o=1.29119251
#         er_Qy_p=4.285781736
#         Wq_eV = self.Wq_keV * 1e3

#         Nq = energy * 1e3 / Wq_eV       

#         m1 = tf.cast(er_m1_a + (er_m1_b - er_m1_a) / pow(1. + pow(self.drift_field / er_m1_d, er_m1_e), er_m1_f),
#                      fd.float_type())
#         m5 = tf.cast(Nq / energy / (1 + self.alpha * tf.math.erf(0.05 * energy)), fd.float_type()) - m1
#         m10 = tf.cast((er_m10_a + (er_m10_b - er_m10_c) /
#                       (1 + pow(self.drift_field / er_m10_d, er_m10_e))),
#                       fd.float_type())

#         Qy = m1 + (er_Qy_a - m1) / pow((1. + pow(energy / (fd.tf_log10(tf.cast(self.drift_field, fd.float_type())) *
#                                                     er_Qy_b + er_Qy_c),
#                                                     er_Qy_d + (er_Qy_e - er_Qy_f) /
#                                                     (1 + pow(self.drift_field / er_Qy_g, er_Qy_h)))),
#                                           er_Qy_i) + \
#             m5 + (0. - m5) / pow((1. + pow(energy / (er_Qy_k + (er_Qy_l - er_Qy_m) /
#                                  (1. + pow(self.drift_field / er_Qy_n, er_Qy_o))), er_Qy_p)), m10)

#         coeff_TI = tf.cast(pow(1. / XENON_REF_DENSITY, 0.3), fd.float_type())
#         coeff_Ni = tf.cast(pow(1. / XENON_REF_DENSITY, 1.4), fd.float_type())
#         coeff_OL = tf.cast(pow(1. / XENON_REF_DENSITY, -1.7) /
#                            fd.tf_log10(1. + coeff_TI * coeff_Ni * pow(XENON_REF_DENSITY, 1.7)), fd.float_type())

#         Qy *= coeff_OL * fd.tf_log10(1. + coeff_TI * coeff_Ni * pow(self.density, 1.7)) * pow(self.density, -1.7)

#         nel_temp = Qy * energy
#         # Don't let number of electrons go negative
#         nel = tf.where(nel_temp < 0,
#                        0 * nel_temp,
#                        nel_temp)

#         return nel

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

        return tf.cast(nq, fd.float_type())

    def fano_factor(self, nq_mean):
        er_free_a = 0.3#0.0015
        Fano = 0.12707 - 0.029623 * self.density - 0.0057042 * pow(self.density, 2.) + 0.0015957 * pow(self.density, 3.)

        return tf.constant(er_free_a,tf.float32) #Fano + er_free_a * tf.sqrt(nq_mean) * pow(self.drift_field, 0.5)

    def exciton_ratio(self, energy):
        return self.alpha * tf.math.erf(0.05 * energy)

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

        mask_quanta = tf.less(nq_mean, 10000*tf.ones_like(nq_mean))
        mask_field_low = tf.greater(self.drift_field*tf.ones_like(nq_mean), 50.*tf.ones_like(nq_mean))
        mask_field_high = tf.less(self.drift_field*tf.ones_like(nq_mean), 2000.*tf.ones_like(nq_mean))

        mask_product = tf.logical_and(mask_quanta, tf.logical_and(mask_field_low, mask_field_high))

        skewness = tf.ones_like(nq_mean, dtype=fd.float_type()) * skew
        skewness_masked = tf.multiply(skewness, tf.cast(mask_product, fd.float_type()))

        if self.detector in ['lz','lz_SR3']:
            skewness_masked = tf.zeros_like(nq_mean, dtype=fd.float_type())

        return skewness_masked

    def variance(self, *args):
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        if self.detector in ['lz','lz_SR3']:
            er_free_b = 0.04311#0.046452
        else:
            er_free_b = 0.0553
        er_free_c = 0.15505#0.205
        er_free_d = 0.46894#0.45
        er_free_e = -0.26564#-0.2

        elec_frac = nel_mean / nq_mean
        ampl = er_free_b
        
        ampl =  tf.cast(0.086036 + (er_free_b - 0.086036) /
                       pow((1. + pow(self.drift_field / 295.2, 251.6)), 0.0069114),
                       fd.float_type()) 
        wide = er_free_c
        cntr = er_free_d
        skew = er_free_e

        mode = cntr + 2. / (tf.sqrt(2. * pi)) * skew * wide / tf.sqrt(1. + skew * skew)
        norm = 1. / (tf.exp(-0.5 * pow(mode - cntr, 2.) / (wide * wide)) *
                     (1. + tf.math.erf(skew * (mode - cntr) / (wide * tf.sqrt(2.)))))

        omega = norm * ampl * tf.exp(-0.5 * pow(elec_frac - cntr, 2.) / (wide * wide)) * \
            (1. + tf.math.erf(skew * (elec_frac - cntr) / (wide * tf.sqrt(2.))))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         omega)

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni


@export
class nestNRSource(nestSource):
    def __init__(self, *args, energy_min=0.01, energy_max=10., num_energies=1000, energy_bin_edges=None, **kwargs):
        if not hasattr(self, 'energies'):
            if energy_bin_edges is not None:
                self.energies = fd.np_to_tf(0.5 * (energy_bin_edges[1:] + energy_bin_edges[:-1]))
                self.rates_vs_energy = tf.ones(len(energy_bin_edges) - 1, fd.float_type())
            else:
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

    def mean_yields(self, energy):
        nr_nuis_alpha = 10.19
        nr_nuis_beta = 1.11
        nr_nuis_gamma = 0.0498
        nr_nuis_delta = -0.0533
        nr_nuis_epsilon = 12.46
        nr_nuis_zeta =  0.2942
        nr_nuis_eta = 1.899
        nr_nuis_theta = 0.3197
        nr_nuis_l = 2.066
        nr_nuis_p = 0.509
        nr_new_nuis_a = 0.996
        nr_new_nuis_b =  0.999
 
        TIB = nr_nuis_gamma * tf.math.pow(self.drift_field, nr_nuis_delta) * pow(self.density / XENON_REF_DENSITY, 0.3)
        Qy = 1. / (TIB * tf.math.pow(energy + nr_nuis_epsilon, nr_nuis_p))
        Qy *= (1. - (1. / tf.math.pow(1. + tf.math.pow(tf.math.divide_no_nan(energy , nr_nuis_zeta), nr_nuis_eta), nr_new_nuis_a)))

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        nq_temp = nr_nuis_alpha * pow(energy, nr_nuis_beta)

        nph_temp = (nq_temp - nel) * (1. - (1. / tf.math.pow(1. + tf.math.pow(tf.math.divide_no_nan(energy , nr_nuis_theta), nr_nuis_l), nr_new_nuis_b)))
        # Don't let number of photons go negative
        nph = tf.where(nph_temp < 0,
                       0 * nph_temp,
                       nph_temp)

        nq = nel + nph

        ni = (4. / TIB) * (tf.exp(nel * TIB / 4.) - 1.)

        nex = nq - ni

        ex_ratio = tf.cast(tf.math.divide_no_nan(nex , ni), fd.float_type())

        ex_ratio = tf.where(tf.logical_and(ex_ratio < self.alpha, energy > 100.),
                            self.alpha * tf.ones_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)
        ex_ratio = tf.where(tf.logical_and(ex_ratio > 1., energy < 1.),
                            tf.ones_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)
        ex_ratio = tf.where(tf.math.is_nan(ex_ratio),
                            tf.zeros_like(ex_ratio, dtype=fd.float_type()),
                            ex_ratio)

        return nel, nq, ex_ratio

    def yield_fano(self, nq_mean):
        if self.detector in ['lz','lz_SR3']:
            nr_free_a = 0.404
            nr_free_b = 0.393
        else:
            nr_free_a = 1.
            nr_free_b = 1.

        ni_fano = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_a
        nex_fano = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_b

        return ni_fano, nex_fano

    @staticmethod
    def skewness(nq_mean):
        nr_free_f =  2.220

        mask = tf.less(nq_mean, 1e4 * tf.ones_like(nq_mean))
        skewness = tf.ones_like(nq_mean, dtype=fd.float_type()) * nr_free_f
        skewness_masked = tf.multiply(skewness, tf.cast(mask, fd.float_type()))

        return skewness_masked

    def variance(self, *args):
        nel_mean = args[0]
        nq_mean = args[1]
        recomb_p = args[2]
        ni = args[3]

        if self.detector in ['lz','lz_SR3']:
            nr_free_c = 0.0383
        else:
            nr_free_c = 0.1
        nr_free_d = 0.497
        nr_free_e =  0.1906

        elec_frac = nel_mean / nq_mean

        omega = nr_free_c * tf.exp(-0.5 * pow(elec_frac - nr_free_d, 2.) / (nr_free_e * nr_free_e))
        omega = tf.where(nq_mean == 0,
                         tf.zeros_like(omega, dtype=fd.float_type()),
                         tf.cast(omega, dtype=fd.float_type()))

        return recomb_p * (1. - recomb_p) * ni + omega * omega * ni * ni


@export
class nestGammaSource(nestERSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mean_yield_electron(self, energy):
        Wq_eV = self.Wq_keV * 1e3
        m3 = 2.
        m4 = 2.
        m6 = 0.

        m1 = 33.951 + (3.3284 - 33.951) / (1. + pow(self.drift_field / 165.34, .72665))
        m2 = 1000 / Wq_eV
        m5 = 23.156 + (10.737 - 23.156) / (1. + pow(self.drift_field / 34.195, .87459))
        densCorr = 240720. / pow(self.density, 8.2076)
        m7 = 66.825 + (829.25 - 66.825) / (1. + pow(self.drift_field / densCorr, .83344))

        m8 = 2.

        Qy = m1 + (m2 - m1) / (1. + pow(energy / m3, m4)) + m5 + (m6 - m5) / (1. + pow(energy / m7, m8))

        nel_temp = Qy * energy
        # Don't let number of electrons go negative
        nel = tf.where(nel_temp < 0,
                       0 * nel_temp,
                       nel_temp)

        return nel


@export
class nestERGammaWeightedSource(nestERSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mean_yield_electron(self, energy):
        weight_param_a = 0.23
        weight_param_b = 0.77
        weight_param_c = 2.95
        weight_param_d = -1.44
        weight_param_e = 421.15
        weight_param_f = 3.27

        weightG = tf.cast(weight_param_a + weight_param_b * tf.math.erf(weight_param_c *
                          (tf.math.log(energy) + weight_param_d)) *
                          (1. - (1. / (1. + pow(self.drift_field / weight_param_e, weight_param_f)))),
                          fd.float_type())
        weightB = tf.cast(1. - weightG, fd.float_type())

        nel_gamma = tf.cast(nestGammaSource.mean_yield_electron(self, energy), fd.float_type())
        nel_beta = tf.cast(nestERSource.mean_yield_electron(self, energy), fd.float_type())

        return nel_gamma * weightG + nel_beta * weightB


@export
class nestFasterERSource(nestERSource):
    model_blocks = (fd_nest.FixedShapeEnergySpectrumFaster,) + nestERSource.model_blocks[1:]


@export
class nestSpatialRateERSource(nestERSource):
    model_blocks = (fd_nest.SpatialRateEnergySpectrumER,) + nestERSource.model_blocks[1:]


@export
class nestSpatialRateNRSource(nestNRSource):
    model_blocks = (fd_nest.SpatialRateEnergySpectrumNR,) + nestNRSource.model_blocks[1:]


@export
class nestTemporalRateDecayERSource(nestERSource):
    model_blocks = (fd_nest.TemporalRateEnergySpectrumDecayER,) + nestERSource.model_blocks[1:]


@export
class nestTemporalRateDecayNRSource(nestNRSource):
    model_blocks = (fd_nest.TemporalRateEnergySpectrumDecayNR,) + nestNRSource.model_blocks[1:]


@export
class nestSpatialTemporalRateDecayERSource(nestERSource):
    model_blocks = (fd_nest.SpatialTemporalRateEnergySpectrumDecayER,) + nestERSource.model_blocks[1:]


@export
class nestSpatialTemporalRateDecayNRSource(nestNRSource):
    model_blocks = (fd_nest.SpatialTemporalRateEnergySpectrumDecayNR,) + nestNRSource.model_blocks[1:]


@export
class nestTemporalRateOscillationERSource(nestERSource):
    model_blocks = (fd_nest.TemporalRateEnergySpectrumOscillationER,) + nestERSource.model_blocks[1:]


@export
class nestTemporalRateOscillationNRSource(nestNRSource):
    model_blocks = (fd_nest.TemporalRateEnergySpectrumOscillationNR,) + nestNRSource.model_blocks[1:]


@export
class nestWIMPSource(nestNRSource):
    model_blocks = (fd_nest.WIMPEnergySpectrum,) + nestNRSource.model_blocks[1:]

    def __init__(self, *args, wimp_mass=40, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'default'

        self.energy_hist = pkl.load(open(os.path.join(os.path.dirname(__file__), 'wimp_spectra/WIMP_spectra.pkl'), 'rb'))[wimp_mass]

        self.n_time_bins = len(self.energy_hist.bin_edges[0])
        e_centers = fd_nest.WIMPEnergySpectrum.bin_centers(self.energy_hist.bin_edges[1])
        self.energies = fd.np_to_tf(e_centers)

        self.array_columns = (('energy_spectrum', len(e_centers)),)

        super().__init__(*args, **kwargs)
