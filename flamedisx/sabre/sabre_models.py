import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
from .. import sabre as fd_sabre

import numpy as np
from scipy import integrate

export, __all__ = fd.exporter()


@export
class SABREBetaSource(fd.BlockModelSource):
    model_blocks = (
        fd_sabre.FixedShapeEnergySpectrum,
        fd_sabre.PhotonsPhotoelectrons,
        fd_sabre.MakeFinalSignal)

    def __init__(self, *args, energies=None, rates_vs_energy=None, **kwargs):
        self.energies= tf.cast(energies, dtype=fd.float_type())
        self.rates_vs_energy = tf.cast(rates_vs_energy, dtype=fd.float_type())

        self.ly_relative_energies_keV = tf.cast(np.geomspace(5., 75., 100),
                                                dtype=fd.float_type())
        self.ly_relative = tf.cast(self.light_yield_relative_interp(self.ly_relative_energies_keV),
                                   dtype=fd.float_type())

        super().__init__(*args, **kwargs)

    def eff_light_yield(self, energy, *, eff_ly=11.25):
        """
        """
        ly_relative_interp = tfp.math.interp_regular_1d_grid(energy, self.ly_relative_energies_keV[0],
                                                             self.ly_relative_energies_keV[-1], self.ly_relative)

        return eff_ly * ly_relative_interp

    def light_yield_relative_interp(self, ly_relative_energies_keV):
        """
        """
        #: Fixed model parameters: materials
        Z_NaI = 64
        M_NaI = 149.89
        rho_NaI = 3.67 # g / cm^3
        I_keV_NaI = 0.452 # keV

        # Fixed parameters: model
        c = 2.8
        dEdx_ons = 36.4 # MeV / cm
        eta_eh = 0.534
        dEdx_birks = 166 # MeV / cm

        def dEdx_MeV_cm(E_keV):
            E_ev = 1000. * E_keV
            I_ev = 1000. * I_keV_NaI

            prefactor = 785. * Z_NaI * rho_NaI / M_NaI
            ln_arg = 1.16 * (E_ev + c * I_ev) / I_ev

            dEdx_ev_angstrom = prefactor * np.log(ln_arg) / E_ev
            dEdx_MeV_cm = 1e-6 * 1e8 * dEdx_ev_angstrom

            return dEdx_MeV_cm

        def eta_cap(E_keV):
            dEdx = dEdx_MeV_cm(E_keV)

            numerator = 1. - eta_eh * np.exp(-dEdx / dEdx_ons)
            denominator = 1. + (dEdx / dEdx_birks)

            return numerator / denominator

        def light_yield(E_keV):
            result = integrate.quad(lambda x: eta_cap(x) / (E_keV - I_keV_NaI), I_keV_NaI, E_keV)

            return result[0]

        vlight_yield = np.vectorize(light_yield)

        ly_relative = vlight_yield(ly_relative_energies_keV)
        ly_relative = ly_relative / vlight_yield(46.5)

        return ly_relative

    final_dimensions = ('integrated_charge',)
    no_step_dimensions = ()


@export
class SABREGammaSource(SABREBetaSource):
    pass