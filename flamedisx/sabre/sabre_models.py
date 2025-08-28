import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
from .. import sabre as fd_sabre

import numpy as np
import pandas as pd
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

    @staticmethod
    def light_yield_relative_interp(ly_relative_energies_keV):
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
    @staticmethod
    def light_yield_relative_interp(ly_relative_energies_keV):
        electron_table = {
            "K_bind": [33.17, 33.17, 33.17, 0.0, 0.0, 0.0, 0.0],
            "K_x": [28.32, 28.61, 32.30, 0.0, 0.0, 0.0, 0.0],
            "L_bind": [0.0, 0.0, 0.0, 5.19, 4.85, 4.56, 0.0],
            "M_bind": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87],
            "M_aug": [3.59, 3.32, 0.0, 3.45, 3.59, 3.32, 0.0],
            "valence_aug_1": [0.63, 0.62, 0.87, 0.87, 0.63, 0.62, 0.87],
            "valence_aug_2": [0.63, 0.62, 0.0, 0.87, 0.63, 0.62, 0.0],
            "prob": [0.24, 0.46, 0.13, 0.023, 0.037, 0.07, 0.04]
        }
        electron_table_df = pd.DataFrame(electron_table)

        def gamma_light_yield_j(E_keV, j, electron_table_df):
            row = electron_table_df.iloc[j]

            prob = row['prob']
            row = row.drop('prob')

            energies = []

            if (E_keV < row['L_bind']) and (j in [3, 4, 5]):
                return 0., prob

            if (E_keV < row['K_bind']) and (j in [0, 1, 2]):
                return 0., prob

            if j in [0, 1, 2]:
                pe_energy = E_keV - row['K_bind']
                energies.append(pe_energy)
                row = row.drop('K_bind')
                other_energies = [e for e in row if e > 0.]
                energies.extend(other_energies)
            elif j in [3, 4, 5]:
                pe_energy = E_keV - row['L_bind']
                energies.append(pe_energy)
                row = row.drop('L_bind')
                other_energies = [e for e in row if e > 0.]
                energies.extend(other_energies)
            elif j == 6:
                pe_energy = E_keV - row['M_bind']
                energies.append(pe_energy)
                row = row.drop('M_bind')
                other_energies = [e for e in row if e > 0.]
                energies.extend(other_energies)
            else:
                raise ValueError('Invalid value of j: must be 0-6')

            weighted_ly = 0.
            for interaction_energy in energies:
                weighted_ly += interaction_energy * SABREBetaSource.light_yield_relative_interp(interaction_energy) / E_keV

            return weighted_ly, prob

        def gamma_light_yield(E_keV, electron_table_df):
            electron_table_df = electron_table_df.copy()

            if E_keV < 33.17:
                probs = electron_table_df['prob'].copy()
                probs[0:3] = 0.
                sum_probs = np.sum(probs)
                probs = probs / sum_probs
                electron_table_df['prob'] = probs

            if E_keV < 5.19:
                probs = electron_table_df['prob'].copy()
                probs[0:4] = 0.
                sum_probs = np.sum(probs)
                probs = probs / sum_probs
                electron_table_df['prob'] = probs

            if E_keV < 4.85:
                probs = electron_table_df['prob'].copy()
                probs[0:5] = 0.
                sum_probs = np.sum(probs)
                probs = probs / sum_probs
                electron_table_df['prob'] = probs

            if E_keV < 4.56:
                probs = electron_table_df['prob'].copy()
                probs[0:6] = 0.
                sum_probs = np.sum(probs)
                probs = probs / sum_probs
                electron_table_df['prob'] = probs

            gamma_ly = 0.
            for j in range(7):
                weighted_ly_j, prob_j = gamma_light_yield_j(E_keV, j, electron_table_df)
                gamma_ly += (weighted_ly_j * prob_j)

            return gamma_ly

        vgamma_light_yield = np.vectorize(gamma_light_yield, excluded={'electron_table_df'})

        ly_relative = vgamma_light_yield(ly_relative_energies_keV, electron_table_df=electron_table_df)

        return ly_relative
