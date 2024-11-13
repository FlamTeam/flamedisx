import tensorflow as tf
import tensorflow_probability as tfp

import pickle as pkl

import flamedisx as fd
from .. import sabre as fd_sabre

export, __all__ = fd.exporter()


@export
class SABRESource(fd.BlockModelSource):
    model_blocks = (
        fd_sabre.FixedShapeEnergySpectrum,
        fd_sabre.MakePhotons,
        fd_sabre.DetectPhotoelectrons,
        fd_sabre.MakeFinalSignal)

    def __init__(self, *args, spectrum_path=None, **kwargs):
        energy_spectrum = pkl.load(open(spectrum_path, 'rb'))
        self.energies= tf.cast(energy_spectrum[0], dtype=fd.float_type())
        self.rates_vs_energy = tf.cast(energy_spectrum[1], dtype=fd.float_type())

        super().__init__(*args, **kwargs)

    def light_yield(self, energy, *, abs_ly=45.):
        """
        """
        ly_relative_energies_keV = tf.experimental.numpy.geomspace(1., 450., num=100, dtype=fd.float_type())
        ly_relative = self.light_yield_relative_interp(ly_relative_energies_keV)

        ly_relative_interp = tfp.math.interp_regular_1d_grid(energy, 1., 450., ly_relative)

        return abs_ly * ly_relative_interp

    def light_yield_relative_interp(self, ly_relative_energies_keV):
        """
        """
        def light_yield_relative(args):
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

            def eta_cap(E_keV):
                dEdx = dEdx_MeV_cm(E_keV)

                numerator = 1. - eta_eh * tf.math.exp(-dEdx / dEdx_ons)
                denominator = 1. + (dEdx / dEdx_birks)

                return numerator / denominator

            def dEdx_MeV_cm(E_keV):
                E_ev = 1000. * E_keV
                I_ev = 1000. * I_keV_NaI

                prefactor = 785. * Z_NaI * rho_NaI / M_NaI
                ln_arg = 1.16 * (E_ev + c * I_ev) / I_ev

                dEdx_ev_angstrom = prefactor * tf.math.log(ln_arg) / E_ev
                dEdx_MeV_cm = 1e-6 * 1e8 * dEdx_ev_angstrom

                return dEdx_MeV_cm

            E_keV = args[0]

            x = tf.linspace(I_keV_NaI, E_keV, num=1000)
            y = eta_cap(x)

            return tfp.math.trapz(y, x=x) / (E_keV - I_keV_NaI)

        ly_relative = tf.vectorized_map(light_yield_relative, elems=[ly_relative_energies_keV])
        ly_relative = ly_relative / light_yield_relative([450.])

        return ly_relative

    final_dimensions = ('integrated_charge',)
    no_step_dimensions = ()