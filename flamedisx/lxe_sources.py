import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()

import configparser, os

config = configparser.ConfigParser(inline_comment_prefixes=';')
config.read(os.path.join(os.path.dirname(__file__), 'config', fd.config_file))


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
    def p_electron(nq, *, er_pel_a=config.getfloat('DEFAULT','er_pel_a_guess'),
                   er_pel_b=config.getfloat('DEFAULT','er_pel_b_guess'),
                   er_pel_c=config.getfloat('DEFAULT','er_pel_c_guess'),
                   er_pel_e0=config.getfloat('DEFAULT','er_pel_e0_guess')):
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
    no_step_dimensions = ()


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
    no_step_dimensions = ()

    # Use a larger default energy range, since most energy is lost
    # to heat.
    energies = tf.cast(tf.linspace(0.7, 150., 100),
                       fd.float_type())
    rates_vs_energy = tf.ones(100, fd.float_type())

    @staticmethod
    def p_electron(nq, *,
                   alpha=config.getfloat('DEFAULT','alpha_guess'),
                   zeta=config.getfloat('DEFAULT','zeta_guess'),
                   beta=config.getfloat('DEFAULT','beta_guess'),
                   gamma=config.getfloat('DEFAULT','gamma_guess'),
                   delta=config.getfloat('DEFAULT','delta_guess'),
                   drift_field=config.getfloat('DEFAULT','drift_field_guess')):
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
class SpatialRateERSource(ERSource):
    model_blocks = (fd.SpatialRateEnergySpectrum,) + ERSource.model_blocks[1:]


@export
class SpatialRateNRSource(NRSource):
    model_blocks = (fd.SpatialRateEnergySpectrum,) + NRSource.model_blocks[1:]


@export
class WIMPSource(NRSource):
    model_blocks = (fd.WIMPEnergySpectrum,) + NRSource.model_blocks[1:]
