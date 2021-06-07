import tensorflow as tf

import configparser, os

import flamedisx as fd
export, __all__ = fd.exporter()


class defaultSource(fd.BlockModelSource):
    def __init__(self, *args, detector='default', **kwargs):
        assert detector in ('default',)

        print(os.path.dirname(__file__))
        print(os.path.join(
        os.path.dirname(__file__), '../config/', detector+'.ini'))
        assert os.path.exists(os.path.join(
        os.path.dirname(__file__), '../config/', detector+'.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../config/', detector+'.ini'))

        # detection.py
        self.photon_detection_eff = config.getfloat('DEFAULT','photon_detection_eff_config')
        self.min_photons = config.getint('DEFAULT','min_photons_config')

        # double_pe.py
        self.double_pe_fraction = config.getfloat('DEFAULT','double_pe_fraction_config')

        # final_signals.py
        self.photoelectron_gain_mean = config.getfloat('DEFAULT','photoelectron_gain_mean_config')
        self.photoelectron_gain_std = config.getfloat('DEFAULT','photoelectron_gain_std_config')
        self.S1_min = config.getfloat('DEFAULT','S1_min_config')
        self.S1_max = config.getfloat('DEFAULT','S1_max_config')
        self.electron_gain_std = config.getfloat('DEFAULT','electron_gain_std_config')
        self.S2_min = config.getfloat('DEFAULT','S2_min_config')
        self.S2_max = config.getfloat('DEFAULT','S2_max_config')

        # energy_spectrum.py
        self.fv_radius =  config.getfloat('DEFAULT','radius_config')
        self.fv_high = config.getfloat('DEFAULT','z_top_config')
        self.fv_low = config.getfloat('DEFAULT','z_bottom_config')
        self.drift_velocity = config.getfloat('DEFAULT','drift_velocity_config')

        super().__init__(*args, **kwargs)


@export
class defaultERSource(defaultSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd.default.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum,
        fd.default.lxe_blocks.quanta_generation.MakeERQuanta,
        fd.default.lxe_blocks.quanta_splitting.MakePhotonsElectronsBetaBinomial,
        fd.default.lxe_blocks.detection.DetectPhotons,
        fd.default.lxe_blocks.double_pe.MakeS1Photoelectrons,
        fd.default.lxe_blocks.final_signals.MakeS1,
        fd.default.lxe_blocks.detection.DetectElectrons,
        fd.default.lxe_blocks.final_signals.MakeS2)

    @staticmethod
    def p_electron(nq, *, er_pel_a=15, er_pel_b=-27.7, er_pel_c=32.5,
                   er_pel_e0=5.):
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
class defaultNRSource(defaultSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd.default.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum,
        fd.default.lxe_blocks.quanta_generation.MakeNRQuanta,
        fd.default.lxe_blocks.quanta_splitting.MakePhotonsElectronsBinomial,
        fd.default.lxe_blocks.detection.DetectPhotons,
        fd.default.lxe_blocks.double_pe.MakeS1Photoelectrons,
        fd.default.lxe_blocks.final_signals.MakeS1,
        fd.default.lxe_blocks.detection.DetectElectrons,
        fd.default.lxe_blocks.final_signals.MakeS2)

    final_dimensions = ('s1', 's2')
    no_step_dimensions = ()

    # Use a larger default energy range, since most energy is lost
    # to heat.
    energies = tf.cast(tf.linspace(0.7, 150., 100),
                       fd.float_type())
    rates_vs_energy = tf.ones(100, fd.float_type())

    @staticmethod
    def p_electron(nq, *,
                   alpha=1.280, zeta=0.045, beta=273 * .9e-4,
                   gamma=0.0141, delta=0.062,
                   drift_field=120):
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
class defaultSpatialRateERSource(defaultERSource):
    model_blocks = (fd.default.lxe_blocks.energy_spectrum.SpatialRateEnergySpectrum,) + defaultERSource.model_blocks[1:]


@export
class defaultSpatialRateNRSource(defaultNRSource):
    model_blocks = (fd.default.lxe_blocks.energy_spectrum.SpatialRateEnergySpectrum,) + defaultNRSource.model_blocks[1:]


@export
class defaultWIMPSource(defaultNRSource):
    model_blocks = (fd.default.lxe_blocks.energy_spectrum.WIMPEnergySpectrum,) + defaultNRSource.model_blocks[1:]
