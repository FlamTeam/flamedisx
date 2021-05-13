import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis

GAS_CONSTANT = 8.314
N_AVAGADRO = 6.0221409e23
A_XENON = 131.293


class nestSource(fd.BlockModelSource):
    def __init__(self, *args, **kwargs):
        assert fd.detector in ('default',)

        # common (known) parameters
        self.temperature = fd.config.getfloat('NEST','temperature_config')
        self.pressure = fd.config.getfloat('NEST','pressure_config')
        self.drift_field = fd.config.getfloat('NEST','drift_field_config')
        # derived (known) parameters
        self.density = fd.calculate_density(self.temperature, self.pressure).item()
        self.drift_velocity = fd.calculate_drift_velocity(self.drift_field,
        self.density, self.temperature).item()
        # detection.py
        self.photon_detection_eff = fd.config.getfloat('NEST','photon_detection_eff_config')
        self.min_photons = fd.config.getint('NEST','min_photons_config')
        # double_pe.py
        self.double_pe_fraction = fd.config.getfloat('NEST','double_pe_fraction_config')
        # final_signals.py
        self.spe_res = fd.config.getfloat('NEST','spe_res_config')
        self.S1_min = fd.config.getfloat('NEST','S1_min_config')
        self.S1_max = fd.config.getfloat('NEST','S1_max_config')
        self.dpe_factor = 1 + fd.config.getfloat('NEST','double_pe_fraction_config')
        self.gas_field = fd.config.getfloat('NEST','gas_field_config')
        self.gas_gap = fd.config.getfloat('NEST','gas_gap_config')
        self.g1_gas = fd.config.getfloat('NEST','g1_gas_config')
        self.S2_min = fd.config.getfloat('NEST','S2_min_config')
        self.S2_max = fd.config.getfloat('NEST','S2_max_config')
        # energy_spectrum.py
        self.radius =  fd.config.getfloat('NEST','radius_config')
        self.z_top = fd.config.getfloat('NEST','z_top_config')
        self.z_bottom = fd.config.getfloat('NEST','z_bottom_config')
        self.z_topDrift = fd.config.getfloat('NEST','z_topDrift_config')

        super().__init__(*args, **kwargs)

    # final_signals.py

    def electron_gain_mean(self):
        rho = self.pressure * 1e5 / (self.temperature * GAS_CONSTANT) * \
        A_XENON * 1e-6
        elYield = (0.137 * self.gas_field*  1e3 - \
        4.70e-18 * (N_AVAGADRO * rho / A_XENON)) * self.gas_gap * 0.1

        return tf.cast(elYield * self.g1_gas, fd.float_type())[o]

    def electron_gain_std(self):
        rho = self.pressure * 1e5 / (self.temperature * GAS_CONSTANT) * \
        A_XENON * 1e-6
        elYield = (0.137 * self.gas_field*  1e3 - \
        4.70e-18 * (N_AVAGADRO * rho / A_XENON)) * self.gas_gap * 0.1

        return tf.sqrt(2 * elYield)[o]


@export
class nestERSource(nestSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd.nest.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum,
        fd.nest.lxe_blocks.quanta_generation.MakeERQuanta,
        fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronsBetaBinomial,
        fd.nest.lxe_blocks.detection.DetectPhotons,
        fd.nest.lxe_blocks.double_pe.MakeS1Photoelectrons,
        fd.nest.lxe_blocks.final_signals.MakeS1,
        fd.nest.lxe_blocks.detection.DetectElectrons,
        fd.nest.lxe_blocks.final_signals.MakeS2)

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
class nestNRSource(nestSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    model_blocks = (
        fd.nest.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum,
        fd.nest.lxe_blocks.quanta_generation.MakeNRQuanta,
        fd.nest.lxe_blocks.quanta_splitting.MakePhotonsElectronsBinomial,
        fd.nest.lxe_blocks.detection.DetectPhotons,
        fd.nest.lxe_blocks.double_pe.MakeS1Photoelectrons,
        fd.nest.lxe_blocks.final_signals.MakeS1,
        fd.nest.lxe_blocks.detection.DetectElectrons,
        fd.nest.lxe_blocks.final_signals.MakeS2)

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
class nestSpatialRateERSource(nestERSource):
    model_blocks = (fd.nest.lxe_blocks.energy_spectrum.SpatialRateEnergySpectrum,) + nestERSource.model_blocks[1:]


@export
class nestSpatialRateNRSource(nestNRSource):
    model_blocks = (fd.nest.lxe_blocks.energy_spectrum.SpatialRateEnergySpectrum,) + nestNRSource.model_blocks[1:]


@export
class nestWIMPSource(nestNRSource):
    model_blocks = (fd.nest.lxe_blocks.energy_spectrum.WIMPEnergySpectrum,) + nestNRSource.model_blocks[1:]
