"""Toy XLZD detector implementation

"""
import numpy as np
import tensorflow as tf

import configparser
import os
import pandas as pd

import flamedisx as fd
from .. import nest as fd_nest

export, __all__ = fd.exporter()


##
# Flamedisx sources
##

class XLZDSource:
    def __init__(self, *args,
                 drift_field_V_cm=80., gas_field_kV_cm=7.5, elife_ns=10000e3, g1=0.31,
                 temperature_K=174.1, pressure_bar=1.79, num_pmts=902, double_pe_fraction=0.2,
                 g1_gas=0.1, s2Fano=2., spe_res=0.38, spe_thr=0.375, spe_eff=1.,
                 cS1_min=0., cS1_max=100., log10_cS2_min=2.5, log10_cS2_max=4.,
                 s2_thr=198., coin_level=4,
                 ignore_maps_acc=False, **kwargs):
        super().__init__(*args, **kwargs)

        if ignore_maps_acc:
            self.ignore_acceptances = True

        assert kwargs['detector'] in ('xlzd',)
        assert kwargs['configuration'] in ('80t', '60t', '40t')

        assert os.path.exists(os.path.join(
            os.path.dirname(__file__), '../nest/config/', kwargs['detector'] + '.ini'))

        config = configparser.ConfigParser(inline_comment_prefixes=';')
        config.read(os.path.join(os.path.dirname(__file__), '../nest/config/',
                                 kwargs['detector'] + '.ini'))

        self.radius = config.getfloat(kwargs['configuration'], 'radius_config')
        self.z_topDrift = config.getfloat(kwargs['configuration'], 'z_topDrift_config')
        self.z_top = config.getfloat(kwargs['configuration'], 'z_top_config')
        self.z_bottom = config.getfloat(kwargs['configuration'], 'z_bottom_config')

        self.configuration = kwargs['configuration']

        self.drift_field = drift_field_V_cm
        self.gas_field = gas_field_kV_cm
        self.elife = elife_ns
        self.g1 = g1 # this represents PMT QE
        self.temperature = temperature_K
        self.pressure = pressure_bar
        self.num_pmts = num_pmts
        self.double_pe_fraction = double_pe_fraction
        self.g1_gas = g1_gas
        self.s2Fano = s2Fano
        self.spe_res = spe_res
        self.spe_thr = spe_thr
        self.spe_eff = spe_eff

        self.density = fd_nest.calculate_density(
            self.temperature, self.pressure)
        self.density_gas = fd_nest.calculate_density_gas(
            self.temperature, self.pressure)
        self.drift_velocity = fd_nest.calculate_drift_velocity(
            self.drift_field, self.density, self.temperature)
        self.Wq_keV, self.alpha = fd_nest.calculate_work(self.density)
        self.extraction_eff = fd_nest.calculate_extraction_eff(self.gas_field, self.temperature)
        self.s1_mean_mult = fd_nest.calculate_s1_mean_mult(self.spe_res)
        self.g2 = fd_nest.calculate_g2(self.gas_field, self.density_gas, self.gas_gap,
                                       self.g1_gas, self.extraction_eff)


        self.cS1_min = cS1_min
        self.cS1_max = cS1_max
        self.log10_cS2_min = log10_cS2_min
        self.log10_cS2_max = log10_cS2_max
        self.s2_thr = s2_thr
        self.coin_table = fd_nest.get_coin_table(coin_level, self.num_pmts,
                                                 self.spe_res, self.spe_thr, self.spe_eff,
                                                 self.double_pe_fraction)

    def s1_acceptance(self, s1, cs1):

        acceptance = tf.where((s1 >= self.spe_thr) & (cs1 >= self.cS1_min) & (cs1 <= self.cS1_max),
                              tf.ones_like(s1, dtype=fd.float_type()),  # if condition non-zero
                              tf.zeros_like(s1, dtype=fd.float_type()))  # if false

        return acceptance

    def s2_acceptance(self, s2, cs2):

        log10_cs2 = np.log10(cs2 + 1e-10)

        acceptance = tf.where((s2 >= self.s2_thr) &
                              (log10_cs2 >= self.log10_cS2_min) & (log10_cs2 <= self.log10_cS2_max),
                              tf.ones_like(s2, dtype=fd.float_type()),  # if condition non-zero
                              tf.zeros_like(s2, dtype=fd.float_type()))  # if false

        return acceptance

    def s1_posDependence(self, z):
        """
        Returns LCE. PMT QE then handled by the g1 value.
        Coefficients come from fit to LCE curve obtained by Theresa Fruth via
        BACCARAT.
        Requires z to be in cm, and in the FV.
        """
        if self.configuration == '80t':
            a = 5.01480764786202e-01
            b = 1.0987171117870357e-03
            c = 2.6949708579314157e-06
            d = -4.6066555019055335e-09
            e = -3.1658521366562203e-12
        elif self.configuration == '60t':
            a = 5.550088308957059e-01
            b = 7.959774439474241e-04
            c = 1.9863250116287806e-06
            d = -1.4497010098307493e-08
            e = -2.028805722848711e-11
        elif self.configuration == '40t':
            a = 6.324164580843357e-01
            b = 3.980052004436636e-04
            c = 8.151870713156558e-07
            d = -4.238165802951504e-08
            e = -1.0960362072784391e-10

        LCE = a + b * z + c * z**2 + d * z**3 + e * z**4

        return LCE

    def add_extra_columns(self, d):
        super().add_extra_columns(d)

        if self.configuration == '80t':
            LCE_average = 0.4829
        elif self.configuration == '60t':
            LCE_average = 0.5601
        elif self.configuration == '40t':
            LCE_average = 0.6532
        d['s1_pos_corr'] = self.s1_posDependence(d['z'].values) / LCE_average # normalise to volume-averaged LCE

        if 's1' in d.columns and 'cs1' not in d.columns:
            d['cs1'] = d['s1'] / d['s1_pos_corr']
        if 's2' in d.columns and 'cs2' not in d.columns:
            d['cs2'] = d['s2'] * np.exp(d['drift_time'] / self.elife)

        if 'cs1' in d.columns and 'cs2' in d.columns and 'ces_er_equivalent' not in d.columns:
             d['ces_er_equivalent'] = (d['cs1'] / (self.g1 * LCE_average) + d['cs2'] / self.g2) * self.Wq_keV / (1. + self.double_pe_fraction)


@export
class XLZDERSource(XLZDSource, fd.nest.nestERSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDNRSource(XLZDSource, fd.nest.nestNRSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


##
# Signal sources
##


@export
class XLZDWIMPSource(XLZDSource, fd.nest.nestWIMPSource):

    def __init__(
        self,
        *args,
        wimp_mass=40,
        sigma=1e-45,
        fid_mass=1.0,
        min_E=1e-2,
        max_E=80.0,
        n_energy_bins=800,
        min_time="2019-09-01T08:28:00",
        max_time="2020-09-01T08:28:00",
        livetime=1.0,
        n_time_bins=25,
        modulation=True,
        **kwargs
    ):
        if "detector" not in kwargs:
            kwargs["detector"] = "xlzd"
        if "configuration" not in kwargs:
            kwargs["configuration"] = "80t"
        super().__init__(
            *args,
            wimp_mass=wimp_mass,
            sigma=sigma,
            fid_mass=fid_mass,
            min_E=min_E,
            max_E=max_E,
            n_energy_bins=n_energy_bins,
            min_time=min_time,
            max_time=max_time,
            livetime=livetime,
            n_time_bins=n_time_bins,
            modulation=modulation,
            **kwargs
        )


@export
class XLZDMigdalSource(XLZDSource, fd.nest.nestMigdalSource):

    def __init__(
        self,
        *args,
        wimp_mass=40,
        sigma=1e-45,
        fid_mass=1.0,
        min_E=1e-2,
        max_E=80.0,
        n_energy_bins=800,
        min_time="2019-09-01T08:28:00",
        max_time="2020-09-01T08:28:00",
        livetime=1.0,
        n_time_bins=25,
        modulation=True,
        migdal_model="Cox",
        **kwargs
    ):
        if "detector" not in kwargs:
            kwargs["detector"] = "xlzd"
        if "configuration" not in kwargs:
            kwargs["configuration"] = "80t"
        super().__init__(
            *args,
            wimp_mass=wimp_mass,
            sigma=sigma,
            fid_mass=fid_mass,
            min_E=min_E,
            max_E=max_E,
            n_energy_bins=n_energy_bins,
            min_time=min_time,
            max_time=max_time,
            livetime=livetime,
            n_time_bins=n_time_bins,
            modulation=modulation,
            migdal_model=migdal_model,
            **kwargs
        )


@export
class XLZDEFTScalarO6Source(XLZDSource, fd.nest.EFTScalarO6Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDALPGalacticDMSource(XLZDSource, fd.nest.ALPGalacticDMSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDHiddenPhotonSource(XLZDSource, fd.nest.HiddenPhotonSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


##
# Background sources
##


@export
class XLZDXe136Source(XLZDSource, fd.nest.Xe136Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDXe124Source(XLZDSource, fd.nest.Xe124Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDPb214Source(XLZDSource, fd.nest.Pb214Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDKr85Source(XLZDSource, fd.nest.Kr85Source):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDvERSource(XLZDSource, fd.nest.vERSource, fd.nest.nestTemporalRateOscillationERSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'

        if amplitude is None:
            self.amplitude = 2. * 0.01671
        else:
            self.amplitude = amplitude

        if phase_ns is None:
            self.phase_ns = pd.to_datetime('2022-01-04T00:00:00').value
        else:
            self.phase_ns = phase_ns

        if period_ns is None:
            self.period_ns = 1. * 3600. * 24. * 365.25 * 1e9
        else:
            self.period_ns = period_ns

        super().__init__(*args, **kwargs)


@export
class XLZDvNRSolarSource(XLZDSource, fd.nest.vNRSolarSource, fd.nest.nestTemporalRateOscillationNRSource):
    def __init__(self, *args, amplitude=None, phase_ns=None, period_ns=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'

        if amplitude is None:
            self.amplitude = 2. * 0.01671
        else:
            self.amplitude = amplitude

        if phase_ns is None:
            self.phase_ns = pd.to_datetime('2022-01-04T00:00:00').value
        else:
            self.phase_ns = phase_ns

        if period_ns is None:
            self.period_ns = 1. * 3600. * 24. * 365.25 * 1e9
        else:
            self.period_ns = period_ns

        super().__init__(*args, **kwargs)


@export
class XLZDvNROtherLNGSSource(XLZDSource, fd.nest.vNROtherLNGSSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDvNROtherSURFSource(XLZDSource, fd.nest.vNROtherSURFSource):
    def __init__(self, *args, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'
        super().__init__(*args, **kwargs)


@export
class XLZDNeutronSource(XLZDSource, fd.nest.NeutronSource, fd.nest.nestSpatialRateDecayNRSource):
    def __init__(self, *args, decay_constant=None, **kwargs):
        if ('detector' not in kwargs):
            kwargs['detector'] = 'xlzd'
        if ('configuration' not in kwargs):
            kwargs['configuration'] = '80t'

        if decay_constant is None:
            self.decay_constant = 3.57 # cm; from XLZD GEANT4 simulations
        else:
            self.decay_constant = decay_constant

        super().__init__(*args, **kwargs)
