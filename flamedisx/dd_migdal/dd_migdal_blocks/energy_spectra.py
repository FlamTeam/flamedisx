import numpy as np
import pandas as pd
import tensorflow as tf

import os

from multihist import Histdd
import pickle as pkl

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis

# KE_Spectrum='' # Gauss+Box method
# KE_Spectrum='_CR_100keVnr_min' # Casey's DD Exact
# KE_Spectrum='_CR_conduitCut_100keVnr_min' # Casey's DD Exact with conduit cut applied
# KE_Spectrum='_Mono' # Monoenergetic 2450s
# KE_Spectrum='_J4NDL_Mono' # Monoenergetic 2450s with J4NDL rescaling
# KE_Spectrum='_J4NDL_CR_conduitCut_100keVnr_min' # Casey's DD Exact

#AbundanceTests https://docs.google.com/spreadsheets/d/17IsdiT9ZeBqYlGMyCHETPMagqxxuouwbTvt3vqgJ2yU/edit#gid=949580901
# KE_Spectrum='_G4_test1' 
# KE_Spectrum='_G4_test2' 
# KE_Spectrum='_G4_test3' 
# KE_Spectrum='_G4_test4' 
# KE_Spectrum='_G4_test4_CR'
# KE_Spectrum='_G4_test5' 

# KE_Spectrum='_G4_mono_BRUTEFORCE'
# KE_Spectrum='_G4_mono_BRUTEFORCE_v2'  # MonoKE * (Data/MonoKE_sim)
# KE_Spectrum='_G4_mono_BRUTEFORCE_v3' # CR100 * (Data/CR100_sim)
# KE_Spectrum='_G4_mono_BRUTEFORCE_v4' # MonoKE * BFv2 * (Data/BFv2_sim)
# KE_Spectrum='_G4_mono_BRUTEFORCE_v5' # MonoKE * BFv2 * BFv4 * (Data/BFv4_sim)

# KE_Spectrum='_G4_CR_Chen240702' # Chen's adjustment of CR w/ Qmod https://docs.google.com/presentation/d/1qOjAONrVtlh2mSRRs1Mu54y3XqRHbCTSILAIaLGjHMc/edit#slide=id.g2e976ee5630_0_72
# KE_Spectrum='_G4_CR_Chen_pmod_240702' #Chen's adjustment w/ pmod
# KE_Spectrum='_G4_CR_Chen_pmod_240710_8th' # Chen's adjustment w/ fermi pmod
# KE_Spectrum='_G4_CR_Chen_pmod_20240715_12th_spline' # Chen's adjustment w/ fermi pmod - low energy correction
# KE_Spectrum='_G4_CR_Chen_pmod_20240715_15th_spline' # Chen's adjustment w/ fermi pmod - low energy correction - no low S1 cut in sim
KE_Spectrum='_G4_CR_Chen_pmod_20240718_19th_spline' # Chen's adjustment w/ fermi pmod - low energy correction - no low S1 cut in sim (best fit as of 240718)

print('Using KE Spectrum: %s'%KE_Spectrum)

@export
class EnergySpectrumFirstMSU(fd.FirstBlock):
    dimensions = ('energy_first',)
    model_attributes = ('energies_first', 'rates_vs_energy_first')

    model_functions = ('get_r_dt_diff_rate', 'get_S2Width_diff_rate')

    r_dt_dist = np.load(os.path.join(
        os.path.dirname(__file__), '../migdal_database/NR_spatial_template.npz'))

    r_edges = r_dt_dist['r_edges']
    dt_edges = r_dt_dist['dt_edges']
    hist_values_r_dt = r_dt_dist['hist_values']

    mh_r_dt = Histdd(bins=[len(r_edges) - 1, len(dt_edges) - 1]).from_histogram(hist_values_r_dt, bin_edges=[r_edges, dt_edges])
    mh_r_dt = mh_r_dt / mh_r_dt.n
    mh_r_dt = mh_r_dt / mh_r_dt.bin_volumes()

    r_dt_diff_rate = mh_r_dt
    r_dt_events_per_bin = mh_r_dt * mh_r_dt.bin_volumes()

    #: Energies from the first scatter
    energies_first = tf.cast(tf.linspace(1.75, 97.95, 65), dtype=fd.float_type()) # Standard
    # energies_first = tf.cast(tf.linspace(0.0, 150.0, 100),dtype=fd.float_type()) # increased range for testing
    
    #: Dummy energy spectrum of 1s. Override for SS
    rates_vs_energy_first = tf.ones(65, dtype=fd.float_type()) / sum(tf.ones(65, dtype=fd.float_type())) # 240405 AV: added /sum --> integrate to 1

    def get_r_dt_diff_rate(self, r_dt_diff_rate):
        return r_dt_diff_rate

    def get_S2Width_diff_rate(self, S2Width_diff_rate):
        return S2Width_diff_rate

    def _compute(self, data_tensor, ptensor, *, energy_first):
        spectrum = tf.repeat(fd.np_to_tf(self.rates_vs_energy_first)[o, :],
                             self.source.batch_size,
                             axis=0)
        
        spectrum *= self.source.mu_before_efficiencies()

#         spectrum *= tf.repeat(self.gimme('get_r_dt_diff_rate', ### turn off to test s1s2 diff rates alone! TODO
#                                          data_tensor=data_tensor,
#                                          ptensor=ptensor)[:, o],
#                               tf.shape(self.energies_first),
#                               axis=1)

        spectrum *= tf.repeat(self.gimme('get_S2Width_diff_rate',  ### turn off to test s1s2 diff rates alone!
                                         data_tensor=data_tensor,
                                         ptensor=ptensor)[:, o],
                              tf.shape(self.energies_first),
                              axis=1)

#################

        # spectrum = tf.repeat(self.gimme('get_r_dt_diff_rate', ### TEST ISOLATED PDF FOR P-VAL TESTING
        #                                  data_tensor=data_tensor,
        #                                  ptensor=ptensor)[:, o],
        #                       tf.shape(self.energies_first),
        #                       axis=1)
        # spectrum = tf.repeat(self.gimme('get_S2Width_diff_rate', ### TEST ISOLATED PDF FOR P-VAL TESTING
        #                                  data_tensor=data_tensor,
        #                                  ptensor=ptensor)[:, o],
        #                       tf.shape(self.energies_first),
        #                       axis=1)

        return spectrum

    def domain(self, data_tensor):
        return {self.dimensions[0]: tf.repeat(fd.np_to_tf(self.energies_first)[o, :],
                                              self.source.batch_size,
                                              axis=0)}

    def _annotate(self, d):
        d['r_dt_diff_rate'] = self.r_dt_diff_rate.lookup(
            *[d['r'], d['drift_time']])

        d['S2Width_diff_rate'] = self.source.S2Width_diff_rate.lookup(d['S2Width'])

    def random_truth(self, n_events, fix_truth=None, **params):
        """Return pandas dataframe with event positions and times
        randomly drawn.
        """
        # The energy spectrum is represented as a 'comb' of delta functions
        # in the differential rate calculation. To get simulator-data agreement,
        # we have to do the same thing here.
        # TODO: can we fix both to be better?
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy_first)
        assert len(spectrum_numpy) == len(self.energies_first), \
            "Energies and spectrum have different length"

        data = dict()
        data['energy_first'] = np.random.choice(
            fd.tf_to_np(self.energies_first),
            size=n_events,
            p=spectrum_numpy / spectrum_numpy.sum(),
            replace=True)
        assert np.all(data['energy_first'] >= 0), "Generated negative energies??"

        r_dt = self.r_dt_events_per_bin.get_random(n_events)
        data ['r'] = r_dt[:, 0]
        data ['drift_time'] = r_dt[:, 1]

        data ['S2Width'] = self.source.S2Width_events_per_bin.get_random(n_events)

        # For a constant-shape spectrum, fixing truth values is easy:
        # we just overwrite the simulated values.
        # Fixing one does not constrain any of the others.
        self.source._overwrite_fixed_truths(data, fix_truth, n_events)

        data = pd.DataFrame(data)
        return data

    def validate_fix_truth(self, d):
        """
        """
        if d is not None:
            raise RuntimeError("Currently don't support fix_truth for this source")
        return dict()


@export
class EnergySpectrumFirstMSU3(EnergySpectrumFirstMSU):
    #: Energies from the first scatter
    energies_first = tf.cast(tf.linspace(3., 95., 24), dtype=fd.float_type())
    #: Dummy energy spectrum of 1s
    rates_vs_energy_first = tf.ones(24, dtype=fd.float_type()) / sum(tf.ones_like(energies_first,dtype=fd.float_type())) # 240405 AV: added /sum --> integrate to 1


@export
class EnergySpectrumFirstSS(EnergySpectrumFirstMSU):
    SS_Spectrum_filename = '../migdal_database/SS_spectrum'+KE_Spectrum+'.pkl'
    # SS_Spectrum_filename = '../migdal_database/SS_spectrum_CR_100keVnr_min.pkl'
    #: Energy spectrum for SS case
    rates_vs_energy_first = pkl.load(open(os.path.join(os.path.dirname(__file__), SS_Spectrum_filename), 'rb'))
    
    # ### Flat NR band instead
    # rates_vs_energy_first = tf.ones(100, dtype=fd.float_type()) / sum( tf.ones(100, dtype=fd.float_type()))  # 240711 testing flat NR spectrum
    
    assert np.isclose(np.sum(rates_vs_energy_first), 1.)


@export
class EnergySpectrumFirstMigdal(EnergySpectrumFirstMSU):
    #: Energies from the first scatter
    energies_first = fd.np_to_tf(np.geomspace(1.04712855e-02, 9.54992586e+01, 100))
    #: Dummy energy spectrum of 1s
    rates_vs_energy_first = tf.ones(100, dtype=fd.float_type()) / sum( tf.ones_like(energies_first,dtype=fd.float_type())) # 240405 AV: added /sum --> integrate to 1


@export
class EnergySpectrumFirstMigdalMSU(EnergySpectrumFirstMSU):
    #: Energies from the first scatter
    energies_first = fd.np_to_tf(np.geomspace(0.11167041, 17.90984679, 24))
    #: Dummy energy spectrum of 1s
    rates_vs_energy_first = tf.ones(24, dtype=fd.float_type()) / sum(tf.ones_like(energies_first,dtype=fd.float_type())) # 240405 AV: added /sum --> integrate to 1


@export
class EnergySpectrumFirstIE_CS(EnergySpectrumFirstMSU):
    #: Energies from the first scatter
    energies_first = fd.np_to_tf(np.geomspace(1.04126487e-02, 2.88111130e+01, 99))
    #: Dummy energy spectrum of 1s
    rates_vs_energy_first = tf.ones(99, dtype=fd.float_type()) / sum(tf.ones_like(energies_first,dtype=fd.float_type()))

    r_dt_dist = np.load(os.path.join(
        os.path.dirname(__file__), '../migdal_database/IE_CS_spatial_template.npz'))

    hist_values_r_dt = r_dt_dist['hist_values']
    r_edges = r_dt_dist['r_edges']
    dt_edges = r_dt_dist['dt_edges']

    mh_r_dt = Histdd(bins=[len(r_edges) - 1, len(dt_edges) - 1]).from_histogram(hist_values_r_dt, bin_edges=[r_edges, dt_edges])
    mh_r_dt = mh_r_dt / mh_r_dt.n
    mh_r_dt = mh_r_dt / mh_r_dt.bin_volumes()

    r_dt_diff_rate = mh_r_dt
    r_dt_events_per_bin = mh_r_dt * mh_r_dt.bin_volumes()


@export
class EnergySpectrumFirstER(EnergySpectrumFirstMSU):
    #: Flat ER energy spectrum
    energies_first = tf.cast(tf.linspace(0.01, 35., 100), fd.float_type())
    rates_vs_energy_first = tf.ones_like(energies_first, fd.float_type()) / sum(tf.ones_like(energies_first,dtype=fd.float_type()))

    r_dt_dist = np.load(os.path.join(
        os.path.dirname(__file__), '../migdal_database/ER_spatial_template.npz'))

    hist_values_r_dt = r_dt_dist['hist_values']
    r_edges = r_dt_dist['r_edges']
    dt_edges = r_dt_dist['dt_edges']

    mh_r_dt = Histdd(bins=[len(r_edges) - 1, len(dt_edges) - 1]).from_histogram(hist_values_r_dt, bin_edges=[r_edges, dt_edges])
    mh_r_dt = mh_r_dt / mh_r_dt.n
    mh_r_dt = mh_r_dt / mh_r_dt.bin_volumes()

    r_dt_diff_rate = mh_r_dt
    r_dt_events_per_bin = mh_r_dt * mh_r_dt.bin_volumes()


@export
class EnergySpectrumSecondMSU(fd.Block):
    dimensions = ('energy_second',)
    model_attributes = ('energies_second', 'rates_vs_energy')

    #: Energies from the second scatter
    energies_second = tf.cast(tf.linspace(1.75, 97.95, 65),
                            dtype=fd.float_type())
    #: Joint energy spectrum for MSU scatters. Override for other double scatters
    MSU2_Spectrum_filename = '../migdal_database/MSU_spectrum'+KE_Spectrum+'.pkl'
    # MSU2_Spectrum_filename = '../migdal_database/MSU_spectrum_CR_100keVnr_min.pkl'
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), MSU2_Spectrum_filename), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ignore_shape_assertion=True)

    def _compute(self, data_tensor, ptensor, *, energy_second):
        spectrum = tf.repeat(fd.np_to_tf(self.rates_vs_energy)[o, :, :],
                             self.source.batch_size,
                             axis=0)
        return spectrum

    def _simulate(self, d):
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy)
        spectrum_flat = spectrum_numpy.flatten()

        Es_first = np.repeat(fd.tf_to_np(self.source.energies_first)[:, np.newaxis], len(fd.tf_to_np(self.energies_second)), axis=1)
        Es_second = np.repeat(fd.tf_to_np(self.energies_second)[np.newaxis, :], len(fd.tf_to_np(self.source.energies_first)), axis=0)
        Es = np.dstack((Es_first, Es_second))
        Es_flat = Es.reshape(-1, Es.shape[-1])

        assert np.shape(Es)[0] == np.shape(spectrum_numpy)[0], \
            "Energies and spectrum have different length"
        assert np.shape(Es)[1] == np.shape(spectrum_numpy)[1], \
            "Energies and spectrum have different length"

        energy_index = np.random.choice(
            np.arange(spectrum_flat.size),
            size=len(d),
            p=spectrum_flat / spectrum_flat.sum(),
            replace=True)
        d['energy_first'] = Es_flat[energy_index][:, 0]
        d['energy_second'] = Es_flat[energy_index][:, 1]

        assert np.all(d['energy_first'] >= 0), "Generated negative energies??"
        assert np.all(d['energy_second'] >= 0), "Generated negative energies??"

    def _annotate(self, d):
        d['energy_second_min'] = fd.tf_to_np(self.energies_second)[0]
        d['energy_second_max'] = fd.tf_to_np(self.energies_second)[-1]

    def _calculate_dimsizes_special(self):
        d = self.source.data

        self.source.dimsizes['energy_second'] = len(self.energies_second)

        d_energy = np.diff(self.energies_second)
        d['energy_second_steps'] = d_energy[0]

        assert np.isclose(self.energies_second[0] + (len(self.energies_second) - 1) * d_energy[0],
                          self.energies_second[-1]), "Logic only works with constant stepping in energy spectrum"


@export
class EnergySpectrumOthersMSU3(fd.Block):
    dimensions = ('energy_others',)
    model_attributes = ('energies_others', 'rates_vs_energy')

    #: Energies from the scatters
    energies_others = tf.cast(tf.linspace(3., 95., 24), dtype=fd.float_type())
    #: Joint energy spectrum for MSU3 scatters. Override for other triple scatters
    MSU3_Spectrum_filename = '../migdal_database/MSU3_spectrum'+KE_Spectrum+'.pkl'
    # MSU3_Spectrum_filename = '../migdal_database/MSU3_spectrum_CR_100keVnr_min.pkl'
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), MSU3_Spectrum_filename), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ignore_shape_assertion=True)

    def _compute(self, data_tensor, ptensor, *, energy_others):
        spectrum = tf.repeat(fd.np_to_tf(self.rates_vs_energy)[o, :, :, :],
                             self.source.batch_size,
                             axis=0)
        return spectrum

    def _simulate(self, d):
        spectrum_numpy = fd.tf_to_np(self.rates_vs_energy)
        spectrum_flat = spectrum_numpy.flatten()

        Es_first = self.source.energies_first
        Es_others = self.energies_others

        E1_mesh, E2_mesh, E3_mesh = np.meshgrid(Es_first, Es_others, Es_others, indexing='ij')
        Es = np.stack((E1_mesh, E2_mesh, E3_mesh), axis=-1)
        Es_flat = Es.reshape(-1, Es.shape[-1])

        assert np.shape(Es)[0] == np.shape(spectrum_numpy)[0], \
            "Energies and spectrum have different length"
        assert np.shape(Es)[1] == np.shape(spectrum_numpy)[1], \
            "Energies and spectrum have different length"
        assert np.shape(Es)[2] == np.shape(spectrum_numpy)[2], \
            "Energies and spectrum have different length"

        energy_index = np.random.choice(
            np.arange(spectrum_flat.size),
            size=len(d),
            p=spectrum_flat / spectrum_flat.sum(),
            replace=True)
        d['energy_first'] = Es_flat[energy_index][:, 0]
        d['energy_second'] = Es_flat[energy_index][:, 1]
        d['energy_third'] = Es_flat[energy_index][:, 2]

        assert np.all(d['energy_first'] >= 0), "Generated negative energies??"
        assert np.all(d['energy_second'] >= 0), "Generated negative energies??"
        assert np.all(d['energy_third'] >= 0), "Generated negative energies??"

    def _annotate(self, d):
        d['energy_others_min'] = fd.tf_to_np(self.energies_others)[0]
        d['energy_others_max'] = fd.tf_to_np(self.energies_others)[-1]

    def _calculate_dimsizes_special(self):
        d = self.source.data

        self.source.dimsizes['energy_others'] = len(self.energies_others)

        d_energy = np.diff(self.energies_others)
        d['energy_others_steps'] = d_energy[0]

        assert np.isclose(self.energies_others[0] + (len(self.energies_others) - 1) * d_energy[0],
                          self.energies_others[-1]), "Logic only works with constant stepping in energy spectrum"


@export
class EnergySpectrumSecondMigdal2(EnergySpectrumSecondMSU):
    #: Energies from the second scatter
    energies_second = tf.cast(tf.linspace(0.75, 98.25, 66),
                            dtype=fd.float_type())
    #: Joint energy spectrum for Migdal2 scatters
    Mig2_Spectrum_filename = '../migdal_database/migdal_2_spectrum'+KE_Spectrum+'.pkl'
    # Mig2_Spectrum_filename = '../migdal_database/migdal_2_spectrum_CR_100keVnr_min.pkl'
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), Mig2_Spectrum_filename), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)


@export
class EnergySpectrumSecondMigdal3(EnergySpectrumSecondMSU):
    #: Energies from the second scatter
    energies_second = tf.cast(tf.linspace(0.75, 98.25, 66),
                            dtype=fd.float_type())
    #: Joint energy spectrum for Migdal3 scatters
    Mig3_Spectrum_filename = '../migdal_database/migdal_3_spectrum'+KE_Spectrum+'.pkl'
    # Mig3_Spectrum_filename = '../migdal_database/migdal_3_spectrum_CR_100keVnr_min.pkl'
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), Mig3_Spectrum_filename), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)


@export
class EnergySpectrumSecondMigdal4(EnergySpectrumSecondMSU):
    #: Energies from the second scatter
    energies_second = tf.cast(tf.linspace(0.75, 98.25, 66),
                            dtype=fd.float_type())
    #: Joint energy spectrum for Migdal4 scatters
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), '../migdal_database/migdal_4_spectrum.pkl'), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)


@export
class EnergySpectrumOthersMigdalMSU(EnergySpectrumOthersMSU3):
    #: Energies from the second scatter
    energies_others = tf.cast(tf.linspace(2.5, 80.5, 27),
                            dtype=fd.float_type())
    energies_second = energies_others
    #: Joint energy spectrum for MigdalMSU scatters
    MigMSU_Spectrum_filename = '../migdal_database/migdal_MSU_spectrum'+KE_Spectrum+'.pkl'
    # MigMSU_Spectrum_filename = '../migdal_database/migdal_MSU_spectrum_CR_100keVnr_min.pkl'
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), MigMSU_Spectrum_filename), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)


@export
class EnergySpectrumSecondIE_CS(EnergySpectrumSecondMSU):
    #: Joint energy spectrum for IE + CS scatters
    if KE_Spectrum=='_Mono':
        KE_Spectrum_IE = ''
    else:
        KE_Spectrum_IE = str(KE_Spectrum)
    rates_vs_energy = pkl.load(open(os.path.join(
        os.path.dirname(__file__), '../migdal_database/IE_CS_spectrum.pkl'), 'rb')) # Use this one regardless of KE spectrum
        # os.path.dirname(__file__), '../migdal_database/IE_CS_spectrum'+KE_Spectrum_IE+'.pkl'), 'rb'))
    assert np.isclose(np.sum(rates_vs_energy), 1.)
