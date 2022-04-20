import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import operator

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakePhotonsElectronsNR(fd.Block):
    is_ER = False

    dimensions = ('electrons_produced', 'photons_produced')
    bonus_dimensions = (('ions_produced', True),)
    depends_on = ((('energy',), 'rate_vs_energy'),)

    max_dim_size = {'ions_produced': 30}

    exclude_data_tensor = ('ions_produced_max',)

    special_model_functions = ('mean_yields', 'recomb_prob', 'skewness', 'variance',
                               'width_correction', 'mu_correction')
    model_functions = special_model_functions

    def setup(self):
        self.array_columns = (('ions_produced_min',
                               max(min(len(self.source.energies),
                                       self.source.max_dim_sizes['energy']),
                                   2)),)

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 electrons_produced, photons_produced,
                 # Bonus dimension
                 ions_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy):

        def compute_single_energy(args, approx=False):
            # Compute the block for a single energy.
            # Set approx to True for an approximate computation at higher energies

            energy = args[0]
            rate_vs_energy = args[1]
            ions_min = args[2]

            ions_min = tf.repeat(ions_min[:, o], tf.shape(ions_produced)[1], axis=1)
            ions_min = tf.repeat(ions_min[:, :, o], tf.shape(ions_produced)[2], axis=2)
            ions_min = tf.repeat(ions_min[:, :, :, o], tf.shape(ions_produced)[3], axis=3)

            # Calculate the ion domain tensor for this energy
            _ions_produced = ions_produced_add + ions_min

            if self.is_ER:
                nel_mean = self.gimme('mean_yield_electron', data_tensor=data_tensor, ptensor=ptensor,
                                      bonus_arg=energy)
                nq_mean = self.gimme('mean_yield_quanta', data_tensor=data_tensor, ptensor=ptensor,
                                     bonus_arg=(energy, nel_mean))
                fano = self.gimme('fano_factor', data_tensor=data_tensor, ptensor=ptensor,
                                  bonus_arg=nq_mean)

                if approx:
                    p_nq = tfp.distributions.Normal(loc=nq_mean,
                                                    scale=tf.sqrt(nq_mean * fano) + 1e-10).prob(nq)
                else:
                    normal_dist_nq = tfp.distributions.Normal(loc=nq_mean,
                                                              scale=tf.sqrt(nq_mean * fano) + 1e-10)
                    p_nq = normal_dist_nq.cdf(nq + 0.5) - normal_dist_nq.cdf(nq - 0.5)

                ex_ratio = self.gimme('exciton_ratio', data_tensor=data_tensor, ptensor=ptensor,
                                      bonus_arg=energy)
                alpha = 1. / (1. + ex_ratio)

                p_ni = tfp.distributions.Binomial(
                    total_count=nq, probs=alpha).prob(_ions_produced)

            else:
                yields = self.gimme('mean_yields', data_tensor=data_tensor, ptensor=ptensor,
                                    bonus_arg=energy)
                nel_mean = yields[0]
                nq_mean = yields[1]
                ex_ratio = yields[2]
                alpha = 1. / (1. + ex_ratio)

                if approx:
                    p_ni = tfp.distributions.Normal(loc=nq_mean*alpha,
                                                    scale=tf.sqrt(nq_mean*alpha) + 1e-10).prob(_ions_produced)

                    p_nq = tfp.distributions.Normal(loc=nq_mean*alpha*ex_ratio,
                                                    scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10).prob(
                                                        nq - _ions_produced)
                else:
                    normal_dist_ni = tfp.distributions.Normal(loc=nq_mean*alpha,
                                                              scale=tf.sqrt(nq_mean*alpha) + 1e-10)
                    p_ni = normal_dist_ni.cdf(_ions_produced + 0.5) - \
                        normal_dist_ni.cdf(_ions_produced - 0.5)

                    normal_dist_nq = tfp.distributions.Normal(loc=nq_mean*alpha*ex_ratio,
                                                              scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10)
                    p_nq = normal_dist_nq.cdf(nq - _ions_produced + 0.5) \
                        - normal_dist_nq.cdf(nq - _ions_produced - 0.5)

            recomb_p = self.gimme('recomb_prob', data_tensor=data_tensor, ptensor=ptensor,
                                  bonus_arg=(nel_mean, nq_mean, ex_ratio))
            skew = self.gimme('skewness', data_tensor=data_tensor, ptensor=ptensor,
                              bonus_arg=nq_mean)
            var = self.gimme('variance', data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=(nel_mean, nq_mean, recomb_p, _ions_produced))
            width_corr = self.gimme('width_correction', data_tensor=data_tensor, ptensor=ptensor,
                                    bonus_arg=skew)
            mu_corr = self.gimme('mu_correction', data_tensor=data_tensor, ptensor=ptensor,
                                 bonus_arg=(skew, var, width_corr))

            mean = (tf.ones_like(_ions_produced, dtype=fd.float_type()) - recomb_p) * _ions_produced - mu_corr
            std_dev = tf.sqrt(var) / width_corr

            if self.is_ER:
                owens_t_terms = 5
            else:
                owens_t_terms = 2

            if approx:
                p_nel = fd.tfp_files.SkewGaussian(loc=mean, scale=std_dev,
                                                  skewness=skew,
                                                  owens_t_terms=owens_t_terms).prob(electrons_produced)
            else:
                p_nel = fd.tfp_files.TruncatedSkewGaussianCC(loc=mean, scale=std_dev,
                                                             skewness=skew,
                                                             limit=_ions_produced,
                                                             owens_t_terms=owens_t_terms).prob(electrons_produced)

            p_mult = p_nq * p_ni * p_nel

            # Contract over ions_produced
            p_final = tf.reduce_sum(p_mult, 3)

            r_final = p_final * rate_vs_energy

            r_final = tf.where(tf.math.is_nan(r_final),
                               tf.zeros_like(r_final, dtype=fd.float_type()),
                               r_final)

            return r_final

        def compute_single_energy_full(args):
            # Compute the block for a single energy, without approximations
            return compute_single_energy(args, approx=False)

        def compute_single_energy_approx(args):
            # Compute the block for a single energy, without continuity corrections
            # or truncated skew Gaussian
            return compute_single_energy(args, approx=True)

        nq = electrons_produced + photons_produced

        ions_min_initial = self.source._fetch('ions_produced_min', data_tensor=data_tensor)[:, 0, o]
        ions_min_initial = tf.repeat(ions_min_initial, tf.shape(ions_produced)[1], axis=1)
        ions_min_initial = tf.repeat(ions_min_initial[:, :, o], tf.shape(ions_produced)[2], axis=2)
        ions_min_initial = tf.repeat(ions_min_initial[:, :, :, o], tf.shape(ions_produced)[3], axis=3)

        # Work out the difference between each point in the ion domain and the lower bound,
        # for the lowest energy
        ions_produced_add = ions_produced - ions_min_initial

        # Energy above which we use the approximate computation
        if self.is_ER:
            cutoff_energy = 5.
        else:
            cutoff_energy = 20.

        energies_below_cutoff = tf.size(tf.where(energy[0, :] < cutoff_energy))
        energies_above_cutoff = tf.size(tf.where(energy[0, :] >= cutoff_energy))

        # We split the sum over energies to implement the approximate computation
        # above the cutoff energy
        energy_full, energy_approx = tf.split(energy[0, :], [energies_below_cutoff, energies_above_cutoff], 0)
        rate_vs_energy_full, rate_vs_energy_approx = \
            tf.split(rate_vs_energy[0, :], [energies_below_cutoff, energies_above_cutoff], 0)
        # Want to get rid of the padding of 0s at the end
        ion_bounds_min = self.source._fetch('ions_produced_min', data_tensor=data_tensor)[:, 0:tf.size(energy[0, :])]
        ion_bounds_min_full, ion_bounds_min_approx = \
            tf.split(ion_bounds_min, [energies_below_cutoff, energies_above_cutoff], 1)

        # Sum the block result per energy over energies, separately for the
        # energies below the cutoff and the energies above the cutoff
        result_full = tf.reduce_sum(tf.vectorized_map(compute_single_energy_full,
                                                      elems=[energy_full,
                                                             rate_vs_energy_full,
                                                             tf.transpose(ion_bounds_min_full)]),
                                    0)
        result_approx = tf.reduce_sum(tf.vectorized_map(compute_single_energy_approx,
                                                        elems=[energy_approx,
                                                               rate_vs_energy_approx,
                                                               tf.transpose(ion_bounds_min_approx)]),
                                      0)

        return (result_full + result_approx)

    def _simulate(self, d):
        # If you forget the .values here, you may get a Python core dump...
        if self.is_ER:
            nel = self.gimme_numpy('mean_yield_electron', d['energy'].values)
            nq = self.gimme_numpy('mean_yield_quanta', (d['energy'].values, nel))
            fano = self.gimme_numpy('fano_factor', nq)

            nq_actual_temp = np.round(stats.norm.rvs(nq, np.sqrt(fano*nq))).astype(int)
            # Don't let number of quanta go negative
            nq_actual = np.where(nq_actual_temp < 0,
                                 nq_actual_temp * 0,
                                 nq_actual_temp)

            ex_ratio = self.gimme_numpy('exciton_ratio', d['energy'].values)
            alpha = 1. / (1. + ex_ratio)

            d['ions_produced'] = stats.binom.rvs(n=nq_actual, p=alpha)

            nex = nq_actual - d['ions_produced']

        else:
            yields = self.gimme_numpy('mean_yields', d['energy'].values)
            nel = yields[0]
            nq = yields[1]
            ex_ratio = yields[2]
            alpha = 1. / (1. + ex_ratio)

            ni_temp = np.round(stats.norm.rvs(nq*alpha, np.sqrt(nq*alpha))).astype(int)
            # Don't let number of ions go negative
            d['ions_produced'] = np.where(ni_temp < 0,
                                          ni_temp * 0,
                                          ni_temp)

            nex_temp = np.round(stats.norm.rvs(nq*alpha*ex_ratio, np.sqrt(nq*alpha*ex_ratio))).astype(int)
            # Don't let number of excitons go negative
            nex = np.where(nex_temp < 0,
                           nex_temp * 0,
                           nex_temp)

            nq_actual = d['ions_produced'] + nex

        recomb_p = self.gimme_numpy('recomb_prob', (nel, nq, ex_ratio))
        skew = self.gimme_numpy('skewness', nq)
        var = self.gimme_numpy('variance', (nel, nq, recomb_p, d['ions_produced'].values))
        width_corr = self.gimme_numpy('width_correction', skew)
        mu_corr = self.gimme_numpy('mu_correction', (skew, var, width_corr))

        el_prod_temp1 = np.round(stats.skewnorm.rvs(skew, (1 - recomb_p) * d['ions_produced'] - mu_corr,
                                 np.sqrt(var) / width_corr)).astype(int)
        # Don't let number of electrons go negative
        el_prod_temp2 = np.where(el_prod_temp1 < 0,
                                 el_prod_temp1 * 0,
                                 el_prod_temp1)
        # Don't let number of electrons be greater than number of ions
        d['electrons_produced'] = np.where(el_prod_temp2 > d['ions_produced'],
                                           d['ions_produced'],
                                           el_prod_temp2)

        ph_prod_temp = nq_actual - d['electrons_produced']
        # Don't let number of photons be less than number of excitons
        d['photons_produced'] = np.where(ph_prod_temp < nex,
                                         nex,
                                         ph_prod_temp)

    def _annotate(self, d):
        pass

    def _annotate_special(self, d):
        # Here we manually calculate ion bounds for each energy we will sum over in the spectrum
        # Simple computation, based on forward simulation procedure

        def get_bounds_ER(energy):
            nel = self.gimme_numpy('mean_yield_electron', energy)
            nq = self.gimme_numpy('mean_yield_quanta', (energy, nel))
            fano = self.gimme_numpy('fano_factor', nq)
            nq_actual_upper = nq + np.sqrt(fano * nq) * self.source.max_sigma
            nq_actual_lower = nq - np.sqrt(fano * nq) * self.source.max_sigma

            ex_ratio = self.gimme_numpy('exciton_ratio', energy)
            alpha = 1. / (1. + ex_ratio)

            ions_mean_upper = nq_actual_upper * alpha
            ions_mean_lower = nq_actual_lower * alpha
            ions_std_upper = np.sqrt(nq_actual_upper * alpha * (1 - alpha))
            ions_std_lower = np.sqrt(nq_actual_lower * alpha * (1 - alpha))

            ions_produced_min = np.floor(ions_mean_lower - self.source.max_sigma * ions_std_lower).astype(int)
            ions_produced_max = np.ceil(ions_mean_upper + self.source.max_sigma * ions_std_upper).astype(int)

            return (ions_produced_min, ions_produced_max)

        def get_bounds_NR(energy):
            nq = self.gimme_numpy('mean_yields', energy)[1]
            ex_ratio = self.gimme_numpy('mean_yields', energy)[2]
            alpha = 1. / (1. + ex_ratio)
            ions_mean = nq * alpha
            ions_std = np.sqrt(nq * alpha)

            ions_produced_min = np.floor(ions_mean - self.source.max_sigma * ions_std).astype(int)
            ions_produced_max = np.ceil(ions_mean + self.source.max_sigma * ions_std).astype(int)

            return (ions_produced_min, ions_produced_max)

        # Compute ion bounds for every energy in the full spectrum, once
        if self.is_ER:
            bounds = [get_bounds_ER(energy) for energy in self.source.energies.numpy()]
        else:
            bounds = [get_bounds_NR(energy) for energy in self.source.energies.numpy()]

        ions_produced_min_full = [x[0] for x in bounds]
        ions_produced_max_full = [x[1] for x in bounds]

        for batch in range(self.source.n_batches):
            d_batch = d[batch * self.source.batch_size:(batch + 1) * self.source.batch_size]

            # These are the same across all events in a batch
            energy_min = d_batch['energy_min'].iloc[0]
            energy_max = d_batch['energy_max'].iloc[0]

            energies_trim = self.source.energies.numpy()[(self.source.energies.numpy() >= energy_min) &
                                                         (self.source.energies.numpy() <= energy_max)]

            # Keep only the ion bounds corresponding to the energies in the trimmed
            # spectrum for this batch
            ions_produced_min_full_trim = np.asarray(ions_produced_min_full)[
                (self.source.energies.numpy() >= energy_min) &
                (self.source.energies.numpy() <= energy_max)]
            ions_produced_max_full_trim = np.asarray(ions_produced_max_full)[
                (self.source.energies.numpy() >= energy_min) &
                (self.source.energies.numpy() <= energy_max)]

            index_step = np.round(np.linspace(0, len(energies_trim) - 1,
                                              min(len(energies_trim), self.source.max_dim_sizes['energy']))).astype(int)

            # Keep only the ion bounds corresponding to the energies in the stepped + trimmed
            # spectrum for this batch
            ions_produced_min = list(np.take(ions_produced_min_full_trim, index_step))
            ions_produced_max = list(np.take(ions_produced_max_full_trim, index_step))

            # For the events in the dataframe that are part of this batch, save the ion bounds at each energy
            # in the dataframe
            indicies = np.arange(batch * self.source.batch_size, (batch + 1) * self.source.batch_size)
            d.loc[batch * self.source.batch_size:(batch + 1) * self.source.batch_size - 1, 'ions_produced_min'] = \
                pd.Series([ions_produced_min]*len(indicies), index=indicies)
            d.loc[batch * self.source.batch_size:(batch + 1) * self.source.batch_size - 1, 'ions_produced_max'] = \
                pd.Series([ions_produced_max]*len(indicies), index=indicies)

        # If mono-energetic, one zero element at the end to get tensor dimensions
        # that match up with non-mono-energetic case; will be discarded later on
        max_num_energies = max(min(len(self.source.energies), self.source.max_dim_sizes['energy']), 2)
        # Pad with 0s at the end to make each one the same size
        [bounds.extend([0]*(max_num_energies - len(bounds))) for bounds in d['ions_produced_min'].values]

        return True

    def _calculate_dimsizes_special(self):
        d = self.source.data

        ions_produced_max = d['ions_produced_max'].to_numpy()
        ions_produced_min = d['ions_produced_min'].to_numpy()

        # Take the dimsize for ions_produced to be the largest dimsize across the energy range
        dimsizes = [max([elem + 1 for elem in list(map(operator.sub, maxs, mins))])
                    for maxs, mins in zip(ions_produced_max, ions_produced_min)]
        # Cap the dimsize if we are above the max_dim_size
        self.source.dimsizes['ions_produced'] = \
            self.source.max_dim_sizes['ions_produced'] * \
            np.greater(dimsizes, self.source.max_dim_sizes['ions_produced']) + \
            dimsizes * np.less_equal(dimsizes, self.source.max_dim_sizes['ions_produced'])

        # Calculate the stepping across the domain
        d['ions_produced_steps'] = tf.where(dimsizes > self.source.dimsizes['ions_produced'],
                                            tf.math.ceil(([elem-1 for elem in dimsizes]) /
                                            (self.source.dimsizes['ions_produced']-1)),
                                            1).numpy()

    def _domain_dict_bonus(self, d):
        electrons_domain = self.source.domain('electrons_produced', d)
        photons_domain = self.source.domain('photons_produced', d)

        ions_min_initial = self.source._fetch('ions_produced_min', data_tensor=d)[:, 0, o]
        steps = self.source._fetch('ions_produced_steps', data_tensor=d)[:, o]
        ions_range = tf.range(tf.reduce_max(self.source._fetch('ions_produced_dimsizes', data_tensor=d))) * steps
        ions_domain_initial = ions_min_initial + ions_range

        electrons = tf.repeat(electrons_domain[:, :, o], tf.shape(photons_domain)[1], axis=2)
        electrons = tf.repeat(electrons[:, :, :, o], tf.shape(ions_domain_initial)[1], axis=3)

        photons = tf.repeat(photons_domain[:, o, :], tf.shape(electrons_domain)[1], axis=1)
        photons = tf.repeat(photons[:, :, :, o], tf.shape(ions_domain_initial)[1], axis=3)

        # We construct the ions domain for only the lowest energy; this is modified later
        ions = tf.repeat(ions_domain_initial[:, o, :], tf.shape(electrons_domain)[1], axis=1)
        ions = tf.repeat(ions[:, :, o, :], tf.shape(photons_domain)[1], axis=2)

        return dict({'electrons_produced': electrons,
                     'photons_produced': photons,
                     'ions_produced': ions})


@export
class MakePhotonsElectronER(MakePhotonsElectronsNR):
    is_ER = True

    special_model_functions = tuple(
        [x for x in MakePhotonsElectronsNR.special_model_functions if x != 'mean_yields'] +
        ['mean_yield_electron', 'mean_yield_quanta', 'fano_factor', 'exciton_ratio'])
    model_functions = special_model_functions
