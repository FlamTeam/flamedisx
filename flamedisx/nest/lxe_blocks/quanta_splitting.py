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

    exclude_data_tensor = ('ions_produced_min', 'ions_produced_max')

    special_model_functions = ('mean_yields', 'recomb_prob', 'skewness', 'variance',
                                'width_correction', 'mu_correction')
    model_functions = special_model_functions

    prior_dimensions = ('ions_produced',)

    use_batch = True

    def _compute(self,
                 data_tensor, ptensor,
                 #
                 i_batch,
                 # Domain
                 electrons_produced, photons_produced,
                 # Bonus dimension
                 ions_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy):

        def compute_single_energy(args):

            energy = args[0]
            rate_vs_energy = args[1]
            energy_index = args[2]

            ions_min_initial = self.ion_bounds_min_tensor[i_batch, :, energy_index, o]
            ions_min_initial = tf.repeat(ions_min_initial, tf.shape(ions_produced)[1], axis=1)
            ions_min_initial = tf.repeat(ions_min_initial[:, :, o], tf.shape(ions_produced)[2], axis=2)
            ions_min_initial = tf.repeat(ions_min_initial[:, :, :, o], tf.shape(ions_produced)[3], axis=3)

            ions_min = self.ion_bounds_min_tensor[i_batch, :, energy_index, o]
            ions_min = tf.repeat(ions_min, tf.shape(ions_produced)[1], axis=1)
            ions_min = tf.repeat(ions_min[:, :, o], tf.shape(ions_produced)[2], axis=2)
            ions_min = tf.repeat(ions_min[:, :, :, o], tf.shape(ions_produced)[3], axis=3)

            _ions_produced = ions_produced - ions_min_initial + ions_min

            if self.is_ER:
                nel_mean = self.gimme('mean_yield_electron', data_tensor=data_tensor, ptensor=ptensor,
                                      bonus_arg=energy)
                nq_mean = self.gimme('mean_yield_quanta', data_tensor=data_tensor, ptensor=ptensor,
                                     bonus_arg=(energy,nel_mean))
                fano = self.gimme('fano_factor', data_tensor=data_tensor, ptensor=ptensor,
                                  bonus_arg=nq_mean)

                # p_nq = tfp.distributions.Normal(
                #     loc=nq_mean, scale=tf.sqrt(nq_mean * fano) + 1e-10).cdf(nq + 0.5) \
                # - tfp.distributions.Normal(
                #     loc=nq_mean, scale=tf.sqrt(nq_mean * fano) + 1e-10).cdf(nq - 0.5)

                p_nq = tfp.distributions.Normal(
                    loc=nq_mean, scale=tf.sqrt(nq_mean * fano) + 1e-10).prob(nq)

                ex_ratio = self.gimme('exciton_ratio', data_tensor=data_tensor, ptensor=ptensor,
                                      bonus_arg=energy)
                alpha = 1. / (1. + ex_ratio)

                p_ni = tfp.distributions.Binomial(
                    total_count=nq, probs=alpha).prob(ions_produced)

            else:
                yields = self.gimme('mean_yields', data_tensor=data_tensor, ptensor=ptensor,
                                     bonus_arg=energy)
                nel_mean = yields[0]
                nq_mean = yields[1]
                ex_ratio = yields[2]
                alpha = 1. / (1. + ex_ratio)

                # p_ni = tfp.distributions.Normal(
                #     loc=nq_mean*alpha, scale=tf.sqrt(nq_mean*alpha) + 1e-10).cdf(ions_produced + 0.5) \
                # -  tfp.distributions.Normal(
                #     loc=nq_mean*alpha, scale=tf.sqrt(nq_mean*alpha) + 1e-10).cdf(ions_produced - 0.5)

                p_ni = tfp.distributions.Normal(
                    loc=nq_mean*alpha, scale=tf.sqrt(nq_mean*alpha) + 1e-10).prob(ions_produced)

                # p_nq = tfp.distributions.Normal(
                #     loc=nq_mean*alpha*ex_ratio, scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10).cdf(nq - ions_produced + 0.5) \
                # - tfp.distributions.Normal(
                #     loc=nq_mean*alpha*ex_ratio, scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10).cdf(nq - ions_produced - 0.5)

                p_nq = tfp.distributions.Normal(
                    loc=nq_mean*alpha*ex_ratio, scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10).prob(nq - ions_produced)

            recomb_p = self.gimme('recomb_prob', data_tensor=data_tensor, ptensor=ptensor,
                                  bonus_arg=(nel_mean, nq_mean, ex_ratio))
            skew = self.gimme('skewness', data_tensor=data_tensor, ptensor=ptensor,
                              bonus_arg=nq_mean)
            var = self.gimme('variance', data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=(nel_mean, nq_mean, recomb_p, ions_produced))
            width_corr = self.gimme('width_correction', data_tensor=data_tensor, ptensor=ptensor,
                                    bonus_arg=skew)
            mu_corr = self.gimme('mu_correction', data_tensor=data_tensor, ptensor=ptensor,
                                       bonus_arg=(skew, var, width_corr))

            mean = (tf.ones_like(ions_produced, dtype=fd.float_type()) - recomb_p) * ions_produced - mu_corr
            std_dev = tf.sqrt(var) / width_corr
            # p_nel = tfp.distributions.TruncatedSkewGaussianCC(
            #         loc=mean, scale=std_dev, skewness=skew, limit=ions_produced).prob(electrons_produced)

            p_nel = tfp.distributions.SkewGaussian(
                    loc=mean, scale=std_dev, skewness=skew).prob(electrons_produced)

            p_mult = p_nq * p_ni * p_nel

            p_final = tf.reduce_sum(p_mult, 3)

            r_final = p_final * rate_vs_energy

            r_final = tf.where(tf.math.is_nan(r_final),
                                   tf.zeros_like(r_final, dtype=fd.float_type()),
                                   r_final)

            return r_final

        nq = electrons_produced + photons_produced

        result = tf.reduce_sum(tf.vectorized_map(compute_single_energy,
                                                 elems=[energy[0,:],rate_vs_energy[0,:],tf.range(tf.shape(energy)[1])]),
                                                 0)

        return result

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
        mu_corr= self.gimme_numpy('mu_correction', (skew, var, width_corr))

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

    def _annotate_prior(self, d):
        for batch in range(self.source.n_batches):
            d_batch = d[batch * self.source.batch_size : (batch + 1) * self.source.batch_size]

            energy_min = min(d_batch['energy_min'])
            energy_max = max(d_batch['energy_max'])

            energies_trim = self.source.energies.numpy()[np.where(self.source.energies >= energy_min) and
                                                         np.where(self.source.energies <= energy_max)]

            index_step = np.round(np.linspace(0, len(energies_trim) - 1,
                                              min(len(energies_trim), self.source.max_dim_size_initial))).astype(int)
            energies_trim_step = energies_trim[index_step]

            ions_produced_min = []
            ions_produced_max = []

            for energy in energies_trim_step:

                if self.is_ER:
                    nel = self.gimme_numpy('mean_yield_electron', energy)
                    nq = self.gimme_numpy('mean_yield_quanta', (energy, nel))
                    ex_ratio = self.gimme_numpy('exciton_ratio', energy)
                    alpha = 1. / (1. + ex_ratio)
                    ions_mean = nq * alpha
                    ions_std = nq * alpha * (1 - alpha)
                else:
                    nq = self.gimme_numpy('mean_yields', energy)[1]
                    ex_ratio = self.gimme_numpy('mean_yields', energy)[2]
                    alpha = 1. / (1. + ex_ratio)
                    ions_mean = nq * alpha
                    ions_std = np.sqrt(nq * alpha)

                ions_produced_min.append(np.floor(ions_mean - self.source.max_sigma * ions_std).astype(int))
                ions_produced_max.append(np.ceil(ions_mean + self.source.max_sigma * ions_std).astype(int))

            indicies = np.arange(batch * self.source.batch_size, (batch + 1) * self.source.batch_size)
            d.loc[batch * self.source.batch_size : (batch + 1) * self.source.batch_size - 1, 'ions_produced_min'] = \
                pd.Series([ions_produced_min]*len(indicies), index=indicies)
            d.loc[batch * self.source.batch_size : (batch + 1) * self.source.batch_size - 1, 'ions_produced_max'] = \
                pd.Series([ions_produced_max]*len(indicies), index=indicies)

        return True

    def _calculate_dimsizes_special(self):
        d = self.source.data

        maxs_batch = d['ions_produced_max'].to_numpy()
        mins_batch = d['ions_produced_min'].to_numpy()

        # Take the maximum dimsize across the energy range per event
        dimsizes = [max([elem + 1 for elem in list(map(operator.sub, maxs, mins))])
                                                 for maxs, mins in zip(maxs_batch, mins_batch)]
        self.source.dimsizes['ions_produced'] = \
            self.source.max_dim_size * np.greater(dimsizes, self.source.max_dim_size) + \
            dimsizes * np.less_equal(dimsizes, self.source.max_dim_size)

        d['ions_produced_steps'] = tf.where(dimsizes > self.source.dimsizes['ions_produced'],
                                            tf.math.ceil(([elem-1 for elem in dimsizes]) / (self.source.dimsizes['ions_produced']-1)),
                                            1).numpy()

    def _populate_special_tensors(self, d):
        ion_bounds_min = [tf.convert_to_tensor(values, dtype=fd.float_type()) for values in d['ions_produced_min'].values]
        ion_bounds_min_tensor = tf.stack(ion_bounds_min)

        self.ion_bounds_min_tensor = tf.reshape(ion_bounds_min_tensor,
            [self.source.n_batches, -1, tf.shape(ion_bounds_min_tensor)[1]])

    def _domain_dict_bonus(self, d, i_batch):
        electrons_domain = self.source.domain('electrons_produced', d)
        photons_domain = self.source.domain('photons_produced', d)

        ions_min_initial = self.ion_bounds_min_tensor[i_batch, :, 0, o]
        steps = self.source._fetch('ions_produced_steps', data_tensor=d)[:, o]
        ions_range = tf.range(tf.reduce_max(self.source._fetch('ions_produced_dimsizes', data_tensor=d))) * steps
        ions_domain_initial = ions_min_initial + ions_range

        electrons = tf.repeat(electrons_domain[:, :, o], tf.shape(photons_domain)[1], axis=2)
        electrons = tf.repeat(electrons[:, :, :, o], tf.shape(ions_domain_initial)[1], axis=3)

        photons = tf.repeat(photons_domain[:, o, :], tf.shape(electrons_domain)[1], axis=1)
        photons = tf.repeat(photons[:, :, :, o], tf.shape(ions_domain_initial)[1], axis=3)

        ions = tf.repeat(ions_domain_initial[:, o, :], tf.shape(electrons_domain)[1], axis=1)
        ions = tf.repeat(ions[:, :, o, :], tf.shape(photons_domain)[1], axis=2)

        return dict({'electrons_produced': electrons,
                     'photons_produced': photons,
                     'ions_produced': ions})

@export
class MakePhotonsElectronER(MakePhotonsElectronsNR):
    is_ER = True

    special_model_functions = tuple(
        [x for x in MakePhotonsElectronsNR.special_model_functions if x != 'mean_yields']
         + ['mean_yield_electron', 'mean_yield_quanta', 'fano_factor', 'exciton_ratio'])
    model_functions = special_model_functions
