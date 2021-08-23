import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class MakePhotonsElectronsNR(fd.Block):
    is_ER = False

    dimensions = ('electrons_produced', 'photons_produced')
    extra_dimensions = (('ions_produced', True),)
    depends_on = ((('energy',), 'rate_vs_energy'),)

    special_model_functions = ('mean_yields', 'recomb_prob', 'skewness', 'variance',
                                'width_correction', 'mu_correction')
    model_functions = special_model_functions

    MC_annotate = True

    MC_annotate_dimensions = ('ions_produced',)

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 electrons_produced, photons_produced,
                 # Dependency domain and value
                 energy, rate_vs_energy,
                 # Bonus dimension
                 ions_produced):

        def compute_single_energy(args):

            energy = args[0]
            rate_vs_energy = args[1]

            if self.is_ER:
                nel_mean = self.gimme('mean_yield_electron', data_tensor=data_tensor, ptensor=ptensor,
                                      bonus_arg=energy)
                nq_mean = self.gimme('mean_yield_quanta', data_tensor=data_tensor, ptensor=ptensor,
                                     bonus_arg=(energy,nel_mean))
                fano = self.gimme('fano_factor', data_tensor=data_tensor, ptensor=ptensor,
                                  bonus_arg=nq_mean)

                p_nq = tfp.distributions.Normal(
                    loc=nq_mean, scale=tf.sqrt(nq_mean * fano) + 1e-10).cdf(nq + 0.5) \
                - tfp.distributions.Normal(
                    loc=nq_mean, scale=tf.sqrt(nq_mean * fano) + 1e-10).cdf(nq - 0.5)

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

                p_ni = tfp.distributions.Normal(
                    loc=nq_mean*alf, scale=tf.sqrt(nq_mean*alpha) + 1e-10).cdf(ions_produced + 0.5) \
                -  tfp.distributions.Normal(
                    loc=nq_mean*alf, scale=tf.sqrt(nq_mean*alpha) + 1e-10).cdf(ions_produced - 0.5)

                p_nq = tfp.distributions.Normal(
                    loc=nq_mean*alf*excitonR, scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10).cdf(nq - ions_produced + 0.5) \
                - tfp.distributions.Normal(
                    loc=nq_mean*alf*excitonR, scale=tf.sqrt(nq_mean*alpha*ex_ratio) + 1e-10).cdf(nq - ions_produced - 0.5)

            recomb_p = self.gimme('recomb_prob', data_tensor=data_tensor, ptensor=ptensor,
                                  bonus_arg=(nel_mean, nq_mean, ex_ratio))
            skew = self.gimme('skewness', data_tensor=data_tensor, ptensor=ptensor,
                              bonus_arg=nq_mean)
            var = self.gimme('variance', data_tensor=data_tensor, ptensor=ptensor,
                             bonus_arg=(nel_mean, nq_mean, recomb_p, ions_produced))
            width_corr = self.gimme('width_correction', data_tensor=data_tensor, ptensor=ptensor,
                                    bonus_arg=skew)
            mu_correction = self.gimme('get_muCorrection', data_tensor=data_tensor, ptensor=ptensor,
                                       bonus_arg=(skew, var, width_corr))

            mean = (tf.ones_like(ions_produced, dtype=fd.float_type()) - recomb_p) * ions_produced - mu_corr
            std_dev = tf.sqrt(variance) / width_corr
            p_nel = tfp.distributions.TruncatedSkewGaussianCC(
                    loc=mean, scale=std_dev, skewness=skew, limit=ions_produced).prob(electrons_produced)

            p_mult = p_nq * p_ni * p_nel

            p_final = tf.reduce_sum(p_mult, 3)

            r_final = p_final * rate_vs_energy

            r_final = tf.where(tf.math.is_nan(r_final),
                                   tf.zeros_like(r_final, dtype=fd.float_type()),
                                   r_final)

            return r_final

        nq = electrons_produced + photons_produced

        result = tf.reduce_sum(tf.vectorized_map(compute_single_energy, elems=[energy[:,0],rate_vs_energy[:,0]]), 0)

        return result

    def _simulate(self, d):
        # If you forget the .values here, you may get a Python core dump...
        if self.is_ER:
            nel = self.gimme_numpy('mean_yield_electron', bonus_arg=d['energy'].values)
            nq = self.gimme_numpy('mean_yield_quanta', bonus_arg=(d['energy'].values, nel))
            fano = self.gimme_numpy('fano_factor', bonus_arg=nq)

            nq_actual_temp = np.round(stats.norm.rvs(nq, np.sqrt(fano*nq))).astype(int)
            # Don't let number of quanta go negative
            nq_actual = np.where(nq_actual_temp < 0,
                                 nq_actual_temp * 0,
                                 nq_actual_temp)

            ex_ratio = self.gimme_numpy('exciton_ratio', bonus_arg=d['energy'].values)
            alpha = 1. / (1. + ex_ratio)

            d['ions_produced'] = stats.binom.rvs(n=nq_actual, p=alpha)

            nex = nq_actual - d['ions_produced']

        else:
            yields = self.gimme_numpy('mean_yields', bonus_arg=d['energy'].values)
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

        recomb_p = self.gimme_numpy('recomb_prob', bonus_arg=(nel, nq, ex_ratio))
        skew = self.gimme_numpy('skewness', bonus_arg=nq)
        var = self.gimme_numpy('variance', bonus_arg=(nel, nq, recomb_p, d['ions_produced'].values))
        width_corr = self.gimme_numpy('width_correction', bonus_arg=skew)
        mu_corr= self.gimme_numpy('mu_correction', bonus_arg=(skew, var, width_corr))

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


@export
class MakePhotonsElectronER(MakePhotonsElectronsNR):
    is_ER = True

    special_model_functions = tuple(
        [x for x in MakePhotonsElectronsNR.special_model_functions if x != 'mean_yields']
         + ['mean_yield_electron', 'mean_yield_quanta', 'fano_factor', 'exciton_ratio'])
    model_functions = special_model_functions
