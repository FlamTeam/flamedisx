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

    special_model_functions = ('mean_yields', 'recomb_prob', 'skewness', 'variance',
                                'width_correction', 'mu_correction')
    model_functions = special_model_functions

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
            n_ex = np.where(nex_temp < 0,
                            nex_temp * 0,
                            nex_temp)

            nq_actual = d['ions_produced'] + n_ex

        recomb_p = self.gimme_numpy('recomb_prob', bonus_arg=(nel, nq, ex_ratio))
        skew = self.gimme_numpy('skewness', bonus_arg=nq)
        var = self.gimme_numpy('variance', bonus_arg=(nel, nq, recomb_p, d['ions_produced'].values))
        width_corr = self.gimme_numpy('width_correction', bonus_arg=skew)
        mu_corr= self.gimme_numpy('mu_correction', bonus_arg=(skew, var, width_corr))

        el_prod_temp1 = np.round(stats.skewnorm.rvs(skewness, (1 - recombP) * d['ion_produced'] - muCorrection,
                                 np.sqrt(Variance) / widthCorrection)).astype(int)
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
        d['photons_produced'] = np.where(ph_prod_temp < n_ex,
                                         n_ex,
                                         ph_prod_temp)


@export
class MakePhotonsElectronER(MakePhotonsElectronsNR):
    is_ER = True

    special_model_functions = tuple(
        [x for x in MakePhotonsElectronsNR.special_model_functions if x != 'mean_yields']
         + ['mean_yield_electron', 'mean_yield_quanta', 'fano_factor', 'exciton_ratio'])
    model_functions = special_model_functions
