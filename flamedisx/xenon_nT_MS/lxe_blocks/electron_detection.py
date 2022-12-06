import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


@export
class DetectElectronsMod(fd.Block):
    dimensions = ('electrons_produced', 's2_photons_produced')

    model_functions = ('electron_detection_eff', 'electron_gain')

    def _compute(self, data_tensor, ptensor,
                 electrons_produced, s2_photons_produced):
        p = self.gimme('electron_detection_eff',
                       data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        gain = self.gimme('electron_gain',
                          data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Need to divide by gain at the end, else it gets absorbed into the stepping correction
        result = tfp.distributions.Binomial(
                total_count=electrons_produced,
                probs=tf.cast(p, dtype=fd.float_type())
            ).prob(tf.round(s2_photons_produced / gain)) / gain
        return result

    def _simulate(self, d):
        p = self.gimme_numpy('electron_detection_eff')

        d['s2_photons_produced'] = np.round(stats.binom.rvs(
            n=d['electrons_produced'],
            p=p) * self.gimme_numpy('electron_gain')).astype(int)

    def _annotate(self, d):
        # Get efficiency
        effs = self.gimme_numpy('electron_detection_eff')
        # Get gain
        gains = self.gimme_numpy('electron_gain')

        for suffix, bound, intify in (('_min', 'lower', np.floor),
                                      ('_max', 'upper', np.ceil)):
            out_bounds = intify(d['s2_photons_produced' + suffix] / gains)
            supports = [np.linspace(out_bound, np.ceil(out_bound / eff * 10.),
                                    1000).astype(int) for out_bound, eff in zip(out_bounds, effs)]
            ns = supports
            ps = [eff * np.ones_like(support) for eff, support in zip(effs, supports)]
            rvs = [out_bound * np.ones_like(support)
                   for out_bound, support in zip(out_bounds, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim='electrons_produced',
                                   bounds_prob=self.source.bounds_prob, bound=bound,
                                   bound_type='binomial', supports=supports,
                                   rvs_binom=rvs, ns_binom=ns, ps_binom=ps)

    def _annotate_special(self, d):
        # Here we obtain improved bounds on electrons produced with a non-flat prior
        for batch in range(self.source.n_batches):
            d_batch = d[batch * self.source.batch_size:(batch + 1) * self.source.batch_size]

            # Get efficiency
            effs = self.gimme_numpy('electron_detection_eff')[
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size]
            # Get gain
            gains = self.gimme_numpy('electron_gain')[
                batch * self.source.batch_size:(batch + 1) * self.source.batch_size]

            for suffix, bound, intify in (('_min', 'lower', np.floor),
                                          ('_max', 'upper', np.ceil)):
                out_bounds = intify(d_batch['s2_photons_produced' + suffix] / gains)
                supports = [np.linspace(out_bound, np.ceil(out_bound / eff * 10.),
                                        1000).astype(int) for out_bound, eff in zip(out_bounds, effs)]
                ns = supports
                ps = [eff * np.ones_like(support) for eff, support in zip(effs, supports)]
                rvs = [out_bound * np.ones_like(support)
                       for out_bound, support in zip(out_bounds, supports)]

                fd.bounds.bayes_bounds_priors(source=self.source, batch=batch,
                                              df=d, in_dim='electrons_produced',
                                              bounds_prob=self.source.bounds_prob, bound=bound,
                                              bound_type='binomial', supports=supports,
                                              rvs_binom=rvs, ns_binom=ns, ps_binom=ps)

            return False
