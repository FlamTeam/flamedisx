import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


class MakePhotoelectrons(fd.Block):
    model_functions = ('double_pe_fraction',)

    quanta_in_name: str
    quanta_out_name: str

    def _compute(self, data_tensor, ptensor,
                 quanta_in, quanta_out):
        p_dpe = self.gimme('double_pe_fraction',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]

        # Double-pe emission only creates additional photoelectrons.
        # Invalid values will get assigned p=0 later.
        extra_pe = quanta_out - quanta_in
        invalid = extra_pe < 0

        # Negative arguments would mess up tfp's Binomial
        extra_pe = tf.where(invalid,
                            tf.zeros_like(extra_pe),
                            extra_pe)

        # (N_pe - N_photons) distributed as Binom(N_photons, p=pdpe)
        result = tfp.distributions.Binomial(
                total_count=quanta_in,
                probs=tf.cast(p_dpe, dtype=fd.float_type())
            ).prob(extra_pe)

        # Set probability of extra_pe < 0 cases to 0
        return tf.where(invalid,
                        tf.zeros_like(quanta_out),
                        result)

    def _simulate(self, d):
        d[self.quanta_out_name] = stats.binom.rvs(
            n=d[self.quanta_in_name],
            p=self.gimme_numpy('double_pe_fraction')) + d[self.quanta_in_name]

    def _annotate(self, d):
        for suffix, bound in (('_min', 'lower'),
                              ('_max', 'upper')):
            out_bounds = d[self.quanta_out_name + suffix]
            supports = [np.linspace(np.ceil(out_bound / 2.), out_bound + 1., 1000).astype(int)
                        for out_bound in out_bounds]
            ns = supports
            ps = [p * np.ones_like(support) for p, support in zip(self.gimme_numpy('double_pe_fraction'), supports)]
            rvs = [out_bound - support for out_bound, support in zip(out_bounds, supports)]

            fd.bounds.bayes_bounds(df=d, in_dim=self.quanta_in_name,
                                   bounds_prob=self.source.bounds_prob, bound=bound,
                                   bound_type='binomial', supports=supports,
                                   rvs_binom=rvs, ns_binom=ns, ps_binom=ps)


@export
class MakeS1Photoelectrons(MakePhotoelectrons):
    dimensions = ('photons_detected', 's1_photoelectrons_produced')

    quanta_in_name = 'photons_detected'
    quanta_out_name = 's1_photoelectrons_produced'

    def _compute(self, data_tensor, ptensor,
                 photons_detected, s1_photoelectrons_produced):
        return super()._compute(
            quanta_in=photons_detected,
            quanta_out=s1_photoelectrons_produced,
            data_tensor=data_tensor, ptensor=ptensor)


@export
class MakeS2Photoelectrons(MakePhotoelectrons):
    dimensions = ('s2_photons_detected', 's2_photoelectrons_detected')

    quanta_in_name = 's2_photons_detected'
    quanta_out_name = 's2_photoelectrons_detected'

    def _compute(self, data_tensor, ptensor,
                 s2_photons_detected, s2_photoelectrons_detected):
        return super()._compute(
            quanta_in=s2_photons_detected,
            quanta_out=s2_photoelectrons_detected,
            data_tensor=data_tensor, ptensor=ptensor)

# Add Blocks for S1 RQs
@export
class MakeS1RQs(MakePhotoelectrons):
    model_functions = ('double_pe_fraction', 'spe_eff', 'spe_res', 'spe_thr', 's1_spike_count',)
    special_model_functions = ('s1_spike_count',)
    dimensions = ('photons_detected', 's1_max_channel_area')

    quanta_in_name = 'photons_detected'
    quanta_out_name = 's1_max_channel_area'
    
    def s1_spike_count(self, n_photon_detected):
        """
        Identical math as in NEST::S1Calc::GetS1. (parametric mode)
        Return a 1D matrix of simulated spike count, 
        with the same dim as "n_photon_detected"
        """
        g1 = 0.5
        g1_XYZ = 1.
        spe_eff = self.gimme_numpy('spe_eff')
        # Might need some protection code
        # eff=max(0, min(eff,1))
        spe_res = self.gimme_numpy('spe_res')
        spe_thr = self.gimme_numpy('spe_thr')
        # Should use a 0D constant P_dpe (as NEST default)
        p_dpe = self.gimme_numpy('double_pe_fraction')

        # calculate below_threshold_percentile
        bigPhi_alpha_spe = 0.5 * (1. + tf.math.erf(-1./spe_res / tf.math.sqrt(2.)))
        bigPhi_xi_spe = 0.5 * (1. + tf.math.erf(spe_thr-1.)) / spe_res / tf.math.sqrt(2.)
        spe_btp = (bigPhi_xi_spe - bigPhi_alpha_spe) / (1. - bigPhi_alpha_spe)
        bigPhi_alpha_dpe = 0.5 * (1. + tf.math.erf(-2./spe_res / tf.math.sqrt(2.)))
        bigPhi_xi_dpe = 0.5 * (1. + tf.math.erf(spe_thr-2.)) / spe_res / tf.math.sqrt(2.)
        dpe_btp = (bigPhi_xi_dpe - bigPhi_alpha_dpe) / (1. - bigPhi_alpha_dpe)
        below_thre_perc = spe_btp * (1. - p_dpe) + dpe_btp * p_dpe
        
        # number of hits
        n = tf.cast(n_photon_detected, dtype=tf.float32) # Binomial needs float
        nHits = tfp.distributions.Binomial(n, g1).sample()
        nSpike = tfp.distributions.Binomial(nHits, spe_eff * (1-below_thre_perc)).sample()
        return tf.cast(nSpike, dtype=tf.int32)
    
    def _compute(self, data_tensor, ptensor,
                 quanta_in, quanta_out):
        """NOT implemented yet"""
        p_dpe = self.gimme('double_pe_fraction',
                           data_tensor=data_tensor, ptensor=ptensor)[:, o, o]
        # p_dpe will be a 1d tensor w/ length same as PMT# (quanta out)

        # Double-pe emission only creates additional photoelectrons.
        # Invalid values will get assigned p=0 later.
        extra_pe = quanta_out - quanta_in
        invalid = extra_pe < 0

        # Negative arguments would mess up tfp's Binomial
        extra_pe = tf.where(invalid,
                            tf.zeros_like(extra_pe),
                            extra_pe)

        # (N_pe - N_photons) distributed as Binom(N_photons, p=pdpe)
        result = tfp.distributions.Binomial(
                total_count=quanta_in,
                probs=tf.cast(p_dpe, dtype=fd.float_type())
            ).prob(extra_pe)

        # Set probability of extra_pe < 0 cases to 0
        return tf.where(invalid,
                        tf.zeros_like(quanta_out),
                        result)
    
    def _simulate(self, d):
        """
        To use this block properly, you need to overwrite double_pe_fraction() 
        to load channel-based p_dpe. Though there's a protection for 0D average p_dpe
        """
        nPh = d[self.quanta_in_name]
        spe_res = tf.constant(self.gimme_numpy('spe_res')[0], dtype=tf.float32) #Want 0D number for broadcasting
        p_dpe = self.gimme_numpy('double_pe_fraction')
        """
        Still needs a bit more complication here. I should make it a special model func
        And need to implement protection against large spike# (>20) or p_dpe=0
        """
        
        nSpike = self.gimme_numpy('s1_spike_count', nPh)
        maxSpike = tf.math.reduce_max(nSpike)
        nSpike = tf.reshape(tf.repeat(nSpike, maxSpike), [tf.size(nPh), maxSpike])
        # Matrix of number of spikes for each n_photon, shape: (n_photon, maxSpike)
        nSim = tf.repeat([tf.range(maxSpike)], tf.size(nPh), axis=0) +1
        nSim = tf.where(nSim - nSpike > 0, 0, 1)
        # Matrix of 0 and 1. each 1 represents a spike for this "line of data (n_photon_det)"
        randomCh = tf.cast(tf.random.uniform(shape=nSim.shape, maxval=541), tf.int32)
        randomCh = tf.math.multiply(nSim,randomCh)
        # Above: Replace "1"s in nSim with random Channel number (or index of p_dpe list)
        if (tf.size(p_dpe) == 1):
            p_dpe = tf.repeat(p_dpe, 541)
        # Replace index with corresponding p_dpe.
        p = tf.gather(p_dpe, randomCh) # p has the same shape as randomCh

        """Also want spe_res per channel"""
        # Simulate channel-wise peArea, identical math as in LZLAMA
        pe = 1 + tfp.distributions.Binomial(1, p).sample()
        peArea = tfp.distributions.TruncatedNormal(tf.math.sqrt(pe) * spe_res, pe, 
                            0., pe + 10 * tf.math.sqrt(pe) * spe_res /(1. + p)).sample()
        MaxChArea = tf.math.reduce_max(peArea, axis=1)

        # MaxChArea should have the shape of n_photon_detected
        d[self.quanta_out_name] = MaxChArea