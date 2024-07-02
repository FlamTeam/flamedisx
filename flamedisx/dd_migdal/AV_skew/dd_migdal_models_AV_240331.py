from copy import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import os

from multihist import Hist1d

import scipy.interpolate as itp
from scipy import integrate

import math as m
pi = tf.constant(m.pi)

from .. import dd_migdal as fd_dd_migdal

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis

import numba as nb

import matplotlib.pyplot as plt ###
import matplotlib as mpl


###
# filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam.npz'
filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam_20240309.npz'
with np.load(filename) as f:
    fit_values_allkeVnr_allparam = f['fit_values_allkeVnr_allparam']
    
def interp_nd(x):
        
    # print('x',tf.shape(x))
    # Define x_grid_points for interpolation
    part1 = tf.cast(tf.experimental.numpy.geomspace(1,4,7), fd.float_type())
    part2 = tf.cast(tf.experimental.numpy.geomspace(4,80,23), fd.float_type())[1:]
    keVnr_choices = tf.concat([part1,part2],axis=0)
    
    # print(np.shape(keVnr_choices))
    # print(tf.shape(keVnr_choices))

    Fi_grid = tf.cast([0.25,0.3,0.4,0.55,0.75,1.], fd.float_type())                  # Fano ion
    Fex_grid = tf.cast([0.3,0.4,0.55,0.75,1.,1.25,1.5,1.75], fd.float_type())        # Fano exciton
    NBamp_grid = tf.cast([0.,0.02,0.04], fd.float_type())   # amplitude for non-binomial NR recombination fluctuations
    NBloc = tf.cast([0.4,0.45,0.5,0.55], fd.float_type())                       # non-binomial: loc of elecfrac
    RawSkew_grid = tf.cast([0.,1.5,3.,5.,8.,13.], fd.float_type())              # raw skewness

    x_grid_points = (Fi_grid, Fex_grid, NBamp_grid, NBloc, RawSkew_grid, keVnr_choices)

    #define reference function values --> f(x_grid_points)
    y_ref = tf.cast(fit_values_allkeVnr_allparam, fd.float_type()), # want shape = (1, ...)

    # nd interpolation of points "x" to determine Edependance of various values

    interp = tfp.math.batch_interp_rectilinear_nd_grid(x=x,
                                                       x_grid_points=x_grid_points,
                                                       y_ref=y_ref,
                                                       axis=1,
                                                       )[0] # o/p shape (1, len(E), 7)
                                                            # want shape   (len(E), 7)

    # print('interpo final shape: ',np.shape(interp))
    # print('interpo final shape: ',tf.shape(interp))
    return interp 
    
def skewnorm_1d(x,x_std,x_mean,x_alpha):

    # define scale and loc params for shifting skew2d from x=0
    skewa = x_alpha
    final_std = x_std
    final_mean = x_mean

    delta = (skewa / tf.sqrt(1. + skewa**2))
    scale = final_std / tf.sqrt(1. - 2 * delta**2 / pi)
    loc = final_mean - scale * delta * tf.sqrt(2 / pi)


    denominator = tf.sqrt( 2. * pi * x_std * x_std )
    exp_prefactor = -1. / 2 
    exp_term_1 = (x - loc) * (x - loc) / (scale*scale)


    norm_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1))

    # Phi(x) = (1 + Erf(x/sqrt(2)))/2
    Erf_arg = (x_alpha * (x-loc)/scale)
    Erf = tf.math.erf( Erf_arg / 1.4142 )

    norm_cdf = ( 1 + Erf ) / 2

    # skew1d = norm(x_mod)*(2*std/scale)*normcdf(xmod)
    probs = norm_pdf * norm_cdf * (2*final_std / scale)

    return probs
        
###

@export
class NRSource(fd.BlockModelSource): # TODO -- ADD SKEW!
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstSS,
        fd_dd_migdal.MakeS1S2SS)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 1.
    
#     #### Old Yield Model
#     @staticmethod
#     def signal_means(energy, a=13.1895962, b=1.06532331,
#                      c_s2_0=3.70318382, c_s2_1=-3.49159718, c_s2_2=0.07861683,
#                      g1=0.1131, g2=47.35,
#                      s1_mean_multiplier=1., s2_mean_multiplier=1.):
#         P = c_s2_0 + c_s2_1 * (fd.tf_log10(energy) - 1.6) + c_s2_2 * pow((fd.tf_log10(energy) - 1.6), 2)
#         s2_mean = s2_mean_multiplier * P * energy * g2

#         s1_mean = s1_mean_multiplier * (a * energy**b - s2_mean / g2) * g1
#         s1_mean = tf.where(s1_mean < 0.01, 0.01 * tf.ones_like(s1_mean, dtype=fd.float_type()), s1_mean)

#         return s1_mean, s2_mean

#     @staticmethod
#     def signal_vars(*args, d_s1=1.20307136, d_s2=38.27449296):
#         s1_mean = args[0]
#         s2_mean = args[1]

#         s1_var = d_s1 * s1_mean

#         s2_var = d_s2 * s2_mean

#         return s1_var, s2_var

#     @staticmethod
#     def signal_corr(energies, anti_corr=-0.20949764):
#         return anti_corr * tf.ones_like(energies)
    
#     @staticmethod # 240213 - AV added
#     def signal_skews(energies, s1_skew=0,s2_skew=0):
        
#         s1_skew *= tf.ones_like(energies)
#         s2_skew *= tf.ones_like(energies)
        
#         return s1_skew, s2_skew
    
    
    ### New Yield Model # 240305 - AV added
    ### params --> {alpha, beta, gamma, delta, epsilon, Fi, Fex, NBamp, RawSkew}
    @staticmethod
    def yield_params(energies, yalpha=11.0, ybeta=1.1, ythomas=0.038, yepsilon=12.6):
        
        Efield = 193.
        rho = 2.9
        
        Qy = 1 / ythomas / tf.sqrt( energies + yepsilon) * ( 1 - 1/(1 + (energies/0.3)**2) )
        Ne_mean = Qy * energies
        
        Ly = yalpha * energies**ybeta - Ne_mean
        Nph_mean = Ly * (1 - 1/(1 + (energies/0.3)**2) )
        
        return Nph_mean, Ne_mean
    
    @staticmethod
    def quanta_params(energies, Fi=0.4,Fex=0.4,NBamp=0.0,NBloc=0.0,RawSkew=2.25):
        
        Fi *= tf.ones_like(energies)
        Fex *= tf.ones_like(energies)
        NBamp *= tf.ones_like(energies)
        NBloc *= tf.ones_like(energies)
        RawSkew *= tf.ones_like(energies)
        
        return Fi, Fex, NBamp, NBloc, RawSkew
    
    
    ###
    
    def get_s2(self, s2):
        return s2

    def s1s2_acceptance(self, s1, s2, s1_min=5, s1_max=200, s2_min=400, s2_max=2e4): # 231208 AV adjusted s1_min from 20 --> 5 phd
        s1_acc = tf.where((s1 < s1_min) | (s1 > s1_max),
                          tf.zeros_like(s1, dtype=fd.float_type()),
                          tf.ones_like(s1, dtype=fd.float_type()))
        s2_acc = tf.where((s2 < s2_min) | (s2 > s2_max),
                          tf.zeros_like(s2, dtype=fd.float_type()),
                          tf.ones_like(s2, dtype=fd.float_type()))
        s1s2_acc = tf.where((s2 > 200*s1**(0.73)), ### turn off for testing  TURN BACK ON!!!!!!!!!!
                            tf.ones_like(s2, dtype=fd.float_type()),
                            tf.zeros_like(s2, dtype=fd.float_type()))
        nr_endpoint = tf.where((s1 > 140) & (s2 > 8e3) & (s2 < 11.5e3),
                            tf.zeros_like(s2, dtype=fd.float_type()),
                            tf.ones_like(s2, dtype=fd.float_type()))

        return (s1_acc * s2_acc * s1s2_acc * nr_endpoint)
        # return (s1_acc * s2_acc) # for testing

    final_dimensions = ('s1',)
    
    def estimate_mu(self, **params):
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,1400,90), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,320,95), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        # S1S2 binning
        s1_edges = tf.cast(tf.linspace(self.defaults['s1_min'], self.defaults['s1_max'], 100), fd.float_type())
        s2_edges = tf.cast(tf.experimental.numpy.geomspace(self.defaults['s2_min'], self.defaults['s2_max'], 100), fd.float_type())
        s1 = 0.5 * (s1_edges[1:] + s1_edges[:-1])
        s2 = 0.5 * (s2_edges[1:] + s2_edges[:-1])
        s1_diffs = tf.experimental.numpy.diff(s1_edges)
        s2_diffs = tf.experimental.numpy.diff(s2_edges)

        S1_pos = tf.repeat(s1[:,o],len(Nph),axis=1)
        S2_pos = tf.repeat(s2[:,o],len(Ne),axis=1)
        
        S1_mesh, S2_mesh = tf.meshgrid(s1,s2,indexing='ij')

        # Energy binning
        energies = fd.np_to_tf(self.energies_first)
        rates_vs_energy = fd.np_to_tf(self.rates_vs_energy_first)
        
        
        # Load params
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energies, 
                                        ptensor=ptensor)
        
        Fi, Fex, NBamp, NBloc, RawSkew = self.gimme('quanta_params',
                                                     bonus_arg=energies, 
                                                     ptensor=ptensor)
        
        xcoords_skew2D = tf.stack((Fi, Fex, NBamp, NBloc, RawSkew, energies), axis=-1)
        skew2D_model_param = interp_nd(x=xcoords_skew2D)
        
        Nph_std = tf.sqrt(skew2D_model_param[:,2]**2 / skew2D_model_param[:,0] * Nph_mean)
        Ne_std = tf.sqrt(skew2D_model_param[:,3]**2 / skew2D_model_param[:,1] * Ne_mean)
        Nph_skew = skew2D_model_param[:,4]
        Ne_skew = skew2D_model_param[:,5]    
        initial_corr = skew2D_model_param[:,6]
        
        ### Quanta Production
        x = NphNe[:,:,0]
        x = tf.repeat(x[:,:,o],len(energies),axis=2)

        y = NphNe[:,:,1]
        y = tf.repeat(y[:,:,o],len(energies),axis=2) 

        x_mean = Nph_mean
        y_mean = Ne_mean
        x_std = Nph_std
        y_std = Ne_std
        anti_corr = initial_corr
        x_skew = Nph_skew
        y_skew = Ne_skew
        
        # adjust dimensionality of input tensors
        skews1 = x_skew[:,o]
        skews2 = y_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = x_std[:,o]
        f_std_y = y_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = x_mean[:,o]
        f_mean_y = y_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(anti_corr[:,o], 2, axis=1)  
        cov = tf.repeat(cov[:,:,o], 2, axis=2)
        cov = cov - (tf.eye(2)*cov-tf.eye(2)) 

        # define scale and loc params for shifting skew2d from (0,0)
        del1 = tf.einsum('...ji,...jk->...ik',skewa,cov)
        del2 = tf.einsum('...ij,...jk->...ik',del1,skewa)       
        bCa = tf.einsum('...jk,...ji->...ki',cov,skewa) 

        aCa = 1. + del2
        delta = (1. / tf.sqrt(aCa)) * bCa

        scale = final_std / tf.sqrt(1. - 2 * delta**2 / pi)
        loc = final_mean - scale * delta * tf.sqrt(2/pi)

        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * x_std * y_std * tf.sqrt(1. - anti_corr * anti_corr) 
        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))
        exp_term_1 = (x - loc[:,0,0]) * (x - loc[:,0,0]) / (scale[:,0,0]*scale[:,0,0])
        exp_term_2 = (y - loc[:,1,0]) * (y - loc[:,1,0]) / (scale[:,1,0]*scale[:,1,0])
        exp_term_3 = -2. * anti_corr * (x - loc[:,0,0]) * (y - loc[:,1,0]) / (scale[:,0,0] * scale[:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (x_skew * (x-loc[:,0,0])/scale[:,0,0]) + (y_skew * (y-loc[:,1,0])/scale[:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,0,0]*scale[:,1,0])) * (final_std[:,0,0]*final_std[:,1,0])
    
        probs *= rates_vs_energy
        probs = tf.reduce_sum(probs, axis=2)
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0]
        
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = 4.61849047 * Nph**(-0.23931848)

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = -2.37542105 *  Ne** (-0.26152676)


        S1_pdf = skewnorm_1d(x=S1_pos,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,o,:],len(s2),1)
        S1_pdf = tf.repeat(S1_pdf[:,:,:,o],len(Ne),3)

        S2_pdf = skewnorm_1d(x=S2_pos,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[o,:,:],len(s1),0)
        S2_pdf = tf.repeat(S2_pdf[:,:,o,:],len(Nph),2)

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf

        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=3)
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2)
        
        
        # account for S1,S2 space fiducial ROI acceptance
        acceptance = self.gimme('s1s2_acceptance',
                                bonus_arg=(S1_mesh, S2_mesh),
                                special_call=True)
        S1S2_pdf *= acceptance
        
#         plt.pcolormesh(s1,s2,tf.transpose(S1S2_pdf).numpy(),cmap='jet',norm=mpl.colors.LogNorm(vmin=1e-9))
#         plt.xlabel('S1 [phd]')
#         plt.ylabel('S2 [phd]')
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlim(s1_edges[0],s1_edges[-1])
#         plt.ylim(s2_edges[0],s2_edges[-1])
        
        
        # rescale by the S1,S2 bin width
        mu_est = s1_diffs[o, :] @ S1S2_pdf
        mu_est = mu_est @ s2_diffs[:, o]
        mu_est = mu_est[0][0]
        
        return mu_est

        
        


@export
class NRNRSource(NRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMSU,
        fd_dd_migdal.EnergySpectrumSecondMSU,
        fd_dd_migdal.MakeS1S2MSU)

    no_step_dimensions = ('energy_second')

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def pdf_for_mu_pre_populate(self, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)
        ptensor = self.ptensor_from_kwargs(**params)
        
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_second)  # shape: {E2_bins}
        
         ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme('yield_params',
                                           bonus_arg=energy_first, 
                                           ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        Fi_1, Fex_1, NBamp_1, NBloc_1, RawSkew_1 = self.gimme('quanta_params',
                                                              bonus_arg=energy_first, 
                                                              ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        
        xcoords_1_skew2D = tf.stack((Fi_1, Fex_1, NBamp_1, NBloc_1, RawSkew_1, energy_first), axis=-1) # shape: {E1_bins, 6}
        xcoords_1_skew2D = tf.reshape(xcoords_1_skew2D, [1, -1, 6])                            # shape: {1, E1_bins, 6}
        skew2D_1_model_param = interp_nd(x=xcoords_1_skew2D) # shape: {E1_bins, 7}
        
        Nph_1_mean = tf.reshape(Nph_1_mean, [-1,1]) # shape: {E1_bins,1}
        Ne_1_mean = tf.reshape(Ne_1_mean, [-1,1])   # shape: {E1_bins,1}
        Nph_1_std = tf.reshape(tf.sqrt(skew2D_1_model_param[:,2]**2 / skew2D_1_model_param[:,0] * Nph_1_mean[:,0]), [-1,1]) # shape: {E1_bins,1}
        Ne_1_std = tf.reshape(tf.sqrt(skew2D_1_model_param[:,3]**2 / skew2D_1_model_param[:,1] * Ne_1_mean[:,0]), [-1,1])   # shape: {E1_bins,1}
        Nph_1_skew = tf.reshape(skew2D_1_model_param[:,4], [-1,1])      # shape: {E1_bins,1}
        Ne_1_skew = tf.reshape(skew2D_1_model_param[:,5], [-1,1])       # shape: {E1_bins,1}
        initial_1_corr = tf.reshape(skew2D_1_model_param[:,6], [-1,1])  # shape: {E1_bins,1}
        
        
        ###########  # Second vertex
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params',
                                           bonus_arg=energy_second, 
                                           ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        Fi_2, Fex_2, NBamp_2, NBloc_2, RawSkew_2 = self.gimme('quanta_params',
                                                              bonus_arg=energy_second, 
                                                              ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        xcoords_2_skew2D = tf.stack((Fi_2, Fex_2, NBamp_2, NBloc_2, RawSkew_2, energy_second), axis=-1) # shape: {E2_bins, 6}
        xcoords_2_skew2D = tf.reshape(xcoords_2_skew2D, [1, -1, 6])                            # shape: {1, E2_bins, 6}
        skew2D_2_model_param = interp_nd(x=xcoords_2_skew2D) # shape: {E2_bins, 7}
        
        Nph_2_mean = tf.reshape(Nph_2_mean, [-1,1]) # shape: {E2_bins,1}
        Ne_2_mean = tf.reshape(Ne_2_mean, [-1,1])  # shape: {E2_bins,1}
        Nph_2_std = tf.reshape(tf.sqrt(skew2D_2_model_param[:,2]**2 / skew2D_2_model_param[:,0] * Nph_2_mean[:,0]), [-1,1]) # shape: {E2_bins,1}
        Ne_2_std = tf.reshape(tf.sqrt(skew2D_2_model_param[:,3]**2 / skew2D_2_model_param[:,1] * Ne_2_mean[:,0]), [-1,1])   # shape: {E2_bins,1}
        Nph_2_skew = tf.reshape(skew2D_2_model_param[:,4], [-1,1])      # shape: {E2_bins,1}
        Ne_2_skew = tf.reshape(skew2D_2_model_param[:,5], [-1,1])       # shape: {E2_bins,1}
        initial_2_corr = tf.reshape(skew2D_2_model_param[:,6], [-1,1])  # shape: {E2_bins,1}
        
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        Nph_mean = Nph_1_mean + tf.transpose(Nph_2_mean) # shape: {E1_bins,E2_bins}
        Nph_std = tf.sqrt(Nph_1_std**2 + tf.transpose(Nph_2_std**2)) # shape: {E1_bins,E2_bins}
        Ne_mean = Ne_1_mean + tf.transpose(Ne_2_mean) # shape: {E1_bins,E2_bins}
        Ne_std = tf.sqrt(Ne_1_std**2 + tf.transpose(Ne_2_std**2)) # shape: {E1_bins,E2_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (Nph_1_skew*Nph_1_mean + tf.transpose(Nph_2_skew*Nph_2_mean))/(Nph_mean) # shape: {E1_bins,E2_bins}
        Ne_skew = (Ne_1_skew*Ne_1_mean + tf.transpose(Ne_2_skew*Ne_2_mean))/(Ne_mean) # shape: {E1_bins,E2_bins}
        initial_corr = (initial_1_corr*energy_first+tf.transpose(initial_2_corr*energy_second))/(energy_first+tf.transpose(energy_second)) # shape: {E1_bins,E2_bins}
        
        spectrum = fd.tf_to_np(self.rates_vs_energy)

        self.defaults = old_defaults

        return Nph_mean, Nph_std, Ne_mean, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum, energy_first, energy_second

    @staticmethod
    # @nb.njit(error_model="numpy",fastmath=True)
    def pdf_for_mu(s1,s2, Nph_mean, Nph_std, Ne_mean, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum, energy_first, energy_second):
        
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,3300,90), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,600,95), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        S1_pos = tf.repeat(s1[:,o],len(Nph),axis=1)
        S2_pos = tf.repeat(s2[:,o],len(Ne),axis=1)
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}

        x_mean = Nph_mean
        y_mean = Ne_mean
        x_std = Nph_std
        y_std = Ne_std
        anti_corr = initial_corr
        x_skew = Nph_skew
        y_skew = Ne_skew
        
        # adjust dimensionality of input tensors
        skews1 = x_skew[:,:,o] # final shape: {E1_bins,E2_bins,1}
        skews2 = y_skew[:,:,o] # final shape: {E1_bins,E2_bins,1}
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        f_std_x = x_std[:,:,o] # final shape: {E1_bins,E2_bins,1}
        f_std_y = y_std[:,:,o] # final shape: {E1_bins,E2_bins,1}
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        f_mean_x = x_mean[:,:,o] # final shape: {E1_bins,E2_bins,1}
        f_mean_y = y_mean[:,:,o] # final shape: {E1_bins,E2_bins,1}
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        cov = tf.repeat(anti_corr[:,:,o], 2, axis=2)  # final shape: {E1_bins,E2_bins,2}
        cov = tf.repeat(cov[:,:,:,o], 2, axis=3)      # final shape: {E1_bins,E2_bins,2,2}
        cov = cov - (tf.eye(2)*cov-tf.eye(2))         # final shape: {E1_bins,E2_bins,2,2}
        
        # define scale and loc params for shifting skew2d from (0,0)
        del1 = tf.einsum('...ji,...jk->...ik',skewa,cov)
        del2 = tf.einsum('...ij,...jk->...ik',del1,skewa)       
        bCa = tf.einsum('...jk,...ji->...ki',cov,skewa) 

        aCa = 1. + del2
        delta = (1. / tf.sqrt(aCa)) * bCa

        scale = final_std / tf.sqrt(1. - 2 * delta**2 / pi) # shape: {E1_bins,E2_bins,2,1}  
        loc = final_mean - scale * delta * tf.sqrt(2/pi)    # shape: {E1_bins,E2_bins,2,1}  
        
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * x_std * y_std * tf.sqrt(1. - anti_corr * anti_corr) 
        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))
        exp_term_1 = (x - loc[:,:,0,0]) * (x - loc[:,:,0,0]) / (scale[:,:,0,0]*scale[:,:,0,0])
        exp_term_2 = (y - loc[:,:,1,0]) * (y - loc[:,:,1,0]) / (scale[:,:,1,0]*scale[:,:,1,0])
        exp_term_3 = -2. * anti_corr * (x - loc[:,:,0,0]) * (y - loc[:,:,1,0]) / (scale[:,:,0,0] * scale[:,:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))
        
        
        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (x_skew * (x-loc[:,:,0,0])/scale[:,:,0,0]) + (y_skew * (y-loc[:,:,1,0])/scale[:,:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,:,0,0]*scale[:,:,1,0])) * (final_std[:,:,0,0]*final_std[:,:,1,0])  # shape: {Nph,Ne,E1_bins,E2_bins}      
        
        probs = probs.numpy()
        probs[np.isnan(probs)]=0
        
        ### Calculate probabilities
        probs *= spectrum # shape: {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=3) # final shape: {Nph,Ne,E1_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}    
        
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35
        
        S1_mean = Nph*g1 # shape: {Nph}
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = 4.61849047 * Nph**(-0.23931848)
        
        S2_mean = Ne*g2 # shape: {Ne}
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = -2.37542105 *  Ne** (-0.26152676)
             
        S1_pdf = skewnorm_1d(x=S1_pos,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,o,:],len(s2),1)
        S1_pdf = tf.repeat(S1_pdf[:,:,:,o],len(Ne),3)
        
        S2_pdf = skewnorm_1d(x=S2_pos,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[o,:,:],len(s1),0)
        S2_pdf = tf.repeat(S2_pdf[:,:,o,:],len(Nph),2)
        
        S1_pdf = S1_pdf.numpy()
        S2_pdf = S2_pdf.numpy()
        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {S1, S2, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = np.sum(S1S2_pdf,axis=3) # final shape: {S1, S2, Nph, Ne}
        S1S2_pdf = np.sum(S1S2_pdf,axis=2) # final shape: {S1, S2, Nph, Ne}
               
       
        return S1S2_pdf

    def estimate_mu(self, error=1e-5, **params):
        """
        """
        # S1S2 binning
        s1_edges = tf.cast(tf.linspace(self.defaults['s1_min'], self.defaults['s1_max'], 100), fd.float_type())
        s2_edges = tf.cast(tf.experimental.numpy.geomspace(self.defaults['s2_min'], self.defaults['s2_max'], 100), fd.float_type())
        s1 = 0.5 * (s1_edges[1:] + s1_edges[:-1])
        s2 = 0.5 * (s2_edges[1:] + s2_edges[:-1])
        s1_diffs = tf.experimental.numpy.diff(s1_edges)
        s2_diffs = tf.experimental.numpy.diff(s2_edges)
        
        S1_mesh, S2_mesh = tf.meshgrid(s1,s2,indexing='ij')
        
        
        Nph_mean, Nph_std, Ne_mean, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum, energy_first, energy_second = \
            self.pdf_for_mu_pre_populate(**params)
        
        S1S2_pdf = self.pdf_for_mu(s1, s2, Nph_mean, Nph_std, Ne_mean, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum, energy_first, energy_second)
        

        
        # account for S1,S2 space fiducial ROI acceptance      
        acceptance = self.gimme('s1s2_acceptance',
                                bonus_arg=(S1_mesh, S2_mesh),
                                special_call=True)
        S1S2_pdf *= acceptance
        
        plt.pcolormesh(s1,s2,tf.transpose(S1S2_pdf).numpy(),cmap='jet',norm=mpl.colors.LogNorm(vmin=1e-9))      
        plt.xlabel('S1 [phd]')
        plt.ylabel('S2 [phd]')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(s1_edges[0],s1_edges[-1])
        plt.ylim(s2_edges[0],s2_edges[-1])
        
        
        # rescale by the S1,S2 bin width
        mu_est = s1_diffs[o, :] @ S1S2_pdf
        mu_est = mu_est @ s2_diffs[:, o]
        mu_est = mu_est[0][0]
        
        return mu_est
    

# @export
# class NRNRNRSource(NRNRSource):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstMSU3,
#         fd_dd_migdal.EnergySpectrumOthersMSU3,
#         fd_dd_migdal.MakeS1S2MSU3)

#     no_step_dimensions = ('energy_others')

#     def pdf_for_mu_pre_populate(self, **params):
#         old_defaults = copy(self.defaults)
#         self.set_defaults(**params)

#         e1, e2, e3 = np.meshgrid(self.energies_first, self.energies_others, self.energies_others, indexing='ij')

#         s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means', e1)
#         s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', e2)
#         s1_mean_third, s2_mean_third = self.gimme_numpy('signal_means', e3)
#         s1_mean = s1_mean_first + s1_mean_second + s1_mean_third
#         s2_mean = s2_mean_first + s2_mean_second + s2_mean_third

#         s1_var_first, s2_var_first = self.gimme_numpy('signal_vars', (s1_mean_first, s2_mean_first))
#         s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
#         s1_var_third, s2_var_third = self.gimme_numpy('signal_vars', (s1_mean_third, s2_mean_third))
#         s1_var = s1_var_first + s1_var_second + s1_var_third
#         s2_var = s2_var_first + s2_var_second + s2_var_third

#         s1_mean = fd.tf_to_np(s1_mean)
#         s2_mean = fd.tf_to_np(s2_mean)
#         s1_var = fd.tf_to_np(s1_var)
#         s2_var = fd.tf_to_np(s2_var)

#         s1_std = np.sqrt(s1_var)
#         s2_std = np.sqrt(s2_var)

#         s1s2_corr_nr = self.gimme_numpy('signal_corr', e1)
#         s1s2_cov_first = s1s2_corr_nr  * np.sqrt(s1_var_first * s2_var_first)
#         s1s2_cov_second = s1s2_corr_nr  * np.sqrt(s1_var_second * s2_var_second)
#         s1s2_cov_third = s1s2_corr_nr  * np.sqrt(s1_var_third * s2_var_third)

#         s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
#         anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)
#         anti_corr = fd.tf_to_np(anti_corr)

#         spectrum = fd.tf_to_np(self.rates_vs_energy)

#         self.defaults = old_defaults

#         return s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum


# @export
# class Migdal2Source(NRNRSource):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstMigdal,
#         fd_dd_migdal.EnergySpectrumSecondMigdal2,
#         fd_dd_migdal.MakeS1S2Migdal)

#     S2Width_dist = np.load(os.path.join(
#         os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

#     hist_values_S2Width = S2Width_dist['hist_values']
#     S2Width_edges = S2Width_dist['S2Width_edges']

#     mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
#     mh_S2Width = mh_S2Width / mh_S2Width.n
#     mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

#     S2Width_diff_rate = mh_S2Width
#     S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

#     ER_NEST = np.load(os.path.join(
#         os.path.dirname(__file__), './migdal_database/ER_NEST.npz'))

#     E_ER = ER_NEST['EkeVee']
#     s1_mean_ER = itp.interp1d(E_ER, ER_NEST['s1mean'])
#     s2_mean_ER = itp.interp1d(E_ER, ER_NEST['s2mean'])
#     s1_var_ER = itp.interp1d(E_ER, ER_NEST['s1std']**2)
#     s2_var_ER = itp.interp1d(E_ER, ER_NEST['s2std']**2)
#     s1s2_corr_ER = itp.interp1d(E_ER, ER_NEST['S1S2corr'])

#     def __init__(self, *args, **kwargs):
#         energies_first = self.model_blocks[0].energies_first
#         energies_first = tf.where(energies_first > 49., 49. * tf.ones_like(energies_first), energies_first)

#         if hasattr(self.model_blocks[1], 'energies_second'):
#             energies_first = tf.repeat(energies_first[:, o], tf.shape(self.model_blocks[1].energies_second), axis=1)

#         self.s1_mean_ER_tf, self.s2_mean_ER_tf = self.signal_means_ER(energies_first)
#         self.s1_var_ER_tf, self.s2_var_ER_tf, self.s1s2_cov_ER_tf = self.signal_vars_ER(energies_first)

#         super().__init__(*args, **kwargs)

#     def signal_means_ER(self, energy):
#         energy_cap = np.where(energy <= 49., energy, 49.)
#         s1_mean = tf.cast(self.s1_mean_ER(energy_cap), fd.float_type())
#         s2_mean = tf.cast(self.s2_mean_ER(energy_cap), fd.float_type())

#         return s1_mean, s2_mean

#     def signal_vars_ER(self, energy):
#         energy_cap = np.where(energy <= 49., energy, 49.)
#         s1_var = tf.cast(self.s1_var_ER(energy_cap), fd.float_type())
#         s2_var = tf.cast(self.s2_var_ER(energy_cap), fd.float_type())
#         s1s2_corr = tf.cast(np.nan_to_num(self.s1s2_corr_ER(energy_cap)), fd.float_type())
#         s1s2_cov = s1s2_corr * tf.sqrt(s1_var * s2_var)

#         return s1_var, s2_var, s1s2_cov

#     def pdf_for_mu_pre_populate(self, **params):
#         old_defaults = copy(self.defaults)
#         self.set_defaults(**params)

#         e1, e2 = np.meshgrid(self.energies_first, self.energies_second, indexing='ij')

#         s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', e1)
#         s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', e2)
#         s1_mean = s1_mean_first + s1_mean_second
#         s2_mean = s2_mean_first + s2_mean_second

#         s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', e1)
#         s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
#         s1_var = s1_var_first + s1_var_second
#         s2_var = s2_var_first + s2_var_second

#         s1_mean = fd.tf_to_np(s1_mean)
#         s2_mean = fd.tf_to_np(s2_mean)
#         s1_var = fd.tf_to_np(s1_var)
#         s2_var = fd.tf_to_np(s2_var)

#         s1_std = np.sqrt(s1_var)
#         s2_std = np.sqrt(s2_var)

#         s1s2_corr_second = self.gimme_numpy('signal_corr', e1)
#         s1s2_cov_second = s1s2_corr_second  * np.sqrt(s1_var_second * s2_var_second)

#         s1s2_cov = s1s2_cov_first + s1s2_cov_second
#         anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)
#         anti_corr = fd.tf_to_np(anti_corr)

#         spectrum = fd.tf_to_np(self.rates_vs_energy)

#         self.defaults = old_defaults

#         return s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum


# @export
# class Migdal3Source(Migdal2Source):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstMigdal,
#         fd_dd_migdal.EnergySpectrumSecondMigdal3,
#         fd_dd_migdal.MakeS1S2Migdal)


# @export
# class Migdal4Source(Migdal2Source):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstMigdal,
#         fd_dd_migdal.EnergySpectrumSecondMigdal4,
#         fd_dd_migdal.MakeS1S2Migdal)


# @export
# class MigdalMSUSource(Migdal2Source):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstMigdalMSU,
#         fd_dd_migdal.EnergySpectrumOthersMigdalMSU,
#         fd_dd_migdal.MakeS1S2MigdalMSU)

#     no_step_dimensions = ('energy_others')

#     S2Width_dist = np.load(os.path.join(
#         os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

#     hist_values_S2Width = S2Width_dist['hist_values']
#     S2Width_edges = S2Width_dist['S2Width_edges']

#     mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
#     mh_S2Width = mh_S2Width / mh_S2Width.n
#     mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

#     S2Width_diff_rate = mh_S2Width
#     S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

#     def pdf_for_mu_pre_populate(self, **params):
#         old_defaults = copy(self.defaults)
#         self.set_defaults(**params)

#         e1, e2, e3 = np.meshgrid(self.energies_first, self.energies_others, self.energies_others, indexing='ij')

#         s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', e1)
#         s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', e2)
#         s1_mean_third, s2_mean_third = self.gimme_numpy('signal_means', e3)
#         s1_mean = s1_mean_first + s1_mean_second + s1_mean_third
#         s2_mean = s2_mean_first + s2_mean_second + s2_mean_third

#         s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', e1)
#         s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
#         s1_var_third, s2_var_third = self.gimme_numpy('signal_vars', (s1_mean_third, s2_mean_third))
#         s1_var = s1_var_first + s1_var_second + s1_var_third
#         s2_var = s2_var_first + s2_var_second + s2_var_third

#         s1_mean = fd.tf_to_np(s1_mean)
#         s2_mean = fd.tf_to_np(s2_mean)
#         s1_var = fd.tf_to_np(s1_var)
#         s2_var = fd.tf_to_np(s2_var)

#         s1_std = np.sqrt(s1_var)
#         s2_std = np.sqrt(s2_var)

#         s1s2_corr_others = self.gimme_numpy('signal_corr', e1)
#         s1s2_cov_second = s1s2_corr_others * np.sqrt(s1_var_second * s2_var_second)
#         s1s2_cov_third = s1s2_corr_others * np.sqrt(s1_var_third * s2_var_third)

#         s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
#         anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)
#         anti_corr = fd.tf_to_np(anti_corr)

#         spectrum = fd.tf_to_np(self.rates_vs_energy)

#         self.defaults = old_defaults

#         return s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum


# @export
# class IECSSource(Migdal2Source):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstIE_CS,
#         fd_dd_migdal.EnergySpectrumSecondIE_CS,
#         fd_dd_migdal.MakeS1S2Migdal)

#     S2Width_dist = np.load(os.path.join(
#         os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

#     hist_values_S2Width = S2Width_dist['hist_values']
#     S2Width_edges = S2Width_dist['S2Width_edges']

#     mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
#     mh_S2Width = mh_S2Width / mh_S2Width.n
#     mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

#     S2Width_diff_rate = mh_S2Width
#     S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()


@export
class ERSource(NRNRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstER,
        fd_dd_migdal.MakeS1S2ER)
    
    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))
    
    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    ER_NEST = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/ER_NEST.npz'))

    E_ER = ER_NEST['EkeVee']
    s1_mean_ER = itp.interp1d(E_ER, ER_NEST['s1mean'])
    s2_mean_ER = itp.interp1d(E_ER, ER_NEST['s2mean'])
    s1_var_ER = itp.interp1d(E_ER, ER_NEST['s1std']**2)
    s2_var_ER = itp.interp1d(E_ER, ER_NEST['s2std']**2)
    s1s2_corr_ER = itp.interp1d(E_ER, ER_NEST['S1S2corr'])
    
    def __init__(self, *args, **kwargs):
        energies_first = self.model_blocks[0].energies_first
        energies_first = tf.where(energies_first > 49., 49. * tf.ones_like(energies_first), energies_first)

        if hasattr(self.model_blocks[1], 'energies_second'):
            energies_first = tf.repeat(energies_first[:, o], tf.shape(self.model_blocks[1].energies_second), axis=1)

        self.s1_mean_ER_tf, self.s2_mean_ER_tf = self.signal_means_ER(energies_first)
        self.s1_var_ER_tf, self.s2_var_ER_tf, self.s1s2_cov_ER_tf = self.signal_vars_ER(energies_first)

        super().__init__(*args, **kwargs)
        
    def signal_means_ER(self, energy):
        energy_cap = np.where(energy <= 49., energy, 49.)
        s1_mean = tf.cast(self.s1_mean_ER(energy_cap), fd.float_type())
        s2_mean = tf.cast(self.s2_mean_ER(energy_cap), fd.float_type())

        return s1_mean, s2_mean

    def signal_vars_ER(self, energy):
        energy_cap = np.where(energy <= 49., energy, 49.)
        s1_var = tf.cast(self.s1_var_ER(energy_cap), fd.float_type())
        s2_var = tf.cast(self.s2_var_ER(energy_cap), fd.float_type())
        s1s2_corr = tf.cast(np.nan_to_num(self.s1s2_corr_ER(energy_cap)), fd.float_type())
        s1s2_cov = s1s2_corr * tf.sqrt(s1_var * s2_var)

        return s1_var, s2_var, s1s2_cov
    
    
    def pdf_for_mu_pre_populate(self, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)

        s1_mean, s2_mean = self.gimme('signal_means_ER', bonus_arg=self.energies_first)
        s1_var, s2_var, s1s2_cov = self.gimme('signal_vars_ER',  bonus_arg=self.energies_first)
        s1_mean = fd.tf_to_np(s1_mean)
        s2_mean = fd.tf_to_np(s2_mean)
        s1_var = fd.tf_to_np(s1_var)
        s2_var = fd.tf_to_np(s2_var)

        s1_std = np.sqrt(s1_var)
        s2_std = np.sqrt(s2_var)

        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)
        anti_corr = fd.tf_to_np(anti_corr)

        spectrum = fd.tf_to_np(self.rates_vs_energy_first)

        self.defaults = old_defaults

        return s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum
    
    @staticmethod
    @nb.njit(error_model="numpy",fastmath=True)
    def pdf_for_mu(s1, s2, s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum):
        denominator = 2. * np.pi * s1_std * s2_std * np.sqrt(1.- anti_corr * anti_corr)
        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))
        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        pdf_vals = 1. / denominator * np.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        s1s2_acc = np.where((s2 > 200*s1**(0.73)), 1., 0.)
        nr_endpoint = np.where((s1 > 140) & (s2 > 8e3) & (s2 < 11.5e3), 0., 1.)

        return np.sum(pdf_vals * s1s2_acc * nr_endpoint * spectrum)

    def estimate_mu(self, error=1e-5, **params):
        """
        """
        s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum = \
            self.pdf_for_mu_pre_populate(**params)

        f = lambda s2, s1: self.pdf_for_mu(s1, s2, s1_mean, s2_mean, s1_var, s2_var,
                                           s1_std, s2_std, anti_corr, spectrum)
        return integrate.dblquad(f, self.defaults['s1_min'], self.defaults['s1_max'],
                                 self.defaults['s2_min'], self.defaults['s2_max'],
                                 epsabs=error, epsrel=error)[0]
    
    
# @export
# class Migdal2Source(NRNRSource):
#     model_blocks = (
#         fd_dd_migdal.EnergySpectrumFirstMigdal,
#         fd_dd_migdal.EnergySpectrumSecondMigdal2,
#         fd_dd_migdal.MakeS1S2Migdal)

    # S2Width_dist = np.load(os.path.join(
    #     os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

#     hist_values_S2Width = S2Width_dist['hist_values']
#     S2Width_edges = S2Width_dist['S2Width_edges']

#     mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
#     mh_S2Width = mh_S2Width / mh_S2Width.n
#     mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

#     S2Width_diff_rate = mh_S2Width
#     S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

#     ER_NEST = np.load(os.path.join(
#         os.path.dirname(__file__), './migdal_database/ER_NEST.npz'))

    # E_ER = ER_NEST['EkeVee']
    # s1_mean_ER = itp.interp1d(E_ER, ER_NEST['s1mean'])
    # s2_mean_ER = itp.interp1d(E_ER, ER_NEST['s2mean'])
    # s1_var_ER = itp.interp1d(E_ER, ER_NEST['s1std']**2)
    # s2_var_ER = itp.interp1d(E_ER, ER_NEST['s2std']**2)
    # s1s2_corr_ER = itp.interp1d(E_ER, ER_NEST['S1S2corr'])

#     def __init__(self, *args, **kwargs):
#         energies_first = self.model_blocks[0].energies_first
#         energies_first = tf.where(energies_first > 49., 49. * tf.ones_like(energies_first), energies_first)

#         if hasattr(self.model_blocks[1], 'energies_second'):
#             energies_first = tf.repeat(energies_first[:, o], tf.shape(self.model_blocks[1].energies_second), axis=1)

#         self.s1_mean_ER_tf, self.s2_mean_ER_tf = self.signal_means_ER(energies_first)
#         self.s1_var_ER_tf, self.s2_var_ER_tf, self.s1s2_cov_ER_tf = self.signal_vars_ER(energies_first)

#         super().__init__(*args, **kwargs)

#     def signal_means_ER(self, energy):
#         energy_cap = np.where(energy <= 49., energy, 49.)
#         s1_mean = tf.cast(self.s1_mean_ER(energy_cap), fd.float_type())
#         s2_mean = tf.cast(self.s2_mean_ER(energy_cap), fd.float_type())

#         return s1_mean, s2_mean

#     def signal_vars_ER(self, energy):
#         energy_cap = np.where(energy <= 49., energy, 49.)
#         s1_var = tf.cast(self.s1_var_ER(energy_cap), fd.float_type())
#         s2_var = tf.cast(self.s2_var_ER(energy_cap), fd.float_type())
#         s1s2_corr = tf.cast(np.nan_to_num(self.s1s2_corr_ER(energy_cap)), fd.float_type())
#         s1s2_cov = s1s2_corr * tf.sqrt(s1_var * s2_var)

#         return s1_var, s2_var, s1s2_cov

#     def pdf_for_mu_pre_populate(self, **params):
#         old_defaults = copy(self.defaults)
#         self.set_defaults(**params)

#         e1, e2 = np.meshgrid(self.energies_first, self.energies_second, indexing='ij')

#         s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', e1)
#         s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', e2)
#         s1_mean = s1_mean_first + s1_mean_second
#         s2_mean = s2_mean_first + s2_mean_second

#         s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', e1)
#         s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
#         s1_var = s1_var_first + s1_var_second
#         s2_var = s2_var_first + s2_var_second

#         s1_mean = fd.tf_to_np(s1_mean)
#         s2_mean = fd.tf_to_np(s2_mean)
#         s1_var = fd.tf_to_np(s1_var)
#         s2_var = fd.tf_to_np(s2_var)

#         s1_std = np.sqrt(s1_var)
#         s2_std = np.sqrt(s2_var)

#         s1s2_corr_second = self.gimme_numpy('signal_corr', e1)
#         s1s2_cov_second = s1s2_corr_second  * np.sqrt(s1_var_second * s2_var_second)

#         s1s2_cov = s1s2_cov_first + s1s2_cov_second
#         anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)
#         anti_corr = fd.tf_to_np(anti_corr)

#         spectrum = fd.tf_to_np(self.rates_vs_energy)

#         self.defaults = old_defaults

#         return s1_mean, s2_mean, s1_var, s2_var, s1_std, s2_std, anti_corr, spectrum
