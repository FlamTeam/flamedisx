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

###################################################################################################
#DRM Parameters #  240419 - AV added to make DRM changes easier
S2_fano_all = 10.0 #1.77 # 21.3 

### S1,S2 Yield
g1 = 0.1131
g2 = 47.35

#Data Acceptance cut params # 240419 - AV added to make it easier to set cuts below NR band
BCUT=200 #160
MCUT=0.73 #0.77

###################################################################################################

### interpolation grids for NR
filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam_20240309.npz'
with np.load(filename) as f:
    fit_values_allkeVnr_allparam = f['fit_values_allkeVnr_allparam']
    
### interpolation grids for ER
filenameER = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVee_weightedER_20240319.npz'
with np.load(filenameER) as f:
    fit_values_allkeVee_fanoversion = f['fit_values_allkeVee']
    
def interp_nd(x):
    """
    interpolation function
    x: energy
    x-grid: [fano_ion, fano_exciton, amplitude of non-binomial fluctuation, location of NB, raw skewness]
    y (output): [Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, correlation]
    """
        
    # Define x_grid_points for interpolation
    part1 = tf.cast(tf.experimental.numpy.geomspace(1,4,7), fd.float_type())
    part2 = tf.cast(tf.experimental.numpy.geomspace(4,80,23), fd.float_type())[1:]
    keVnr_choices = tf.concat([part1,part2],axis=0)

    Fi_grid = tf.cast([0.25,0.3,0.4,0.55,0.75,1.], fd.float_type())             # Fano ion
    Fex_grid = tf.cast([0.3,0.4,0.55,0.75,1.,1.25,1.5,1.75], fd.float_type())   # Fano exciton
    NBamp_grid = tf.cast([0.,0.02,0.04], fd.float_type())                       # amplitude for non-binomial NR recombination fluctuations
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
    return interp 
  
def interp_nd_ER(x):
    """
    interpolation function
    x: energy. shape must be (energy,1)
    x-grid: energy
    y (output): [Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, correlation]
    """
        
    # Define x_grid_points for interpolation
    x_grid_points = (tf.cast([[  0.05      ,  0.1       ,  0.15      ,  0.2       ,  0.25      ,  0.3       ,
                                0.35      ,  0.4       ,  0.45      ,  0.5       ,  0.56060147,  0.62854801,
                                0.70472987,  0.7901452 ,  0.88591312,  0.99328839,  1.11367786,  1.24865888,
                                1.4       ,  1.65412633,  1.95438137,  2.30913835,  2.7282904 ,  3.22352642,
                                3.8086571 ,  4.5       ,  4.96840281,  5.48556144,  6.05655087,  6.6869743 ,
                                7.3830182 ,  8.15151298,  9.        , 10.19124635, 11.5401669 , 13.06763152,
                               14.79727245, 16.75584986, 18.97366596, 21.48503377, 24.32880798, 27.54898616,
                               31.19538939, 35.32443313, 40.        ]],
                            fd.float_type()),) # energy grids

    #define reference function values --> f(x_grid_points)
    y_ref = tf.cast(fit_values_allkeVee_fanoversion, fd.float_type()), # want shape = (1, ...)

    # nd interpolation of points "x" to determine Edependance of various values
    interp = tfp.math.batch_interp_rectilinear_nd_grid(x=x,
                                                       x_grid_points=x_grid_points,
                                                       y_ref=y_ref,
                                                       axis=1,
                                                       )[0] # o/p shape (1, len(E), 7)
                                                            # want shape   (len(E), 7)

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

    S2Width_diff_rate = mh_S2Width * (mh_S2Width.bin_volumes()) # 240416 AV Added so sum == 1
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 1.
    
    ### New Yield Model # 240305 - AV added # 240326 - JB replace gamma delta to thomas-imel
    ### params --> {alpha, beta, Thomas-Imel, epsilon, Fi, Fex, NBamp, RawSkew}
    @staticmethod
    def yield_params(energies, yalpha=11.0, ybeta=1.1, ythomas=0.0467, yepsilon=12.6):
        
        ############# NEST paper (v1) https://arxiv.org/pdf/2211.10726.pdf
        pp = 0.5
        zeta = 0.3
        eta = 2
        # gamma = 0.0480 ==> ythomas
        delta = -0.0533
        rho = 2.9
        rho_0 = 2.9
        nu = 0.3
        theta = 0.3
        ll = 2
        ##############
        
        
        # ############# SR3 DD Tuning (https://docs.google.com/presentation/d/14IoMsX77OtQJOeZb3nEV6cmtomqN0Y5DRY0rqt0XQOU/edit#slide=id.g29ac68d8356_0_3)
        # pp = 0.511
        # zeta = 0.33
        # eta = 2.18
        # # gamma = 0.0480 ==> ythomas
        # delta = -0.0533
        # rho = 2.9
        # rho_0 = 2.9
        # nu = 0.3
        # theta = 0.313
        # ll = 2.62
        # ##############
        
        TI = ythomas*193**delta * (rho/rho_0)**nu
        # TI = ythomas
        
        Qy = 1 / TI /  (energies + yepsilon)**pp * ( 1 - 1/(1 + (energies/zeta)**eta) )
        Ne_mean = Qy * energies
        
        Ly = yalpha * energies**ybeta - Ne_mean
        Nph_mean = Ly * (1 - 1/(1 + (energies/theta)**ll) )
        
        return Nph_mean, Ne_mean
    
    @staticmethod
    def quanta_params(energies, Fi=0.4,Fex=0.4,NBamp=0.0,NBloc=0.0,RawSkew=2.25):
        
        Fi *= tf.ones_like(energies)
        Fex *= tf.ones_like(energies)
        NBamp *= tf.ones_like(energies)
        NBloc *= tf.ones_like(energies)
        RawSkew *= tf.ones_like(energies)
        
        xcoords_skew2D = tf.stack((Fi, Fex, NBamp, NBloc, RawSkew, energies), axis=-1)
        skew2D_model_param = interp_nd(x=xcoords_skew2D) #shape (energies, 7), [Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, correlation]
        
        Nph_fano = skew2D_model_param[:,2]**2 / skew2D_model_param[:,0] #shape (energies)
        Ne_fano  = skew2D_model_param[:,3]**2 / skew2D_model_param[:,1] #shape (energies) # Ne fano = 0.001 FOR TESTING
        Nph_skew = skew2D_model_param[:,4]     #shape (energies)
        Ne_skew  = skew2D_model_param[:,5]     #shape (energies)
        initial_corr = skew2D_model_param[:,6] #shape (energies)
        
        tf.print('Nph_fano',Nph_fano)
        tf.print('Ne_fano',Ne_fano)
        
        return Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr
    
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
        
        s1s2_acc = tf.where((s2 > BCUT*s1**(MCUT)),
                            tf.ones_like(s2, dtype=fd.float_type()),
                            tf.zeros_like(s2, dtype=fd.float_type()))
        nr_endpoint = tf.where((s1 > 140) & (s2 > 8e3) & (s2 < 11.5e3),
                            tf.zeros_like(s2, dtype=fd.float_type()),
                            tf.ones_like(s2, dtype=fd.float_type()))

        return (s1_acc * s2_acc * s1s2_acc * nr_endpoint)
        # return (s1_acc * s2_acc) # FOR TESTING ONLY

    final_dimensions = ('s1',)
    
    @staticmethod
    def pdf_for_nphne(x, y, x_mean, y_mean, x_std, y_std, x_skew, y_skew, anti_corr):      
        # define scale and loc params for shifting skew2d from (0,0)
        # parameters are defined in a form of matrix multiplication
        # bCa = [[1,corr],[corr,1]] @[[skew1],[skew2]] 
        # del2 = [skew1, skew2] @ [[1,corr],[corr,1]] @[[skew1],[skew2]] 
        # aCa = 1. + del2 
        # scale = final_std / tf.sqrt(1. - 2 * delta**2 / pi) #shape (energies,2,1)
        # loc = final_mean - scale * delta * tf.sqrt(2/pi) #shape (energies,2,1)
        
        # the shape of below variables is (energies)=(E1,E2,...), depending on number of vertices
        bCa1 = x_skew + anti_corr * y_skew
        bCa2 = anti_corr * x_skew + y_skew
        aCa = 1. + x_skew * bCa1 + y_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = x_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = y_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = x_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = y_mean - scale2 * delta2 * tf.sqrt(2/pi)
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * x_std * y_std * tf.sqrt(1. - anti_corr * anti_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * anti_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (x_skew * (x-loc1)/scale1) + (y_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        NphNe_pdf = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (x_std*y_std) #shape (Nph,Ne,energies)
                        
        return NphNe_pdf
        
    @staticmethod
    def pdf_for_s1s2_from_nphne(s1,s2,Nph,Ne,NphNe_pdf):
        S1_pos = tf.repeat(s1[:,o],len(Nph),axis=1) #shape (s1,Nph)
        S2_pos = tf.repeat(s2[:,o],len(Ne),axis=1) #shape (s2,Ne)

        S1_mean = Nph*g1                          # shape (Nph)
        S1_fano = 1.12145985 * Nph**(-0.00629895) # shape (Nph)
        S1_std = tf.sqrt(S1_mean*S1_fano)         # shape (Nph)
        S1_skew = 4.61849047 * Nph**(-0.23931848) # shape (Nph)

        S2_mean = Ne*g2                             # shape (Ne)
        S2_fano = S2_fano_all #1.77 # 21.3          # shape (Ne) #0.001
        S2_std = tf.sqrt(S2_mean*S2_fano)           # shape (Ne)
        S2_skew = -2.37542105 *  Ne** (-0.26152676) # shape (Ne)

        S1_pdf = skewnorm_1d(x=S1_pos,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew) #shape (s1,Nph)
        S1_pdf = tf.repeat(S1_pdf[:,o,:],len(s2),1) #shape (s1,s2,Nph)
        S1_pdf = tf.repeat(S1_pdf[:,:,:,o],len(Ne),3) #shape (s1,s2,Nph,Ne)

        S2_pdf = skewnorm_1d(x=S2_pos,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew) #shape (s2,Ne)
        S2_pdf = tf.repeat(S2_pdf[o,:,:],len(s1),0) #shape (s1,s2,Ne)
        S2_pdf = tf.repeat(S2_pdf[:,:,o,:],len(Nph),2) #shape (s1,s2,Nph,Ne)

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf #shape (s1,s2,Nph,Ne)

        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=[2,3]) #shape (s1,s2)
        
        return S1S2_pdf

    def estimate_mu(self, **params):
        
        # Quanta Binning
        Nph_bw = 10.0
        Ne_bw = 4.0  
        Nph_edges = tf.cast(tf.range(0,2500,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,500,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)

        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)
        
        # S1S2 binning
        s1_edges = tf.cast(tf.linspace(self.defaults['s1_min'], self.defaults['s1_max'], 100), fd.float_type()) #shape (s1+1)
        s2_edges = tf.cast(tf.experimental.numpy.geomspace(self.defaults['s2_min'], self.defaults['s2_max'], 101), fd.float_type()) #shape (s2+1)
        s1 = 0.5 * (s1_edges[1:] + s1_edges[:-1]) #shape (s1)
        s2 = 0.5 * (s2_edges[1:] + s2_edges[:-1]) #shape (s2)
        s1_diffs = tf.experimental.numpy.diff(s1_edges) #shape (s1)
        s2_diffs = tf.experimental.numpy.diff(s2_edges) #shape (s2)

        S1_mesh, S2_mesh = tf.meshgrid(s1,s2,indexing='ij') #shape (s1,s2)
        
        # Energy binning
        energies = fd.np_to_tf(self.energies_first)
        spectrum = fd.np_to_tf(self.rates_vs_energy_first)
        
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energies),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energies),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        ptensor = self.ptensor_from_kwargs(**params)
        
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energies, 
                                        ptensor=ptensor) #shape (energies)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                                         bonus_arg=energies, 
                                                                         ptensor=ptensor) #shape (energies)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        # Generate NphNe pdfs
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        NphNe_pdf *= spectrum #shape (Nph,Ne,energies)
        NphNe_pdf = tf.reduce_sum(NphNe_pdf, axis=tf.range(2,tf.rank(NphNe_pdf))) #shape (Nph,Ne)
        
        NphNe_probs = NphNe_pdf*Nph_diffs[0]*Ne_diffs[0] #shape (Nph,Ne), Nph and Ne grids must be linspace!
        
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_probs = NphNe_probs[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
                
        S1S2_pdf = self.pdf_for_s1s2_from_nphne(s1,s2,Nph,Ne,NphNe_probs)
        
        # account for S1,S2 space fiducial ROI acceptance
        acceptance = self.gimme('s1s2_acceptance',
                                bonus_arg=(S1_mesh, S2_mesh),
                                special_call=True)
        S1S2_pdf *= acceptance
        
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

    S2Width_diff_rate = mh_S2Width * (mh_S2Width.bin_volumes()) # 240416 AV Added so sum == 1
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()
    

    def fftConvolve_nphnePDFs(self, NphNe, Nph_bw, Ne_bw, **params):
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_second)  # shape: {E2_bins}
        spectrum = fd.np_to_tf(self.rates_vs_energy)
        
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energy_first),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energy_first),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        # note, pdf is calculated once because energy bins for the first and the second vertices are the same
        ptensor = self.ptensor_from_kwargs(**params)
        
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energy_first, 
                                        ptensor=ptensor) #shape (energies)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                                         bonus_arg=energy_first, 
                                                                         ptensor=ptensor) #shape (energies)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        # Nph-Ne pdf
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        probs = NphNe_pdf*Nph_bw*Ne_bw #shape (Nph,Ne, energy), Nph and Ne grids must be linspace!
        probs *= 1/tf.reduce_sum(probs,axis=[0,1]) # normalize the probability for each recoil energy

        # FFT convolution 
        NphNe_all_prob_1_tp = tf.transpose(probs, perm=[2,0,1], conjugate=False) 
        NphNe_all_prob_1_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_prob_1_tp,tf.complex64))

        NphNe_all_prob_1_tp_fft2d_repeat = tf.repeat(NphNe_all_prob_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_prob_2_tp_fft2d_repeat = tf.repeat(NphNe_all_prob_1_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne} 
        NphNe_all_prob_12 = tf.math.real(tf.signal.ifft2d(NphNe_all_prob_1_tp_fft2d_repeat*NphNe_all_prob_2_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_prob = tf.einsum('ijkl,ij->kl',NphNe_all_prob_12,spectrum) 
        
        return NphNe_prob
      
    @staticmethod
    def NphNeBinning():
        Nph_bw = 14.0
        Ne_bw = 4.0
        Nph_edges = tf.cast(tf.range(0,3500,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,600,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        return Nph_bw, Ne_bw, Nph_edges, Ne_edges
        
    
    def estimate_mu(self, **params):
        # Quanta Binning
        Nph_bw, Ne_bw, Nph_edges, Ne_edges = self.NphNeBinning()

        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)
        
        # S1S2 binning
        s1_edges = tf.cast(tf.linspace(self.defaults['s1_min'], self.defaults['s1_max'], 100), fd.float_type()) #shape (s1+1)
        s2_edges = tf.cast(tf.experimental.numpy.geomspace(self.defaults['s2_min'], self.defaults['s2_max'], 101), fd.float_type()) #shape (s2+1)
        s1 = 0.5 * (s1_edges[1:] + s1_edges[:-1]) #shape (s1)
        s2 = 0.5 * (s2_edges[1:] + s2_edges[:-1]) #shape (s2)
        s1_diffs = tf.experimental.numpy.diff(s1_edges) #shape (s1)
        s2_diffs = tf.experimental.numpy.diff(s2_edges) #shape (s2)

        S1_mesh, S2_mesh = tf.meshgrid(s1,s2,indexing='ij') #shape (s1,s2)

        NphNe_probs = self.fftConvolve_nphnePDFs(NphNe, Nph_bw, Ne_bw, **params)
        
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_probs = NphNe_probs[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
                
        S1S2_pdf = self.pdf_for_s1s2_from_nphne(s1,s2,Nph,Ne,NphNe_probs)
        
        # account for S1,S2 space fiducial ROI acceptance
        acceptance = self.gimme('s1s2_acceptance',
                                bonus_arg=(S1_mesh, S2_mesh),
                                special_call=True)
        S1S2_pdf *= acceptance
        
        # rescale by the S1,S2 bin width
        mu_est = s1_diffs[o, :] @ S1S2_pdf
        mu_est = mu_est @ s2_diffs[:, o]
        mu_est = mu_est[0][0]
        
        return mu_est      

    

@export
class NRNRNRSource(NRNRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMSU3,
        fd_dd_migdal.EnergySpectrumOthersMSU3,
        fd_dd_migdal.MakeS1S2MSU3)

    no_step_dimensions = ('energy_others')

    def fftConvolve_nphnePDFs(self, NphNe, Nph_bw, Ne_bw, **params):
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_others)  # shape: {E2_bins}
        energy_third = fd.np_to_tf(self.energies_others)  # shape: {E3_bins}
        spectrum = fd.np_to_tf(self.rates_vs_energy)
        
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energy_first),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energy_first),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        # note, pdf is calculated once because energy bins for all three vertices are the same
        ptensor = self.ptensor_from_kwargs(**params)
        
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energy_first, 
                                        ptensor=ptensor) #shape (energies)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                                         bonus_arg=energy_first, 
                                                                         ptensor=ptensor) #shape (energies)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        # Nph-Ne pdf
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        probs = NphNe_pdf*Nph_bw*Ne_bw #shape (Nph,Ne, energy), Nph and Ne grids must be linspace!
        probs *= 1/tf.reduce_sum(probs,axis=[0,1]) # normalize the probability for each recoil energy

        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs, perm=[2,0,1], conjugate=False)  #shape (energy,Nph,Ne)
        NphNe_all_pdf_1_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))
        
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}        
        
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[o,:,:,:],tf.shape(energy_second)[0],axis=0) # final shape: {E2_bins, E3_bins, Nph, Ne}
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_3_tp_fft2d_repeat[o,:,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_123 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat*NphNe_all_pdf_3_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}
        
        NphNe_prob = tf.einsum('ijklm,ijk->lm',NphNe_all_pdf_123,spectrum) # shape (Nph, Ne)
        
        return NphNe_prob

    @staticmethod
    def NphNeBinning():
        Nph_bw = 25.0
        Ne_bw = 5.0
        Nph_edges = tf.cast(tf.range(0,5000,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,800,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        return Nph_bw, Ne_bw, Nph_edges, Ne_edges      
      

@export
class Migdal2Source(NRNRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdal,
        fd_dd_migdal.EnergySpectrumSecondMigdal2,
        fd_dd_migdal.MakeS1S2Migdal)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/SS_Mig_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width * (mh_S2Width.bin_volumes()) # 240416 AV Added so sum == 1
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def __init__(self, *args, **kwargs):
        energies_first = self.model_blocks[0].energies_first
        energies_first = tf.where(energies_first > 49., 49. * tf.ones_like(energies_first), energies_first)

        self.Nph_mean_ER_tf, self.Ne_mean_ER_tf, Nph_fano, Ne_fano, self.Nph_skew_ER_tf, self.Ne_skew_ER_tf, self.initial_corr_ER_tf = self.quanta_params_ER(energies_first) 
        self.Nph_std_ER_tf = tf.sqrt(Nph_fano * self.Nph_mean_ER_tf)
        self.Ne_std_ER_tf = tf.sqrt(Ne_fano * self.Ne_mean_ER_tf)
        
        super().__init__(*args, **kwargs)

    def quanta_params_ER(self, energy):
        """
        assume energy is rank 1
        """
        energy_cap = tf.where(energy <= 49., energy, 49. * tf.ones_like(energy))
        energy_cap = tf.reshape(energy_cap, (-1,1))
        skew2D_model_param = interp_nd_ER(x=energy_cap) #shape (energies, 7), [Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, correlation]
        
        Nph_mean = skew2D_model_param[:,0]
        Ne_mean = skew2D_model_param[:,1]
        Nph_fano = skew2D_model_param[:,2]     #shape (energies)
        Ne_fano  = skew2D_model_param[:,3]     #shape (energies)
        Nph_skew = skew2D_model_param[:,4]     #shape (energies)
        Ne_skew  = skew2D_model_param[:,5]     #shape (energies)
        initial_corr = skew2D_model_param[:,6] #shape (energies)
        
        return Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr

    def fftConvolve_nphnePDFs(self, NphNe, Nph_bw, Ne_bw, **params):
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_second)  # shape: {E2_bins}
        spectrum = fd.np_to_tf(self.rates_vs_energy)
        
        # Nph-Ne pdf
        ###########  # First vertex
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energy_first),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energy_first),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        ptensor = self.ptensor_from_kwargs(**params)
              
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params_ER',
                                                                                   bonus_arg=energy_first, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (E1_bins)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (E1_bins)
        
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        probs_1 = NphNe_pdf*Nph_bw*Ne_bw #shape (Nph,Ne, energy), Nph and Ne grids must be linspace!
        probs_1 *= 1/tf.reduce_sum(probs_1,axis=[0,1]) # normalize the probability for each recoil energy
        
        ###########  # Second vertex
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energy_second),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energy_second),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        # note, pdf is calculated once because energy bins for all three vertices are the same
        ptensor = self.ptensor_from_kwargs(**params)
        
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energy_second, 
                                        ptensor=ptensor) #shape (energies)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                                         bonus_arg=energy_second, 
                                                                         ptensor=ptensor) #shape (energies)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        probs_2 = NphNe_pdf*Nph_bw*Ne_bw #shape (Nph,Ne, energy), Nph and Ne grids must be linspace!
        probs_2 *= 1/tf.reduce_sum(probs_2,axis=[0,1]) # normalize the probability for each recoil energy        

        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs_1, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_2_tp = tf.transpose(probs_2, perm=[2,0,1], conjugate=False) 
        
        NphNe_all_pdf_1_tp_fft2d = tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))
        NphNe_all_pdf_2_tp_fft2d = tf.signal.fft2d(tf.cast(NphNe_all_pdf_2_tp,tf.complex64))
        
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne} 
        NphNe_all_pdf_12 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_prob = tf.einsum('ijkl,ij->kl',NphNe_all_pdf_12,spectrum)
        
        return NphNe_prob    

@export
class Migdal3Source(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdal,
        fd_dd_migdal.EnergySpectrumSecondMigdal3,
        fd_dd_migdal.MakeS1S2Migdal)


@export
class Migdal4Source(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdal,
        fd_dd_migdal.EnergySpectrumSecondMigdal4,
        fd_dd_migdal.MakeS1S2Migdal)


@export
class MigdalMSUSource(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMigdalMSU,
        fd_dd_migdal.EnergySpectrumOthersMigdalMSU,
        fd_dd_migdal.MakeS1S2MigdalMSU)

    no_step_dimensions = ('energy_others')

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width * (mh_S2Width.bin_volumes()) # 240416 AV Added so sum == 1
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def fftConvolve_nphnePDFs(self, NphNe, Nph_bw, Ne_bw, **params):
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_others)  # shape: {E2_bins}
        energy_third = fd.np_to_tf(self.energies_others)  # shape: {E3_bins}
        spectrum = fd.np_to_tf(self.rates_vs_energy)
        
        # Nph-Ne pdf
        ###########  # First vertex
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energy_first),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energy_first),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        ptensor = self.ptensor_from_kwargs(**params)
              
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params_ER',
                                                                                   bonus_arg=energy_first, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (E1_bins)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (E1_bins)
        
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        probs_1 = NphNe_pdf*Nph_bw*Ne_bw #shape (Nph,Ne, energy), Nph and Ne grids must be linspace!
        probs_1 *= 1/tf.reduce_sum(probs_1,axis=[0,1]) # normalize the probability for each recoil energy
        
        ###########  # second vertex
        # note, third pdf is not calculated because energy bins for the second and third vertices are the same
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energy_second),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energy_second),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        # note, pdf is calculated once because energy bins for all three vertices are the same
        ptensor = self.ptensor_from_kwargs(**params)
        
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energy_second, 
                                        ptensor=ptensor) #shape (energies)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                                         bonus_arg=energy_second, 
                                                                         ptensor=ptensor) #shape (energies)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        probs_2 = NphNe_pdf*Nph_bw*Ne_bw #shape (Nph,Ne, energy), Nph and Ne grids must be linspace!
        probs_2 *= 1/tf.reduce_sum(probs_2,axis=[0,1]) # normalize the probability for each recoil energy
        
        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs_1, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_2_tp = tf.transpose(probs_2, perm=[2,0,1], conjugate=False) 
        
        NphNe_all_pdf_1_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))
        NphNe_all_pdf_2_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_2_tp,tf.complex64))
        
        
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}        
        
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d[o,:,:,:],tf.shape(energy_second)[0],axis=0) # final shape: {E2_bins, E3_bins, Nph, Ne}
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_3_tp_fft2d_repeat[o,:,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_123 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat*NphNe_all_pdf_3_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}
        
        NphNe_prob = tf.einsum('ijklm,ijk->lm',NphNe_all_pdf_123,spectrum)
        
        return NphNe_prob    
      
    @staticmethod
    def NphNeBinning():
        Nph_bw = 30.0
        Ne_bw = 7.0
        Nph_edges = tf.cast(tf.range(0,5000,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,800,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        return Nph_bw, Ne_bw, Nph_edges, Ne_edges         

@export
class IECSSource(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstIE_CS,
        fd_dd_migdal.EnergySpectrumSecondIE_CS,
        fd_dd_migdal.MakeS1S2Migdal)

    S2Width_dist = np.load(os.path.join(
        os.path.dirname(__file__), './migdal_database/MSU_IECS_S2Width_template.npz'))

    hist_values_S2Width = S2Width_dist['hist_values']
    S2Width_edges = S2Width_dist['S2Width_edges']

    mh_S2Width = Hist1d(bins=len(S2Width_edges) - 1).from_histogram(hist_values_S2Width, bin_edges=S2Width_edges)
    mh_S2Width = mh_S2Width / mh_S2Width.n
    mh_S2Width = mh_S2Width / mh_S2Width.bin_volumes()

    S2Width_diff_rate = mh_S2Width * (mh_S2Width.bin_volumes()) # 240416 AV Added so sum == 1
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()


@export
class ERSource(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstER,
        fd_dd_migdal.MakeS1S2ER)

    def estimate_mu(self, **params):
        
        # Quanta Binning
        Nph_bw = 10.0
        Ne_bw = 4.0
        Nph_edges = tf.cast(tf.range(0,2500,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,700,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)

        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)
        
        # S1S2 binning
        s1_edges = tf.cast(tf.linspace(self.defaults['s1_min'], self.defaults['s1_max'], 100), fd.float_type()) #shape (s1+1)
        s2_edges = tf.cast(tf.experimental.numpy.geomspace(self.defaults['s2_min'], self.defaults['s2_max'], 101), fd.float_type()) #shape (s2+1)
        s1 = 0.5 * (s1_edges[1:] + s1_edges[:-1]) #shape (s1)
        s2 = 0.5 * (s2_edges[1:] + s2_edges[:-1]) #shape (s2)
        s1_diffs = tf.experimental.numpy.diff(s1_edges) #shape (s1)
        s2_diffs = tf.experimental.numpy.diff(s2_edges) #shape (s2)

        S1_mesh, S2_mesh = tf.meshgrid(s1,s2,indexing='ij') #shape (s1,s2)
        
        # Energy binning
        energies = fd.np_to_tf(self.energies_first)
        spectrum = fd.np_to_tf(self.rates_vs_energy_first)
        
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energies),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energies),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        ptensor = self.ptensor_from_kwargs(**params)
              
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params_ER',
                                                                                   bonus_arg=energies, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (E1_bins)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (E1_bins)
                
        # NphNe PDF
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr) #shape (Nph,Ne,energies)

        NphNe_pdf *= spectrum #shape (Nph,Ne,energies)
        NphNe_pdf = tf.reduce_sum(NphNe_pdf, axis=tf.range(2,tf.rank(NphNe_pdf))) #shape (Nph,Ne)
        
        NphNe_probs = NphNe_pdf*Nph_diffs[0]*Ne_diffs[0] #shape (Nph,Ne), Nph and Ne grids must be linspace!
                
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_probs = NphNe_probs[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
          
        S1S2_pdf = self.pdf_for_s1s2_from_nphne(s1,s2,Nph,Ne,NphNe_probs)
        
        # account for S1,S2 space fiducial ROI acceptance
        acceptance = self.gimme('s1s2_acceptance',
                                bonus_arg=(S1_mesh, S2_mesh),
                                special_call=True)
        S1S2_pdf *= acceptance
        
        # rescale by the S1,S2 bin width
        mu_est = s1_diffs[o, :] @ S1S2_pdf
        mu_est = mu_est @ s2_diffs[:, o]
        mu_est = mu_est[0][0]
        
        return mu_est    
    