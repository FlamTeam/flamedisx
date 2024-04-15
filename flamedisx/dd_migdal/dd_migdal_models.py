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

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def mu_before_efficiencies(self, **params):
        return 1.
    
    '''
    #### Old Yield Model
    @staticmethod
    def signal_means(energy, a=13.1895962, b=1.06532331,
                     c_s2_0=3.70318382, c_s2_1=-3.49159718, c_s2_2=0.07861683,
                     g1=0.1131, g2=47.35,
                     s1_mean_multiplier=1., s2_mean_multiplier=1.):
        P = c_s2_0 + c_s2_1 * (fd.tf_log10(energy) - 1.6) + c_s2_2 * pow((fd.tf_log10(energy) - 1.6), 2)
        s2_mean = s2_mean_multiplier * P * energy * g2

        s1_mean = s1_mean_multiplier * (a * energy**b - s2_mean / g2) * g1
        s1_mean = tf.where(s1_mean < 0.01, 0.01 * tf.ones_like(s1_mean, dtype=fd.float_type()), s1_mean)

        return s1_mean, s2_mean

    @staticmethod
    def signal_vars(*args, d_s1=1.20307136, d_s2=38.27449296):
        s1_mean = args[0]
        s2_mean = args[1]

        s1_var = d_s1 * s1_mean

        s2_var = d_s2 * s2_mean

        return s1_var, s2_var

    @staticmethod
    def signal_corr(energies, anti_corr=-0.20949764):
        return anti_corr * tf.ones_like(energies)
    
    @staticmethod # 240213 - AV added
    def signal_skews(energies, s1_skew=0,s2_skew=0):
        
        s1_skew *= tf.ones_like(energies)
        s2_skew *= tf.ones_like(energies)
        
        return s1_skew, s2_skew
    '''
    
    ### New Yield Model # 240305 - AV added # 240326 - JB replace gamma delta to thomas-imel
    ### params --> {alpha, beta, Thomas-Imel, epsilon, Fi, Fex, NBamp, RawSkew}
    @staticmethod
    def yield_params(energies, yalpha=11.0, ybeta=1.1, ythomas=0.0467, yepsilon=12.6):
        
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
        
        xcoords_skew2D = tf.stack((Fi, Fex, NBamp, NBloc, RawSkew, energies), axis=-1)
        skew2D_model_param = interp_nd(x=xcoords_skew2D) #shape (energies, 7), [Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, correlation]
        
        Nph_fano = skew2D_model_param[:,2]**2 / skew2D_model_param[:,0] #shape (energies)
        Ne_fano  = skew2D_model_param[:,3]**2 / skew2D_model_param[:,1] #shape (energies)
        Nph_skew = skew2D_model_param[:,4]     #shape (energies)
        Ne_skew  = skew2D_model_param[:,5]     #shape (energies)
        initial_corr = skew2D_model_param[:,6] #shape (energies)
        
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
        
        s1s2_acc = tf.where((s2 > 200*s1**(0.73)), ### turn off for testing  TURN BACK ON!!!!!!!!!!
                            tf.ones_like(s2, dtype=fd.float_type()),
                            tf.zeros_like(s2, dtype=fd.float_type()))
        nr_endpoint = tf.where((s1 > 140) & (s2 > 8e3) & (s2 < 11.5e3),
                            tf.zeros_like(s2, dtype=fd.float_type()),
                            tf.ones_like(s2, dtype=fd.float_type()))

        return (s1_acc * s2_acc * s1s2_acc * nr_endpoint)
        # return (s1_acc * s2_acc) # for testing

    final_dimensions = ('s1',)
    
    @staticmethod
    def pdf_for_nphne(x, y, x_mean, y_mean, x_std, y_std, x_skew, y_skew, anti_corr, spectrum):      
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
        
        NphNe_pdf *= spectrum #shape (Nph,Ne,energies)
        NphNe_pdf = tf.reduce_sum(NphNe_pdf, axis=tf.range(2,tf.rank(NphNe_pdf))) #shape (Nph,Ne)
                
        return NphNe_pdf
        
    @staticmethod
    def pdf_for_s1s2_from_nphne(s1,s2,Nph,Ne,NphNe_pdf):
        S1_pos = tf.repeat(s1[:,o],len(Nph),axis=1) #shape (s1,Nph)
        S2_pos = tf.repeat(s2[:,o],len(Ne),axis=1) #shape (s2,Ne)
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1                          # shape (Nph)
        S1_fano = 1.12145985 * Nph**(-0.00629895) # shape (Nph)
        S1_std = tf.sqrt(S1_mean*S1_fano)         # shape (Nph)
        S1_skew = 4.61849047 * Nph**(-0.23931848) # shape (Nph)

        S2_mean = Ne*g2                             # shape (Ne)
        S2_fano = 21.3                              # shape (Ne)
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

    def pdf_for_nphne_pre_populate(self, NphNe, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)        
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Energy binning
        energies = fd.np_to_tf(self.energies_first)
              
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energies),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energies),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energies, 
                                        ptensor=ptensor) #shape (energies)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                                         bonus_arg=energies, 
                                                                         ptensor=ptensor) #shape (energies)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        spectrum = fd.np_to_tf(self.rates_vs_energy_first)
        # tf.print('estimate mu: sum rates_vs_energy_first',tf.reduce_sum(self.rates_vs_energy_first))
        
        self.defaults = old_defaults
        
        return x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum
      
    def estimate_mu(self, **params):
        
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(0,2500,250), fd.float_type()) # shape (Nph+1) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(0,800,200), fd.float_type()) # shape (Ne+1)
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

        x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum = self.pdf_for_nphne_pre_populate(NphNe,**params)
        
        NphNe_pdf = self.pdf_for_nphne(x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum)

        NphNe_probs = NphNe_pdf*Nph_diffs[0]*Ne_diffs[0] #shape (Nph,Ne), Nph and Ne grids must be linspace!
        # NphNe_probs = NphNe_probs/tf.reduce_sum(NphNe_probs) # 240408 AV added to normalize Nph,Ne pdf to 1
        # tf.print('NphNe_probs sum:', tf.reduce_sum(NphNe_probs))
                
        S1S2_pdf = self.pdf_for_s1s2_from_nphne(s1,s2,Nph,Ne,NphNe_probs)
        
        # account for S1,S2 space fiducial ROI acceptance
        acceptance = self.gimme('s1s2_acceptance',
                                bonus_arg=(S1_mesh, S2_mesh),
                                special_call=True)
        S1S2_pdf *= acceptance
        
        '''
        #*******test plot*******#
        plt.pcolormesh(Nph,Ne,tf.transpose(NphNe_pdf).numpy(),cmap='jet')
        plt.xlabel('Nph')
        plt.ylabel('Ne')
        plt.xlim(Nph_edges[0],Ne_edges[-1])
        plt.ylim(Nph_edges[0],Ne_edges[-1])
        plt.colorbar()
        plt.show()
        
        plt.pcolormesh(Nph,Ne,tf.transpose(NphNe_probs).numpy(),cmap='jet')
        plt.xlabel('Nph')
        plt.ylabel('Ne')
        plt.xlim(Nph_edges[0],Ne_edges[-1])
        plt.ylim(Nph_edges[0],Ne_edges[-1])
        plt.colorbar()
        plt.show()
        
        plt.pcolormesh(s1,s2,tf.transpose(S1S2_pdf*tf.reshape(s1_diffs,[-1,1])*s2_diffs).numpy(),cmap='jet')
        plt.xlabel('S1 [phd]')
        plt.ylabel('S2 [phd]')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(s1_edges[0],s1_edges[-1])
        plt.ylim(s2_edges[0],s2_edges[-1])
        plt.colorbar()
        plt.show()
        '''
        
        
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
    
    def pdf_for_nphne_pre_populate(self, NphNe, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)        
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_second)  # shape: {E2_bins}
              
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme('yield_params',
                                            bonus_arg=energy_first, 
                                            ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_first, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_1_var = Nph_1_fano * Nph_1_mean #shape (E1_bins)
        Ne_1_var = Ne_1_fano * Ne_1_mean #shape (E1_bins)
        
        ###########  # Second vertex
        # Load params
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params',
                                            bonus_arg=energy_second, 
                                            ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_second, 
                                                                                   ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
                
        Nph_2_var = Nph_2_fano * Nph_2_mean #shape (E2_bins)
        Ne_2_var = Ne_2_fano * Ne_2_mean #shape (E2_bins)
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        Nph_mean = tf.reshape(Nph_1_mean,[-1,1]) + tf.reshape(Nph_2_mean,[1,-1]) # shape: {E1_bins,E2_bins}
        Nph_std = tf.sqrt(tf.reshape(Nph_1_var,[-1,1]) + tf.reshape(Nph_2_var,[1,-1])) # shape: {E1_bins,E2_bins}
        Ne_mean = tf.reshape(Ne_1_mean,[-1,1]) + tf.reshape(Ne_2_mean,[1,-1]) # shape: {E1_bins,E2_bins}
        Ne_std = tf.sqrt(tf.reshape(Ne_1_var,[-1,1]) + tf.reshape(Ne_2_var,[1,-1])) # shape: {E1_bins,E2_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (tf.reshape(Nph_1_skew*Nph_1_mean,[-1,1]) 
                  + tf.reshape(Nph_2_skew*Nph_2_mean,[1,-1]))/(Nph_mean) # shape: {E1_bins,E2_bins}
        Ne_skew = (tf.reshape(Ne_1_skew*Ne_1_mean,[-1,1]) 
                 + tf.reshape(Ne_2_skew*Ne_2_mean,[1,-1]))/(Ne_mean) # shape: {E1_bins,E2_bins}
        totE = tf.reshape(energy_first,[-1,1]) + tf.reshape(energy_second,[1,-1]) # shape: {E1_bins,E2_bins}
        initial_corr = (tf.reshape(initial_1_corr*energy_first,[-1,1]) 
                      + tf.reshape(initial_2_corr*energy_second,[1,-1]))/(totE) # shape: {E1_bins,E2_bins}
        
        spectrum = fd.np_to_tf(self.rates_vs_energy)
        
        # tf.print('estimate mu: sum rates_vs_energy',tf.reduce_sum(self.rates_vs_energy))
        
        
        self.defaults = old_defaults
        
        return x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum
    

@export
class NRNRNRSource(NRNRSource):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstMSU3,
        fd_dd_migdal.EnergySpectrumOthersMSU3,
        fd_dd_migdal.MakeS1S2MSU3)

    no_step_dimensions = ('energy_others')

    def pdf_for_nphne_pre_populate(self, NphNe, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)        
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_others)  # shape: {E2_bins}
        energy_third = fd.np_to_tf(self.energies_others)  # shape: {E3_bins}
              
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        x = tf.repeat(x[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        y = tf.repeat(y[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}
        
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme('yield_params',
                                            bonus_arg=energy_first, 
                                            ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_first, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_1_var = Nph_1_fano * Nph_1_mean #shape (E1_bins)
        Ne_1_var = Ne_1_fano * Ne_1_mean #shape (E1_bins)
        
        ###########  # Second vertex
        # Load params
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params',
                                            bonus_arg=energy_second, 
                                            ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_second, 
                                                                                   ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
                
        Nph_2_var = Nph_2_fano * Nph_2_mean #shape (E2_bins)
        Ne_2_var = Ne_2_fano * Ne_2_mean #shape (E2_bins)
        
        ###########  # Third vertex
        # Load params
        Nph_3_mean, Ne_3_mean = self.gimme('yield_params',
                                            bonus_arg=energy_third, 
                                            ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
        
        Nph_3_fano, Ne_3_fano, Nph_3_skew, Ne_3_skew, initial_3_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_third, 
                                                                                   ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
                
        Nph_3_var = Nph_3_fano * Nph_3_mean #shape (E3_bins)
        Ne_3_var = Ne_3_fano * Ne_3_mean #shape (E3_bins)
        
        # compute Skew2D for triple scatters
        # sum the mean and variances of both scatters
        Nph_mean = tf.reshape(Nph_1_mean,[-1,1,1]) + tf.reshape(Nph_2_mean,[1,-1,1]) + tf.reshape(Nph_3_mean,[1,1,-1]) # shape: {E1_bins,E2_bins,E3_bins}
        Nph_std = tf.sqrt(tf.reshape(Nph_1_var,[-1,1,1]) + tf.reshape(Nph_2_var,[1,-1,1]) + tf.reshape(Nph_3_var,[1,1,-1])) # shape: {E1_bins,E2_bins,E3_bins}
        Ne_mean = tf.reshape(Ne_1_mean,[-1,1,1]) + tf.reshape(Ne_2_mean,[1,-1,1]) + tf.reshape(Ne_3_mean,[1,1,-1]) # shape: {E1_bins,E2_bins,E3_bins}
        Ne_std = tf.sqrt(tf.reshape(Ne_1_var,[-1,1,1]) + tf.reshape(Ne_2_var,[1,-1,1]) + tf.reshape(Ne_3_var,[1,1,-1])) # shape: {E1_bins,E2_bins,E3_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (tf.reshape(Nph_1_skew*Nph_1_mean,[-1,1,1]) 
                  + tf.reshape(Nph_2_skew*Nph_2_mean,[1,-1,1]) 
                  + tf.reshape(Nph_3_skew*Nph_3_mean,[1,1,-1]))/(Nph_mean) # shape: {E1_bins,E2_bins,E3_bins}
        Ne_skew = (tf.reshape(Ne_1_skew*Ne_1_mean,[-1,1,1]) 
                 + tf.reshape(Ne_2_skew*Ne_2_mean,[1,-1,1]) 
                 + tf.reshape(Ne_3_skew*Ne_3_mean,[1,1,-1]))/(Ne_mean) # shape: {E1_bins,E2_bins,E3_bins}
        totE = tf.reshape(energy_first,[-1,1,1]) + tf.reshape(energy_second,[1,-1,1]) + tf.reshape(energy_third,[1,1,-1]) # shape: {E1_bins,E2_bins,E3_bins}
        initial_corr = (tf.reshape(initial_1_corr*energy_first,[-1,1,1]) 
                      + tf.reshape(initial_2_corr*energy_second,[1,-1,1]) 
                      + tf.reshape(initial_3_corr*energy_third,[1,1,-1]))/(totE) # shape: {E1_bins,E2_bins,E3_bins}
        
        spectrum = fd.np_to_tf(self.rates_vs_energy)

        self.defaults = old_defaults
        
        return x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum


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

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def __init__(self, *args, **kwargs):
        energies_first = self.model_blocks[0].energies_first
        energies_first = tf.where(energies_first > 49., 49. * tf.ones_like(energies_first), energies_first)

        
        # ####-------test--------####
        # print('np.shape(energies_first)',np.shape(energies_first))
        
        # if hasattr(self.model_blocks[1], 'energies_second'):
        #     energies_first = tf.repeat(energies_first[:, o], tf.shape(self.model_blocks[1].energies_second), axis=1)
            
        # ####-------test--------####
        # print('np.shape(energies_first)',np.shape(energies_first))

        # old parameters - 240328 JB
        # self.s1_mean_ER_tf, self.s2_mean_ER_tf = self.signal_means_ER(energies_first)
        # self.s1_var_ER_tf, self.s2_var_ER_tf, self.s1s2_cov_ER_tf = self.signal_vars_ER(energies_first)
        # new parameters - 240328 JB
        self.Nph_mean_ER_tf, self.Ne_mean_ER_tf, Nph_fano, Ne_fano, self.Nph_skew_ER_tf, self.Ne_skew_ER_tf, self.initial_corr_ER_tf = self.quanta_params_ER(energies_first) 
        self.Nph_std_ER_tf = tf.sqrt(Nph_fano * self.Nph_mean_ER_tf)
        self.Ne_std_ER_tf = tf.sqrt(Ne_fano * self.Ne_mean_ER_tf)
        
        # ####-------test--------####
        # print('np.shape(self.initial_corr_ER_tf)',np.shape(self.initial_corr_ER_tf))        

        super().__init__(*args, **kwargs)

    '''
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
    '''
    
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

    def pdf_for_nphne_pre_populate(self, NphNe, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)        
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first) # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_second)  # shape: {E2_bins}
              
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean, Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme('quanta_params_ER',
                                                                                   bonus_arg=energy_first, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_1_var = Nph_1_fano * Nph_1_mean #shape (E1_bins)
        Ne_1_var = Ne_1_fano * Ne_1_mean #shape (E1_bins)
        
        ###########  # Second vertex
        # Load params
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params',
                                            bonus_arg=energy_second, 
                                            ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_second, 
                                                                                   ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
                
        Nph_2_var = Nph_2_fano * Nph_2_mean #shape (E2_bins)
        Ne_2_var = Ne_2_fano * Ne_2_mean #shape (E2_bins)
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        Nph_mean = tf.reshape(Nph_1_mean,[-1,1]) + tf.reshape(Nph_2_mean,[1,-1]) # shape: {E1_bins,E2_bins}
        Nph_std = tf.sqrt(tf.reshape(Nph_1_var,[-1,1]) + tf.reshape(Nph_2_var,[1,-1])) # shape: {E1_bins,E2_bins}
        Ne_mean = tf.reshape(Ne_1_mean,[-1,1]) + tf.reshape(Ne_2_mean,[1,-1]) # shape: {E1_bins,E2_bins}
        Ne_std = tf.sqrt(tf.reshape(Ne_1_var,[-1,1]) + tf.reshape(Ne_2_var,[1,-1])) # shape: {E1_bins,E2_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (tf.reshape(Nph_1_skew*Nph_1_mean,[-1,1]) 
                  + tf.reshape(Nph_2_skew*Nph_2_mean,[1,-1]))/(Nph_mean) # shape: {E1_bins,E2_bins}
        Ne_skew = (tf.reshape(Ne_1_skew*Ne_1_mean,[-1,1]) 
                 + tf.reshape(Ne_2_skew*Ne_2_mean,[1,-1]))/(Ne_mean) # shape: {E1_bins,E2_bins}
        totE = tf.reshape(energy_first,[-1,1]) + tf.reshape(energy_second,[1,-1]) # shape: {E1_bins,E2_bins}
        initial_corr = (tf.reshape(initial_1_corr*energy_first,[-1,1]) 
                      + tf.reshape(initial_2_corr*energy_second,[1,-1]))/(totE) # shape: {E1_bins,E2_bins}
        
        spectrum = fd.np_to_tf(self.rates_vs_energy)
        
        self.defaults = old_defaults
        
        return x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum      


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

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()

    def pdf_for_nphne_pre_populate(self, NphNe, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)        
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Energy binning
        energy_first = fd.np_to_tf(self.energies_first)    # shape: {E1_bins}
        energy_second = fd.np_to_tf(self.energies_others)  # shape: {E2_bins}
        energy_third = fd.np_to_tf(self.energies_others)  # shape: {E3_bins}
              
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        x = tf.repeat(x[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        y = tf.repeat(y[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}
        
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean, Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme('quanta_params_ER',
                                                                                   bonus_arg=energy_first, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_1_var = Nph_1_fano * Nph_1_mean #shape (E1_bins)
        Ne_1_var = Ne_1_fano * Ne_1_mean #shape (E1_bins)
        
        ###########  # Second vertex
        # Load params
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params',
                                            bonus_arg=energy_second, 
                                            ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_second, 
                                                                                   ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
                
        Nph_2_var = Nph_2_fano * Nph_2_mean #shape (E2_bins)
        Ne_2_var = Ne_2_fano * Ne_2_mean #shape (E2_bins)
        
        ###########  # Third vertex
        # Load params
        Nph_3_mean, Ne_3_mean = self.gimme('yield_params',
                                            bonus_arg=energy_third, 
                                            ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
        
        Nph_3_fano, Ne_3_fano, Nph_3_skew, Ne_3_skew, initial_3_corr = self.gimme('quanta_params',
                                                                                   bonus_arg=energy_third, 
                                                                                   ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
                
        Nph_3_var = Nph_3_fano * Nph_3_mean #shape (E3_bins)
        Ne_3_var = Ne_3_fano * Ne_3_mean #shape (E3_bins)
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        Nph_mean = tf.reshape(Nph_1_mean,[-1,1,1]) + tf.reshape(Nph_2_mean,[1,-1,1]) + tf.reshape(Nph_3_mean,[1,1,-1]) # shape: {E1_bins,E2_bins,E3_bins}
        Nph_std = tf.sqrt(tf.reshape(Nph_1_var,[-1,1,1]) + tf.reshape(Nph_2_var,[1,-1,1]) + tf.reshape(Nph_3_var,[1,1,-1])) # shape: {E1_bins,E2_bins,E3_bins}
        Ne_mean = tf.reshape(Ne_1_mean,[-1,1,1]) + tf.reshape(Ne_2_mean,[1,-1,1]) + tf.reshape(Ne_3_mean,[1,1,-1]) # shape: {E1_bins,E2_bins,E3_bins}
        Ne_std = tf.sqrt(tf.reshape(Ne_1_var,[-1,1,1]) + tf.reshape(Ne_2_var,[1,-1,1]) + tf.reshape(Ne_3_var,[1,1,-1])) # shape: {E1_bins,E2_bins,E3_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (tf.reshape(Nph_1_skew*Nph_1_mean,[-1,1,1]) 
                  + tf.reshape(Nph_2_skew*Nph_2_mean,[1,-1,1]) 
                  + tf.reshape(Nph_3_skew*Nph_3_mean,[1,1,-1]))/(Nph_mean) # shape: {E1_bins,E2_bins,E3_bins}
        Ne_skew = (tf.reshape(Ne_1_skew*Ne_1_mean,[-1,1,1]) 
                 + tf.reshape(Ne_2_skew*Ne_2_mean,[1,-1,1]) 
                 + tf.reshape(Ne_3_skew*Ne_3_mean,[1,1,-1]))/(Ne_mean) # shape: {E1_bins,E2_bins,E3_bins}
        totE = tf.reshape(energy_first,[-1,1,1]) + tf.reshape(energy_second,[1,-1,1]) + tf.reshape(energy_third,[1,1,-1]) # shape: {E1_bins,E2_bins,E3_bins}
        initial_corr = (tf.reshape(initial_1_corr*energy_first,[-1,1,1]) 
                      + tf.reshape(initial_2_corr*energy_second,[1,-1,1]) 
                      + tf.reshape(initial_3_corr*energy_third,[1,1,-1]))/(totE) # shape: {E1_bins,E2_bins,E3_bins}
        
        spectrum = fd.np_to_tf(self.rates_vs_energy)

        self.defaults = old_defaults
        
        return x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum


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

    S2Width_diff_rate = mh_S2Width
    S2Width_events_per_bin = mh_S2Width * mh_S2Width.bin_volumes()


@export
class ERSource(Migdal2Source):
    model_blocks = (
        fd_dd_migdal.EnergySpectrumFirstER,
        fd_dd_migdal.MakeS1S2ER)
    
    def pdf_for_nphne_pre_populate(self, NphNe, **params):
        old_defaults = copy(self.defaults)
        self.set_defaults(**params)        
        ptensor = self.ptensor_from_kwargs(**params)
        
        # Energy binning
        energies = fd.np_to_tf(self.energies_first)
              
        ### Quanta Production
        x = NphNe[:,:,0] #shape (Nph,Ne)
        x = tf.repeat(x[:,:,o],len(energies),axis=2) #shape (Nph,Ne,energies), grid of Nph center points

        y = NphNe[:,:,1] #shape (Nph,Ne)
        y = tf.repeat(y[:,:,o],len(energies),axis=2)  #shape (Nph,Ne,energies), grid of Ne center points
        
        # Load params
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params_ER',
                                                                                   bonus_arg=energies, 
                                                                                   ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
                
        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (energies)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (energies)
        
        spectrum = fd.np_to_tf(self.rates_vs_energy_first)
        
        self.defaults = old_defaults
        
        return x, y, Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr, spectrum