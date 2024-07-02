import typing as ty

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

import math as m
pi = tf.constant(m.pi)

import flamedisx as fd
export, __all__ = fd.exporter()
o = tf.newaxis


import sys ########

### interpolation grids for NR
# filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam.npz'
filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam_20240309.npz'
with np.load(filename) as f:
    fit_values_allkeVnr_allparam = f['fit_values_allkeVnr_allparam']
    
### interpolation grids for ER
filenameER = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVee_weightedER_20240319.npz'
with np.load(filenameER) as f:
    fit_values_allkeVee_fanoversion = f['fit_values_allkeVee']
    
def interp_nd(x):
    ''' 
    inputs:
    x -- energy values to inerpolate along the rectilinear grid, shape: (1, Energy_bins, 6)
    
    outputs:
    interp -- f(x), shape: (Energy_bins, 7)
    '''
    
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
                                                       axis=1
                                                       )[0] # o/p shape  (1, Energy_bins, 7)
                                                            # want shape    (Energy_bins, 7)
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
class MakeS1S2MSU(fd.Block):
    """
    """
    model_attributes = ('check_acceptances',)

    dimensions = ('energy_first', 's1')
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_second',), 'rate_vs_energy'))

    # special_model_functions = ('signal_means', 'signal_vars', 'signal_corr', 'signal_skews', # 240213 - AV added skew
    #                            'yield_params','quanta_params') # 240305 - AV added new yield model
    special_model_functions = ('yield_params','quanta_params')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable
    
    # def pdf_for_nphne(self, a,b,c):
    #     print('this is test function pdf_for_nphne')
    #     return 0

    def _simulate(self, d):
      
        ####----test----####
        # self.pdf_for_nphne(1,2,3)
      
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme_numpy('yield_params', energies_first) # shape: {E1_bins} (matches bonus_arg)
        Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme_numpy('quanta_params', energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_1_std = tf.sqrt(Nph_1_fano * Nph_1_mean)
        Ne_1_std = tf.sqrt(Ne_1_fano * Ne_1_mean) 
        
        Nph_2_mean, Ne_2_mean = self.gimme_numpy('yield_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme_numpy('quanta_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_std = tf.sqrt(Nph_2_fano * Nph_2_mean)
        Ne_2_std = tf.sqrt(Ne_2_fano * Ne_2_mean) 
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        # 240331 JB: assume shape(energies_first) == shape(energies_second)
        Nph_mean = Nph_1_mean + Nph_2_mean # shape: {E_bins}
        Nph_std = tf.sqrt(Nph_1_std**2 + Nph_2_std**2) # shape: {E_bins}
        Ne_mean = Ne_1_mean + Ne_2_mean # shape: {E_bins}
        Ne_std = tf.sqrt(Ne_1_std**2 + Ne_2_std**2) # shape: {E_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (Nph_1_skew*Nph_1_mean + Nph_2_skew*Nph_2_mean)/(Nph_mean) # shape: {E_bins}
        Ne_skew = (Ne_1_skew*Ne_1_mean + Ne_2_skew*Ne_2_mean)/(Ne_mean) # shape: {E_bins}
        initial_corr = (initial_1_corr*energies_first+initial_2_corr*energies_second)/(energies_first+energies_second) # shape: {E_bins}


        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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

        cov_star = np.ones((np.shape(cov)[0],3,3))
        cov_star[:,0,1:] = tf.transpose(delta, perm=[0,2,1])[:,0,:]
        cov_star[:,1:,0] = delta[:,:,0]
        cov_star[:,1:,1:] = cov
        
        dim=2
        x = np.array([stats.multivariate_normal(np.zeros(dim+1), cov_i).rvs(size=1) for cov_i in cov_star])
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        
        NphNe_samples = x1*scale[:,:,0]+loc[:,:,0] 

        Nph = NphNe_samples[:,0]
        Ne = NphNe_samples[:,1]
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = (4.61849047 * Nph**(-0.23931848)).numpy()

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = (-2.37542105 *  Ne** (-0.26152676)).numpy()
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi)).numpy()
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi)).numpy()
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi)).numpy()
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi)).numpy()

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        S1_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S1_loc,S1_scale,S1_skew)])
        S2_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S2_loc,S2_scale,S2_skew)])

        d['s1'] = S1_sample
        d['s2'] = S2_sample
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _annotate(self, d):
        pass

    def _compute(self,
                     data_tensor, ptensor,
                     # Domain
                     s1,
                     # Dependency domains and values
                     energy_first, rate_vs_energy_first,
                     energy_second, rate_vs_energy):

        ####----test----####
        # self.pdf_for_nphne(1,2,3)      
      
        
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,2000,150), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,800,155), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}

        # ####--------test---------####
        # print('1 np.shape(energy_first)',np.shape(energy_first))
        # print('1 np.shape(energy_second)',np.shape(energy_second))
        # print('1 np.shape(rate_vs_energy_first)',np.shape(rate_vs_energy_first))
        # print('1 np.shape(rate_vs_energy)',np.shape(rate_vs_energy))
        
        # Energy binning
        energy_second = tf.repeat(energy_second[:,o,:], tf.shape(energy_first[0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None} 
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}

        # ####--------test---------####
        # print('2 np.shape(energy_first)',np.shape(energy_first))
        # print('2 np.shape(energy_second)',np.shape(energy_second))
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
        

        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
              
        # ####--------test---------####
        # print('3 np.shape(energy_first)',np.shape(energy_first))
        # print('3 np.shape(energy_second)',np.shape(energy_second))
        # print('3 np.shape(rate_vs_energy_first)',np.shape(rate_vs_energy_first))
        # print('3 np.shape(rate_vs_energy_second)',np.shape(rate_vs_energy_second))
        # print('3 np.shape(rate_vs_energy)',np.shape(rate_vs_energy))
          
          
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_first, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme('quanta_params',
                                                     bonus_arg=energy_first, 
                                                     data_tensor=data_tensor,
                                                     ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)

        Nph_1_var = (Nph_1_fano * Nph_1_mean) #shape (E1_bins)
        Ne_1_var = (Ne_1_fano * Ne_1_mean) #shape (E1_bins)         
        
        ###########  # Second vertex
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_second, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params', 
                                                              bonus_arg=energy_second, 
                                                              data_tensor=data_tensor,
                                                              ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_var = (Nph_2_fano * Nph_2_mean) #shape (E2_bins)
        Ne_2_var = (Ne_2_fano * Ne_2_mean) #shape (E2_bins)       
        
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
        

        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        
        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,:,o] # final shape: {E1_bins,E2_bins,1}
        skews2 = Ne_skew[:,:,o] # final shape: {E1_bins,E2_bins,1}
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        f_std_x = Nph_std[:,:,o] # final shape: {E1_bins,E2_bins,1}
        f_std_y = Ne_std[:,:,o] # final shape: {E1_bins,E2_bins,1}
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        f_mean_x = Nph_mean[:,:,o] # final shape: {E1_bins,E2_bins,1}
        f_mean_y = Ne_mean[:,:,o] # final shape: {E1_bins,E2_bins,1}
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}
        

        cov = tf.repeat(initial_corr[:,:,o], 2, axis=2)  # final shape: {E1_bins,E2_bins,2}
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
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr) 
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr))
        exp_term_1 = (x - loc[:,:,0,0]) * (x - loc[:,:,0,0]) / (scale[:,:,0,0]*scale[:,:,0,0])
        exp_term_2 = (y - loc[:,:,1,0]) * (y - loc[:,:,1,0]) / (scale[:,:,1,0]*scale[:,:,1,0])
        exp_term_3 = -2. * initial_corr * (x - loc[:,:,0,0]) * (y - loc[:,:,1,0]) / (scale[:,:,0,0] * scale[:,:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))
        
        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc[:,:,0,0])/scale[:,:,0,0]) + (Ne_skew * (y-loc[:,:,1,0])/scale[:,:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,:,0,0]*scale[:,:,1,0])) * (final_std[:,:,0,0]*final_std[:,:,1,0])  # shape: {Nph,Ne,E1_bins,E2_bins}      
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        ### Calculate probabilities
        probs *= rate_vs_energy # shape: {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=3) # final shape: {Nph,Ne,E1_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}
        
        # print('sum NphNe_pdf',tf.reduce_sum(NphNe_pdf))

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

             
        S1_pdf = skewnorm_1d(x=s1,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,:,o],len(Ne),2) # final shape: {batch_size, Nph, Ne}
        
        S2_pdf = skewnorm_1d(x=s2,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[:,o,:],len(Nph),1) # final shape: {batch_size, Nph, Ne}

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {batch_size, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2) # final shape: {batch_size, Nph}
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=1) # final shape: {batch_size,}

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size,}
        S1S2_pdf *= acceptance
     
        S1S2_pdf = tf.repeat(S1S2_pdf[:,o],tf.shape(energy_first)[0],1) 
        S1S2_pdf = tf.repeat(S1S2_pdf[:,:,o],1,1) # final shape: {batch_size,E_bins,1} 

        return tf.transpose(S1S2_pdf, perm=[0, 2, 1]) # initial shape: {batch_size,E_bins,1} --> final shape: {batch_size,1,E_bins}   

    def check_data(self):
        if not self.check_acceptances:
         return
        s_acc = self.gimme_numpy('s1s2_acceptance')
        if np.any(s_acc <= 0):
         raise ValueError(f"Found event with non-positive signal "
                          f"acceptance: did you apply and configure "
                          "your cuts correctly?")


@export
class MakeS1S2MSU3(MakeS1S2MSU):
    """
    """
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),
                  (('energy_others',), 'rate_vs_energy'))

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        energies_third= d['energy_third'].values        
        
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme_numpy('yield_params', energies_first) # shape: {E1_bins} (matches bonus_arg)
        Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme_numpy('quanta_params', energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_1_std = tf.sqrt(Nph_1_fano * Nph_1_mean)
        Ne_1_std = tf.sqrt(Ne_1_fano * Ne_1_mean) 
        
        Nph_2_mean, Ne_2_mean = self.gimme_numpy('yield_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme_numpy('quanta_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_std = tf.sqrt(Nph_2_fano * Nph_2_mean)
        Ne_2_std = tf.sqrt(Ne_2_fano * Ne_2_mean) 
        
        Nph_3_mean, Ne_3_mean = self.gimme_numpy('yield_params', energies_third) # shape: {E3_bins} (matches bonus_arg)
        Nph_3_fano, Ne_3_fano, Nph_3_skew, Ne_3_skew, initial_3_corr = self.gimme_numpy('quanta_params', energies_third) # shape: {E3_bins} (matches bonus_arg)
        
        Nph_3_std = tf.sqrt(Nph_3_fano * Nph_3_mean)
        Ne_3_std = tf.sqrt(Ne_3_fano * Ne_3_mean) 
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        # 240331 JB: assume shape(energies_first) == shape(energies_second) == shape(energies_third)
        Nph_mean = Nph_1_mean + Nph_2_mean + Nph_3_mean # shape: {E_bins}
        Nph_std = tf.sqrt(Nph_1_std**2 + Nph_2_std**2 + Nph_3_std**2) # shape: {E_bins}
        Ne_mean = Ne_1_mean + Ne_2_mean + Ne_3_mean # shape: {E_bins}
        Ne_std = tf.sqrt(Ne_1_std**2 + Ne_2_std**2 + Ne_3_std**2) # shape: {E_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (Nph_1_skew*Nph_1_mean + Nph_2_skew*Nph_2_mean + Nph_3_skew*Nph_3_mean)/(Nph_mean) # shape: {E_bins}
        Ne_skew = (Ne_1_skew*Ne_1_mean + Ne_2_skew*Ne_2_mean + Ne_3_skew*Ne_3_mean)/(Ne_mean) # shape: {E_bins}
        initial_corr = (initial_1_corr*energies_first+initial_2_corr*energies_second+initial_3_corr*energies_third)/(energies_first+energies_second+energies_third) # shape: {E_bins}


        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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

        cov_star = np.ones((np.shape(cov)[0],3,3))
        cov_star[:,0,1:] = tf.transpose(delta, perm=[0,2,1])[:,0,:]
        cov_star[:,1:,0] = delta[:,:,0]
        cov_star[:,1:,1:] = cov
        
        dim=2
        x = np.array([stats.multivariate_normal(np.zeros(dim+1), cov_i).rvs(size=1) for cov_i in cov_star])
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        
        NphNe_samples = x1*scale[:,:,0]+loc[:,:,0] 

        Nph = NphNe_samples[:,0]
        Ne = NphNe_samples[:,1]
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = (4.61849047 * Nph**(-0.23931848)).numpy()

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = (-2.37542105 *  Ne** (-0.26152676)).numpy()
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi)).numpy()
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi)).numpy()
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi)).numpy()
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi)).numpy()

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        S1_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S1_loc,S1_scale,S1_skew)])
        S2_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S2_loc,S2_scale,S2_skew)])

        d['s1'] = S1_sample
        d['s2'] = S2_sample
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_others, rate_vs_energy):
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,2000,150), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,800,155), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}

        # ####--------test---------####
        # print('1 np.shape(energy_first)',np.shape(energy_first)) # batch, E1, none
        # print('1 np.shape(energy_others)',np.shape(energy_others)) # batch, none
        # print('1 np.shape(rate_vs_energy_first)',np.shape(rate_vs_energy_first)) # batch, E1
        # print('1 np.shape(rate_vs_energy)',np.shape(rate_vs_energy)) # batch, E1, E2, E3
        # print('\n1 energy_first[0]',energy_first[0]) # E1, none
        # print('1 energy_others[0]\n',energy_others[0]) # none
        
        # Energy binning
        energy_second = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None} 
        energy_third = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,0,:]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E3_bins, None} 
        
        # ####--------test---------####
        # print('2 np.shape(energy_second)',np.shape(energy_second))
        # print('2 np.shape(energy_third)',np.shape(energy_third))
        # print('\n2 energy_second[0]',energy_second[0]) # E2_bins, None
        # print('2 energy_third[0]\n',energy_third[0]) # E3_bins, None
        
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
    
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        energy_third = fd.np_to_tf(energy_third)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E3_bins} 
        rate_vs_energy_third = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        # ####--------test---------####
        # print('3 np.shape(energy_first)',np.shape(energy_first))
        # print('3 np.shape(energy_second)',np.shape(energy_second))
        # print('3 np.shape(energy_third)',np.shape(energy_third))
        # print('3 np.shape(rate_vs_energy_first)',np.shape(rate_vs_energy_first))
        # print('3 np.shape(rate_vs_energy_second)',np.shape(rate_vs_energy_second))
        # print('3 np.shape(rate_vs_energy_third)',np.shape(rate_vs_energy_third))
        # print('3 np.shape(rate_vs_energy)',np.shape(rate_vs_energy))
        # print('\n2 energy_first',energy_first) # E2_bins
        # print('2 energy_second',energy_second) # E2_bins
        # print('2 energy_third\n',energy_third) # E3_bins
        
        
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_first, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme('quanta_params',
                                                     bonus_arg=energy_first, 
                                                     data_tensor=data_tensor,
                                                     ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)

        Nph_1_var = (Nph_1_fano * Nph_1_mean) #shape (E1_bins)
        Ne_1_var = (Ne_1_fano * Ne_1_mean) #shape (E1_bins)         
        
        ###########  # Second vertex
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_second, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params', 
                                                              bonus_arg=energy_second, 
                                                              data_tensor=data_tensor,
                                                              ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_var = (Nph_2_fano * Nph_2_mean) #shape (E2_bins)
        Ne_2_var = (Ne_2_fano * Ne_2_mean) #shape (E2_bins)     
        
        ###########  # Third vertex
        Nph_3_mean, Ne_3_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_second, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
        Nph_3_fano, Ne_3_fano, Nph_3_skew, Ne_3_skew, initial_3_corr = self.gimme('quanta_params', 
                                                              bonus_arg=energy_third, 
                                                              data_tensor=data_tensor,
                                                              ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
        
        Nph_3_var = (Nph_3_fano * Nph_3_mean) #shape (E3_bins)
        Ne_3_var = (Ne_3_fano * Ne_3_mean) #shape (E3_bins)          
        
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
        

        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        x = tf.repeat(x[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        y = tf.repeat(y[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}
        
        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        skews2 = Ne_skew[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,2,1}

        f_std_x = Nph_std[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        f_std_y = Ne_std[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,2,1}

        f_mean_x = Nph_mean[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        f_mean_y = Ne_mean[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,2,1}
        

        cov = tf.repeat(initial_corr[:,:,:,o], 2, axis=3)  # final shape: {E1_bins,E2_bins,E3_bins,2}
        cov = tf.repeat(cov[:,:,:,:,o], 2, axis=4)      # final shape: {E1_bins,E2_bins,E3_bins,2,2}
        cov = cov - (tf.eye(2)*cov-tf.eye(2))         # final shape: {E1_bins,E2_bins,E3_bins,2,2}

        
        # define scale and loc params for shifting skew2d from (0,0)
        del1 = tf.einsum('...ji,...jk->...ik',skewa,cov)
        del2 = tf.einsum('...ij,...jk->...ik',del1,skewa)       
        bCa = tf.einsum('...jk,...ji->...ki',cov,skewa) 

        aCa = 1. + del2
        delta = (1. / tf.sqrt(aCa)) * bCa

        scale = final_std / tf.sqrt(1. - 2 * delta**2 / pi) # shape: {E1_bins,E2_bins,E3_bins,2,1}  
        loc = final_mean - scale * delta * tf.sqrt(2/pi)    # shape: {E1_bins,E2_bins,E3_bins,2,1}  
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr) 
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr))
        exp_term_1 = (x - loc[:,:,:,0,0]) * (x - loc[:,:,:,0,0]) / (scale[:,:,:,0,0]*scale[:,:,:,0,0])
        exp_term_2 = (y - loc[:,:,:,1,0]) * (y - loc[:,:,:,1,0]) / (scale[:,:,:,1,0]*scale[:,:,:,1,0])
        exp_term_3 = -2. * initial_corr * (x - loc[:,:,:,0,0]) * (y - loc[:,:,:,1,0]) / (scale[:,:,:,0,0] * scale[:,:,:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))
        
        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc[:,:,:,0,0])/scale[:,:,:,0,0]) + (Ne_skew * (y-loc[:,:,:,1,0])/scale[:,:,:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,:,:,0,0]*scale[:,:,:,1,0])) * (final_std[:,:,:,0,0]*final_std[:,:,:,1,0])  # shape: {Nph,Ne,E1_bins,E2_bins,E3_bins}      
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        ### Calculate probabilities
        probs *= rate_vs_energy # shape: {Nph,Ne,E1_bins,E2_bins,E3_bins}
        probs = tf.reduce_sum(probs, axis=4) # final shape: {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=3) # final shape: {Nph,Ne,E1_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}
        
        # print('sum NphNe_pdf',tf.reduce_sum(NphNe_pdf))

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

             
        S1_pdf = skewnorm_1d(x=s1,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,:,o],len(Ne),2) # final shape: {batch_size, Nph, Ne}
        
        S2_pdf = skewnorm_1d(x=s2,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[:,o,:],len(Nph),1) # final shape: {batch_size, Nph, Ne}

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {batch_size, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2) # final shape: {batch_size, Nph}
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=1) # final shape: {batch_size,}

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size,}
        S1S2_pdf *= acceptance
     
        S1S2_pdf = tf.repeat(S1S2_pdf[:,o],tf.shape(energy_first)[0],1) 
        S1S2_pdf = tf.repeat(S1S2_pdf[:,:,o],1,1) # final shape: {batch_size,E_bins,1} 

        return tf.transpose(S1S2_pdf, perm=[0, 2, 1]) # initial shape: {batch_size,E_bins,1} --> final shape: {batch_size,1,E_bins}  


@export
class MakeS1S2SS(MakeS1S2MSU):
    """
    """
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),)
    
    def _simulate(self, d): 
        energies = d['energy_first'].values
        
        # Load params
        Nph_mean, Ne_mean = self.gimme_numpy('yield_params', energies) # shape: {E_bins} (matches bonus_arg)
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('quanta_params', energies) # shape: {E_bins} (matches bonus_arg)
        
        Nph_std = tf.sqrt(Nph_fano * Nph_mean)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) 

        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o] 
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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

        cov_star = np.ones((np.shape(cov)[0],3,3))
        cov_star[:,0,1:] = tf.transpose(delta, perm=[0,2,1])[:,0,:]
        cov_star[:,1:,0] = delta[:,:,0]
        cov_star[:,1:,1:] = cov
        
        dim=2
        x = np.array([stats.multivariate_normal(np.zeros(dim+1), cov_i).rvs(size=1) for cov_i in cov_star])
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        
        NphNe_samples = x1*scale[:,:,0]+loc[:,:,0] 

        Nph = NphNe_samples[:,0]
        Ne = NphNe_samples[:,1]
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = (4.61849047 * Nph**(-0.23931848)).numpy()

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = (-2.37542105 *  Ne** (-0.26152676)).numpy()
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi)).numpy()
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi)).numpy()
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi)).numpy()
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi)).numpy()

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        S1_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S1_loc,S1_scale,S1_skew)])
        S2_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S2_loc,S2_scale,S2_skew)])
        

        d['s1'] = S1_sample
        d['s2'] = S2_sample
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')
        
    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):

        ####----test----####
        # self.pdf_for_nphne(1,2,3)      
      
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,2000,150), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,800,155), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E_bins, None} --> final shape: {batch_size}
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        
        
       
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}
            
        # Energy binning
        energy_first = fd.np_to_tf(energy_first)[0,:,0]               # inital shape: {batch_size, E_bins, None} --> final shape: {E_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E_bins} --> final shape: {E_bins}

        # Load params
        Nph_mean, Ne_mean = self.gimme('yield_params',
                                        bonus_arg=energy_first, 
                                        data_tensor=data_tensor,
                                        ptensor=ptensor) # shape: {E_bins} (matches bonus_arg)
        
        Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme('quanta_params',
                                                     bonus_arg=energy_first, 
                                                     data_tensor=data_tensor,
                                                     ptensor=ptensor) # shape: {E_bins} (matches bonus_arg)

        Nph_std = tf.sqrt(Nph_fano * Nph_mean) #shape (E_bins)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) #shape (E_bins)     
        
        ### Quanta Production
        x = NphNe[:,:,0]
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        y = NphNe[:,:,1]
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr) 
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr))
        exp_term_1 = (x - loc[:,0,0]) * (x - loc[:,0,0]) / (scale[:,0,0]*scale[:,0,0])
        exp_term_2 = (y - loc[:,1,0]) * (y - loc[:,1,0]) / (scale[:,1,0]*scale[:,1,0])
        exp_term_3 = -2. * initial_corr * (x - loc[:,0,0]) * (y - loc[:,1,0]) / (scale[:,0,0] * scale[:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc[:,0,0])/scale[:,0,0]) + (Ne_skew * (y-loc[:,1,0])/scale[:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,0,0]*scale[:,1,0])) * (final_std[:,0,0]*final_std[:,1,0])
        probs *= rate_vs_energy_first # shape: {Nph,Ne,E_bins,} {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}
        
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
        
        
        S1_pdf = skewnorm_1d(x=s1,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,:,o],len(Ne),2) # final shape: {batch_size, Nph, Ne}
        
        S2_pdf = skewnorm_1d(x=s2,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[:,o,:],len(Nph),1) # final shape: {batch_size, Nph, Ne}

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {batch_size, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2) # final shape: {batch_size, Nph}
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=1) # final shape: {batch_size,}
        
        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size,}
        S1S2_pdf *= acceptance
     
        S1S2_pdf = tf.repeat(S1S2_pdf[:,o],tf.shape(energy_first)[0],1) 
        S1S2_pdf = tf.repeat(S1S2_pdf[:,:,o],1,1) # final shape: {batch_size,E_bins,1} 

        return tf.transpose(S1S2_pdf, perm=[0, 2, 1]) # initial shape: {batch_size,E_bins,1} --> final shape: {batch_size,1,E_bins} 


@export
class MakeS1S2Migdal(MakeS1S2MSU):
    """
    """
    special_model_functions = ('yield_params','quanta_params',
                               'quanta_params_ER')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        
        # Load params
        Nph_1_mean, Ne_1_mean, Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme_numpy('quanta_params_ER', energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_1_std = tf.sqrt(Nph_1_fano * Nph_1_mean)
        Ne_1_std = tf.sqrt(Ne_1_fano * Ne_1_mean) 
        
        Nph_2_mean, Ne_2_mean = self.gimme_numpy('yield_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme_numpy('quanta_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_std = tf.sqrt(Nph_2_fano * Nph_2_mean)
        Ne_2_std = tf.sqrt(Ne_2_fano * Ne_2_mean) 
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        # 240331 JB: assume shape(energies_first) == shape(energies_second)
        Nph_mean = Nph_1_mean + Nph_2_mean # shape: {E_bins}
        Nph_std = tf.sqrt(Nph_1_std**2 + Nph_2_std**2) # shape: {E_bins}
        Ne_mean = Ne_1_mean + Ne_2_mean # shape: {E_bins}
        Ne_std = tf.sqrt(Ne_1_std**2 + Ne_2_std**2) # shape: {E_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (Nph_1_skew*Nph_1_mean + Nph_2_skew*Nph_2_mean)/(Nph_mean) # shape: {E_bins}
        Ne_skew = (Ne_1_skew*Ne_1_mean + Ne_2_skew*Ne_2_mean)/(Ne_mean) # shape: {E_bins}
        initial_corr = (initial_1_corr*energies_first+initial_2_corr*energies_second)/(energies_first+energies_second) # shape: {E_bins}


        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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

        cov_star = np.ones((np.shape(cov)[0],3,3))
        cov_star[:,0,1:] = tf.transpose(delta, perm=[0,2,1])[:,0,:]
        cov_star[:,1:,0] = delta[:,:,0]
        cov_star[:,1:,1:] = cov
        
        dim=2
        x = np.array([stats.multivariate_normal(np.zeros(dim+1), cov_i).rvs(size=1) for cov_i in cov_star])
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        
        NphNe_samples = x1*scale[:,:,0]+loc[:,:,0] 

        Nph = NphNe_samples[:,0]
        Ne = NphNe_samples[:,1]
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = (4.61849047 * Nph**(-0.23931848)).numpy()

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = (-2.37542105 *  Ne** (-0.26152676)).numpy()
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi)).numpy()
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi)).numpy()
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi)).numpy()
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi)).numpy()

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        S1_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S1_loc,S1_scale,S1_skew)])
        S2_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S2_loc,S2_scale,S2_skew)])

        d['s1'] = S1_sample
        d['s2'] = S2_sample
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_second, rate_vs_energy):
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,2000,150), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,800,155), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}

        # Energy binning
        energy_second = tf.repeat(energy_second[:,o,:], tf.shape(energy_first[0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None}, None will be assigned to E2_bins
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
    
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
        

        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
              
        
        ###########  # First vertex
        # Load params
        Nph_1_mean = self.source.Nph_mean_ER_tf
        # Nph_1_mean = tf.repeat(Nph_1_mean[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        Ne_1_mean = self.source.Ne_mean_ER_tf
        # Ne_1_mean = tf.repeat(Ne_1_mean[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        Nph_1_std = self.source.Nph_std_ER_tf
        # Nph_1_std = tf.repeat(Nph_1_std[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        Nph_1_var = Nph_1_std*Nph_1_std
        Ne_1_std = self.source.Ne_std_ER_tf
        # Ne_1_std = tf.repeat(Ne_1_std[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        Ne_1_var = Ne_1_std*Ne_1_std
        Nph_1_skew = self.source.Nph_skew_ER_tf
        # Nph_1_skew = tf.repeat(Nph_1_skew[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        Ne_1_skew = self.source.Ne_skew_ER_tf
        # Ne_1_skew = tf.repeat(Ne_1_skew[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        initial_1_corr = self.source.initial_corr_ER_tf
        # initial_1_corr = tf.repeat(initial_1_corr[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        
        ###########  # Second vertex
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_second, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params', 
                                                              bonus_arg=energy_second, 
                                                              data_tensor=data_tensor,
                                                              ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_var = (Nph_2_fano * Nph_2_mean) #shape (E2_bins)
        Ne_2_var = (Ne_2_fano * Ne_2_mean) #shape (E2_bins)       
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        Nph_mean = tf.reshape(Nph_1_mean,[-1,1]) + tf.reshape(Nph_2_mean,[1,-1]) # shape: {E1_bins,E2_bins}
        Nph_std = tf.sqrt(tf.reshape(Nph_1_var,[-1,1]) + tf.reshape(Nph_2_var,[1,-1])) # shape: {E1_bins,E2_bins}
        Ne_mean = tf.reshape(Ne_1_mean,[-1,1]) + tf.reshape(Ne_2_mean,[1,-1]) # shape: {E1_bins,E2_bins}
        Ne_std = tf.sqrt(tf.reshape(Ne_1_var,[-1,1]) + tf.reshape(Ne_2_var,[1,-1])) # shape: {E1_bins,E2_bins}
        
        # #####-------test-------#####
        # print('np.shape(initial_1_corr)', np.shape(initial_1_corr))
        # print('np.shape(energy_first)', np.shape(energy_first))
        # print('np.shape(initial_2_corr)', np.shape(initial_2_corr))
        # print('np.shape(energy_second)', np.shape(energy_second))
        
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (tf.reshape(Nph_1_skew*Nph_1_mean,[-1,1]) 
                  + tf.reshape(Nph_2_skew*Nph_2_mean,[1,-1]))/(Nph_mean) # shape: {E1_bins,E2_bins}
        Ne_skew = (tf.reshape(Ne_1_skew*Ne_1_mean,[-1,1]) 
                 + tf.reshape(Ne_2_skew*Ne_2_mean,[1,-1]))/(Ne_mean) # shape: {E1_bins,E2_bins}
        totE = tf.reshape(energy_first,[-1,1]) + tf.reshape(energy_second,[1,-1]) # shape: {E1_bins,E2_bins}
        initial_corr = (tf.reshape(initial_1_corr*energy_first,[-1,1]) 
                      + tf.reshape(initial_2_corr*energy_second,[1,-1]))/(totE) # shape: {E1_bins,E2_bins}
        

        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        
        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,:,o] # final shape: {E1_bins,E2_bins,1}
        skews2 = Ne_skew[:,:,o] # final shape: {E1_bins,E2_bins,1}
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        f_std_x = Nph_std[:,:,o] # final shape: {E1_bins,E2_bins,1}
        f_std_y = Ne_std[:,:,o] # final shape: {E1_bins,E2_bins,1}
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}

        f_mean_x = Nph_mean[:,:,o] # final shape: {E1_bins,E2_bins,1}
        f_mean_y = Ne_mean[:,:,o] # final shape: {E1_bins,E2_bins,1}
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,:,o] # final shape: {E1_bins,E2_bins,2,1}
        

        cov = tf.repeat(initial_corr[:,:,o], 2, axis=2)  # final shape: {E1_bins,E2_bins,2}
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
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr) 
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr))
        exp_term_1 = (x - loc[:,:,0,0]) * (x - loc[:,:,0,0]) / (scale[:,:,0,0]*scale[:,:,0,0])
        exp_term_2 = (y - loc[:,:,1,0]) * (y - loc[:,:,1,0]) / (scale[:,:,1,0]*scale[:,:,1,0])
        exp_term_3 = -2. * initial_corr * (x - loc[:,:,0,0]) * (y - loc[:,:,1,0]) / (scale[:,:,0,0] * scale[:,:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))
        
        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc[:,:,0,0])/scale[:,:,0,0]) + (Ne_skew * (y-loc[:,:,1,0])/scale[:,:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,:,0,0]*scale[:,:,1,0])) * (final_std[:,:,0,0]*final_std[:,:,1,0])  # shape: {Nph,Ne,E1_bins,E2_bins}      
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        ### Calculate probabilities
        probs *= rate_vs_energy # shape: {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=3) # final shape: {Nph,Ne,E1_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}
        
        # print('sum NphNe_pdf',tf.reduce_sum(NphNe_pdf))

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

             
        S1_pdf = skewnorm_1d(x=s1,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,:,o],len(Ne),2) # final shape: {batch_size, Nph, Ne}
        
        S2_pdf = skewnorm_1d(x=s2,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[:,o,:],len(Nph),1) # final shape: {batch_size, Nph, Ne}

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {batch_size, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2) # final shape: {batch_size, Nph}
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=1) # final shape: {batch_size,}

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size,}
        S1S2_pdf *= acceptance
     
        S1S2_pdf = tf.repeat(S1S2_pdf[:,o],tf.shape(energy_first)[0],1) 
        S1S2_pdf = tf.repeat(S1S2_pdf[:,:,o],1,1) # final shape: {batch_size,E_bins,1} 

        return tf.transpose(S1S2_pdf, perm=[0, 2, 1]) # initial shape: {batch_size,E_bins,1} --> final shape: {batch_size,1,E_bins}  


@export
class MakeS1S2MigdalMSU(MakeS1S2MSU3):
    """
    """
    special_model_functions = ('yield_params','quanta_params',
                               'quanta_params_ER')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        energies_third= d['energy_third'].values        
        
        # Load params
        Nph_1_mean, Ne_1_mean, Nph_1_fano, Ne_1_fano, Nph_1_skew, Ne_1_skew, initial_1_corr = self.gimme_numpy('quanta_params_ER', energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_1_std = tf.sqrt(Nph_1_fano * Nph_1_mean)
        Ne_1_std = tf.sqrt(Ne_1_fano * Ne_1_mean) 
        
        Nph_2_mean, Ne_2_mean = self.gimme_numpy('yield_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme_numpy('quanta_params', energies_second) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_std = tf.sqrt(Nph_2_fano * Nph_2_mean)
        Ne_2_std = tf.sqrt(Ne_2_fano * Ne_2_mean) 
        
        Nph_3_mean, Ne_3_mean = self.gimme_numpy('yield_params', energies_third) # shape: {E3_bins} (matches bonus_arg)
        Nph_3_fano, Ne_3_fano, Nph_3_skew, Ne_3_skew, initial_3_corr = self.gimme_numpy('quanta_params', energies_third) # shape: {E3_bins} (matches bonus_arg)
        
        Nph_3_std = tf.sqrt(Nph_3_fano * Nph_3_mean)
        Ne_3_std = tf.sqrt(Ne_3_fano * Ne_3_mean) 
        
        # compute Skew2D for double scatters
        # sum the mean and variances of both scatters
        # 240331 JB: assume shape(energies_first) == shape(energies_second) == shape(energies_third)
        Nph_mean = Nph_1_mean + Nph_2_mean + Nph_3_mean # shape: {E_bins}
        Nph_std = tf.sqrt(Nph_1_std**2 + Nph_2_std**2 + Nph_3_std**2) # shape: {E_bins}
        Ne_mean = Ne_1_mean + Ne_2_mean + Ne_3_mean # shape: {E_bins}
        Ne_std = tf.sqrt(Ne_1_std**2 + Ne_2_std**2 + Ne_3_std**2) # shape: {E_bins}
        
        # take the average skew and correlation weighted by two scatters
        # this is only an approximation
        Nph_skew = (Nph_1_skew*Nph_1_mean + Nph_2_skew*Nph_2_mean + Nph_3_skew*Nph_3_mean)/(Nph_mean) # shape: {E_bins}
        Ne_skew = (Ne_1_skew*Ne_1_mean + Ne_2_skew*Ne_2_mean + Ne_3_skew*Ne_3_mean)/(Ne_mean) # shape: {E_bins}
        initial_corr = (initial_1_corr*energies_first+initial_2_corr*energies_second+initial_3_corr*energies_third)/(energies_first+energies_second+energies_third) # shape: {E_bins}


        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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

        cov_star = np.ones((np.shape(cov)[0],3,3))
        cov_star[:,0,1:] = tf.transpose(delta, perm=[0,2,1])[:,0,:]
        cov_star[:,1:,0] = delta[:,:,0]
        cov_star[:,1:,1:] = cov
        
        dim=2
        x = np.array([stats.multivariate_normal(np.zeros(dim+1), cov_i).rvs(size=1) for cov_i in cov_star])
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        
        NphNe_samples = x1*scale[:,:,0]+loc[:,:,0] 

        Nph = NphNe_samples[:,0]
        Ne = NphNe_samples[:,1]
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = (4.61849047 * Nph**(-0.23931848)).numpy()

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = (-2.37542105 *  Ne** (-0.26152676)).numpy()
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi)).numpy()
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi)).numpy()
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi)).numpy()
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi)).numpy()

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        S1_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S1_loc,S1_scale,S1_skew)])
        S2_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S2_loc,S2_scale,S2_skew)])

        d['s1'] = S1_sample
        d['s2'] = S2_sample
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_others, rate_vs_energy):
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,2000,150), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,800,155), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}

        # ####--------test---------####
        # print('1 np.shape(energy_first)',np.shape(energy_first)) # batch, E1, none
        # print('1 np.shape(energy_others)',np.shape(energy_others)) # batch, none
        # print('1 np.shape(rate_vs_energy_first)',np.shape(rate_vs_energy_first)) # batch, E1
        # print('1 np.shape(rate_vs_energy)',np.shape(rate_vs_energy)) # batch, E1, E2, E3
        
        # Energy binning
        energy_second = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None} 
        energy_third = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,0,:]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E3_bins, None} 
        
        # ####--------test---------####
        # print('2 np.shape(energy_second)',np.shape(energy_second))
        # print('2 np.shape(energy_third)',np.shape(energy_third))
        
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
    
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        energy_third = fd.np_to_tf(energy_third)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E3_bins} 
        rate_vs_energy_third = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        # ####--------test---------####
        # print('3 np.shape(energy_first)',np.shape(energy_first))
        # print('3 np.shape(energy_second)',np.shape(energy_second))
        # print('3 np.shape(energy_third)',np.shape(energy_third))
        # print('3 np.shape(rate_vs_energy_first)',np.shape(rate_vs_energy_first))
        # print('3 np.shape(rate_vs_energy_second)',np.shape(rate_vs_energy_second))
        # print('3 np.shape(rate_vs_energy_third)',np.shape(rate_vs_energy_third))
        # print('3 np.shape(rate_vs_energy)',np.shape(rate_vs_energy))
        

        ###########  # First vertex
        # Load params
        Nph_1_mean = self.source.Nph_mean_ER_tf
        # Nph_1_mean = tf.repeat(Nph_1_mean[o, :, :], tf.shape(s1)[0], axis=0)
        # Nph_1_mean = tf.repeat(Nph_1_mean[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        
        Ne_1_mean = self.source.Ne_mean_ER_tf
        # Ne_1_mean = tf.repeat(Ne_1_mean[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        # Ne_1_mean = tf.repeat(Ne_1_mean[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        
        Nph_1_std = self.source.Nph_std_ER_tf
        # Nph_1_std = tf.repeat(Nph_1_std[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        # Nph_1_std = tf.repeat(Nph_1_std[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        Nph_1_var = Nph_1_std*Nph_1_std
        
        Ne_1_std = self.source.Ne_std_ER_tf
        # Ne_1_std = tf.repeat(Ne_1_std[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        # Ne_1_std = tf.repeat(Ne_1_std[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        Ne_1_var = Ne_1_std*Ne_1_std
        
        Nph_1_skew = self.source.Nph_skew_ER_tf
        # Nph_1_skew = tf.repeat(Nph_1_skew[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        # Nph_1_skew = tf.repeat(Nph_1_skew[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        
        Ne_1_skew = self.source.Ne_skew_ER_tf
        # Ne_1_skew = tf.repeat(Ne_1_skew[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        # Ne_1_skew = tf.repeat(Ne_1_skew[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        
        initial_1_corr = self.source.initial_corr_ER_tf
        # initial_1_corr = tf.repeat(initial_1_corr[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]       
        # initial_1_corr = tf.repeat(initial_1_corr[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        
        ###########  # Second vertex
        Nph_2_mean, Ne_2_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_second, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        Nph_2_fano, Ne_2_fano, Nph_2_skew, Ne_2_skew, initial_2_corr = self.gimme('quanta_params', 
                                                              bonus_arg=energy_second, 
                                                              data_tensor=data_tensor,
                                                              ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        Nph_2_var = (Nph_2_fano * Nph_2_mean) #shape (E2_bins)
        Ne_2_var = (Ne_2_fano * Ne_2_mean) #shape (E2_bins)     
        
        ###########  # Third vertex
        Nph_3_mean, Ne_3_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_second, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
        Nph_3_fano, Ne_3_fano, Nph_3_skew, Ne_3_skew, initial_3_corr = self.gimme('quanta_params', 
                                                              bonus_arg=energy_third, 
                                                              data_tensor=data_tensor,
                                                              ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg)
        
        Nph_3_var = (Nph_3_fano * Nph_3_mean) #shape (E3_bins)
        Ne_3_var = (Ne_3_fano * Ne_3_mean) #shape (E3_bins)          
        
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
        

        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        x = tf.repeat(x[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}
        y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins}
        y = tf.repeat(y[:,:,:,:,o],tf.shape(energy_third)[0],axis=4) # final shape: {Nph, Ne, E1_bins, E2_bins, E3_bins}
        
        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        skews2 = Ne_skew[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,2,1}

        f_std_x = Nph_std[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        f_std_y = Ne_std[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,2,1}

        f_mean_x = Nph_mean[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        f_mean_y = Ne_mean[:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,1}
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,:,:,o] # final shape: {E1_bins,E2_bins,E3_bins,2,1}
        

        cov = tf.repeat(initial_corr[:,:,:,o], 2, axis=3)  # final shape: {E1_bins,E2_bins,E3_bins,2}
        cov = tf.repeat(cov[:,:,:,:,o], 2, axis=4)      # final shape: {E1_bins,E2_bins,E3_bins,2,2}
        cov = cov - (tf.eye(2)*cov-tf.eye(2))         # final shape: {E1_bins,E2_bins,E3_bins,2,2}

        
        # define scale and loc params for shifting skew2d from (0,0)
        del1 = tf.einsum('...ji,...jk->...ik',skewa,cov)
        del2 = tf.einsum('...ij,...jk->...ik',del1,skewa)       
        bCa = tf.einsum('...jk,...ji->...ki',cov,skewa) 

        aCa = 1. + del2
        delta = (1. / tf.sqrt(aCa)) * bCa

        scale = final_std / tf.sqrt(1. - 2 * delta**2 / pi) # shape: {E1_bins,E2_bins,E3_bins,2,1}  
        loc = final_mean - scale * delta * tf.sqrt(2/pi)    # shape: {E1_bins,E2_bins,E3_bins,2,1}  
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr) 
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr))
        exp_term_1 = (x - loc[:,:,:,0,0]) * (x - loc[:,:,:,0,0]) / (scale[:,:,:,0,0]*scale[:,:,:,0,0])
        exp_term_2 = (y - loc[:,:,:,1,0]) * (y - loc[:,:,:,1,0]) / (scale[:,:,:,1,0]*scale[:,:,:,1,0])
        exp_term_3 = -2. * initial_corr * (x - loc[:,:,:,0,0]) * (y - loc[:,:,:,1,0]) / (scale[:,:,:,0,0] * scale[:,:,:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))
        
        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc[:,:,:,0,0])/scale[:,:,:,0,0]) + (Ne_skew * (y-loc[:,:,:,1,0])/scale[:,:,:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,:,:,0,0]*scale[:,:,:,1,0])) * (final_std[:,:,:,0,0]*final_std[:,:,:,1,0])  # shape: {Nph,Ne,E1_bins,E2_bins,E3_bins}      
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        ### Calculate probabilities
        probs *= rate_vs_energy # shape: {Nph,Ne,E1_bins,E2_bins,E3_bins}
        probs = tf.reduce_sum(probs, axis=4) # final shape: {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=3) # final shape: {Nph,Ne,E1_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}
        
        # print('sum NphNe_pdf',tf.reduce_sum(NphNe_pdf))

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

             
        S1_pdf = skewnorm_1d(x=s1,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,:,o],len(Ne),2) # final shape: {batch_size, Nph, Ne}
        
        S2_pdf = skewnorm_1d(x=s2,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[:,o,:],len(Nph),1) # final shape: {batch_size, Nph, Ne}

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {batch_size, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2) # final shape: {batch_size, Nph}
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=1) # final shape: {batch_size,}

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size,}
        S1S2_pdf *= acceptance
     
        S1S2_pdf = tf.repeat(S1S2_pdf[:,o],tf.shape(energy_first)[0],1) 
        S1S2_pdf = tf.repeat(S1S2_pdf[:,:,o],1,1) # final shape: {batch_size,E_bins,1} 

        return tf.transpose(S1S2_pdf, perm=[0, 2, 1]) # initial shape: {batch_size,E_bins,1} --> final shape: {batch_size,1,E_bins}  


@export
class MakeS1S2ER(MakeS1S2SS):
    """
    """
    special_model_functions = ('quanta_params_ER',)
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies = d['energy_first'].values
        
        # Load params
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('quanta_params_ER', energies) # shape: {E_bins} (matches bonus_arg)
        
        Nph_std = tf.sqrt(Nph_fano * Nph_mean)
        Ne_std = tf.sqrt(Ne_fano * Ne_mean) 

        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o] 
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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

        cov_star = np.ones((np.shape(cov)[0],3,3))
        cov_star[:,0,1:] = tf.transpose(delta, perm=[0,2,1])[:,0,:]
        cov_star[:,1:,0] = delta[:,:,0]
        cov_star[:,1:,1:] = cov
        
        dim=2
        x = np.array([stats.multivariate_normal(np.zeros(dim+1), cov_i).rvs(size=1) for cov_i in cov_star])
        x0, x1 = x[:, 0], x[:, 1:]
        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        
        NphNe_samples = x1*scale[:,:,0]+loc[:,:,0] 

        Nph = NphNe_samples[:,0]
        Ne = NphNe_samples[:,1]
        
        ### S1,S2 Yield
        g1 = 0.1131
        g2 = 47.35

        S1_mean = Nph*g1     
        S1_fano = 1.12145985 * Nph**(-0.00629895)
        S1_std = tf.sqrt(S1_mean*S1_fano)
        S1_skew = (4.61849047 * Nph**(-0.23931848)).numpy()

        S2_mean = Ne*g2
        S2_fano = 21.3
        S2_std = tf.sqrt(S2_mean*S2_fano)
        S2_skew = (-2.37542105 *  Ne** (-0.26152676)).numpy()
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi)).numpy()
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi)).numpy()
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi)).numpy()
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi)).numpy()

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
        
#         print(np.all(S1_scale>0))
#         print(np.all(S2_scale>0))
#         print(S1_scale[S1_scale<=0])
#         print(S2_scale[S2_scale<=0])
#         print(S1_scale[np.isnan(S1_scale)])
#         print(S2_scale[np.isnan(S2_scale)])
#         print(S1_scale[~np.isnan(S1_scale)])
#         print(S2_scale[~np.isnan(S2_scale)])
#         print(S1_scale)
#         print(S2_scale)
         
        S1_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S1_loc,S1_scale,S1_skew)])
        S2_sample = np.array([stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1) for loc,scale,skew in zip(S2_loc,S2_scale,S2_skew)])
        
#         test_isnan = np.isnan(S2_scale)
#         for i in range(len(S2_loc)): 
#             loc = S2_loc[i]
#             scale = S2_scale[i]
#             skew = S2_skew[i]
#             try:
#                 stats.skewnorm(a=skew,loc=loc,scale=scale).rvs(1)
#             except:
#                 print('loc,scale,skew', loc,scale,skew)
#                 print('S1_std,S2_std,S1_delta,S2_delta', S1_std[i],S2_std[i],S1_delta[i],S2_delta[i])
#                 print('isnan', test_isnan[i])
#                 print('')
        

        d['s1'] = S1_sample
        d['s2'] = S2_sample
        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):

        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,2000,150), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,800,155), fd.float_type())
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E_bins, None} --> final shape: {batch_size}
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        
        
       
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}
            
        # Energy binning
        energy_first = fd.np_to_tf(energy_first)[0,:,0]               # inital shape: {batch_size, E_bins, None} --> final shape: {E_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E_bins} --> final shape: {E_bins}

        # Load params
        Nph_mean = self.source.Nph_mean_ER_tf
        # Nph_mean = tf.repeat(Nph_mean[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        Ne_mean = self.source.Ne_mean_ER_tf
        # Ne_mean = tf.repeat(Ne_mean[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        Nph_std = self.source.Nph_std_ER_tf
        # Nph_std = tf.repeat(Nph_std[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        Ne_std = self.source.Ne_std_ER_tf
        # Ne_std = tf.repeat(Ne_std[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        Nph_skew = self.source.Nph_skew_ER_tf
        # Nph_skew = tf.repeat(Nph_skew[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        Ne_skew = self.source.Ne_skew_ER_tf
        # Ne_skew = tf.repeat(Ne_skew[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        initial_corr = self.source.initial_corr_ER_tf
        # initial_corr = tf.repeat(initial_corr[o, :], tf.shape(s1)[0], axis=0)[:, :, o]  
        
        ### Quanta Production
        x = NphNe[:,:,0]
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        y = NphNe[:,:,1]
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        # adjust dimensionality of input tensors
        skews1 = Nph_skew[:,o]
        skews2 = Ne_skew[:,o]
        skewa = tf.concat([skews1, skews2], axis=-1)[:,:,o]

        f_std_x = Nph_std[:,o]
        f_std_y = Ne_std[:,o]
        final_std = tf.concat([f_std_x, f_std_y], axis=-1)[:,:,o]

        f_mean_x = Nph_mean[:,o]
        f_mean_y = Ne_mean[:,o]
        final_mean = tf.concat([f_mean_x, f_mean_y], axis=-1)[:,:,o]

        cov = tf.repeat(initial_corr[:,o], 2, axis=1)  
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
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr) 
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr))
        exp_term_1 = (x - loc[:,0,0]) * (x - loc[:,0,0]) / (scale[:,0,0]*scale[:,0,0])
        exp_term_2 = (y - loc[:,1,0]) * (y - loc[:,1,0]) / (scale[:,1,0]*scale[:,1,0])
        exp_term_3 = -2. * initial_corr * (x - loc[:,0,0]) * (y - loc[:,1,0]) / (scale[:,0,0] * scale[:,1,0])

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc[:,0,0])/scale[:,0,0]) + (Ne_skew * (y-loc[:,1,0])/scale[:,1,0])
        Erf = tf.math.erf( Erf_arg / 1.4142 )

        norm_cdf = ( 1 + Erf ) / 2

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale[:,0,0]*scale[:,1,0])) * (final_std[:,0,0]*final_std[:,1,0])
        probs *= rate_vs_energy_first # shape: {Nph,Ne,E_bins,} {Nph,Ne,E1_bins,E2_bins}
        probs = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        NphNe_pdf = probs*Nph_diffs[0]*Ne_diffs[0] #  final shape: {Nph,Ne}
        
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
        
        
        S1_pdf = skewnorm_1d(x=s1,x_mean=S1_mean,x_std=S1_std,x_alpha=S1_skew)
        S1_pdf = tf.repeat(S1_pdf[:,:,o],len(Ne),2) # final shape: {batch_size, Nph, Ne}
        
        S2_pdf = skewnorm_1d(x=s2,x_mean=S2_mean,x_std=S2_std,x_alpha=S2_skew)
        S2_pdf = tf.repeat(S2_pdf[:,o,:],len(Nph),1) # final shape: {batch_size, Nph, Ne}

        S1S2_pdf = S1_pdf * S2_pdf * NphNe_pdf  # final shape: {batch_size, Nph, Ne}
        
        # sum over all Nph, Ne
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=2) # final shape: {batch_size, Nph}
        S1S2_pdf = tf.reduce_sum(S1S2_pdf,axis=1) # final shape: {batch_size,}
        
        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size,}
        S1S2_pdf *= acceptance
     
        S1S2_pdf = tf.repeat(S1S2_pdf[:,o],tf.shape(energy_first)[0],1) 
        S1S2_pdf = tf.repeat(S1S2_pdf[:,:,o],1,1) # final shape: {batch_size,E_bins,1} 

        return tf.transpose(S1S2_pdf, perm=[0, 2, 1]) # initial shape: {batch_size,E_bins,1} --> final shape: {batch_size,1,E_bins} 