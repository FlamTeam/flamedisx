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

###
# filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam.npz'
filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam_20240309.npz'
with np.load(filename) as f:
    fit_values_allkeVnr_allparam = f['fit_values_allkeVnr_allparam']
    
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

    special_model_functions = ('yield_params','quanta_params') # 240305 - AV added new yield model'
                               # signal_means', 'signal_vars', 'signal_corr', 'signal_skews', # 240213 - AV added skew, 240327 - removed old yield model
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    # Whether to check acceptances are positive at the observed events.
    # This is recommended, but you'll have to turn it off if your
    # likelihood includes regions where only anomalous sources make events.
    check_acceptances = True

    # Prevent pycharm warnings:
    source: fd.Source
    gimme: ty.Callable
    gimme_numpy: ty.Callable

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        s1_var_first, s2_var_first = self.gimme_numpy('signal_vars', (s1_mean_first, s2_mean_first))
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second

        s1s2_corr_nr = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_first = s1s2_corr_nr  * np.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * np.sqrt(s1_var_second * s2_var_second)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

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
        
        
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,3300,90), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,600,95), fd.float_type())
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
        energy_second = tf.repeat(energy_second[:,o,:], tf.shape(energy_first[0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None} 
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
    
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
        

        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
              
        
        ###########  # First vertex
        # Load params
        Nph_1_mean, Ne_1_mean = self.gimme('yield_params', 
                                           bonus_arg=energy_first, 
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E1_bins} (matches bonus_arg)
        Fi_1, Fex_1, NBamp_1, NBloc_1, RawSkew_1 = self.gimme('quanta_params', 
                                                              bonus_arg=energy_first, 
                                                              data_tensor=data_tensor,
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
                                           data_tensor=data_tensor,
                                           ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        Fi_2, Fex_2, NBamp_2, NBloc_2, RawSkew_2 = self.gimme('quanta_params', 
                                                              bonus_arg=energy_second, 
                                                              data_tensor=data_tensor,
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
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        ### Calculate probabilities
        probs *= rate_vs_energy # shape: {Nph,Ne,E1_bins,E2_bins}
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

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean_third, s2_mean_third = self.gimme_numpy('signal_means', energies_third)
        s1_mean = s1_mean_first + s1_mean_second + s1_mean_third
        s2_mean = s2_mean_first + s2_mean_second + s2_mean_third

        s1_var_first, s2_var_first = self.gimme_numpy('signal_vars', (s1_mean_first, s2_mean_first))
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var_third, s2_var_third = self.gimme_numpy('signal_vars', (s1_mean_third, s2_mean_third))
        s1_var = s1_var_first + s1_var_second + s1_var_third
        s2_var = s2_var_first + s2_var_second + s2_var_third

        s1s2_corr_nr = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_first = s1s2_corr_nr  * np.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * np.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_nr  * np.sqrt(s1_var_third * s2_var_third)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_others, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_others[0, :,]), axis=2)
        energies_first = tf.repeat(energies_first[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)

        energies_second= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_second= tf.repeat(energies_second[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)[:, :, :, :, o]

        energies_third= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_third= tf.repeat(energies_third[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)[:, :, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s1 = tf.repeat(s1[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)

        s1_mean_first, s2_mean_first = self.gimme('signal_means',
                                                  bonus_arg=energies_first,
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean_third, s2_mean_third = self.gimme('signal_means',
                                                 bonus_arg=energies_third,
                                                 data_tensor=data_tensor,
                                                 ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second + s1_mean_third)
        s2_mean = (s2_mean_first + s2_mean_second + s2_mean_third)

        s1_var_first, s2_var_first = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_first, s2_mean_first),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var_third, s2_var_third = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_third, s2_mean_third),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second + s1_var_third)
        s2_var = (s2_var_first + s2_var_second + s2_var_third)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_corr_nr = self.gimme('signal_corr',
                                  bonus_arg=energies_first,
                                  data_tensor=data_tensor,
                                  ptensor=ptensor)
        s1s2_cov_first = s1s2_corr_nr  * tf.sqrt(s1_var_first * s2_var_first)
        s1s2_cov_second = s1s2_corr_nr  * tf.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_nr  * tf.sqrt(s1_var_third * s2_var_third)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / (s1_std * s2_std)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2E3 = probs * rate_vs_energy[:, :, :, :, o]
        R_E1E2 = tf.reduce_sum(R_E1E2E3, axis=3)
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])


@export
class MakeS1S2SS(MakeS1S2MSU):
    """
    """
    depends_on = ((('energy_first',), 'rate_vs_energy_first'),)

    def _simulate(self, d): 
        energies = d['energy_first'].values
        
        # Load params
        Nph_mean, Ne_mean = self.gimme_numpy('yield_params', energies) # shape: {E_bins} (matches bonus_arg)
        Fi, Fex, NBamp, NBloc, RawSkew = self.gimme_numpy('quanta_params', energies) # shape: {E_bins} (matches bonus_arg)
        
        xcoords_skew2D = tf.stack((Fi, Fex, NBamp, NBloc, RawSkew, energies), axis=-1) # shape: {E_bins, 6}
        xcoords_skew2D = tf.reshape(xcoords_skew2D, [1, -1, 6])                            # shape: {1, E_bins, 6}

        skew2D_model_param = interp_nd(x=xcoords_skew2D) # shape: {E_bins, 7}

        Nph_mean = tf.reshape(Nph_mean, [-1,1])        
        Ne_mean = tf.reshape(Ne_mean, [-1,1])
        Nph_std = tf.sqrt(skew2D_model_param[:,2]**2 / skew2D_model_param[:,0] * Nph_mean[:,0])
        Ne_std = tf.sqrt(skew2D_model_param[:,3]**2 / skew2D_model_param[:,1] * Ne_mean[:,0])
        Nph_skew = skew2D_model_param[:,4]
        Ne_skew = skew2D_model_param[:,5]    
        initial_corr = skew2D_model_param[:,6]

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

        f_mean_x = x_mean
        f_mean_y = y_mean 
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

        S2_loc[np.isnan(S1_scale)] = 0.
        S2_skew[np.isnan(S1_scale)] = 0.
        S2_scale[np.isnan(S1_scale)] = 0.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 0.
         
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
        
        # Quanta Binning
        Nph_edges = tf.cast(tf.linspace(30,1400,90), fd.float_type()) # to save computation time, I only did a rough integration over Nph and Ne
        Ne_edges = tf.cast(tf.linspace(10,320,95), fd.float_type())
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
        
        Fi, Fex, NBamp, NBloc, RawSkew = self.gimme('quanta_params',
                                                     bonus_arg=energy_first, 
                                                     data_tensor=data_tensor,
                                                     ptensor=ptensor) # shape: {E_bins} (matches bonus_arg)
        

        # Current shape for each component: {batch_size, E_centers, None}
        xcoords_skew2D = tf.stack((Fi, Fex, NBamp, NBloc, RawSkew, energy_first), axis=-1) # shape: {E_bins, 6}
        xcoords_skew2D = tf.reshape(xcoords_skew2D, [1, -1, 6])                            # shape: {1, E_bins, 6}
        
        skew2D_model_param = interp_nd(x=xcoords_skew2D) # shape: {E_bins, 7}
        
        Nph_mean = tf.reshape(Nph_mean, [-1,1])        
        Ne_mean = tf.reshape(Ne_mean, [-1,1])
        Nph_std = tf.sqrt(skew2D_model_param[:,2]**2 / skew2D_model_param[:,0] * Nph_mean[:,0])
        Ne_std = tf.sqrt(skew2D_model_param[:,3]**2 / skew2D_model_param[:,1] * Ne_mean[:,0])
        Nph_skew = skew2D_model_param[:,4]
        Ne_skew = skew2D_model_param[:,5]    
        initial_corr = skew2D_model_param[:,6]
        
        
        ### Quanta Production
        x = NphNe[:,:,0]
        x = tf.repeat(x[:,:,o],tf.shape(skew2D_model_param)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        y = NphNe[:,:,1]
        y = tf.repeat(y[:,:,o],tf.shape(skew2D_model_param)[0],axis=2) # final shape: {Nph, Ne, E_bins}

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

        f_mean_x = x_mean
        f_mean_y = y_mean 
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
    special_model_functions = ('signal_means', 'signal_vars',
                               'signal_means_ER', 'signal_vars_ER',
                               'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean = s1_mean_first + s1_mean_second
        s2_mean = s2_mean_first + s2_mean_second

        s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', energies_first)
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var = s1_var_first + s1_var_second
        s2_var = s2_var_first + s2_var_second

        s1s2_corr_second = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_second = s1s2_corr_second  * np.sqrt(s1_var_second * s2_var_second)

        s1s2_cov = s1s2_cov_first + s1s2_cov_second
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_second, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_second[0, :]), axis=2)
        energies_second = tf.repeat(energy_second[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)[:, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_second[0, :]), axis=2)

        s1_mean_first = self.source.s1_mean_ER_tf
        s1_mean_first = tf.repeat(s1_mean_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s2_mean_first = self.source.s2_mean_ER_tf
        s2_mean_first = tf.repeat(s2_mean_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second)
        s2_mean = (s2_mean_first + s2_mean_second)

        s1_var_first = self.source.s1_var_ER_tf
        s1_var_first = tf.repeat(s1_var_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s2_var_first = self.source.s2_var_ER_tf
        s2_var_first = tf.repeat(s2_var_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second)
        s2_var = (s2_var_first + s2_var_second)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_cov_first = self.source.s1s2_cov_ER_tf
        s1s2_cov_first = tf.repeat(s1s2_cov_first[o, :, :], tf.shape(s1)[0], axis=0)[:, :, :, o]
        s1s2_corr_second = self.gimme('signal_corr',
                                      bonus_arg=energies_first,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)
        s1s2_cov_second = s1s2_corr_second  * tf.sqrt(s1_var_second * s2_var_second)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second
        anti_corr = s1s2_cov / (s1_std * s2_std)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2 = probs * rate_vs_energy[:, :, :, o]
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])


@export
class MakeS1S2MigdalMSU(MakeS1S2MSU3):
    """
    """
    special_model_functions = ('signal_means', 'signal_vars',
                               'signal_means_ER', 'signal_vars_ER',
                               'signal_corr')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        energies_third= d['energy_third'].values

        s1_mean_first, s2_mean_first = self.gimme_numpy('signal_means_ER', energies_first)
        s1_mean_second, s2_mean_second = self.gimme_numpy('signal_means', energies_second)
        s1_mean_third, s2_mean_third = self.gimme_numpy('signal_means', energies_third)
        s1_mean = s1_mean_first + s1_mean_second + s1_mean_third
        s2_mean = s2_mean_first + s2_mean_second + s2_mean_third

        s1_var_first, s2_var_first, s1s2_cov_first = self.gimme_numpy('signal_vars_ER', energies_first)
        s1_var_second, s2_var_second = self.gimme_numpy('signal_vars', (s1_mean_second, s2_mean_second))
        s1_var_third, s2_var_third = self.gimme_numpy('signal_vars', (s1_mean_third, s2_mean_third))
        s1_var = s1_var_first + s1_var_second + s1_var_third
        s2_var = s2_var_first + s2_var_second + s2_var_third

        s1s2_corr_others = self.gimme_numpy('signal_corr', energies_first)
        s1s2_cov_second = s1s2_corr_others * np.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_others * np.sqrt(s1_var_third * s2_var_third)

        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies_first))
        Y = np.random.normal(size=len(energies_first))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domains and values
                 energy_first, rate_vs_energy_first,
                 energy_others, rate_vs_energy):
        energies_first = tf.repeat(energy_first[:, :, o], tf.shape(energy_others[0, :,]), axis=2)
        energies_first = tf.repeat(energies_first[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)

        energies_second= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_second= tf.repeat(energies_second[:, :, :, o], tf.shape(energy_others[0, :]), axis=3)[:, :, :, :, o]

        energies_third= tf.repeat(energy_others[:, o, :], tf.shape(energy_first[0, :, 0]), axis=1)
        energies_third= tf.repeat(energies_third[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)[:, :, :, :, o]

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1 = tf.repeat(s1[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s1 = tf.repeat(s1[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)
        s2 = tf.repeat(s2[:, :, o, :], tf.shape(energy_others[0, :]), axis=2)
        s2 = tf.repeat(s2[:, :, :, o, :], tf.shape(energy_others[0, :]), axis=3)

        s1_mean_first = self.source.s1_mean_ER_tf
        s1_mean_first = tf.repeat(s1_mean_first[o, :, :], tf.shape(s1)[0], axis=0)
        s1_mean_first = tf.repeat(s1_mean_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s2_mean_first = self.source.s2_mean_ER_tf
        s2_mean_first = tf.repeat(s2_mean_first[o, :, :], tf.shape(s1)[0], axis=0)
        s2_mean_first = tf.repeat(s2_mean_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s1_mean_second, s2_mean_second = self.gimme('signal_means',
                                                    bonus_arg=energies_second,
                                                    data_tensor=data_tensor,
                                                    ptensor=ptensor)
        s1_mean_third, s2_mean_third = self.gimme('signal_means',
                                                 bonus_arg=energies_third,
                                                 data_tensor=data_tensor,
                                                 ptensor=ptensor)
        s1_mean = (s1_mean_first + s1_mean_second + s1_mean_third)
        s2_mean = (s2_mean_first + s2_mean_second + s2_mean_third)

        s1_var_first = self.source.s1_var_ER_tf
        s1_var_first = tf.repeat(s1_var_first[o, :, :], tf.shape(s1)[0], axis=0)
        s1_var_first = tf.repeat(s1_var_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s2_var_first = self.source.s2_var_ER_tf
        s2_var_first = tf.repeat(s2_var_first[o, :, :], tf.shape(s1)[0], axis=0)
        s2_var_first = tf.repeat(s2_var_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s1_var_second, s2_var_second = self.gimme('signal_vars',
                                                  bonus_arg=(s1_mean_second, s2_mean_second),
                                                  data_tensor=data_tensor,
                                                  ptensor=ptensor)
        s1_var_third, s2_var_third = self.gimme('signal_vars',
                                                bonus_arg=(s1_mean_third, s2_mean_third),
                                                data_tensor=data_tensor,
                                                ptensor=ptensor)
        s1_var = (s1_var_first + s1_var_second + s1_var_third)
        s2_var = (s2_var_first + s2_var_second + s2_var_third)

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_cov_first = self.source.s1s2_cov_ER_tf
        s1s2_cov_first = tf.repeat(s1s2_cov_first[o, :, :], tf.shape(s1)[0], axis=0)
        s1s2_cov_first = tf.repeat(s1s2_cov_first[:, :, :, o], tf.shape(s1)[3], axis=3)[:, :, :, :, o]
        s1s2_corr_others = self.gimme('signal_corr',
                                      bonus_arg=energies_first,
                                      data_tensor=data_tensor,
                                      ptensor=ptensor)
        s1s2_cov_second = s1s2_corr_others * tf.sqrt(s1_var_second * s2_var_second)
        s1s2_cov_third = s1s2_corr_others  * tf.sqrt(s1_var_third * s2_var_third)
        s1s2_cov = s1s2_cov_first + s1s2_cov_second + s1s2_cov_third
        anti_corr = s1s2_cov / (s1_std * s2_std)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        R_E1E2E3 = probs * rate_vs_energy[:, :, :, :, o]
        R_E1E2 = tf.reduce_sum(R_E1E2E3, axis=3)
        R_E1 = tf.reduce_sum(R_E1E2, axis=2)

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(R_E1)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(R_E1)[2], axis=2)
        R_E1 *= acceptance

        return tf.transpose(R_E1, perm=[0, 2, 1])


@export
class MakeS1S2ER(MakeS1S2SS):
    """
    """
    special_model_functions = ('signal_means_ER', 'signal_vars_ER')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies = d['energy_first'].values

        s1_mean, s2_mean = self.gimme_numpy('signal_means_ER', energies)
        s1_var, s2_var, s1s2_cov = self.gimme_numpy('signal_vars_ER', energies)
        anti_corr = s1s2_cov / np.sqrt(s1_var * s2_var)

        X = np.random.normal(size=len(energies))
        Y = np.random.normal(size=len(energies))

        d['s1'] = np.sqrt(s1_var) * X + s1_mean
        d['s2'] = np.sqrt(s2_var) * (anti_corr * X + np.sqrt(1. - anti_corr * anti_corr) * Y) + s2_mean

        d['p_accepted'] *= self.gimme_numpy('s1s2_acceptance')

    def _compute(self,
                 data_tensor, ptensor,
                 # Domain
                 s1,
                 # Dependency domain and value
                 energy_first, rate_vs_energy_first):

        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor)
        s2 = tf.repeat(s2[:, o], tf.shape(s1)[1], axis=1)
        s2 = tf.repeat(s2[:, :, o], tf.shape(s1)[2], axis=2)

        s1_mean = self.source.s1_mean_ER_tf
        s1_mean = tf.repeat(s1_mean[o, :], tf.shape(s1)[0], axis=0)[:, :, o]
        s2_mean = self.source.s2_mean_ER_tf
        s2_mean = tf.repeat(s2_mean[o, :], tf.shape(s1)[0], axis=0)[:, :, o]

        s1_var = self.source.s1_var_ER_tf
        s1_var = tf.repeat(s1_var[o,:], tf.shape(s1)[0], axis=0)[:, :, o]
        s2_var = self.source.s2_var_ER_tf
        s2_var = tf.repeat(s2_var[o,:], tf.shape(s1)[0], axis=0)[:, :, o]

        s1_std = tf.sqrt(s1_var)
        s2_std = tf.sqrt(s2_var)

        s1s2_cov = self.source.s1s2_cov_ER_tf
        s1s2_cov = tf.repeat(s1s2_cov[o,:], tf.shape(s1)[0], axis=0)[:, :, o]
        anti_corr = s1s2_cov / (s1_std * s2_std)

        denominator = 2. * pi * s1_std * s2_std * tf.sqrt(1. - anti_corr * anti_corr)

        exp_prefactor = -1. / (2 * (1. - anti_corr * anti_corr))

        exp_term_1 = (s1 - s1_mean) * (s1 - s1_mean) / s1_var
        exp_term_2 = (s2 - s2_mean) * (s2 - s2_mean) / s2_var
        exp_term_3 = -2. * anti_corr * (s1 - s1_mean) * (s2 - s2_mean) / (s1_std * s2_std)

        probs = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3))

        # Add detection/selection efficiency
        acceptance = self.gimme('s1s2_acceptance',
                                data_tensor=data_tensor, ptensor=ptensor)
        acceptance = tf.repeat(acceptance[:, o], tf.shape(probs)[1], axis=1)
        acceptance = tf.repeat(acceptance[:, :, o], tf.shape(probs)[2], axis=2)
        probs *= acceptance

        return tf.transpose(probs, perm=[0, 2, 1])
