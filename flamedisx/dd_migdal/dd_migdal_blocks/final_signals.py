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


###################################################################################################

###################################################################################################



### interpolation grids for NR
filename = '/global/cfs/cdirs/lz/users/cding/studyNEST_skew2D_notebooks/fit_values_allkeVnr_allparam_20240705.npz'
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

    Fi_grid = tf.cast([0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.55,0.75,1.], fd.float_type())                                                    # Fano ion
    Fex_grid = tf.cast([0.55,0.75,1.,1.5,2.0,2.5,3.0,3.5,4.0,6.0,8.0,12.0,16.0], fd.float_type())   # Fano exciton
    NBamp_grid = tf.cast([0.,0.02,0.04,0.06,0.08], fd.float_type())                                              # amplitude for non-binomial NR recombination fluctuations
    NBloc = tf.cast([0.4,0.45,0.5,0.55,0.6], fd.float_type())                                                              # non-binomial: loc of elecfrac
    RawSkew_grid = tf.cast([-8.,-5.,-3.,-1.5,0.,1.5,3.,5.,8.], fd.float_type())                                                                 # raw skewness
    
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

    special_model_functions = ('yield_and_quanta_params','detector_params')
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
        
        ##### First Vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph1 = x1*scale1 + loc1
        Ne1 = x2*scale2 + loc2
        
        Nph1[Nph1<=0]=0.1
        Ne1[Ne1<=0]=0.1
        
        
        
        ##### Second Vertex
        # Load params        
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_second) # shape: {E2_bins} (matches bonus_arg)
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph2 = x1*scale1 + loc1
        Ne2 = x2*scale2 + loc2
        
        Nph2[Nph2<=0]=0.1
        Ne2[Ne2<=0]=0.1        
        
        Nph = Nph1 + Nph2
        Ne = Ne1 + Ne2
        NphNe_stack = tf.ragged.stack([Nph.astype('float32'),Ne.astype('float32')])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme_numpy('detector_params',NphNe_stack) # shape: {Nph} 


        S1_mean = Nph*g1     
        S1_std = np.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = np.sqrt(S2_mean*S2_fano)
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi))
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi))
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi))
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi))

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
        
        x0 = np.random.normal(size=len(S1_mean))
        x1 = np.random.normal(size=len(S1_mean))
        y = (S1_skew * np.abs(x0) + x1) / np.sqrt(1+S1_skew*S1_skew)
        S1_sample = y*S1_scale + S1_loc
        
        x0 = np.random.normal(size=len(S2_mean))
        x1 = np.random.normal(size=len(S2_mean))
        y = (S2_skew * np.abs(x0) + x1) / np.sqrt(1+S2_skew*S2_skew)
        S2_sample = y*S2_scale + S2_loc
        
        S1_sample[S1_sample<=0]=1.
        S2_sample[S2_sample<=0]=1.
        
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
        # Quanta Binning
        Nph_bw = 14.0
        Ne_bw = 4.0
        Nph_edges = tf.cast(tf.range(0,4500,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,800,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)       

        # Energy binning
        energy_second = tf.repeat(energy_second[:,o,:], tf.shape(energy_first[0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E1_bins, None} 
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}

        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
        
        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
        
        # Load params
        # for MSU2, use Nph_mean to represent both Nph_1_mean and Nph_2_mean
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme('yield_and_quanta_params',
                                                                                         bonus_arg=energy_first, 
                                                                                         data_tensor=data_tensor,
                                                                                         ptensor=ptensor) # shape: {E_bins} (matches bonus_arg)
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}  # FFT: for multipole recoils are of the same type

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT: for multipole recoils are of the same type
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs = probs*Nph_bw*Ne_bw
        probs *= 1/tf.reduce_sum(probs,axis=[0,1]) # normalize the probability for each recoil energy
        
        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_1_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))

        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne} 
        NphNe_all_pdf_12 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_pdf = tf.einsum('ijkl,ij->kl',NphNe_all_pdf_12,rate_vs_energy)

        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_pdf = NphNe_pdf[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
        NphNe_stack = tf.ragged.stack([Nph,Ne])
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme('detector_params',
                                                                  bonus_arg=NphNe_stack,
                                                                  data_tensor=data_tensor,
                                                                  ptensor=ptensor) # shape: {integer} 
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}        
                    
        S1_mean = Nph*g1 # shape: {Nph}
        S1_std = tf.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2 # shape: {Ne}
        S2_std = tf.sqrt(S2_mean*S2_fano)
             
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
        
        ##### First vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_first) # shape: {E1_bins} (matches bonus_arg)

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph1 = x1*scale1 + loc1
        Ne1 = x2*scale2 + loc2
        
        Nph1[Nph1<=0]=0.1
        Ne1[Ne1<=0]=0.1
        
        ##### Second vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_second) # shape: {E2_bins} (matches bonus_arg)

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph2 = x1*scale1 + loc1
        Ne2 = x2*scale2 + loc2
        
        Nph2[Nph2<=0]=0.1
        Ne2[Ne2<=0]=0.1            
        
        ##### Third vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_third) # shape: {E3_bins} (matches bonus_arg)       

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph3 = x1*scale1 + loc1
        Ne3 = x2*scale2 + loc2
        
        Nph3[Nph3<=0]=0.1
        Ne3[Ne3<=0]=0.1        
        
        Nph = Nph1 + Nph2 + Nph3
        Ne = Ne1 + Ne2 + Ne3
        NphNe_stack = tf.ragged.stack([Nph.astype('float32'),Ne.astype('float32')])

        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme_numpy('detector_params',NphNe_stack) # shape: {integer} 
        
        S1_mean = Nph*g1     
        S1_std = np.sqrt(S1_mean*S1_fano)
        
        S2_mean = Ne*g2
        S2_std = np.sqrt(S2_mean*S2_fano)
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi))
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi))
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi))
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi))

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        x0 = np.random.normal(size=len(S1_mean))
        x1 = np.random.normal(size=len(S1_mean))
        y = (S1_skew * np.abs(x0) + x1) / np.sqrt(1+S1_skew*S1_skew)
        S1_sample = y*S1_scale + S1_loc
        
        x0 = np.random.normal(size=len(S2_mean))
        x1 = np.random.normal(size=len(S2_mean))
        y = (S2_skew * np.abs(x0) + x1) / np.sqrt(1+S2_skew*S2_skew)
        S2_sample = y*S2_scale + S2_loc

        S1_sample[S1_sample<=0]=1.
        S2_sample[S2_sample<=0]=1.        
        
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
        Nph_bw = 25.0
        Ne_bw = 5.0
        Nph_edges = tf.cast(tf.range(0,5000,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,800,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)       
        
        # Energy binning
        energy_second = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None} 
        energy_third = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,0,:]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E3_bins, None} 
        
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
    
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        energy_third = fd.np_to_tf(energy_third)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E3_bins} 
        rate_vs_energy_third = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   

        # Load params
        # for MSU2, use Nph_mean to represent both Nph_1_mean and Nph_2_mean
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme('yield_and_quanta_params',
                                                                                         bonus_arg=energy_first, 
                                                                                         data_tensor=data_tensor,
                                                                                         ptensor=ptensor) # shape: {E_bins} (matches bonus_arg)
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}  # FFT: for multipole recoils are of the same type

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT: for multipole recoils are of the same type
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs = probs*Nph_bw*Ne_bw
        probs *= 1/tf.reduce_sum(probs,axis=[0,1]) # normalize the probability for each recoil energy
        
        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_1_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))
        NphNe_all_pdf_2_tp_fft2d = NphNe_all_pdf_1_tp_fft2d
        NphNe_all_pdf_3_tp_fft2d = NphNe_all_pdf_1_tp_fft2d
    
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}        
        
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_3_tp_fft2d[o,:,:,:],tf.shape(energy_second)[0],axis=0) # final shape: {E2_bins, E3_bins, Nph, Ne}
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_3_tp_fft2d_repeat[o,:,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_123 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat*NphNe_all_pdf_3_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}
        NphNe_pdf = tf.einsum('ijklm,ijk->lm',NphNe_all_pdf_123,rate_vs_energy)

        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_pdf = NphNe_pdf[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
        NphNe_stack = tf.ragged.stack([Nph,Ne])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme('detector_params',
                                                                  bonus_arg=NphNe_stack,
                                                                  data_tensor=data_tensor,
                                                                  ptensor=ptensor) # shape: {integer}  # shape: {integer} 
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}        
                    
        S1_mean = Nph*g1 # shape: {Nph}
        S1_std = tf.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2 # shape: {Ne}
        S2_std = tf.sqrt(S2_mean*S2_fano)

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
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies) # shape: {E_bins} (matches bonus_arg) 

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph = x1*scale1 + loc1
        Ne = x2*scale2 + loc2
        
        Nph[Nph<=0]=0.1
        Ne[Ne<=0]=0.1
        NphNe_stack = tf.ragged.stack([Nph.astype('float32'),Ne.astype('float32')])

        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme_numpy('detector_params',NphNe_stack) # shape: {integer} 
        

        S1_mean = Nph*g1     
        S1_std = np.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = np.sqrt(S2_mean*S2_fano)

        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi))
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi))
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi))
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi))

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        x0 = np.random.normal(size=len(S1_mean))
        x1 = np.random.normal(size=len(S1_mean))
        y = (S1_skew * np.abs(x0) + x1) / np.sqrt(1+S1_skew*S1_skew)
        S1_sample = y*S1_scale + S1_loc
        
        x0 = np.random.normal(size=len(S2_mean))
        x1 = np.random.normal(size=len(S2_mean))
        y = (S2_skew * np.abs(x0) + x1) / np.sqrt(1+S2_skew*S2_skew)
        S2_sample = y*S2_scale + S2_loc
        
        
        S1_sample[S1_sample<=0]=1.
        S2_sample[S2_sample<=0]=1.
        
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
        Nph_bw = 20. # 240426 AV increased from 10.0
        Ne_bw = 4.0  #I would like to reduce the Nph, Ne bin width for SS for a few reasons: We have lots of SS events, they are our dominant background. Having a better precision improves convergence speed, and it will not cost computation speed because it has only 1 vertex (dimension is low).
        
        Nph_edges = tf.cast(tf.range(0,2500,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,500,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
        
            
        # Energy binning
        energy_first = fd.np_to_tf(energy_first)[0,:,0]               # inital shape: {batch_size, E_bins, None} --> final shape: {E_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E_bins} --> final shape: {E_bins}
        rate_vs_energy_first = rate_vs_energy_first/tf.reduce_sum(rate_vs_energy_first) # 240416 AV added to ensure ER/NR SS is normalized probably
        
        
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme('yield_and_quanta_params',
                                                                                         bonus_arg=energy_first, 
                                                                                         data_tensor=data_tensor,
                                                                                         ptensor=ptensor) # shape: {E_bins} (matches bonus_arg)   
        
        ### Quanta Production
        x = NphNe[:,:,0]
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        y = NphNe[:,:,1]
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        probs = probs*Nph_bw*Ne_bw
        probs *= 1/tf.reduce_sum(probs,axis=[0,1]) # normalize the probability for each recoil energy                
        probs *= rate_vs_energy_first # shape: {Nph,Ne,E_bins,} {Nph,Ne,E1_bins,E2_bins}
        NphNe_pdf = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}        
        
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_pdf = NphNe_pdf[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
        NphNe_stack = tf.ragged.stack([Nph,Ne])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme('detector_params',
                                                                  bonus_arg=NphNe_stack,
                                                                  data_tensor=data_tensor,
                                                                  ptensor=ptensor) # shape: {integer} 
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}     

        S1_mean = Nph*g1     
        S1_std = tf.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = tf.sqrt(S2_mean*S2_fano)
        
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
    special_model_functions = ('yield_and_quanta_params','detector_params',
                               'quanta_params_ER','quanta_params_ER_mig')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        
        # Load params
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('quanta_params_ER_mig', energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_std = np.sqrt(Nph_fano * Nph_mean)
        Ne_std = np.sqrt(Ne_fano * Ne_mean) 
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph1 = x1*scale1 + loc1
        Ne1 = x2*scale2 + loc2
        
        
        Nph1[Nph1<=0]=0.1
        Ne1[Ne1<=0]=0.1
        
        ##### Second Vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_second) # shape: {E2_bins} (matches bonus_arg )
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph2 = x1*scale1 + loc1
        Ne2 = x2*scale2 + loc2
        
        Nph2[Nph2<=0]=0.1
        Ne2[Ne2<=0]=0.1        
        
        Nph = Nph1 + Nph2
        Ne = Ne1 + Ne2
        NphNe_stack = tf.ragged.stack([Nph.astype('float32'),Ne.astype('float32')])
   
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme_numpy('detector_params',NphNe_stack) # shape: {integer} 

        
        S1_mean = Nph*g1     
        S1_std = np.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = np.sqrt(S2_mean*S2_fano)
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi))
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi))
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi))
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi))

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
        
        x0 = np.random.normal(size=len(S1_mean))
        x1 = np.random.normal(size=len(S1_mean))
        y = (S1_skew * np.abs(x0) + x1) / np.sqrt(1+S1_skew*S1_skew)
        S1_sample = y*S1_scale + S1_loc
        
        x0 = np.random.normal(size=len(S2_mean))
        x1 = np.random.normal(size=len(S2_mean))
        y = (S2_skew * np.abs(x0) + x1) / np.sqrt(1+S2_skew*S2_skew)
        S2_sample = y*S2_scale + S2_loc
        
        S1_sample[S1_sample<=0]=1.
        S2_sample[S2_sample<=0]=1.
        
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
        Nph_bw = 30.0 #  240507 AV changed from 14
        Ne_bw = 10.0   #  240507 AV changed from 4
        Nph_edges = tf.cast(tf.range(0,5000,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1) # 240507 AV changed from 3500
        Ne_edges = tf.cast(tf.range(0,1500,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)     # 240507 AV changed from 600
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)       

        # Energy binning
        energy_second = tf.repeat(energy_second[:,o,:], tf.shape(energy_first[0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E1_bins, None} 
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   

        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:]      # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E1_bins, E2_bins}   
        
        ###########  # First vertex
        # Load params
        Nph_mean = self.source.Nph_mean_ER_tf
        Ne_mean = self.source.Ne_mean_ER_tf
        Nph_std = self.source.Nph_std_ER_tf
        Ne_std = self.source.Ne_std_ER_tf
        Nph_skew = self.source.Nph_skew_ER_tf
        Ne_skew = self.source.Ne_skew_ER_tf
        initial_corr = self.source.initial_corr_ER_tf
        
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT 1st recoil: for multiple recoils of different type, e.g. 1ER+1NR

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT 1st recoil: for multiple recoils of different type, e.g. 1ER+1NR
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs_1 = probs*Nph_bw*Ne_bw
        probs_1 *= 1/tf.reduce_sum(probs_1,axis=[0,1]) # normalize the probability for each recoil energy        
        
        ###########  # Second vertex
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme('yield_and_quanta_params',
                                                                                         bonus_arg=energy_second, 
                                                                                         data_tensor=data_tensor,
                                                                                         ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_second)[0],axis=2) # final shape: {Nph, Ne, E2_bins} # FFT 2nd recoil: for multiple recoils of different type, e.g. 1ER+1NR

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_second)[0],axis=2) # final shape: {Nph, Ne, E2_bins} # FFT 2nd recoil: for multiple recoils of different type, e.g. 1ER+1NR

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs_2 = probs*Nph_bw*Ne_bw
        probs_2 *= 1/tf.reduce_sum(probs_2,axis=[0,1]) # normalize the probability for each recoil energy
        
        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs_1, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_2_tp = tf.transpose(probs_2, perm=[2,0,1], conjugate=False) 
        
        NphNe_all_pdf_1_tp_fft2d = tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))
        NphNe_all_pdf_2_tp_fft2d = tf.signal.fft2d(tf.cast(NphNe_all_pdf_2_tp,tf.complex64))
        
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne} 
        NphNe_all_pdf_12 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_pdf = tf.einsum('ijkl,ij->kl',NphNe_all_pdf_12,rate_vs_energy)
        
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_pdf = NphNe_pdf[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
        NphNe_stack = tf.ragged.stack([Nph,Ne])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme('detector_params',
                                                                  bonus_arg=NphNe_stack,
                                                                  data_tensor=data_tensor,
                                                                  ptensor=ptensor) # shape: {integer} 
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}        
                    
        S1_mean = Nph*g1 # shape: {Nph}
        S1_std = tf.sqrt(S1_mean*S1_fano)
        
        S2_mean = Ne*g2 # shape: {Ne}
        S2_std = tf.sqrt(S2_mean*S2_fano)

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
    special_model_functions = ('yield_and_quanta_params','detector_params',
                               'quanta_params_ER','quanta_params_ER_mig')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies_first = d['energy_first'].values
        energies_second = d['energy_second'].values
        energies_third= d['energy_third'].values        
        
        ##### First Vertex
        # Load params
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('quanta_params_ER_mig', energies_first) # shape: {E1_bins} (matches bonus_arg)
        
        Nph_std = np.sqrt(Nph_fano * Nph_mean)
        Ne_std = np.sqrt(Ne_fano * Ne_mean) 

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph1 = x1*scale1 + loc1
        Ne1 = x2*scale2 + loc2
        
        Nph1[Nph1<=0]=0.1
        Ne1[Ne1<=0]=0.1
        
        ##### Second Vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_second) # shape: {E2_bins} (matches bonus_arg ) 
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph2 = x1*scale1 + loc1
        Ne2 = x2*scale2 + loc2
        
        Nph2[Nph2<=0]=0.1
        Ne2[Ne2<=0]=0.1                
        
        ##### Third Vertex
        # Load params
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('yield_and_quanta_params',energies_third) # shape: {E3_bins} (matches bonus_arg)
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph3 = x1*scale1 + loc1
        Ne3 = x2*scale2 + loc2
        
        Nph3[Nph3<=0]=0.1
        Ne3[Ne3<=0]=0.1        
        
        Nph = Nph1 + Nph2 + Nph3
        Ne = Ne1 + Ne2 + Ne3
        NphNe_stack = tf.ragged.stack([Nph.astype('float32'),Ne.astype('float32')])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme_numpy('detector_params',NphNe_stack) # shape: {integer} 
        

        S1_mean = Nph*g1     
        S1_std = np.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = np.sqrt(S2_mean*S2_fano)
        
        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi))
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi))
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi))
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi))

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
         
        x0 = np.random.normal(size=len(S1_mean))
        x1 = np.random.normal(size=len(S1_mean))
        y = (S1_skew * np.abs(x0) + x1) / np.sqrt(1+S1_skew*S1_skew)
        S1_sample = y*S1_scale + S1_loc
        
        x0 = np.random.normal(size=len(S2_mean))
        x1 = np.random.normal(size=len(S2_mean))
        y = (S2_skew * np.abs(x0) + x1) / np.sqrt(1+S2_skew*S2_skew)
        S2_sample = y*S2_scale + S2_loc

        S1_sample[S1_sample<=0]=1.
        S2_sample[S2_sample<=0]=1.        
        
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
        Nph_bw = 30.0
        Ne_bw = 7.0
        Nph_edges = tf.cast(tf.range(0,5000,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,800,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1]) # shape (Nph)
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1]) # shape (Ne) 
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges) # shape (Nph)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges) # shape (Ne)
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij') #shape (Nph,Ne)
        NphNe = tf.stack((tempX, tempY),axis=2) #shape (Nph,Ne,2)       

        # Energy binning
        energy_second = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,:,0]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E2_bins, None} 
        energy_third = tf.repeat(energy_others[:,o,:], tf.shape(rate_vs_energy[0,0,0,:]),axis=1) # inital shape: {batch_size, None} --> final shape: {batch_size, E3_bins, None} 
        
        energy_first = fd.np_to_tf(energy_first)[0,:,0]                 # inital shape: {batch_size, E1_bins, None} --> final shape: {E1_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E1_bins} --> final shape: {E1_bins}
    
        
        energy_second = fd.np_to_tf(energy_second)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E2_bins} 
        rate_vs_energy_second = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        energy_third = fd.np_to_tf(energy_third)[0,0,:]               # inital shape: {batch_size, E1_bins, E2_bins} --> final shape: {E3_bins} 
        rate_vs_energy_third = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        rate_vs_energy = fd.np_to_tf(rate_vs_energy)[0,:,:,:]      # inital shape: {batch_size, E1_bins, E2_bins, E3_bins} --> final shape: {E1_bins, E2_bins, E3_bins}   
        
        
        ###########  # First vertex
        # Load params
        Nph_mean = self.source.Nph_mean_ER_tf
        Ne_mean = self.source.Ne_mean_ER_tf
        Nph_std = self.source.Nph_std_ER_tf
        Ne_std = self.source.Ne_std_ER_tf
        Nph_skew = self.source.Nph_skew_ER_tf
        Ne_skew = self.source.Ne_skew_ER_tf
        initial_corr = self.source.initial_corr_ER_tf
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT 1st recoil: for multiple recoils of different type, e.g. 1ER+1NR

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT 1st recoil: for multiple recoils of different type, e.g. 1ER+1NR
        
        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs_1 = probs*Nph_bw*Ne_bw
        probs_1 *= 1/tf.reduce_sum(probs_1,axis=[0,1]) # normalize the probability for each recoil energy        

        ###########  # Second vertex
        Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme('yield_and_quanta_params',
                                                                                         bonus_arg=energy_second, 
                                                                                         data_tensor=data_tensor,
                                                                                         ptensor=ptensor) # shape: {E2_bins} (matches bonus_arg)
        
        ### Quanta Production
        x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
        x = tf.repeat(x[:,:,o],tf.shape(energy_second)[0],axis=2) # final shape: {Nph, Ne, E2_bins} # FFT 2nd recoil: for multiple recoils of different type, e.g. 1ER+1NR

        y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
        y = tf.repeat(y[:,:,o],tf.shape(energy_second)[0],axis=2) # final shape: {Nph, Ne, E2_bins} # FFT 2nd recoil: for multiple recoils of different type, e.g. 1ER+1NR

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs_2 = probs*Nph_bw*Ne_bw
        probs_2 *= 1/tf.reduce_sum(probs_2,axis=[0,1]) # normalize the probability for each recoil energy
        
        if 0: # Do not calculate the thrid vetex because it is same with the second.
            ###########  # Third vertex
            Nph_mean, Ne_mean, Nph_std, Ne_std, Nph_skew, Ne_skew, initial_corr = self.gimme('yield_and_quanta_params',
                                                                                         bonus_arg=energy_third, 
                                                                                         data_tensor=data_tensor,
                                                                                         ptensor=ptensor) # shape: {E3_bins} (matches bonus_arg) 

            ### Quanta Production
            x = NphNe[:,:,0] # Nph counts --> final shape: {Nph, Ne}
            #x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins}  # FFT: for multipole recoils are of the same type
            #x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT 1st recoil: for multiple recoils of different type, e.g. 1ER+1NR
            #x = tf.repeat(x[:,:,o],tf.shape(energy_second)[0],axis=2) # final shape: {Nph, Ne, E2_bins} # FFT 2nd recoil: for multiple recoils of different type, e.g. 1ER+1NR
            x = tf.repeat(x[:,:,o],tf.shape(energy_third)[0],axis=2) # final shape: {Nph, Ne, E3_bins} # FFT 3rd recoil: for multiple recoils of different type, e.g. 1ER+1NR
            #x = tf.repeat(x[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins} # not FFT: use it for approximation method only.

            y = NphNe[:,:,1] # Ne counts --> final shape: {Nph, Ne}
            #y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT: for multipole recoils are of the same type
            #y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E1_bins} # FFT 1st recoil: for multiple recoils of different type, e.g. 1ER+1NR
            #y = tf.repeat(y[:,:,o],tf.shape(energy_second)[0],axis=2) # final shape: {Nph, Ne, E2_bins} # FFT 2nd recoil: for multiple recoils of different type, e.g. 1ER+1NR
            y = tf.repeat(y[:,:,o],tf.shape(energy_third)[0],axis=2) # final shape: {Nph, Ne, E3_bins} # FFT 3rd recoil: for multiple recoils of different type, e.g. 1ER+1NR
            #y = tf.repeat(y[:,:,:,o],tf.shape(energy_second)[0],axis=3) # final shape: {Nph, Ne, E1_bins, E2_bins} # not FFT: use it for approximation method only.

            bCa1 = Nph_skew + initial_corr * Ne_skew
            bCa2 = initial_corr * Nph_skew + Ne_skew
            aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
            delta1 = (1. / tf.sqrt(aCa)) * bCa1
            delta2 = (1. / tf.sqrt(aCa)) * bCa2
            scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
            scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
            loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
            loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 

            # multivariate normal
            # f(x,y) = ...
            # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
            denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
            exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
            exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
            exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
            exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

            mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

            # 1d norm cdf
            # Phi(x) = (1 + Erf(x/sqrt(2)))/2
            Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
            Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

            norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

            # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
            probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)     
            probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)

            probs_3 = probs*Nph_bw*Ne_bw
            probs_3 *= 1/tf.reduce_sum(probs_3,axis=[0,1]) # normalize the probability for each recoil energy

        # FFT convolution 
        NphNe_all_pdf_1_tp = tf.transpose(probs_1, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_1_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_1_tp,tf.complex64))
        NphNe_all_pdf_2_tp = tf.transpose(probs_2, perm=[2,0,1], conjugate=False) 
        NphNe_all_pdf_2_tp_fft2d =  tf.signal.fft2d(tf.cast(NphNe_all_pdf_2_tp,tf.complex64))
        NphNe_all_pdf_3_tp_fft2d =  NphNe_all_pdf_2_tp_fft2d
        
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d[:,o,:,:],tf.shape(energy_second)[0],axis=1) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_1_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_1_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}        
        
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d[o,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, Nph, Ne}
        NphNe_all_pdf_2_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_2_tp_fft2d_repeat[:,:,o,:,:],tf.shape(energy_third)[0],axis=2) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_3_tp_fft2d[o,:,:,:],tf.shape(energy_second)[0],axis=0) # final shape: {E2_bins, E3_bins, Nph, Ne}
        NphNe_all_pdf_3_tp_fft2d_repeat = tf.repeat(NphNe_all_pdf_3_tp_fft2d_repeat[o,:,:,:,:],tf.shape(energy_first)[0],axis=0) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}     
        
        NphNe_all_pdf_123 = tf.math.real(tf.signal.ifft2d(NphNe_all_pdf_1_tp_fft2d_repeat*NphNe_all_pdf_2_tp_fft2d_repeat*NphNe_all_pdf_3_tp_fft2d_repeat)) # final shape: {E1_bins, E2_bins, E3_bins, Nph, Ne}
        NphNe_pdf = tf.einsum('ijklm,ijk->lm',NphNe_all_pdf_123,rate_vs_energy)
                    
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_pdf = NphNe_pdf[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
        NphNe_stack = tf.ragged.stack([Nph,Ne])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme('detector_params',
                                                                  bonus_arg=NphNe_stack,
                                                                  data_tensor=data_tensor,
                                                                  ptensor=ptensor) # shape: {integer} 
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}        
                    
        S1_mean = Nph*g1 # shape: {Nph}
        S1_std = tf.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2 # shape: {Ne}
        S2_std = tf.sqrt(S2_mean*S2_fano)

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
    special_model_functions = ('quanta_params_ER', 'detector_params')
    model_functions = ('get_s2', 's1s2_acceptance',) + special_model_functions

    def _simulate(self, d):
        energies = d['energy_first'].values
        
        # Load params
        Nph_mean, Ne_mean, Nph_fano, Ne_fano, Nph_skew, Ne_skew, initial_corr = self.gimme_numpy('quanta_params_ER', energies) # shape: {E_bins} (matches bonus_arg)
        
        Nph_std = np.sqrt(Nph_fano * Nph_mean)
        Ne_std = np.sqrt(Ne_fano * Ne_mean) 

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / np.sqrt(aCa)) * bCa1
        delta2 = (1. / np.sqrt(aCa)) * bCa2
        scale1 = Nph_std / np.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / np.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * np.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * np.sqrt(2/pi)

        a = np.sqrt(1 - delta1*delta1)
        b = (initial_corr - delta1*delta2) / a
        c = np.sqrt(1 - delta2*delta2 - b*b)

        x0 = np.random.normal(size=len(Nph_mean))
        Y = np.random.normal(size=len(Nph_mean))
        Z = np.random.normal(size=len(Nph_mean))

        x1 = np.array(delta1*x0 + a*Y)
        x2 = np.array(delta2*x0 + b*Y + c*Z)

        inds = (x0 <= 0)
        x1[inds] = -1 * x1[inds]
        x2[inds] = -1 * x2[inds]

        Nph = x1*scale1 + loc1
        Ne = x2*scale2 + loc2
        
        Nph[Nph<=0]=0.1
        Ne[Ne<=0]=0.1
        NphNe_stack = tf.ragged.stack([Nph.astype('float32'),Ne.astype('float32')])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme_numpy('detector_params',NphNe_stack) # shape: {integer} 
        

        S1_mean = Nph*g1     
        S1_std = np.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = np.sqrt(S2_mean*S2_fano)

        S1_delta = (S1_skew / np.sqrt(1 + S1_skew**2))
        S1_scale = (S1_std / np.sqrt(1. - 2 * S1_delta**2 / np.pi))
        S1_loc = (S1_mean - S1_scale * S1_delta * np.sqrt(2/np.pi))
        
        S2_delta = (S2_skew / np.sqrt(1 + S2_skew**2))
        S2_scale = (S2_std / np.sqrt(1. - 2 * S2_delta**2 / np.pi))
        S2_loc = (S2_mean - S2_scale * S2_delta * np.sqrt(2/np.pi))

        S2_loc[np.isnan(S2_scale)] = 0.
        S2_skew[np.isnan(S2_scale)] = 0.
        S2_scale[np.isnan(S2_scale)] = 1.
        
        S1_loc[np.isnan(S1_scale)] = 0.
        S1_skew[np.isnan(S1_scale)] = 0.
        S1_scale[np.isnan(S1_scale)] = 1.
        
        x0 = np.random.normal(size=len(S1_mean))
        x1 = np.random.normal(size=len(S1_mean))
        y = (S1_skew * np.abs(x0) + x1) / np.sqrt(1+S1_skew*S1_skew)
        S1_sample = y*S1_scale + S1_loc
        
        x0 = np.random.normal(size=len(S2_mean))
        x1 = np.random.normal(size=len(S2_mean))
        y = (S2_skew * np.abs(x0) + x1) / np.sqrt(1+S2_skew*S2_skew)
        S2_sample = y*S2_scale + S2_loc
        
        S1_sample[S1_sample<=0]=1.
        S2_sample[S2_sample<=0]=1.
        
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
        Nph_bw = 10.0
        Ne_bw = 4.0
        Nph_edges = tf.cast(tf.range(0,2500,Nph_bw)-Nph_bw/2., fd.float_type()) # shape (Nph+1)
        Ne_edges = tf.cast(tf.range(0,700,Ne_bw)-Ne_bw/2., fd.float_type()) # shape (Ne+1)
        Nph = 0.5 * (Nph_edges[1:] + Nph_edges[:-1])
        Ne  = 0.5 * (Ne_edges[1:] + Ne_edges[:-1])
        Nph_diffs = tf.experimental.numpy.diff(Nph_edges)
        Ne_diffs  = tf.experimental.numpy.diff(Ne_edges)
        
        tempX, tempY = tf.meshgrid(Nph,Ne,indexing='ij')
        NphNe = tf.stack((tempX, tempY),axis=2)
            
        # Energy binning
        energy_first = fd.np_to_tf(energy_first)[0,:,0]               # inital shape: {batch_size, E_bins, None} --> final shape: {E_bins} 
        rate_vs_energy_first = fd.np_to_tf(rate_vs_energy_first)[0,:] # inital shape: {batch_size, E_bins} --> final shape: {E_bins}
        rate_vs_energy_first = rate_vs_energy_first/tf.reduce_sum(rate_vs_energy_first) # 240416 AV added to ensure ER/NR SS is normalized probably
        

        # Load params
        Nph_mean = self.source.Nph_mean_ER_tf
        Ne_mean = self.source.Ne_mean_ER_tf
        Nph_std = self.source.Nph_std_ER_tf
        Ne_std = self.source.Ne_std_ER_tf
        Nph_skew = self.source.Nph_skew_ER_tf
        Ne_skew = self.source.Ne_skew_ER_tf
        initial_corr = self.source.initial_corr_ER_tf
        
        
        ### Quanta Production
        x = NphNe[:,:,0]
        x = tf.repeat(x[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        y = NphNe[:,:,1]
        y = tf.repeat(y[:,:,o],tf.shape(energy_first)[0],axis=2) # final shape: {Nph, Ne, E_bins}

        bCa1 = Nph_skew + initial_corr * Ne_skew
        bCa2 = initial_corr * Nph_skew + Ne_skew
        aCa = 1. + Nph_skew * bCa1 + Ne_skew * bCa2
        delta1 = (1. / tf.sqrt(aCa)) * bCa1
        delta2 = (1. / tf.sqrt(aCa)) * bCa2
        scale1 = Nph_std / tf.sqrt(1. - 2 * delta1**2 / pi)
        scale2 = Ne_std / tf.sqrt(1. - 2 * delta2**2 / pi)
        loc1 = Nph_mean - scale1 * delta1 * tf.sqrt(2/pi)
        loc2 = Ne_mean - scale2 * delta2 * tf.sqrt(2/pi) 
        
        # multivariate normal
        # f(x,y) = ...
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution (bivariate case)
        denominator = 2. * pi * Nph_std * Ne_std * tf.sqrt(1. - initial_corr * initial_corr)  #shape (energies)
        exp_prefactor = -1. / (2 * (1. - initial_corr * initial_corr)) #shape (energies)
        exp_term_1 = (x - loc1) * (x - loc1) / (scale1*scale1) #shape (Nph,Ne,energies)
        exp_term_2 = (y - loc2) * (y - loc2) / (scale2*scale2) #shape (Nph,Ne,energies)
        exp_term_3 = -2. * initial_corr * (x - loc1) * (y - loc2) / (scale1 * scale2) #shape (Nph,Ne,energies)

        mvn_pdf = 1. / denominator * tf.exp(exp_prefactor * (exp_term_1 + exp_term_2 + exp_term_3)) #shape (Nph,Ne,energies)

        # 1d norm cdf
        # Phi(x) = (1 + Erf(x/sqrt(2)))/2
        Erf_arg = (Nph_skew * (x-loc1)/scale1) + (Ne_skew * (y-loc2)/scale2) #shape (Nph,Ne,energies)
        Erf = tf.math.erf( Erf_arg / 1.4142 ) #shape (Nph,Ne,energies)

        norm_cdf = ( 1 + Erf ) / 2 #shape (Nph,Ne,energies)

        # skew2d = f(xmod,ymod)*Phi(xmod)*(2/scale)
        probs = mvn_pdf * norm_cdf * (2 / (scale1*scale2)) * (Nph_std*Ne_std) #shape (Nph,Ne,energies)
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        
        probs = probs*Nph_bw*Ne_bw
        probs *= 1/tf.reduce_sum(probs,axis=[0,1]) # normalize the probability for each recoil energy                
        probs *= rate_vs_energy_first # shape: {Nph,Ne,E_bins,} {Nph,Ne,E1_bins,E2_bins}
        NphNe_pdf = tf.reduce_sum(probs, axis=2) # final shape: {Nph,Ne}
        
        # avoid Nph=0 and Ne=0 for the S1, S2 calculation
        NphNe_pdf = NphNe_pdf[1:,1:]
        Nph = Nph[1:]
        Ne = Ne[1:]
        NphNe_stack = tf.ragged.stack([Nph,Ne])
        
        g1, g2, S1_fano, S2_fano, S1_skew, S2_skew = self.gimme('detector_params',
                                                                  bonus_arg=NphNe_stack,
                                                                  data_tensor=data_tensor,
                                                                  ptensor=ptensor) # shape: {integer} 
        
        s1 = s1[:,0,0]  # initial shape: {batch_size, E1_bins, None} --> final shape: {batch_size}
        s1 = tf.repeat(s1[:,o],len(Nph),axis=1) # shape: {batch_size, Nph}
        
        s2 = self.gimme('get_s2', data_tensor=data_tensor, ptensor=ptensor) # shape: {batch_size}
        s2 = tf.repeat(s2[:,o],len(Ne),axis=1) # shape: {batch_size, Ne}     

        S1_mean = Nph*g1     
        S1_std = tf.sqrt(S1_mean*S1_fano)

        S2_mean = Ne*g2
        S2_std = tf.sqrt(S2_mean*S2_fano)
        
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