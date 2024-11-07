from scipy.interpolate import RegularGridInterpolator
import numpy as np
##=============Cuts and acceptances===========
def WS2024_S2splitting_reconstruction_efficiency(S2c, driftTime_us, hist):
    """
        Returns the reconstruction efficiency based on S2 splitting
        S2c: cs2 from data [phd]
        dritTime_us: drift time from data [us]
        hist: input histogram of S2c[Ne-] vs Drift Time [us]
        adapted from BGSkimmer
    """
    ## Make values more python friendly
    weights = np.reshape(hist[0], hist[0].size)
    zmax =  np.max(weights)
    zmin = np.min(weights[weights>0.])
    ## Read into interpolator
    xentries_vals = np.array(hist[1][:-1])
    yentries_vals = np.array(hist[2][:-1])
    Interp = RegularGridInterpolator((xentries_vals,yentries_vals), hist[0])
    
    ## Convert S2c to mean N_e
    mean_SE_area = 44.5 # phd/e-
    mean_Ne = S2c/mean_SE_area

    ## Initialize acceptance values
    acceptance = np.ones_like(mean_Ne)
    ## curves not defined for above 100 e- 
    ## Assume 100% eff. (probably ok...)
    acceptance[mean_Ne>np.max(xentries_vals)] = 1.
    ## Also not defined for < 10e-
    ## ok with 15e- ROI threshold
    acceptance[mean_Ne<np.min(xentries_vals)] = 0.
    
    temp_drift = driftTime_us
    temp_drift[temp_drift<np.min(yentries_vals)] = np.min(yentries_vals)
    temp_drift[temp_drift>np.max(yentries_vals)] = np.max(yentries_vals)
    
    mask = (mean_Ne>=np.min(xentries_vals)) & (mean_Ne<=np.max(xentries_vals))
    ## Acceptances are provided in percent - divide by 100.
    acceptance[mask] = Interp(np.vstack([mean_Ne[mask],temp_drift[mask]]).T)/100.
    
    
    return acceptance

##=====================CUTS=====================
##Build S2raw acceptance
def WS2024_trigger_acceptance(S2raw):
    """
        Returns the Trigger efficiency of S2s
        S2rw: s2 from data [phd]
        adapted from BGSkimmer
    """
    # 50% threshold = 106.89 +/- 0.43 phd
    # 95% threshold = 160.7 +/- 3.5 phd
    # k = 0.0547 +/- 0.0035 phd-1
    
    x0 = 160.7
    k = 0.0547
    
    accValues =  ( 1./( 1 + np.exp( -k*(S2raw-x0) ) ) )
    
    ## make sure acceptances can't go above 1. or below 0.
    accValues[accValues>1.] = 1. 
    accValues[accValues<0.] = 0.
    
    return accValues


# Define polynomial coefficients associated with each azimuthal wall position;
# these start at the 7 o'clock position and progress anti-clockwise
phi_coeffs = [[-1.78880746e-13, 4.91268301e-10, -4.96134607e-07, 2.26430932e-04, -4.71792008e-02, 7.33811298e+01],
              [-1.72264463e-13, 4.59149636e-10, -4.59325165e-07, 2.14612376e-04, -4.85599108e-02, 7.35290867e+01],
              [-3.17099156e-14, 7.26336129e-11, -6.99495385e-08, 3.85531008e-05, -1.33386004e-02, 7.18002889e+01],
              [-6.12280314e-14, 1.67968911e-10, -1.83625538e-07, 1.00457608e-04, -2.86728022e-02, 7.22754350e+01],
              [-1.89897962e-14, 1.52777215e-11, -2.79681508e-09, 1.25689887e-05, -1.33093804e-02, 7.17662251e+01],
              [-2.32118621e-14, 7.30043322e-11, -9.40606298e-08, 6.29728588e-05, -2.28150175e-02, 7.22661091e+01],
              [-8.29749194e-14, 2.31096069e-10, -2.47867121e-07, 1.27576029e-04, -3.24702414e-02, 7.26357609e+01],
              [-2.00718008e-13, 5.44135757e-10, -5.59484466e-07, 2.73028553e-04, -6.46879791e-02, 7.45264998e+01],
              [-7.77420021e-14, 1.97357045e-10, -1.90016273e-07, 8.99659454e-05, -2.30169916e-02, 7.25038258e+01],
              [-5.27296334e-14, 1.49415580e-10, -1.58205132e-07, 8.00275441e-05, -2.13559394e-02, 7.23995451e+01],
              [-6.00198219e-14, 1.55333004e-10, -1.60367908e-07, 7.97754165e-05, -1.94435594e-02, 7.22714399e+01],
              [-8.89919309e-14, 2.40830027e-10, -2.57060475e-07, 1.33002951e-04, -3.32969110e-02, 7.28696020e+01]]


# Use the above set of coefficients to define azimuthal wall position contours
phi_walls = [np.poly1d(phi_coeffs[i]) for i in range(len(phi_coeffs))]
FV_poly = np.poly1d([-4.44147071e-14,  1.43684777e-10, -1.82739476e-07,
                                 1.02160174e-04, -2.31617857e-02, -2.05932471e+00])
def WS2024_fiducial_volume_cut(x,y,dt):
    """
        Fiducial volume cute for WS2024
        x: x position [cm]
        y: y position [cm]
        dt: drift time [us]
        adapted from BGSkimmer
    """
    # Define radial contour for N_tot = 0.01 expected counts (drift-∆R_phi space)
    contour=FV_poly(dt)
    #===CALCULATE dR_DPHI
    # Define azimuthal slices
    n_phi_slices = 12
    phi_slices = np.linspace(-np.pi, np.pi, n_phi_slices + 1) + np.pi/4
    phi_slices[phi_slices > np.pi] -= 2*np.pi

    # Calculate event radii and angles, then mask them according to each slice
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi_cuts = [((phi >= phi_slices[i]) & (phi < phi_slices[i + 1])) if not (phi_slices[i] > phi_slices[i + 1]) else \
               ~((phi <= phi_slices[i]) & (phi > phi_slices[i + 1])) for i in range(n_phi_slices)]

    # Calculate dR_phi by replacing relevant points in loops over phi slices, then return
    dR_phi = np.zeros_like(x)
    for i, p in enumerate(phi_cuts):
        dR_phi[p] = R[p] - phi_walls[i](dt[p])

    # Segment events by whether or not they're in the expandable part
    expandable = (dt > 71) & (dt < 900)

    # Get the radial cut as a mask between two parts
    expansion = 0
    mask = ((dR_phi < (contour + expansion)) & expandable) | ((dR_phi < contour) & ~expandable)

    #cut the drift time 
    dt_cut = (dt > 71) & (dt < 1034)

    return dt_cut&mask&(dR_phi<=0)

def WS2024_resistor_XY_cut(x,y):
    """
        Cut around hot (radioacitvity not like josh) resistors
        x: x position [cm]
        y: y position [cm]
    """
    res1X = -69.8
    res1Y = 3.5
    res1R = 6
    res2X = -67.5
    res2Y = -14.3
    res2R = 6

    not_inside_res1 = np.where(np.sqrt( (x-res1X)*(x-res1X) + (y-res1Y)*(y-res1Y) ) < res1R, 0., 1.)
    not_inside_res2 = np.where(np.sqrt( (x-res2X)*(x-res2X) + (y-res2Y)*(y-res2Y) ) < res2R, 0., 1.)

    return not_inside_res1 * not_inside_res2