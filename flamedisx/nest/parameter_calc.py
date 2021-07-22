import sys

import numpy as np

import flamedisx as fd
export, __all__ = fd.exporter()

GAS_CONSTANT = 8.31446261815324
N_AVAGADRO = 6.0221409e23
XENON_VDW_A = 0.4250
XENON_VDW_B = 5.105e-5
A_XENON = 131.293
Z_XENON = 54


@export
def calculate_density(temp, pressure):
    """Returns density in g/cm^3

    """
    if (temp < 161.40):
        raise ValueError("Solid phase!")

    if (temp < 289.7):
        VaporP_bar = pow(10, 4.0519 - 667.16 / temp)
    else:
        VaporP_bar = sys.float_info.max

    if (pressure < VaporP_bar):
        raise ValueError("Gas phase!")

    density = 2.9970938084691329e2 * np.exp(-8.2598864714323525e-2 * temp) - \
        1.8801286589442915e6 * np.exp(
            -((temp - 4.0820251276172212e2) / 2.7863170223154846e1)**2) - \
        5.4964506351743057e3 * np.exp(
            -((temp - 6.3688597345042672e2) / 1.1225818853661815e2)**2) + \
        8.3450538370682614e2 * np.exp(
            -((temp + 4.8840568924597342e1) / 7.3804147172071107e3)**2) \
        - 8.3086310405942265e2

    return density


@export
def calculate_density_gas(temp, pressure):
    """Returns gaseous density in g/cm^3

    """
    p_Pa = pressure * 1e5
    density = 1 / (
        pow(GAS_CONSTANT * temp, 3) /
        (p_Pa * pow(GAS_CONSTANT * temp, 2) + XENON_VDW_A * p_Pa * p_Pa) +
        XENON_VDW_B) \
        * A_XENON * 1e-6

    return density


@export
def calculate_drift_velocity(drift_field, density, temp):
    """Returns drift_velocity in cm/ns

    """
    polyExp = [
        [-3.1046, 27.037, -2.1668, 193.27, -4.8024, 646.04, 9.2471],  # 100 K
        [-2.7394, 22.760, -1.7775, 222.72, -5.0836, 724.98, 8.7189],  # 120 K
        [-2.3646, 164.91, -1.6984, 21.473, -4.4752, 1202.2, 7.9744],  # 140 K
        [-1.8097, 235.65, -1.7621, 36.855, -3.5925, 1356.2, 6.7865],  # 155 K
        [-1.5000, 37.021, -1.1430, 6.4590, -4.0337, 855.43, 5.4238],  # 157 K
        [-1.4939, 47.879, 0.12608, 8.9095, -1.3480, 1310.9, 2.7598],  # 163 K
        [-1.5389, 26.602, -.44589, 196.08, -1.1516, 1810.8, 2.8912],  # 165 K
        [-1.5000, 28.510, -.21948, 183.49, -1.4320, 1652.9, 2.884],   # 167 K
        [-1.1781, 49.072, -1.3008, 3438.4, -.14817, 312.12, 2.8049],  # 184 K
        [1.2466, 85.975, -.88005, 918.57, -3.0085, 27.568, 2.3823],   # 200 K
        [334.60, 37.556, 0.92211, 345.27, -338.00, 37.346, 1.9834]]  # 230 K

    Temperatures = np.array(
        [100., 120., 140., 155., 157., 163., 165., 167., 184., 200., 230.])

    if not (100 <= temp < 230):
        raise ValueError("Temprature out of range (100-230 K).")

    right = np.where(Temperatures > temp)
    left = np.where(Temperatures <= temp)
    i = left[0][-1]
    j = right[0][0]

    Ti = Temperatures[i]
    Tf = Temperatures[j]

    vi = polyExp[i][0] * np.exp(-drift_field / polyExp[i][1]) + \
        polyExp[i][2] * np.exp(-drift_field / polyExp[i][3]) + \
        polyExp[i][4] * np.exp(-drift_field / polyExp[i][5]) + \
        polyExp[i][6]
    vf = polyExp[j][0] * np.exp(-drift_field / polyExp[j][1]) + \
        polyExp[j][2] * np.exp(-drift_field / polyExp[j][3]) + \
        polyExp[j][4] * np.exp(-drift_field / polyExp[j][5]) + \
        polyExp[j][6]

    if (temp == Ti):
        return vi
    elif (temp == Tf):
        return vf
    elif (vf < vi):
        offset = (
            np.sqrt((Tf * (vf - vi) - Ti * (vf - vi) - 4.) * (vf - vi)) +
            np.sqrt(Tf - Ti) * (vf + vi)) / (2. * np.sqrt(Tf - Ti))
        slope = -(np.sqrt(Tf - Ti) * np.sqrt(
            (Tf * (vf - vi) - Ti * (vf - vi) - 4.) * (vf - vi)) -
            (Tf + Ti) * (vf - vi)) / (2. * (vf - vi))
        speed = 1. / (temp - slope) + offset
    else:
        slope = (vf - vi) / (Tf - Ti)
        speed = slope * (temp - Ti) + vi

    if (speed <= 0.):
        raise ValueError("Negative drift velocity!")

    return speed*1e-4


@export
def calculate_work(density):
    eDensity = density * N_AVAGADRO * Z_XENON / A_XENON
    Wq_eV = (18.7263 - 1.01e-23 * eDensity) * 1.1716263232
    Wq_keV = Wq_eV * 1e-3

    return Wq_keV
