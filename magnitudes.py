""" 
functions that compute quantities dealing with magnitudes.  
"""

from __future__ import division, print_function
import numpy as np
from default_cosmo import default_cosmo  # define a default cosology for utilities
from astropy.table import Table
from scipy import interpolate

__all__ = ( 'apparent_to_absolute_magnitude',
            'absolute_to_apparent_magnitude',
            'luminosity_to_absolute_magnitude',
            'absolute_magnitude_to_luminosity',
            'absolute_magnitude_lim', 'get_sun_mag', )
__author__ = ('Duncan Campbell')


def apparent_to_absolute_magnitude(m, d_L):
    """
    calculate the absolute magnitude
    
    Parameters
    ----------
    m: array_like
        apparent magnitude
    
    d_L: array_like
        luminosity distance to object in Mpc
    
    Returns
    -------
    Mag: np.array of absolute magnitudes
    """
    
    M = m - 5.0*(np.log10(d_L)+5.0)
    
    return M


def absolute_to_apparent_magnitude(M, d_L):
    """
    calculate the apparent magnitude given an absolute magnitude
    
    Parameters
    ----------
    M: array_like
        absolute magnitude
    
    d_L: array_like
        luminosity distance to object in Mpc
    
    Returns
    -------
    mag: np.array of apparent magnitudes
    """
    
    m = M + 5.0*(np.log10(d_L)+5.0)
    
    return m


def luminosity_to_absolute_magnitude(L, band, system='SDSS_Blanton_2003_z0.1'):
    """
    calculate the absolute magnitude
    
    Parameters
    ----------
    L: array_like
        luminosity
    
    band: string
       filter band
    
    system: string, optional
        filter systems: default is 'SDSS_Blanton_2003_z0.1'
          1. Binney_and_Merrifield_1998
          2. SDSS_Blanton_2003_z0.1
    
    Returns
    -------
    Mag: np.array of absolute magnitudes
    """
    
    Msun = get_sun_mag(band,system)
    Lsun = 1.0
    M = -2.5*np.log10(L/Lsun) + Msun
            
    return M


def absolute_magnitude_to_luminosity(M, band, system='SDSS_Blanton_2003_z0.1'):
    """
    calculate the Luminosity
    
    Parameters
    ----------
    M: array_like
        absolute magnitude
    
    band: string
       filter band
    
    system: string, optional
        filter systems: default is 'SDSS_Blanton_2003_z0.1'
          1. Binney_and_Merrifield_1998
          2. SDSS_Blanton_2003_z0.1
    
    Returns
    -------
    L: np.array of Luminosities in $log(L_{\odot})$
    """
    
    Msun = get_sun_mag(band,system)
    L = (M-Msun)/(-2.5) #in log(L/Lsun)
            
    return L


def absolute_magnitude_lim(z, app_mag_lim, cosmo=None):
    """
    give the absolute magnitude limit as a function of redshift for a flux-limited survey.
    
    Parameters
    ----------
    z: array_like
        redshift
    
    app_mag_lim: float
       apparent magnitude limit
    
    cosmo: cosmology object
    
    Returns
    -------
    M,z: np.array, np.array
        absolute magnitude in mag+5loh(h) units
    """
    if cosmo==None:
       cosmo = default_cosmo
    
    d_L = cosmo.luminosity_distance(z).value
    M = apparent_to_absolute_magnitude(app_mag_lim, d_L)
    
    return M-5.0*np.log10(cosmo.h)

def get_sun_mag(filter,system):
    """
    get the solar value for a filter in a system.
    
    Parameters
    ----------
    filter: string
    
    system: string
    
    Returns
    -------
    Msun: float
    """
    if system=='Binney_and_Merrifield_1998':
    #see Binney and Merrifield 1998
        if filter=='U':
            return 5.61
        elif filter=='B':
            return 5.48
        elif filter=='V':
            return 4.83
        elif filter=='R':
            return 4.42
        elif filter=='I':
            return 4.08
        elif filter=='J':
            return 3.64
        elif filter=='H':
            return 3.32
        elif filter=='K':
            return 3.28
        else:
            raise ValueError('Filter does not exist in this system.')
    if system=='SDSS_Blanton_2003_z0.1':
    #see Blanton et al. 2003 equation 14
        if filter=='u':
            return 6.80
        elif filter=='g':
            return 5.45
        elif filter=='r':
            return 4.76
        elif filter=='i':
            return 4.58
        elif filter=='z':
            return 4.51
        else:
            raise ValueError('Filter does not exist in this system.')
    else:
        raise ValueError('Filter system not included in this package.')


def color_k_correct(z, galaxy_type='non-star-forming'):
    """
    interpolated color and k-corrections from table 1 in Eisenstein + (2001)

    Parameters
    ----------
    z : array_like
        array of redshifts

    galaxy_type : string, optional
        string indicating which galaxy type to return color and k-corrections
        options are 'non-star-forming' or 'star-forming' 

    Returns
    -------
    delta_g, u_minus_g, g_minus_r, r_minus_i
        arrays of color an d k-corrections
    """

    z = np.atleast_1d(z)

    tabulated_z = [0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,
                   0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60]

    if galaxy_type == 'non-star-forming':
        delta_g   = [0.000,0.039,0.081,0.128,0.182,0.249,0.322,0.402,0.487,0.575,0.665,0.752,0.836,
                     0.912,0.980,1.056,1.146,1.233,1.285,1.322,1.350,1.382,1.433,1.484,1.535,1.584,1.634,1.692,1.747,1.808,1.881]
        u_minus_g = [1.929,1.928,1.940,1.955,1.965,1.961,1.957,1.953,1.957,1.964,1.969,1.976,1.995,
                     2.030,2.069,2.109,2.147,2.185,2.248,2.312,2.386,2.461,2.541,2.628,2.703,2.750,2.773,2.774,2.770,2.763,2.746]
        g_minus_r = [0.775,0.810,0.843,0.881,0.924,0.977,1.036,1.102,1.173,1.249,1.328,1.400,1.475,
                     1.533,1.583,1.642,1.719,1.778,1.800,1.805,1.792,1.767,1.755,1.737,1.715,1.684,1.657,1.642,1.629,1.626,1.637]
        r_minus_i = [0.387,0.389,0.403,0.417,0.432,0.440,0.451,0.469,0.486,0.499,0.515,0.533,0.542,
                     0.553,0.568,0.581,0.588,0.605,0.618,0.637,0.671,0.718,0.773,0.836,0.905,0.971,1.039,1.096,1.151,1.194,1.236]
    if galaxy_type == 'star-forming':
        delta_g   = [0.000,0.034,0.071,0.113,0.161,0.221,0.286,0.358,0.433,0.511,0.591,0.666,0.738,
                     0.804,0.865,0.929,1.005,1.077,1.120,1.150,1.172,1.194,1.229,1.261,1.293,1.322,1.350,1.384,1.414,1.447,1.486]
        u_minus_g = [1.758,1.754,1.757,1.756,1.748,1.727,1.704,1.677,1.655,1.631,1.603,1.575,1.552,
                     1.535,1.517,1.494,1.459,1.421,1.402,1.383,1.369,1.352,1.327,1.300,1.267,1.227,1.181,1.127,1.075,1.020,0.959]
        g_minus_r = [0.727,0.759,0.788,0.822,0.860,0.907,0.960,1.019,1.082,1.149,1.218,1.281,1.345,
                     1.397,1.440,1.491,1.555,1.604,1.621,1.623,1.609,1.582,1.561,1.532,1.499,1.458,1.419,1.388,1.358,1.335,1.320]
        r_minus_i = [0.374,0.375,0.388,0.401,0.415,0.421,0.432,0.448,0.464,0.475,0.489,0.505,0.513,
                     0.522,0.535,0.545,0.551,0.565,0.575,0.591,0.621,0.662,0.711,0.768,0.831,0.891,0.953,1.005,1.055,1.095,1.134]

    # put data into a table
    t = Table([tabulated_z, delta_g, u_minus_g, g_minus_r, r_minus_i], 
              names=('z', 'delta_g', 'u_minus_g', 'g_minus_r', 'r_minus_i'))

    # build interpolation functions
    f_1 = interpolate.InterpolatedUnivariateSpline(t['z'], t['delta_g'],   k=1)
    f_2 = interpolate.InterpolatedUnivariateSpline(t['z'], t['u_minus_g'], k=1)
    f_3 = interpolate.InterpolatedUnivariateSpline(t['z'], t['g_minus_r'], k=1)
    f_4 = interpolate.InterpolatedUnivariateSpline(t['z'], t['r_minus_i'], k=1)

    return f_1(z), f_2(z), f_3(z), f_4(z)



