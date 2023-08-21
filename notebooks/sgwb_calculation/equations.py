# Equations
from constants import *
import math
import numpy as np

def pc(H0):
    '''
    Equation from Renzini et al. 2022.
    Calculates the critical energy density.
    
    Parameters
    ----------
    H0 : double
        Hubble constant (km s^-1 Mpc^-1)
        
    Returns
    -------
    double
        critical energy density (in km kg s Mpc^-1 m^-3)
    '''
    return 3*(H0*c)**2/(8*math.pi*G)

def pc_SI(H0):
    '''
    Equation from Renzini et al. 2022.
    Calculates the critical energy density.
    
    Parameters
    ----------
    H0 : double
        Hubble constant (in km s^-1 Mpc^-1)
        
    Returns
    -------
    double
        critical energy density (in kg m^-3)
    '''
    H0 = H0*km/Mpc # convert to Hz
    return 3*(H0*c)**2/(8*math.pi*G)

def R(alpha, beta, z, zp, R0):
    '''
    Equation from Callister et al. 2020.
    Calculates the merger rate at a given redshift.
    
    Parameters
    ----------
    alpha : double
        spectral index before peak zp
    beta : double
        spectral index after peak zp
    z : double
        redshift
    zp : double
        peak redshift of merger rate
    R0 : double
        current merger rate (z = 0) (Gpc^- 3 yr^-1)
        
    Returns
    -------
    double
        merger rate at the given redshift (in Gpc^-3 yr^-1)
    '''
    return C(alpha, beta, zp)*R0 * (1+z)**alpha / (1 + (1+z)/(1+zp)**(alpha+beta))

def R_SI(alpha, beta, z, zp, R0):
    '''
    Equation from Callister et al. 2020.
    Calculates the merger rate at a given redshift.
    
    Parameters
    ----------
    alpha : double
        spectral index before peak z_p
    beta : double
        spectral index after peak z_p
    z : double
        redshift
    zp : double
        peak redshift of merger rate
    R0 : double
        current merger rate (z = 0) (Gpc^-3 yr^-1)
        
    Returns
    -------
    double
        merger rate at the given redshift (in Hz m^-3)
    '''
    R0 = R0/Gpc**3/yr # convert to m^-3 s^-1 = m^-3 Hz
    return C(alpha, beta, zp)*R0 * (1+z)**alpha / (1 + ((1+z)/(1+zp))**(alpha+beta))

def C(alpha, beta, zp):
    '''
    Equation from Callister et al. 2020.
    Calculates the normalization constant for the merger rate.
    
    Parameters
    ----------
    alpha : double
        spectral index before peak z_p
    beta : double
        spectral index after peak z_p
    zp : double
        peak redshift of merger rate
    
    Returns
    -------
    double
        normalization constant for the merger rate
    '''
    return 1 + (1+zp)**(-alpha-beta)

def Hubble_rate(z, H0, omegaR, omegaM, omegak, omegaL):
    '''
    Equation from Renzini et al. 2022.
    Calculates the Hubble rate as a function of redshift.
    
    Parameters
    ----------
    z : double
        redshift
    H0 : double
        Hubble constant (Hubble rate at z = 0) (in km s^-1 Mpc^-1)
    omegaR : double
        radiation component of energy density
    omegaM : double
        matter component of energy density
    omegak : double
        spacetime curvature component of energy density
    omegaL : double
        dark energy component of energy density, cosmological constant
    
    Returns
    -------
    double
        Hubble rate for a given redshift z (in km s^-1 Mpc^-1)
    '''
    return H0*(omegaR*(1+z)**4 + omegaM*(1+z)**3 + omegak*(1+z)**2 + omegaL)**(1/2)

def Hubble_rate_SI(z, H0, omegaR, omegaM, omegak, omegaL):
    '''
    Equation from Renzini et al. 2022.
    Calculates the Hubble rate as a function of redshift.
    
    Parameters
    ----------
    z : double
        redshift
    H0 : double
        Hubble constant (Hubble rate at z = 0) (in km s^-1 Mpc^-1)
    omegaR : double
        radiation component of energy density
    omegaM : double
        matter component of energy density
    omegak : double
        spacetime curvature component of energy density
    omegaL : double
        dark energy component of energy density, cosmological constant
    
    Returns
    -------
    double
        Hubble rate for a given redshift z (in Hz m^-3)
    '''
    H0 = H0*km/Mpc # convert to Hz
    return H0*(omegaR*(1+z)**4 + omegaM*(1+z)**3 + omegak*(1+z)**2 + omegaL)**(1/2)

def ave_dEdf(f, Mtots, events):
    '''
    Equation from Renzini et al. 2022.
    Sums the spectral energy density of individual events with a given set of parameters.
    
    Parameters
    ----------
    f : double
        frequency (in Hz)
    f_merge : double
        merger frequency (in Hz)
    f_ring : double
        ringdown frequency (in Hz)
    f_cutoff : double
        cutoff frequency (in Hz)
    sigma : double
        width of Lorentzian function around f_ring (in Hz)
    events : List<double>
        list of events (chirp masses) that incorporates the mass distribution (in Solar masses)
        
    Returns
    -------
    double
        average spectral density
    '''
    total_sum = 0
    for i in range(len(events)):
        Mtot = Mtots[i]
        M = events[i]
        value = dEdf(M, f, Mtot)
        total_sum += value
    return total_sum

def ave_dEdf_SI(f, Mtots, events):
    '''
    Equation from Renzini et al. 2022.
    Sums the spectral energy density of individual events with a given set of parameters.
    
    Parameters
    ----------
    f : double
        frequency (in Hz)
    f_merge : double
        merger frequency (in Hz)
    f_ring : double
        ringdown frequency (in Hz)
    f_cutoff : double
        cutoff frequency (in Hz)
    sigma : double
        width of Lorentzian function around f_ring (in Hz)
    events : List<double>
        list of events (chirp masses) that incorporates the mass distribution (in Solar masses)
        
    Returns
    -------
    double
        average spectral density
    '''
    total_sum = 0
    for i in range(len(events)):
        Mtot = Mtots[i]
        M = events[i]
        value = dEdf_SI(M, f, Mtot)
        total_sum += value
    return total_sum

def dEdf(M, f, Mtot):
    '''
    Equation from Callister et al. 2016.
    Calculates the spectral energy density for a single event.
    
    Parameters
    ----------
    M : double
        chirp mass (in Solar masses)
    f : double
        frequency (in Hz)
    f_merge : double
        merger frequency (in Hz)
    f_ring : double
        ringdown frequency (in Hz)
    f_cutoff : double
        cutoff frequency (in Hz)
    sigma : double
        width of Lorentzian function around f_ring (in Hz)
        
    Returns
    -------
    double
        spectral energy density for a given chirp mass and frequency (in m^2 Hz Msun^5/3 kg^-2/3)
    '''
    return (G*math.pi)**(2/3) * M**(5/3) * H(f, Mtot)/3

def dEdf_SI(M, f, Mtot):
    '''
    Equation from Callister et al. 2016.
    Calculates the spectral energy density for a single event.
    
    Parameters
    ----------
    M : double
        chirp mass (in Solar masses)
    f : double
        frequency (in Hz)
    f_merge : double
        merger frequency (in Hz)
    f_ring : double
        ringdown frequency (in Hz)
    f_cutoff : double
        cutoff frequency (in Hz)
    sigma : double
        width of Lorentzian function around f_ring (in Hz)
        
    Returns
    -------
    double
        spectral energy density for a given chirp mass and frequency (in m^2 kg Hz)
    '''
    M *= Msun # convert to kg
    dEdf = (G*math.pi)**(2/3) * M**(5/3) * H(f, Mtot)/3
    return dEdf

def H(f, Mtot):
    '''
    Equation from Callister et al. 2016.
    Calculates H for a given frequency.
    
    Parameters
    ----------
    f : double
        frequency (in Hz)
    f_merge : double
        merger frequency (in Hz)
    f_ring : double
        ringdown frequency (in Hz)
    f_cutoff : double
        cutoff frequency (in Hz)
    sigma : double
        width of Lorentzian function around f_ring (in Hz)
        
    Returns
    -------
    double
        H(f) (in Hz^-1/3)
    '''
    # Parameters from Tom Callister
    # Waveform model from Ajith+ 2008 (10.1103/PhysRevD.77.104017)
    # Define IMR parameters
    # See Eq. 4.19 and Table 1
    eta = 0.25
    f_merge = (0.29740*eta**2. + 0.044810*eta + 0.095560)/(np.pi*Mtot*MsunToSec)
    f_ring = (0.59411*eta**2. + 0.089794*eta + 0.19111)/(np.pi*Mtot*MsunToSec)
    f_cutoff = (0.84845*eta**2. + 0.12828*eta + 0.27299)/(np.pi*Mtot*MsunToSec)
    sigma = (0.50801*eta**2. + 0.077515*eta + 0.022369)/(np.pi*Mtot*MsunToSec)
    
    if f < f_merge:
        return f**(-1/3)
    elif f >= f_merge and f < f_ring:
        return f**(2/3)/f_merge
    elif f >= f_ring and f < f_cutoff:
        return 1/(f_merge*f_ring**(4/3)) * (f/(1 + ((f-f_ring)/(sigma/2))**2))**2
    else:
        return 0
    
def calculate_M(m1, m2):
    '''
    Calculate the chirp mass.
    
    Parameters
    ----------
    m1 : double
        first component mass
    m2 : double
        second component mass
    
    Return
    ------
    double
        chirp mass (in same unit as component masses)
    '''
    return (m1*m2)**(3/5)/(m1+m2)**(1/5)

def calculate_Mtot(m1, m2):
    '''
    Calculate the total mass.
    
    Parameters
    ----------
    m1 : double
        first component mass
    m2 : double
        second component mass
    
    Return
    ------
    double
        total mass (in same unit as component masses)
    '''
    return m1 + m2

def calculate_m2(m1, q):
    '''
    Calculate the second component mass, assuming q = m2/m1.
    
    Parameters
    ----------
    m1 : double
        first component mass
    q : double
        mass ratio
    
    Return
    ------
    double
        total mass (in same unit as m1)
    '''
    return m1*q

def get_R_array(zs, alpha, beta, zp, R0):
    '''
    Calculates the merger rate for an array of redshifts.
    
    Parameters
    ----------
    zs : double array
        redshift array
    alpha : double
        spectral index before peak z_p
    beta : double
        spectral index after peak z_p
    zp : double
        peak redshift of merger rate
    R0 : double
        current merger rate (z = 0) (Gpc^-3 yr^-1)
        
    Return
    ------
    np array
        merger rate array (same unit as R0)
    '''
    mergerRate = []
    for i in range(len(zs)):
        mergerRate.append(R(alpha, beta, zs[i], zp, R0))
    mergerRate = np.array(mergerRate)
    return mergerRate