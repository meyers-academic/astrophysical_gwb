# Imports
import numpy as np

# Constants
c = 2.998e8 # speed of light
G = 6.6743015e-11 # Gravitational constant
Mpc = 3.086e16*1e6 # Mpc/m
Gpc = Mpc*1e3 # Gpc/m
km = 1.e3 # km/m
H0 = 67.4*km/Mpc # Units of 1/s, Hubble constant
Gyr = 1.e9*365.25*24*3600 # Gyr/s
year = 365.*24*3600 # y/s
OmgR = 9.182e-5 # radiation component of Hubble rate
OmgM = 0.3111 # matter component of Hubble rate
Omgk = 0 # curvature component of Hubble rate
OmgL = 0.6889 # dark matter component of Hubble rate

Msun = 1.989e30 # solar mass, in kg
MsunToSec = Msun*G/np.power(c,3.)
rhoC = 3.*np.power(H0*c,2.)/(8.*np.pi*G)*np.power(Mpc,3.) # Converted to J/Mpc^3, critical density