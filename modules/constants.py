import numpy as np
# Constants
# Fundamental Constants
G = 6.6743015e-11 # Gravitational constant (in m^3 kg^-1 s^-2)
c = 2.99792458e8 # Speed of light (in m s^-1)

# Unit Conversions
km = 1e3 # km --> m
Mpc = 3.08567758128e22 # Mpc --> m
Gpc = 3.08567758128e25 # Gpc --> m
Msun = 1.989e30 # Msun --> kg
MsunToSec = Msun*G/np.power(c,3.)
yr = 60*60*24*365.25 # yr --> s

# Merger Rate Constants
# alpha = 2 # spectral index before peak zp
beta = 3.4 # spectral index after peak zp
# zp = 1.8 # peak redshift of merger rate
R0 = 28.3 # current merger rate (in Gpc^-3 yr^-1)

# Redshift Parameters
z_max = 10 # maximum plausible redshift

# Hubble Rate Constants
H0 = 67.4 # Hubble constant (Hubble rate at z = 0) (in km s^-1 Mpc^-1)
omegaR = 9.182e-5 # radiation component of energy density
omegaM = 0.3111 # matter component of energy density
omegak = 0 # spacetime curvature component of energy density
omegaL = 0.6889 # dark energy component of energy density, cosmological constant

# Mass Parameters
BBH_min = 5 # minimum BBH mass
BBH_max = 44 # maximum BBH mass
