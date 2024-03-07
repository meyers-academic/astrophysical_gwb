import bilby.core.prior
from bilby.gw.prior import AlignedSpin
import numpy as np
from constants import *
import redshiftPrior
import math
from scipy.special import erf

class SpinPrior(bilby.core.prior.Prior):
    '''
    Realistic spin distribution probabilities from Callister and Farr.
    '''
    def __init__(self, minimum, maximum, **kwargs):
        self.minimum = minimum
        self.maximum = maximum
        super(SpinPrior, self).__init__(minimum=minimum, maximum=maximum, **kwargs)

    def prob(self, chi):
        chi0 = 0.15
        gamma = 0.18
        f_iso = 0.67
        mu = 0.59
        sigma = 0.58

        C = (np.arctan((1-chi0)/gamma) + np.arctan(chi0/gamma))**(-1)
        pchi = (C/gamma)*(1+((chi-chi0)/gamma)**2)**(-1)
        return pchi

    def _get_chi_arrays(self):
        chi0 = 0.15
        gamma = 0.18
        f_iso = 0.67
        mu = 0.59
        sigma = 0.58

        chis = np.linspace(self.minimum, self.maximum, 1_000_000)
        pchis = self.prob(chis)
        return chis, pchis

    def sample(self, size):
        chis, pchis = self._get_chi_arrays()
        max_pchi = np.max(pchis)
        total = 0
        samples = []
        while total < size:
            chiIndex = np.random.randint(0, 1_000_000)
            chi = chis[chiIndex]
            pchi = np.random.rand()*max_pchi
            if(pchi < pchis[chiIndex]):
                samples.append(chi)
                total = total + 1
        return np.array(samples)

class ThetaPrior(bilby.core.prior.Prior):
    '''
    Theta distribution prior from Callister and Farr
    '''
    chi0 = 0.15
    gamma = 0.18
    f_iso = 0.67
    mu = 0.59
    sigma = 0.58

    def __init__(self, minimum, maximum, **kwargs):
        self.minimum = minimum
        self.maximum = maximum
        super(ThetaPrior, self).__init__(minimum=minimum, maximum=maximum, **kwargs)

    def phi(self, xi):
        "Gaussian PDF"
        return (1/np.sqrt(2*np.pi))*np.exp(-(1/2)*xi**2)

    def Phi(self, x):
        "Gaussian CDF"
        return (1/2)*(1+erf(x/np.sqrt(2)))

    def prob(self, costheta):
        a = self.minimum
        b = self.maximum

        Nab = (1/self.sigma)*(self.phi((costheta-self.mu)/self.sigma))/(self.Phi((b - self.mu)/self.sigma) - self.Phi((a - self.mu)/self.sigma))
        pcostheta = self.f_iso/2 + (1 - self.f_iso)*Nab
        return pcostheta

    def _get_costheta_arrays(self):
        costhetas = np.linspace(self.minimum, self.maximum, 1000)
        pcosthetas = []
        for costheta in costhetas:
            pcostheta = self.prob(costheta)
            pcosthetas.append(pcostheta)
        return costhetas, pcosthetas

    def sample(self, size):
        costhetas, pcosthetas = self._get_costheta_arrays()
        max_pcostheta = np.max(pcosthetas)
        total = 0
        samples = []
        while total < size:
            costhetaIndex = np.random.randint(1000)
            costheta = costhetas[costhetaIndex]
            theta = np.arccos(costheta)
            pcostheta = np.random.rand()*max_pcostheta
            if(pcostheta < pcosthetas[costhetaIndex]):
                # samples.append(costheta)
                samples.append(theta)
                total = total + 1
        return np.array(samples)

def starter_prior(T_obs=1000):
    priors = bilby.gw.prior.PriorDict()
    priors['mass_1_source'] = bilby.core.prior.PowerLaw(alpha=-2.3, minimum=BBH_min, maximum=BBH_max)
    priors['mass_ratio'] = bilby.core.prior.PowerLaw(alpha=1.5, minimum=0, maximum=1)
    priors['theta_jn'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='theta_jn')
    priors['phi_12'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phi_12')
    priors['phi_jl'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phi_jl')
    priors['redshift'] = redshiftPrior.BrokenPowerLawRedshiftPrior(R0=R0, alpha=alpha, beta=beta, zp=zp, minimum=0, maximum=z_max, name='redshift')
    priors['geocent_time'] = bilby.core.prior.Uniform(minimum=0, maximum=T_obs, name='geocent_time')
    priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=6.283185307179586, name='phase', latex_label='$\\phi$', unit=None, boundary='periodic')
    priors['psi'] = bilby.core.prior.Uniform(minimum=0, maximum=3.141592653589793, name='psi', latex_label='$\\psi$', unit=None, boundary='periodic')
    return priors

def non_spinning_prior(n_injections=0):
    priors = starter_prior()
    priors['a_1'] = 0
    priors['a_2'] = 0
    priors['tilt_1'] = 0
    priors['tilt_2'] = 0
    return priors

def uniform_spin_priors(n_injections=0):
    priors = starter_prior()
    priors['a_1'] = bilby.core.prior.Uniform(minimum=0, maximum=1)
    priors['a_2'] = bilby.core.prior.Uniform(minimum=0, maximum=1)
    priors['tilt_1'] = bilby.core.prior.Sine(minimum=0, maximum=3.141592653589793, name='tilt_1', latex_label='$\\theta_1$', unit=None, boundary=None)
    priors['tilt_2'] = bilby.core.prior.Sine(minimum=0, maximum=3.141592653589793, name='tilt_1', latex_label='$\\theta_1$', unit=None, boundary=None)
    return priors

def callister_farr():
    priors = starter_prior()
    priors['a_1'] = SpinPrior(minimum=0, maximum=1, name='a_1')
    priors['a_2'] = SpinPrior(minimum=0, maximum=1, name='a_2')
    priors['tilt_1'] = ThetaPrior(minimum=-1, maximum=1, name='tilt_1')
    priors['tilt_2'] = ThetaPrior(minimum=-1, maximum=1, name='tilt_2')
    return priors

def maximal_spin_magnitude_uniform_tilt():
    priors = starter_prior()
    priors['a_1'] = 1
    priors['a_2'] = 1
    priors['tilt_1'] = bilby.core.prior.Sine(minimum=0, maximum=3.141592653589793, name='tilt_1', latex_label='$\\theta_1$', unit=None, boundary=None)
    priors['tilt_2'] = bilby.core.prior.Sine(minimum=0, maximum=3.141592653589793, name='tilt_1', latex_label='$\\theta_1$', unit=None, boundary=None)
    return priors

def maximal_spin_magnitude_aligned():
    priors = starter_prior()
    priors['a_1'] = 1
    priors['a_2'] = 1
    priors['tilt_1'] = 0
    priors['tilt_2'] = 0
    return priors

def maximal_spin_magnitude_anti_aligned():
    priors = starter_prior()
    priors['a_1'] = 1
    priors['a_2'] = 1
    priors['tilt_1'] = np.pi
    priors['tilt_2'] = np.pi
    return priors

def maximal_spin_magnitude_in_plane():
    priors = starter_prior()
    priors['a_1'] = 1
    priors['a_2'] = 1
    priors['tilt_1'] = np.pi/2 + 1e-5
    priors['tilt_2'] = np.pi/2 + 1e-5
    return priors

