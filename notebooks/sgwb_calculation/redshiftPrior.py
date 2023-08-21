# Define redshift prior
from bilby.gw.prior import Cosmological
import numpy as np
class PowerLawRedshiftPrior(Cosmological):
    def __init__(self, minimum, maximum, R0, alpha, beta, zp, **kwargs):
        """
        R0 is in units of Gpc^{-3} yr^{-1}
        """
        self.R0 = R0*1e-9 # convert to Mpc^{-3} yr^{-1}
        self.alpha = alpha
        self.beta = beta
        self.zp = zp
        super(PowerLawRedshiftPrior, self).__init__(minimum=minimum, maximum=maximum, **kwargs)
        
    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        C = 1 + (1 + self.zp)**(-self.alpha-self.beta)
        r = C*self.R0* (1+zs)**(self.alpha) / (1 + ((1+zs)/(1+self.zp))**(self.alpha+self.beta))
        p_dz = (1/(1+zs))*r* 4 * np.pi * self.cosmology.differential_comoving_volume(zs)
        return zs, p_dz
    
def calculate_num_injections(T_obs, zs, p_dz):
    '''
    Calculate the number of injections for the observation time.
    
    Parameters
    ----------
    T_obs : double
        observation time (in yrs)
    zs : array
        redshift array
    p_dz : array
        redshift distribution
    Returns
    -------
    double
        theoretical number of events for the observation time
    '''
    p_dz_centers = (p_dz[1:] + p_dz[:-1])/2.
    total_sum = np.sum(np.diff(zs)*p_dz_centers)
    N = T_obs*total_sum
    return N