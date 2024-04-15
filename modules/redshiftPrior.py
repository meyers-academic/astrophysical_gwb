from astropy import units as u
from bilby.gw.prior import Cosmological
import numpy as np
from gwpy.timeseries import TimeSeries



class BrokenPowerLawRedshiftPrior(Cosmological):
    """
    Broken power law for merger rate as a function on redshift
    """

    def __init__(self,
                 minimum,
                 maximum,
                 R0=28.3,
                 alpha=3,
                 beta=3.4,
                 zp=2.4,
                 **kwargs):
        """
        R0 is in units of Gpc^{-3} yr^{-1}
        """
        self.R0 = R0 * 1e-9 * u.Mpc**-3 * u.yr**-1  # convert to Mpc^{-3} yr^{-1}
        self.alpha = alpha
        self.beta = beta
        self.zp = zp
        super(BrokenPowerLawRedshiftPrior, self).__init__(minimum=minimum,
                                                          maximum=maximum,
                                                          **kwargs)

    def _get_rate_vs_redshift(self, zs):
        C = 1 + (1 + self.zp)**(-self.alpha - self.beta)
        r_of_z = C * self.R0 * (1 + zs)**(self.alpha) / (1 + (
            (1 + zs) / (1 + self.zp))**(self.alpha + self.beta))
        return r_of_z

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.9999,
                         self._maximum['redshift'] * 1.0001, 2000)
        C = 1 + (1 + self.zp)**(-self.alpha - self.beta)
        r = C * self.R0 * (1 + zs)**(self.alpha) / (1 + (
            (1 + zs) / (1 + self.zp))**(self.alpha + self.beta))
        p_dz = (
            1 / (1 + zs)
        ) * r * 4 * np.pi * self.cosmology.differential_comoving_volume(
            zs) * u.sr
        # p_dz = (1 / (1 + zs)) * r * self.cosmology.differential_comoving_volume(zs) * u.sr
        return zs, p_dz

    def total_rate(self):
        """
        Get total rate integrated over all redshift in units of events / s^{-1}
        """
        zs, p_dz = self._get_redshift_arrays()
        p_dz_centers = (p_dz[1:] + p_dz[:-1]) / 2.
        total_sum = np.sum(np.diff(zs) * p_dz_centers)
        return total_sum.value / (86400 * 365.25)


class PowerLawRedshiftPrior(Cosmological):
    """
    Power law for merger rate as a function of redshift
    """

    def __init__(self, minimum, maximum, R0, alpha, **kwargs):
        """
        R0 is in units of Gpc^{-3} yr^{-1}
        """
        self.R0 = R0 * 1e-9 * u.Mpc**-3 * u.yr**-1  # convert to Mpc^{-3} yr^{-1}
        self.alpha = alpha
        super(PowerLawRedshiftPrior, self).__init__(minimum=minimum,
                                                    maximum=maximum,
                                                    **kwargs)

    def _get_rate_vs_redshift(self, zs):
        R_of_z = self.R0 * (1 + zs)**self.alpha
        return R_of_z

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        # C = 1 + (1 + self.zp) ** (-self.alpha - self.beta)
        r = self._get_rate_vs_redshift(zs)
        p_dz = (
            1 / (1 + zs)
        ) * r * 4 * np.pi * self.cosmology.differential_comoving_volume(
            zs) * u.sr * u.Mpc**-3
        return zs, p_dz
