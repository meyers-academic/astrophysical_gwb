import bilby.gw
import numpy as np
from loguru import logger
from tqdm import tqdm

from pygwb.constants import H0

# Define redshift prior
from bilby.gw.prior import Cosmological
import numpy as np

import sys

import gwBackground_module as gwb

import astropy.units as u


class BrokenPowerLawRedshiftPrior(Cosmological):
    """
    Broken power law for merger rate as a function on redshift
    """

    def __init__(self, minimum, maximum, R0, alpha, beta, zp, **kwargs):
        """
        R0 is in units of Gpc^{-3} yr^{-1}
        """
        self.R0 = R0 * 1e-9 * u.Mpc ** -3 * u.yr ** -1  # convert to Mpc^{-3} yr^{-1}
        self.alpha = alpha
        self.beta = beta
        self.zp = zp
        super(BrokenPowerLawRedshiftPrior, self).__init__(minimum=minimum, maximum=maximum, **kwargs)

    def _get_rate_vs_redshift(self, zs):
        C = 1 + (1 + self.zp) ** (-self.alpha - self.beta)
        r_of_z = C * self.R0 * (1 + zs) ** (self.alpha) / (1 + ((1 + zs) / (1 + self.zp)) ** (self.alpha + self.beta))
        return r_of_z

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        C = 1 + (1 + self.zp) ** (-self.alpha - self.beta)
        r = C * self.R0 * (1 + zs) ** (self.alpha) / (1 + ((1 + zs) / (1 + self.zp)) ** (self.alpha + self.beta))
        p_dz = (1 / (1 + zs)) * r * 4 * np.pi * self.cosmology.differential_comoving_volume(zs) * u.sr
        return zs, p_dz


class PowerLawRedshiftPrior(Cosmological):
    """
    Power law for merger rate as a function of redshift
    """

    def __init__(self, minimum, maximum, R0, alpha, **kwargs):
        """
        R0 is in units of Gpc^{-3} yr^{-1}
        """
        self.R0 = R0 * 1e-9 * u.Mpc ** -3 * u.yr ** -1  # convert to Mpc^{-3} yr^{-1}
        self.alpha = alpha
        super(PowerLawRedshiftPrior, self).__init__(minimum=minimum, maximum=maximum, **kwargs)

    def _get_rate_vs_redshift(self, zs):
        R_of_z = self.R0 * (1 + zs) ** self.alpha
        return R_of_z

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        # C = 1 + (1 + self.zp) ** (-self.alpha - self.beta)
        r = self._get_rate_vs_redshift(zs)
        p_dz = (1 / (1 + zs)) * r * 4 * np.pi * self.cosmology.differential_comoving_volume(zs) * u.sr * u.Mpc ** -3
        return zs, p_dz


def calculate_num_injections(T_obs, zs, p_dz):
    '''
    Calculate the number of injections for the observation time.

    Parameters
    ----------
    T_obs : double
        observation time (in yrs)
    zs : np.ndarray
        redshift array
    p_dz : array
        redshift distribution
    Returns
    -------
    double
        theoretical number of events for the observation time
    '''
    p_dz_centers = (p_dz[1:] + p_dz[:-1]) / 2.
    total_sum = np.sum(np.diff(zs) * p_dz_centers)
    return T_obs * total_sum


def draw_injections(prior_dict, Tobs):
    """
    Figure out how many injections to draw, then
    draw that number of injections

    Parameters
    ----------
    prior_dict : bilby.gw.prior.BBHPriorDict
        bilby prior dictionary
    Tobs : double
        Observation time (yrs)

    Returns
    -------
    injections : dict
        dictionary containing information on injections
    """
    zs, p_dz = prior_dict['redshift']._get_redshift_arrays()
    N = calculate_num_injections(Tobs, zs, p_dz)

    logger.info(f"We are averaging over {N} waveforms for {np.round(Tobs, 3)} years")
    N_inj = np.random.poisson(N.value)
    injections = prior_dict.sample(N_inj)
    injections["signal_type"] = 'CBC'
    return injections


def sample_dict_compute_injected_omega(prior_dict, Tobs, duration=10, f_ref=25, sampling_frequency=2048,
                                       approximant='IMRPhenomD'):
    # Calculate number of injections
    injections = draw_injections(prior_dict, Tobs)

    # set up waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": approximant,
            "reference_frequency": 50,
            "minimum_frequency": 1
        }
    )  # breaks down at M_tot * (1+z) ~ 1000, according to Xiao-Xiao and Haowen

    # convert to seconds, set up frequency array for waveform

    freqs_psd = waveform_generator.frequency_array
    omega_gw_freq = np.zeros(len(freqs_psd))

    try:
        N_inj = len(injections['geocent_time']['content'])
    except:
        N_inj = len(injections['geocent_time'])

    logger.info('Compute the total injected Omega for ' + str(N_inj) + ' injections')

    # Loop over injections
    for i in tqdm(range(N_inj)):

        inj_params = {}
        # Generate the individual parameters dictionary for each injection
        for k in injections.keys():
            if k == 'signal_type':
                continue
            try:
                inj_params[k] = injections[k]['content'][i]
            except:
                inj_params[k] = injections[k][i]

        # Get frequency domain waveform
        polarizations = waveform_generator.frequency_domain_strain(inj_params)

        # Final PSD of the injection
        psd = np.abs(polarizations['plus']) ** 2 + np.abs(polarizations['cross']) ** 2

        # Add to Omega_spectrum
        omega_gw_freq += 2 * np.pi ** 2 * freqs_psd ** 3 * psd / (3 * H0.si.value ** 2)

    Tobs_seconds = Tobs * 86400 * 365.25  # years to seconds
    omega_gw_freq *= 2 / Tobs_seconds

    return freqs_psd, omega_gw_freq, injections


def calculate_omega_gridded(prior_dict, fref=25):
    # Callister Method
    # Set up OmegaGW object

    # TODO fix this.
    zs = np.linspace(0, 10, num=1000)

    inspiralOnly = False
    m1_min = prior_dict['mass_1'].minimum
    m1_max = prior_dict['mass_1'].maximum
    m2_min = prior_dict['mass_2'].minimum
    m2_max = prior_dict['mass_2'].maximum
    minimum_component_mass = prior_dict['mass_2'].minimum
    maximum_component_mass = prior_dict['mass_1'].maximum
    omg = gwb.OmegaGW_BBH(minimum_component_mass, maximum_component_mass, zs)

    # Calculate merger rate
    mergerRate = prior_dict['redshift']._get_rate_vs_redshift(zs)

    # Calculate probability grid
    # Priors defined in (m1, q) space
    probs = np.empty((omg.m1s_2d.shape[0], omg.qs_2d.shape[1]))  # initialize array
    for i in range(omg.m1s_2d.shape[0]):  # for each m1
        for j in range(omg.qs_2d.shape[1]):  # for each q
            prob = prior_dict.prob({'mass_1': omg.m1s_2d[i][j], 'mass_ratio': omg.qs_2d[i][j]})  # calculate probability
            probs[i][j] = prob  # insert probability into probability array
    probs = probs * (omg.Mtots_2d / (1. + omg.qs_2d))  # multiply by the Jacobian

    # Limit probabilities to physical masses
    probs[omg.m1s_2d < m1_min] = 0
    probs[omg.m1s_2d > m1_max] = 0
    probs[omg.m2s_2d < m2_min] = 0
    probs[omg.m2s_2d > m2_max] = 0

    # Normalize probabilities
    probs /= np.sum(probs)
    omg.probs = probs

    # Calculate Omega(f)
    freqs_TC = np.logspace(0, np.log10(1000), 200)
    OmegaGW_TC = omg.eval(prior_dict['redshift'].R0.value, mergerRate, freqs_TC)

    # Find OmegaGW(fref)
    fref_approx_TC = 100
    fref_approx_index_TC = 0
    for i in range(len(freqs_TC)):
        if (np.absolute(freqs_TC[i] - fref) < fref_approx_TC):
            fref_approx_TC = np.absolute(freqs_TC[i] - fref)
            fref_approx_index_TC = i

    return freqs_TC, OmegaGW_TC

    # Plot
    # fig, ax = plt.subplots()
    # ax.loglog(freqs_TC, OmegaGW_TC, color='#7dd4fa')
    # ax.loglog(freqs_TC, Omega_ref_TC * (freqs_TC / fref) ** (2 / 3), label='2/3 Power Law', color='#000000')
    # ax.set_title(r'GW Energy Density Spectrum (Callister Method)')
    # ax.set_xlabel(r'Frequency (Hz)')
    # ax.set_ylabel(r'$\Omega_{GW}(f)$')
    # ax.set_xlim(10, 1000)
    # ax.legend()
    # fig.show()