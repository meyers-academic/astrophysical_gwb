import numpy as np
import bilby
from tqdm import tqdm
from loguru import logger
from constants import *

from pygwb.constants import H0
import equations

import sys
sys.path.append('../modules')

def monteCarlo(
    # prior_dict,
    injections,
    Tobs,
    duration=10,
    f_ref=25,
    sampling_frequency=2048,
    approximant="IMRPhenomXPHM",
):
    # Calculate number of injections
    # injections = draw_injections(prior_dict, Tobs)

    try:
        injections['mass_1'] = injections['mass_1']*(1+injections['redshift'])
    except:
        for j in range(len(injections['mass_1'])):
            injections['mass_1'][j] = injections['mass_1'][j]*(1 + injections['redshift'][j])
  
    # set up waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": approximant,
            "reference_frequency": 50,
            "minimum_frequency": 1,
        },
    )  # breaks down at M_tot * (1+z) ~ 1000, according to Xiao-Xiao and Haowen

    # convert to seconds, set up frequency array for waveform
    freqs_psd = waveform_generator.frequency_array
    omega_gw_freq = np.zeros(len(freqs_psd))

    try:
        N_inj = len(injections["geocent_time"]["content"])
    except:
        N_inj = len(injections["geocent_time"])

    # logger.info("Compute the total injected Omega for " + str(N_inj) + " injections")

    # Loop over injections
    for i in tqdm(range(N_inj)):
        inj_params = {}
        # Generate the individual parameters dictionary for each injection
        for k in injections.keys():
            if k == "signal_type":
                continue
            try:
                inj_params[k] = injections[k]["content"][i]
            except:
                inj_params[k] = injections[k][i]

        # Get frequency domain waveform
        polarizations = waveform_generator.frequency_domain_strain(inj_params)

        # Final PSD of the injection
        psd = np.abs(polarizations["plus"]) ** 2 + np.abs(polarizations["cross"]) ** 2

        # Add to Omega_spectrum
        omega_gw_freq += 2 * np.pi**2 * freqs_psd**3 * psd / (3 * H0.si.value**2)

    Tobs_seconds = Tobs * 86400 * 365.25  # years to seconds
    omega_gw_freq *= 2 / Tobs_seconds

    # return freqs_psd, omega_gw_freq, injections
    return freqs_psd.tolist(), omega_gw_freq.tolist()

def MC2(
    prior_dict,
    # injections,
    Tobs,
    duration=2,
    f_ref=25,
    sampling_frequency=2048,
    approximant="IMRPhenomXPHM",
):
    injections = draw_injections(prior_dict, Tobs)
    dEdfs, freqs_psd = get_dedf(injections, duration=duration, sampling_frequency=sampling_frequency, approximant=approximant)
    omgw = np.sum(2 * np.pi**2 * freqs_psd**3 * dEdfs / (3 * H0.si.value**2), axis=0)
    Tobs_seconds = Tobs * 86400 * 365.25  # years to seconds
    omgw *= 2 / Tobs_seconds
    return freqs_psd, omgw

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
    N = calculate_num_injections(Tobs, prior_dict)

    logger.info(f"We are averaging over {N} waveforms for {np.round(Tobs, 3)} years")
    N_inj = np.random.poisson(N.value)
    injections = prior_dict.sample(N_inj)
    injections["signal_type"] = "CBC"
    return injections

def calculate_total_rate(priors):
    zs, p_dz = priors['redshift']._get_redshift_arrays()
    p_dz_centers = (p_dz[1:] + p_dz[:-1])/2.
    total_sum = np.sum(np.diff(zs) * p_dz_centers)
    return total_sum

def calculate_expected_num_events(T_obs, priors):
    '''
    T_obs : float
        observation time (in seconds)
    priors: bilby.core.prior.PriorDict
        bilby prior dictionary

    Returns:
    --------
    N : float
        Expected number of events in that time
    '''
    rate = calculate_total_rate(priors)
    return T_obs * rate

def calculate_num_injections(T_obs, priors):
    zs, p_dz = priors['redshift']._get_redshift_arrays()
    p_dz_centers = (p_dz[1:] + p_dz[:-1])/2.
    total_sum = np.sum(np.diff(zs) * p_dz_centers)
    N = T_obs * total_sum
    return N

def get_weights(dEdfs, reference_priors, target_priors, zs, m1s, qs):
    """
    get weights that we use to reweight our spectrum
    
    dEdfs : array-like
        array of h+^2 + hx^2 for the reference injectionset
    reference_priors : bilby.core.prior.PriorDict
        bilby prior dictionary for the reference priors
    target_priors : bilby.core.prior.PriorDict
        bilby prior dictionary for the target priors we want to use to calculate Omega_gw
    zs : array-like
        redshift of each injection
    m1s : array-like
        mass1s of each injection
    m2s : array-like
        mass2s of each injection
    qs : mass ratios of each injection
    
    """
    weights = []
    N_inj = dEdfs.shape[0]
    for i in range(N_inj):
        z = zs[i]
        m1 = m1s[i]
        # m2 = m2s[i]
        q = qs[i]
        # Mtot = Mtots[i]
        # M = Ms[i]

        # Probability of drawing {z, m1, m2}
        p_z = target_priors['redshift'].prob(z)
        p_m1 = target_priors['mass_1'].prob(m1)
        p_q = target_priors['mass_ratio'].prob(q)

        pdraw_z = reference_priors['redshift'].prob(z)
        pdraw_m1 = reference_priors['mass_1'].prob(m1)
        pdraw_q = reference_priors['mass_ratio'].prob(q)

        # # Weight calculation
        # r = equations.R_SI(alpha, beta, z, zp, R0)
        # h = equations.Hubble_rate_SI(z, H0, omegaR, omegaM, omegak, omegaL)
        h0 = equations.Hubble_rate_SI(0, H0, omegaR, omegaM, omegak, omegaL)

        wi = (p_z/pdraw_z) * (p_m1/pdraw_m1) * (p_q/pdraw_q)
        weights.append(wi)
    return np.array(weights)

def get_dedf(injections, duration=2, sampling_frequency=2048, approximant='IMRPhenomD'):
    N_inj = np.size(injections['mass_1'])

    injections['mass_1'] = injections['mass_1']*(1+injections['redshift'])
    
    # set up waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": approximant,
            "reference_frequency": 50,
            "minimum_frequency": 1,
        },
    )  # breaks down at M_tot * (1+z) ~ 1000, according to Xiao-Xiao and Haowen

    # convert to seconds, set up frequency array for waveform

    freqs_psd = waveform_generator.frequency_array
    dEdfs = []

    try:
        N_inj = len(injections["geocent_time"]["content"])
    except:
        N_inj = len(injections["geocent_time"])
    for i in tqdm(range(N_inj)):
        inj_params = {}
        # Generate the individual parameters dictionary for each injection
        for k in injections.keys():
            if k == "signal_type":
                continue
            try:
                inj_params[k] = injections[k]["content"][i]
            except:
                inj_params[k] = injections[k][i]

        # Get frequency domain waveform
        polarizations = waveform_generator.frequency_domain_strain(inj_params)

        # Final PSD of the injection
        psd = np.abs(polarizations["plus"]) ** 2 + np.abs(polarizations["cross"]) ** 2

        # Add to Omega_spectrum
        dEdfs.append(psd)
    return np.array(dEdfs), freqs_psd
