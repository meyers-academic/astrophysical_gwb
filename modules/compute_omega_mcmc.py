import bilby.gw
import numpy as np
from loguru import logger
from tqdm import tqdm

from pygwb.constants import H0

def compute_injected_omega(injection_dict, Tobs, duration=10, f_ref=25, return_spectrum=False, sampling_frequency=2048,
                           approximant='IMRPhenomD'):
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                     waveform_arguments={
                         "waveform_approximant": approximant,
                         "reference_frequency": 50,
                     }
    )
    Tobs = Tobs * 86400 * 365.25 # years to seconds
    freqs_psd = waveform_generator.frequency_array
    omega_gw_freq = np.zeros(len(freqs_psd))

    try:
        N_inj = len(injection_dict['geocent_time']['content'])
    except:
        N_inj = len(injection_dict['geocent_time'])

    omega_gw_freq = np.zeros(len(freqs_psd))
    logger.info('Compute the total injected Omega for ' + str(N_inj) + ' injections')

    # Loop over injections
    for i in tqdm(range(N_inj)):

        inj_params = {}
        # Generate the individual parameters dictionary for each injection
        for k in injection_dict.keys():
            if k == 'signal_type':
                continue
            try:
                inj_params[k] = injection_dict[k]['content'][i]
            except:
                inj_params[k] = injection_dict[k][i]

        # Get frequency domain waveform
        polarizations = waveform_generator.frequency_domain_strain(inj_params)

        # Final PSD of the injection
        psd = np.abs(polarizations['plus'])**2 + np.abs(polarizations['cross'])**2

        # Add to Omega_spectrum
        omega_gw_freq += 2 * np.pi**2 * freqs_psd**3 * psd / (3 * H0.si.value**2)

    omega_gw_freq *= 2 / Tobs

    logger.debug('Compute Omega_ref at f_ref=' + format(f_ref, '.0f') + ' Hz')
    df = freqs_psd[1] - freqs_psd[0]
    fmin = freqs_psd[0]

    i_fref = int((f_ref - fmin) / df)
    logger.debug('True f_ref=' + format(freqs_psd[i_fref], '.1f'))

    Omega_ref = omega_gw_freq[i_fref]
    logger.info(r'Omega_ref=' + format(Omega_ref, '.2e'))


    if return_spectrum == True:
        return freqs_psd, omega_gw_freq

    else:
        return Omega_ref