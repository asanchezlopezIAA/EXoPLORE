"""ETC-based photon uncertainties for direct ANDES sodium observations.

The direct sodium runner calls this after building its noiseless observation.
The instrument ETC table supplies the order-centre S/N per resolution element.
The shared ANDES helper distributes that anchor over pixels and wavelength.
"""

import numpy as np

from exoplore.instruments.wavegrid import pixel_snr_one_order
from exoplore.observation.sodium_noise import SodiumNoiseEstimate


def andes_sodium_photon_uncertainty(native_wave_um, selected_pixels, expected_flux,
                                   telluric_transmission, snr_per_resel,
                                   pixels_per_resel, center_um):
    """Use the ANDES order-centre ETC anchor on the synthetic flux scale.

    The existing ANDES helper supplies the blaze and relative telluric shape.
    The expected flux supplies stellar absorption, transit and throughput.
    A = S_cont**2 / T expresses the effective photon response, giving
    sigma(f) = sqrt(f / A). ETC exposure duration is checked by the caller.
    """
    # Keep the full native order coordinates to calculate the blaze shape.
    wave = np.asarray(native_wave_um, float)
    # These indices select the simulated interval within that complete order.
    indices = np.asarray(selected_pixels)
    # Flux and terrestrial transmission have shape (exposure, selected pixel).
    flux = np.asarray(expected_flux, float)
    tell = np.asarray(telluric_transmission, float)
    # Check that the arrays, S/N anchor and detector sampling describe one order.
    if (wave.ndim != 1 or wave.size < 3 or not np.isfinite(wave).all()
            or np.any(np.diff(wave) <= 0) or np.any(wave <= 0)
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or indices.size == 0 or np.any(indices < 0)
            or np.any(indices >= wave.size) or np.any(np.diff(indices) <= 0)
            or flux.ndim != 2 or flux.shape[1] != indices.size
            or flux.shape != tell.shape or not np.isfinite(flux).all()
            or np.any(flux <= 0) or not np.isfinite(tell).all()
            or np.any(tell <= 0) or np.any(tell > 1)
            or not np.isfinite(snr_per_resel) or snr_per_resel <= 0
            or not np.isfinite(pixels_per_resel) or pixels_per_resel <= 0
            or not np.isfinite(center_um) or not wave[0] <= center_um <= wave[-1]):
        raise ValueError('Supply positive ANDES ETC and photon inputs on native pixels')
    # Locate the ETC order centre on the instrument wavelength grid.
    center = int(np.argmin(abs(wave-center_um)))
    # The observation must include the centre used to anchor the S/N.
    if center not in indices:
        raise ValueError('The sodium interval must contain the ANDES ETC centre')
    # Complete order coordinates determine the blaze width. Only selected
    # pixels enter the returned spectra and their photon uncertainties.
    full_tell = np.ones((len(flux), wave.size))
    full_tell[:, indices] = tell
    # Convert S/N per resolution element to pixels and apply the blaze and
    # relative telluric shape using the existing ANDES calculation.
    continuum = pixel_snr_one_order(
        wave, full_tell, snr_per_resel,
        px_per_resel=pixels_per_resel, center_pix=center)[:, indices]
    # A = S_cont squared / T gives effective counts per unit relative flux.
    response = continuum**2/tell
    # Expected flux / A is the photon variance on the synthetic flux scale.
    sigma = np.sqrt(flux/response)
    # Return the standard deviation, local S/N, continuum S/N and response.
    return SodiumNoiseEstimate(sigma, np.sqrt(flux*response), continuum, response)
