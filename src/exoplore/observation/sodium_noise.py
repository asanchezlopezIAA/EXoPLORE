"""Photon-dominated uncertainties for direct CARMENES sodium transmission.

Called by core/atomic_simulation.py after instrumental sampling. The ETC
continuum anchor sets the photon response. The expected flux then gives the
uncertainty for each exposure and wavelength pixel on the observation scale.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SodiumNoiseEstimate:
    """Uncertainty, local S/N and effective photon response for the same flux."""

    # Gaussian standard deviation in the same units as expected_flux.
    std_noise: np.ndarray
    # Expected S/N including the stellar lines, transit and terrestrial absorption.
    local_snr_pixel: np.ndarray
    # Per-pixel continuum S/N from the ETC, blaze and relative telluric scaling.
    continuum_snr_pixel: np.ndarray
    # Effective counts per unit relative flux, used to calculate photon variance.
    photon_response: np.ndarray


def sodium_photon_uncertainty(etc, order, native_wave_um, expected_flux,
                              telluric_transmission, airmass,
                              exposure_seconds, resolving_power,
                              selected_pixels=None):
    """Derive Gaussian sigma from the ETC continuum anchor and expected photons.

    expected_flux contains the stellar lines, transit, atmosphere, tellurics
    and throughput in out-of-transit stellar-continuum units. It precedes
    the random noise draw. The analytic blaze is represented in the detector
    response. The supplied flux already includes telluric attenuation.

    The ETC maximum sets the continuum S/N under the adopted atmospheric
    reference. If S_cont is the relative-telluric/blaze S/N and T is the
    transmission, the effective photon response is A = S_cont**2 / T.
    Then variance(f) = f/A and local S/N = sqrt(f*A).

    Supply the complete native wavelength grid to position the blaze. For a
    cropped simulation, selected_pixels identifies its native columns. The
    crop must contain every ETC anchor used by the supplied airmass grid.
    Background and detector variance are absorbed into the effective ETC
    anchor under this photon-dominated approximation.
    """
    # Use floating-point arrays for the full order and the selected observation.
    wave = np.asarray(native_wave_um, float)
    flux = np.asarray(expected_flux, float)
    tell = np.asarray(telluric_transmission, float)
    # Remember where the cropped wavelengths sit within the full detector order.
    indices = (np.arange(wave.size) if selected_pixels is None
               else np.asarray(selected_pixels))
    # Check pixel membership, exposure dimensions and physically allowed fluxes.
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or indices.size == 0 or np.any(indices < 0)
            or np.any(indices >= wave.size) or np.any(np.diff(indices) <= 0)
            or flux.shape != (len(airmass), indices.size)
            or tell.shape != flux.shape or not np.isfinite(flux).all()
            or np.any(flux < 0) or not np.isfinite(tell).all()
            or np.any(tell <= 0) or np.any(tell > 1)):
        raise ValueError('Supply nonnegative expected flux and positive transmission on native pixels')
    if not 0 <= order < etc.wave_nm.shape[0]:
        raise ValueError('Physical order index is outside the CARMENES ETC bundle')
    # Select this physical order at every downloaded ETC airmass.
    curves = etc.snr_per_resel[:, order]
    # Give unavailable ETC samples zero weight when locating the maximum.
    valid_curves = np.where(np.isfinite(curves) & (curves > 0), curves, 0.)
    # Each ETC airmass supplies its own maximum-S/N wavelength.
    peaks = np.argmax(valid_curves, axis=1)
    # Convert the ETC wavelengths from nm to micrometres.
    anchor_waves = etc.wave_nm[order, peaks]/1000
    # Find the corresponding native detector pixels.
    anchor_indices = [int(np.argmin(abs(wave-value))) for value in anchor_waves]
    # The anchor must lie inside the simulated crop for its transmission to be known.
    if not np.isin(anchor_indices, indices).all():
        raise ValueError('The sodium wavelength interval must contain the ETC anchor wavelengths')

    # Use the complete order coordinates for the blaze and the known transmission
    # in the observed crop. All ETC anchor wavelengths lie inside that crop.
    full_tell = np.ones((flux.shape[0], wave.size))
    full_tell[:, indices] = tell
    # Use the shared ETC conversion, then retain the observation columns.
    continuum_snr = etc.pixel_snr(
        order, wave, full_tell, airmass, exposure_seconds, resolving_power)[:, indices]
    # Recover the effective photon response A from S_cont squared divided by T.
    response = continuum_snr**2/tell
    if not np.isfinite(response).all() or np.any(response <= 0):
        raise ValueError('The sodium ETC anchor gives an invalid photon response')
    # Transmission already appears in flux. Dividing S_cont squared by T
    # expresses detector sensitivity per unit of that relative flux.
    # Photon variance is expected flux divided by the effective response.
    sigma = np.sqrt(flux/response)
    # The same response gives expected flux divided by its standard deviation.
    local_snr = np.sqrt(flux*response)
    # Return matching arrays for noise generation and later diagnostics.
    return SodiumNoiseEstimate(sigma, local_snr, continuum_snr, response)
