"""Anchor relative-flux photon uncertainties to an ANDES central ETC resel."""
from dataclasses import dataclass

import numpy as np

from exoplore.instruments.wavegrid import pixel_snr_for_andes_order


@dataclass(frozen=True)
class AndesPhotonNoise:
    """Exposure-by-pixel uncertainties, local S/N and effective photon response."""
    sigma: np.ndarray
    snr: np.ndarray
    response: np.ndarray


def andes_photon_noise(wavelength_orders_um, order_id, expected_flux,
                      stellar_reference, tellurics, etc_snr, etc_midpoints_nm,
                      *, pixels_per_resel, stellar_divided=False):
    """Use the central resel as a photon-count anchor for each exposure.

    expected_flux is the noiseless matrix on the simulator's actual flux scale.
    stellar_reference contains the unocculted stellar spectrum on that order's
    observer-frame pixels. The ETC anchor is assigned to this stellar model
    behind the exposure's central telluric transmission, at unit throughput.
    This assignment is an approximation because the ETC's own spectra are absent.

    The effective response gives photons per unit model flux. Relative blaze
    redistributes that response across the order. Stellar lines, tellurics,
    planetary transmission and throughput enter through the expected photons.
    For spectra expressed relative to their stellar spectrum, stellar_divided
    reconstructs those photons and returns sigma in the supplied flux units.
    """
    flux = np.asarray(expected_flux, dtype=float)
    star = np.asarray(stellar_reference, dtype=float)
    tell = np.asarray(tellurics, dtype=float)
    if (flux.ndim != 2 or star.shape != flux.shape or tell.shape != flux.shape
            or not np.isfinite(flux).all() or np.any(flux < 0)
            or not np.isfinite(star).all() or np.any(star <= 0)
            or not np.isfinite(tell).all() or np.any(tell < 0) or np.any(tell > 1)
            or not np.isfinite(pixels_per_resel) or pixels_per_resel <= 0):
        raise ValueError('Supply finite matching flux, positive stellar reference and telluric matrices')
    wave = np.asarray(wavelength_orders_um[order_id], dtype=float)
    if wave.shape != (flux.shape[1],):
        raise ValueError('ANDES wavelength and flux pixels must match')
    centre = int(np.argmin(abs(wave-etc_midpoints_nm[order_id]*1e-3)))
    if np.any(tell[:, centre] <= 0):
        raise ValueError('The ANDES ETC anchor requires positive central telluric transmission')
    # Unit transmission isolates the blaze and the resel-to-pixel conversion.
    anchor_snr = pixel_snr_for_andes_order(
        wavelength_orders_um, order_id, np.ones_like(tell), etc_snr,
        etc_midpoints_nm, px_per_resel=pixels_per_resel)
    if not np.isfinite(anchor_snr).all() or np.any(anchor_snr <= 0):
        raise ValueError('ANDES ETC anchors must be finite and positive')
    response = anchor_snr**2/(star[:, centre]*tell[:, centre])[:, None]
    if not np.isfinite(response).all() or np.any(response <= 0):
        raise ValueError('The central ETC assignment gives a nonfinite photon response')
    conversion = star if stellar_divided else np.ones_like(star)
    sigma = np.sqrt(flux*conversion/response)/conversion
    snr = np.divide(flux, sigma, out=np.zeros_like(flux), where=sigma > 0)
    return AndesPhotonNoise(sigma, snr, response)
