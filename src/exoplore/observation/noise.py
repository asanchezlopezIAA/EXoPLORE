"""
exoplore.observation.noise
===========================

Photon-noise and readout-noise models for synthetic high-resolution
spectroscopic observations.

The noise model follows this approach:

1. The SNR grid provided by the ETC (instrument-specific) encodes the
   photon noise per pixel per exposure.
2. A Gaussian random noise realisation is drawn and scaled by the SNR.
3. A user-supplied ``noise_scaling_factor`` can inflate or deflate the
   noise (useful for sensitivity studies).

All functions work with flux units that are normalised to 1 (i.e. the
stellar spectrum is assumed to be unity at the continuum).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SNR → noise conversion
# ---------------------------------------------------------------------------


def photon_noise(
    snr_per_pixel: np.ndarray,
    noise_scaling_factor: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw a Gaussian photon-noise realisation from an SNR map.

    Parameters
    ----------
    snr_per_pixel:
        SNR per spectral pixel per exposure.  Shape can be
        ``(n_pixels,)`` or ``(n_spectra, n_pixels)``.
    noise_scaling_factor:
        Multiplicative scaling applied to the noise σ.  Default 1.0
        (no scaling).
    rng:
        NumPy random Generator.  If ``None``, a new default Generator
        is created.

    Returns
    -------
    np.ndarray
        Gaussian noise realisation, same shape as ``snr_per_pixel``.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = noise_scaling_factor / np.asarray(snr_per_pixel, dtype=float)
    return rng.normal(0.0, sigma)


def total_noise(
    snr_per_pixel: np.ndarray,
    read_noise_electrons: float = 0.0,
    dark_current_electrons_per_s: float = 0.0,
    exposure_time_seconds: float = 0.0,
    noise_scaling_factor: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine photon noise, read noise, and dark current into total noise.

    Parameters
    ----------
    snr_per_pixel:
        Photon-limited SNR per pixel per exposure.
    read_noise_electrons:
        Read noise in electrons (RMS).  0 → negligible read noise.
    dark_current_electrons_per_s:
        Dark current rate in e⁻/s.  0 → negligible dark current.
    exposure_time_seconds:
        Exposure time in seconds.  Required if dark current > 0.
    noise_scaling_factor:
        Global scaling factor applied to the total σ.

    Returns
    -------
    noise_realisation:
        Gaussian noise draw, same shape as ``snr_per_pixel``.
    sigma_total:
        Total σ per pixel (useful as the uncertainty array for CCF).
    """
    if rng is None:
        rng = np.random.default_rng()

    snr = np.asarray(snr_per_pixel, dtype=float)
    sigma_photon = 1.0 / snr

    # Read noise contribution (in relative flux units)
    # Approximation: σ_read ≈ read_noise / (signal counts)
    # Since SNR = signal/sigma_photon, signal ~ SNR²
    signal_counts = snr ** 2  # approximate photon count
    sigma_read = np.where(
        signal_counts > 0,
        read_noise_electrons / signal_counts,
        0.0,
    )

    # Dark current contribution
    dark_electrons = dark_current_electrons_per_s * exposure_time_seconds
    sigma_dark = np.where(
        signal_counts > 0,
        np.sqrt(dark_electrons) / signal_counts,
        0.0,
    )

    sigma_total = noise_scaling_factor * np.sqrt(
        sigma_photon ** 2 + sigma_read ** 2 + sigma_dark ** 2
    )
    noise_realisation = rng.normal(0.0, sigma_total)
    return noise_realisation, sigma_total


def snr_per_pixel(
    flux: np.ndarray,
    exposure_time_seconds: float,
    telescope_area_cm2: float,
    throughput: float = 0.1,
    read_noise_electrons: float = 5.0,
    dark_current_electrons_per_s: float = 0.01,
) -> np.ndarray:
    """Estimate per-pixel SNR from source flux.

    A simplified photon-counting SNR formula:

    .. math::

        \\text{SNR} = \\frac{S t}{\\sqrt{S t + N_{\\rm read}^2 + D t}}

    where *S* = detected photon rate, *t* = exposure time,
    *N_read* = read noise, *D* = dark current rate.

    Parameters
    ----------
    flux:
        Source flux density in erg/s/cm²/Å, shape ``(n_pixels,)``.
    exposure_time_seconds:
        Exposure time in seconds.
    telescope_area_cm2:
        Effective collecting area in cm².
    throughput:
        Total system throughput (telescope × instrument × detector).
    read_noise_electrons:
        Read noise in electrons.
    dark_current_electrons_per_s:
        Dark current rate in e⁻/s.

    Returns
    -------
    np.ndarray
        Estimated SNR per pixel, shape ``(n_pixels,)``.
    """
    flux = np.asarray(flux, dtype=float)
    # Approximate photon energy as hν ~ 3.3e-12 erg at 1 μm
    h_erg_s = 6.626e-27
    c_cm_s = 3e10
    energy_per_photon = h_erg_s * c_cm_s / 1e-4  # crude ~1 μm

    signal = flux * telescope_area_cm2 * throughput * exposure_time_seconds / energy_per_photon
    noise_sq = signal + read_noise_electrons ** 2 + dark_current_electrons_per_s * exposure_time_seconds
    return signal / np.sqrt(noise_sq)


def compute_global_exposure_limit(R, pixels_per_res, Kp, P, max_dv_bary=0.):
    """Compute a worst-case exposure time that avoids one-pixel velocity smearing.

    Assumes a sinusoidal planet RV with semi-amplitude *Kp* and period *P*,
    plus a known maximum barycentric velocity rate *max_dv_bary*.

    Parameters
    ----------
    R : float
        Instrument resolving power.
    pixels_per_res : float
        Detector pixels per resolution element.
    Kp : float
        Planet RV semi-amplitude in km/s.
    P : float
        Orbital period in days.
    max_dv_bary : float, optional
        Maximum barycentric velocity rate in km/s per day.  Default 0.

    Returns
    -------
    t_max : float
        Uniform exposure limit in seconds that avoids >1-pixel Doppler shift.
    dvdt_total : float
        Maximum total dv/dt in km/s per day used in the calculation.
    """
    from exoplore.instruments import compute_pixel_velocity_scale
    dvdt_planet = 2 * np.pi * Kp / P          # km/s per day
    dvdt_total = dvdt_planet + abs(max_dv_bary)
    dvdt_sec = dvdt_total / 86400.0            # km/s per second
    v_pixel = compute_pixel_velocity_scale(R, pixels_per_res)
    t_max = v_pixel / dvdt_sec
    return t_max, dvdt_total


# ---------------------------------------------------------------------------
# v0.23 additions
# ---------------------------------------------------------------------------

def add_throughput(
    F: np.ndarray,
    jitter_frac: float = 0.02,
    mode: str = "white",
    red_smooth_sigma: float = 2.0,
    seed: int = None,
) -> np.ndarray:
    """
    Multiply each exposure in F by a random throughput factor.

    Parameters
    ----------
    F : numpy.ndarray
        Input flux array of shape ``(n_spectra, n_pixels)``.
    jitter_frac : float, optional
        1-sigma fractional jitter (e.g. 0.02 for 2 %).  Default 0.02.
    mode : str, optional
        ``'white'`` for uncorrelated draws each exposure;
        ``'red'`` for a temporally correlated (smoothed) sequence.
        Default ``'white'``.
    red_smooth_sigma : float, optional
        Gaussian kernel sigma (in exposures) for the ``'red'`` jitter.
        Default 2.0.
    seed : int, optional
        Random seed for reproducibility.  Default ``None``.

    Returns
    -------
    F_mod : numpy.ndarray
        Throughput-modulated flux array, same shape as *F*.
    """
    import scipy.ndimage

    rng = np.random.default_rng(seed)
    n_spectra = F.shape[0]

    # draw raw jitter factors around unity
    raw = rng.normal(loc=1.0, scale=jitter_frac, size=n_spectra)

    if mode == "white":
        T = raw

    elif mode == "red":
        # smooth to introduce temporal correlation
        T = scipy.ndimage.gaussian_filter1d(raw, sigma=red_smooth_sigma)
    else:
        raise ValueError("mode must be 'white' or 'red'")

    # apply multiplicative throughput to each exposure
    F_mod = F * T[:, None]
    return F_mod
