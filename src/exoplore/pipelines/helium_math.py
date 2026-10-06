"""exoplore.pipelines.helium_math
============================

Numerical primitives for direct, vacuum He I transmission analysis.

All wavelengths are nm; velocities are km/s, positive for recession. These
functions implement the frame and multiplet conventions of the czesla2024 pipeline. Interpolation never bridges
an invalid native pixel. Variance propagation below is marginal only; inference
must also account for shared reference noise and interpolation covariance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

C_KMS = 299792.458


def air_to_vacuum_nm(wavelength: np.ndarray) -> np.ndarray:
    """Standard dry-air Edlén conversion; never apply this to cr2res vacuum data."""
    w = np.asarray(wavelength, float)
    if np.any(~np.isfinite(w)) or np.any((w < 200) | (w > 2000)):
        raise ValueError("Edlen conversion here is restricted to 200–2000 nm")
    s2 = (1000 / w) ** 2
    return w * (1 + 1e-8 * (8342.13 + 2406030 / (130 - s2) + 15997 / (38.9 - s2)))


def stellar_wavelength(wave_nm: np.ndarray, berv_kms: float, gamma_kms: float) -> np.ndarray:
    """Undo stellar recession after applying Astropy's additive barycentric RV.

    Optical Doppler convention: lambda_star = lambda_top * (1+BERV/c) /
    (1+gamma/c). Positive gamma shifts an observed red line back to the blue.
    """
    return np.asarray(wave_nm) * (1 + berv_kms / C_KMS) / (1 + gamma_kms / C_KMS)


def planet_wavelength(stellar_nm: np.ndarray, planet_rv_kms: float) -> np.ndarray:
    """Remove the planet's receding velocity using the optical convention."""
    return np.asarray(stellar_nm) / (1 + planet_rv_kms / C_KMS)


def czesla_indices(n_exposures: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based reference 1–10,36–40 and full-transit 16–32 indices."""
    if n_exposures != 40:
        raise ValueError("Czesla running numbers require exactly 40 chronological exposures")
    return np.r_[0:10, 35:40], np.arange(15, 32)


def window_mask(wavelength_nm: np.ndarray, windows_nm: list[list[float]]) -> np.ndarray:
    """Mark inclusive windows on the supplied frame's native grid."""
    result = np.zeros(np.shape(wavelength_nm), bool)
    for lo, hi in windows_nm:
        if lo >= hi:
            raise ValueError("windows must have positive widths")
        result |= (wavelength_nm >= lo) & (wavelength_nm <= hi)
    return result


def resample(wave: np.ndarray, flux: np.ndarray, variance: np.ndarray,
             destination: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation with squared variance weights and no gap bridging."""
    if np.any(np.diff(wave) <= 0):
        raise ValueError("wavelength grid must strictly increase")
    index = np.searchsorted(wave, destination, side="right") - 1
    index = np.clip(index, 0, len(wave) - 2)
    t = (destination - wave[index]) / (wave[index + 1] - wave[index])
    good = ((t >= 0) & (t <= 1) & np.isfinite(flux[index]) & np.isfinite(flux[index + 1])
            & np.isfinite(variance[index]) & np.isfinite(variance[index + 1]))
    y = np.where(good, (1 - t) * flux[index] + t * flux[index + 1], np.nan)
    v = np.where(good, (1 - t) ** 2 * variance[index] + t ** 2 * variance[index + 1], np.nan)
    return y, v


def weighted_mean(flux: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance mean of finite pixels; empty coverage remains NaN."""
    good = np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
    weight = np.divide(1, variance, out=np.zeros_like(variance), where=good)
    den = weight.sum(axis=0)
    out = np.divide(np.sum(np.where(good, flux, 0) * weight, axis=0), den,
                    out=np.full_like(den, np.nan), where=den > 0)
    var = np.divide(1, den, out=np.full_like(den, np.nan), where=den > 0)
    return out, var


def multiplet(wave_nm: np.ndarray, column_cm2: float, shift_kms: float,
              intrinsic_sigma_kms: float, filling_factor: float,
              systematic_sigma_kms: float, lines_nm: np.ndarray,
              oscillator_strengths: np.ndarray) -> np.ndarray:
    """Czesla slab: T=1-f+f*exp(-sum tau_j), shared shift and Gaussian width.

    tau_j=N*pi*e^2/(m_e*c)*f_j*lambda_j/(sqrt(2*pi)*sigma_v)*exp(-v_j^2/(2*sigma_v^2)).
    The integrated cross-section constant is 0.02654029 cm² Hz (Gaussian cgs).
    As in Czesla Eq. 2, instrumental/smearing broadening enters optical depth;
    this is an approximation appropriate to weak saturation, not a 3D model.
    """
    if column_cm2 < 0 or intrinsic_sigma_kms <= 0 or not 0 < filling_factor <= 1:
        raise ValueError("invalid physical multiplet parameters")
    sigma = np.hypot(intrinsic_sigma_kms, systematic_sigma_kms)
    velocity = C_KMS * (np.asarray(wave_nm)[:, None] / np.asarray(lines_nm)[None, :] - 1)
    tau = column_cm2 * 0.02654029 * np.asarray(oscillator_strengths) * (np.asarray(lines_nm) * 1e-7)
    tau = np.sum(tau / (np.sqrt(2 * np.pi) * sigma * 1e5) *
                 np.exp(-0.5 * ((velocity - shift_kms) / sigma) ** 2), axis=1)
    return 1 - filling_factor + filling_factor * np.exp(-tau)
