"""exoplore.pipelines.helium_math
=================================

Numerical primitives for direct, vacuum He I transmission analysis.

All wavelengths are nm. Velocities are km/s, positive for recession. These
functions implement the frame and multiplet conventions of the czesla2024 pipeline. Interpolation never bridges
an invalid native pixel. Variance propagation below is marginal only. inference
must also account for shared reference noise and interpolation covariance.


Seidel22 also calls the frame, interpolation, window and weighted-mean
functions in this file. Supplying exposure_weights selects its continuum-S/N
weighting. The default weighted_mean calculation uses inverse variance.
"""
from __future__ import annotations

# Role in the calculation
# Convert wavelength frames and resample direct-helium spectra. These functions use
# vacuum wavelengths in nm and velocities in km/s, and carry valid-pixel and detector-
# gap masks through interpolation.


import numpy as np
from scipy.optimize import least_squares

C_KMS = 299792.458


def air_to_vacuum_nm(wavelength: np.ndarray) -> np.ndarray:
    """Standard dry-air Edlén conversion. Never apply this to cr2res vacuum data."""
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
    # Apply the additive barycentric factor, then remove the supplied stellar
    # recession factor. Seidel22 includes the full observed stellar motion in
    # gamma_kms and supplies berv_kms=0 when using this helper.
    return np.asarray(wave_nm) * (1 + berv_kms / C_KMS) / (1 + gamma_kms / C_KMS)


def planet_wavelength(stellar_nm: np.ndarray, planet_rv_kms: float) -> np.ndarray:
    """Remove the planet's receding velocity using the optical convention."""
    # Divide by the planet/star Doppler factor to put the line at its rest wavelength.
    return np.asarray(stellar_nm) / (1 + planet_rv_kms / C_KMS)


def window_mask(wavelength_nm: np.ndarray, windows_nm: list[list[float]]) -> np.ndarray:
    """Mark inclusive windows on the supplied frame's native grid."""
    # Start with all wavelength channels outside the requested windows.
    result = np.zeros(np.shape(wavelength_nm), bool)
    # Combine the inclusive lower and upper bounds of every supplied interval.
    for lo, hi in windows_nm:
        if lo >= hi:
            raise ValueError("windows must have positive widths")
        # A channel is selected when it lies inside at least one window.
        result |= (wavelength_nm >= lo) & (wavelength_nm <= hi)
    return result


def resample(wave: np.ndarray, flux: np.ndarray, variance: np.ndarray,
             destination: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation with squared variance weights and valid endpoints."""
    if np.any(np.diff(wave) <= 0):
        raise ValueError("wavelength grid must strictly increase")
    # Locate the input pixel immediately below each destination wavelength.
    index = np.searchsorted(wave, destination, side="right") - 1
    # Keep indices within the available neighbouring input-pixel pairs.
    index = np.clip(index, 0, len(wave) - 2)
    # t gives the fractional position between two original pixels. The same
    # endpoints interpolate the flux. Their squared coefficients propagate the
    # diagonal variances. This helper does not return full pixel covariance.
    t = (destination - wave[index]) / (wave[index + 1] - wave[index])
    good = ((t >= 0) & (t <= 1) & np.isfinite(flux[index]) & np.isfinite(flux[index + 1])
            & np.isfinite(variance[index]) & np.isfinite(variance[index + 1]))
    # Valid endpoints alone do not identify a physical detector gap. The
    # CARMENES gap_resample wrapper additionally excludes such gap intervals.
    y = np.where(good, (1 - t) * flux[index] + t * flux[index + 1], np.nan)
    v = np.where(good, (1 - t) ** 2 * variance[index] + t ** 2 * variance[index + 1], np.nan)
    return y, v


def weighted_mean(flux: np.ndarray, variance: np.ndarray, *,
                  exposure_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Combine finite pixels and propagate their independent variances.

    Helium callers use the default inverse-variance weights. Seidel22 supplies
    one continuum S/N weight per exposure. Both choices renormalise weights at
    each pixel so that masked samples contribute zero weight.
    """
    # Use explicitly supplied exposure weights for Seidel22 reference and transit averages.
    if exposure_weights is not None:
        # Require one positive finite weight for each exposure.
        selected = np.asarray(exposure_weights, dtype=float)
        if (selected.shape != (len(flux),) or np.any(~np.isfinite(selected))
                or np.any(selected <= 0)):
            raise ValueError("Exposure weights must be finite and positive, one per exposure")
        # Define usable exposure/pixel pairs before normalizing the averaging weights.
        good = np.isfinite(flux) & np.isfinite(variance) & (variance >= 0)
        # Apply the exposure weight at each valid wavelength, with zero contribution from masks.
        weight = np.where(good, selected[:, None], 0.0)
        # Sum available weights separately at each wavelength channel.
        # Sum inverse variances at each output wavelength.
        den = weight.sum(axis=0)
        # Normalize those weights to unity where usable data exist.
        weight = np.divide(weight, den, out=np.zeros_like(weight), where=den > 0)
        # Calculate the weighted flux average over exposure rows.
        out = np.sum(weight * np.where(good, flux, 0), axis=0)
        # Independent variances combine with the square of each normalized flux weight.
        var = np.sum(weight**2 * np.where(good, variance, 0), axis=0)
        # Mark channels with no contributing exposure as missing.
        out[den == 0] = np.nan
        var[den == 0] = np.nan
        return out, var
    # The default weighting uses the inverse of each positive variance.
    good = np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
    # Assign zero weight to invalid pixels before calculating the inverse-variance mean.
    weight = np.divide(1, variance, out=np.zeros_like(variance), where=good)
    den = weight.sum(axis=0)
    # Divide the weighted flux sum by its total inverse-variance weight.
    out = np.divide(np.sum(np.where(good, flux, 0) * weight, axis=0), den,
                    out=np.full_like(den, np.nan), where=den > 0)
    # The variance of this inverse-variance mean is the reciprocal total weight.
    var = np.divide(1, den, out=np.full_like(den, np.nan), where=den > 0)
    return out, var


def gap_resample(wave, flux, variance, destination):
    """Interpolate flux and variance within contiguous detector sections.

    A separation larger than five median pixel widths identifies a detector
    gap. CARMENES and Seidel22 use the same interpolation and gap criterion.
    """
    # First interpolate using the valid neighbouring endpoints and squared variance weights.
    y, v = resample(wave, flux, variance, destination)
    # Measure local pixel spacing on the source wavelength grid.
    widths = np.diff(wave)
    # Flag intervals wider than five times the median spacing as detector gaps.
    gaps = np.flatnonzero(widths > 5*np.median(widths))
    # Remove interpolated values lying strictly inside each gap.
    for i in gaps:
        # Keep the measured endpoints while excluding wavelengths between the detector sections.
        missing = (destination > wave[i]) & (destination < wave[i+1])
        y[missing] = np.nan
        v[missing] = np.nan
    return y, v


def multiplet(wave_nm: np.ndarray, column_cm2: float, shift_kms: float,
              intrinsic_sigma_kms: float, filling_factor: float,
              systematic_sigma_kms: float, lines_nm: np.ndarray,
              oscillator_strengths: np.ndarray) -> np.ndarray:
    """Czesla slab: T=1-f+f*exp(-sum tau_j), shared shift and Gaussian width.

    tau_j=N*pi*e^2/(m_e*c)*f_j*lambda_j/(sqrt(2*pi)*sigma_v)*exp(-v_j^2/(2*sigma_v^2)).
    The integrated cross-section constant is 0.02654029 cm² Hz (Gaussian cgs).
    As in Czesla Eq. 2, instrumental/smearing broadening enters optical depth.
    The slab approximation applies in the weak-saturation regime.
    """
    if column_cm2 < 0 or intrinsic_sigma_kms <= 0 or not 0 < filling_factor <= 1:
        raise ValueError("invalid physical multiplet parameters")
    sigma = np.hypot(intrinsic_sigma_kms, systematic_sigma_kms)
    velocity = C_KMS * (np.asarray(wave_nm)[:, None] / np.asarray(lines_nm)[None, :] - 1)
    tau = column_cm2 * 0.02654029 * np.asarray(oscillator_strengths) * (np.asarray(lines_nm) * 1e-7)
    tau = np.sum(tau / (np.sqrt(2 * np.pi) * sigma * 1e5) *
                 np.exp(-0.5 * ((velocity - shift_kms) / sigma) ** 2), axis=1)
    return 1 - filling_factor + filling_factor * np.exp(-tau)
