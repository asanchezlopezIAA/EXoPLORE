"""
exoplore.pipelines.blasp24
==========================

Blain, Sánchez-López & Mollière (2024) data-preparation pipeline for
high-resolution spectroscopic time series.

The BLASP24 pipeline (Blain, Sánchez-López & Mollière 2024, AJ)
normalises and optionally telluric-corrects each spectrum using a
weighted polynomial fit.  Unlike BL19, it propagates uncertainties
through the fit, accounting for degrees of freedom lost in the
polynomial correction.

This module provides standalone functions for the normalisation and
telluric-correction steps of the pipeline.

References
----------
Blain, D., Sánchez-López, A. & Mollière, P. (2024), AJ.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def blasp24_normalise(
    wavelengths: np.ndarray,
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    polynomial_degree: int = 2,
    mask_threshold: float = 1e-16,
    propagate_uncertainties: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BLASP24 normalisation: divide each spectrum by a polynomial fit.

    A polynomial of degree ``polynomial_degree`` is fitted to each
    spectrum over ``good_pixels`` using optional inverse-variance weights,
    then each spectrum is divided by its fit.

    Uncertainty propagation corrects for degrees of freedom lost to the
    polynomial fit, following the weighted variance formula in
    Blain et al. (2024).

    Parameters
    ----------
    wavelengths:
        Wavelength array, shape ``(n_pixels,)``.
    data:
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    uncertainties:
        Per-pixel σ, same shape as ``data``.
    good_pixels:
        Integer indices of unmasked columns (relative to full pixel
        grid).
    weights:
        Per-pixel, per-spectrum weights, shape
        ``(n_spectra, n_pixels)``.  If ``None``, uniform weights
        are used.
    polynomial_degree:
        Degree of the normalisation polynomial.
    mask_threshold:
        Columns where the fitted polynomial is below this value are
        added to the mask (avoids dividing by near-zero fits).
    propagate_uncertainties:
        If True, scale propagated uncertainties by
        ``sqrt(valid_points / n_pixels)`` to account for degrees of
        freedom.

    Returns
    -------
    data_normalised:
        Normalised data, shape ``(n_spectra, n_pixels)``.
        Masked columns retain their input values divided by 1.
    uncertainties_normalised:
        Propagated uncertainties, same shape as ``data_normalised``.
    mask_updated:
        Updated integer mask (input mask ∪ new mask from low-fit
        values).
    good_pixels_updated:
        Updated good-pixel integer index array.
    """
    data = np.asarray(data, dtype=float).copy()
    unc = np.asarray(uncertainties, dtype=float).copy()
    n_spectra, n_pixels = data.shape
    dof = polynomial_degree + 1  # degrees of freedom consumed

    # Build mask from complement of good_pixels
    all_idx = np.arange(n_pixels, dtype=int)
    mask = np.setdiff1d(all_idx, good_pixels)

    if weights is None:
        weights = np.ones((n_spectra, n_pixels), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).copy()

    # Zero weights in masked region
    if mask.size > 0:
        weights[:, mask] = 0.0
        data[:, mask] = 1.0  # avoid hidden NaN/zero in masked region

    norm_fits = np.ones_like(data)

    wave_good = wavelengths[good_pixels]
    for n in range(n_spectra):
        coeff = np.polyfit(
            x=wave_good,
            y=data[n, good_pixels],
            deg=polynomial_degree,
            w=weights[n, good_pixels],
        )
        fit_fn = np.poly1d(coeff)
        norm_fits[n, good_pixels] = fit_fn(wave_good)

    # Expand mask where polynomial fit is near zero
    low_fit = np.any(norm_fits < mask_threshold, axis=0)
    new_bad = np.where(low_fit)[0]
    if new_bad.size > 0:
        mask = np.unique(np.concatenate([mask, new_bad]))
        good_pixels = np.setdiff1d(all_idx, mask)

    # Apply normalisation
    data /= norm_fits
    unc /= np.abs(norm_fits)

    if propagate_uncertainties and good_pixels.size > 0:
        valid_points = good_pixels.size - dof
        if valid_points > 0:
            unc *= np.sqrt(valid_points / n_pixels)

    return data, unc, mask, good_pixels


# ---------------------------------------------------------------------------
# Telluric correction (airmass-based polynomial)
# ---------------------------------------------------------------------------


def blasp24_telluric_correct(
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    airmass: np.ndarray,
    weights: Optional[np.ndarray] = None,
    mask_threshold: float = 0.8,
    polynomial_degree: int = 2,
    propagate_uncertainties: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BLASP24 telluric correction: airmass-based polynomial fit.

    For each spectral pixel in ``good_pixels``, fits a polynomial in
    airmass and divides to correct for telluric absorption variation.

    Parameters
    ----------
    data:
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    uncertainties:
        Per-pixel σ, same shape as ``data``.
    good_pixels:
        Integer indices of good columns (full pixel grid).
    airmass:
        Airmass array, shape ``(n_spectra,)``.
    weights:
        Per-pixel weights, shape ``(n_spectra, n_pixels)``.
        ``None`` → uniform weights.
    mask_threshold:
        Minimum acceptable fit value; pixels with smaller fits are
        added to the mask.
    polynomial_degree:
        Degree of the airmass polynomial.
    propagate_uncertainties:
        If True, propagate uncertainties through the division.

    Returns
    -------
    data_corrected, uncertainties_corrected, mask_updated, good_pixels_updated.
    """
    data = np.asarray(data, dtype=float).copy()
    unc = np.asarray(uncertainties, dtype=float).copy()
    n_spectra, n_pixels = data.shape
    airmass = np.asarray(airmass, dtype=float)

    all_idx = np.arange(n_pixels, dtype=int)
    mask = np.setdiff1d(all_idx, good_pixels)

    if weights is None:
        weights = np.ones((n_spectra, n_pixels), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).copy()

    if mask.size > 0:
        weights[:, mask] = 0.0

    tell_fits = np.ones_like(data)
    for k in good_pixels:
        coeff = np.polyfit(
            x=airmass,
            y=data[:, k],
            deg=polynomial_degree,
            w=weights[:, k],
        )
        fit_fn = np.poly1d(coeff)
        tell_fits[:, k] = fit_fn(airmass)

    # Expand mask
    low_fit = np.any(tell_fits < mask_threshold, axis=0)
    new_bad = np.where(low_fit)[0]
    if new_bad.size > 0:
        mask = np.unique(np.concatenate([mask, new_bad]))
        good_pixels = np.setdiff1d(all_idx, mask)

    data /= tell_fits
    if propagate_uncertainties:
        unc /= np.abs(tell_fits)

    return data, unc, mask, good_pixels


# ---------------------------------------------------------------------------
# Convenience: run full BLASP24 pipeline in one call
# ---------------------------------------------------------------------------


def run_blasp24_pipeline(
    wavelengths: np.ndarray,
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    airmass: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    polynomial_degree: int = 2,
    apply_telluric_correction: bool = False,
    telluric_mask_threshold: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full BLASP24 normalisation (+ optional telluric) pipeline.

    Parameters
    ----------
    wavelengths:
        Wavelength array, shape ``(n_pixels,)``.
    data:
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    uncertainties:
        Per-pixel σ, same shape as ``data``.
    good_pixels:
        Integer indices of unmasked columns.
    airmass:
        Airmass array, shape ``(n_spectra,)``.  Required if
        ``apply_telluric_correction=True``.
    weights:
        Per-pixel weights for polynomial fitting.  ``None`` → uniform.
    polynomial_degree:
        Degree of the normalisation (and telluric) polynomial.
    apply_telluric_correction:
        If True, run the airmass-based telluric correction after
        normalisation.
    telluric_mask_threshold:
        Threshold for the telluric correction mask step.

    Returns
    -------
    data_prepared, uncertainties_prepared, mask, good_pixels.
    """
    data_norm, unc_norm, mask, gp = blasp24_normalise(
        wavelengths, data, uncertainties, good_pixels,
        weights=weights,
        polynomial_degree=polynomial_degree,
    )
    if apply_telluric_correction:
        if airmass is None:
            raise ValueError(
                "airmass must be provided when apply_telluric_correction=True."
            )
        return blasp24_telluric_correct(
            data_norm, unc_norm, gp, airmass,
            weights=weights,
            mask_threshold=telluric_mask_threshold,
            polynomial_degree=polynomial_degree,
        )
    return data_norm, unc_norm, mask, gp
