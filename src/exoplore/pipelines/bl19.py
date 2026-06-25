"""
exoplore.pipelines.bl19
=======================

Brogi & Line (2019) data-preparation pipeline for high-resolution
spectroscopic time series.

The BL19 pipeline (Brogi & Line 2019, AJ, 157, 114) normalises each
spectrum by the mean of its brightest pixels (or by a polynomial fit
to the spectral envelope when ``use_envelope_fit=True``), then performs
two rounds of telluric correction via polynomial fitting to the median
spectrum and to each pixel's time evolution.

This module provides standalone ``pipeline_BL19_norm`` and
``pipeline_BL19_tellcorr`` functions that operate without the
``inp_dat`` dependency.

References
----------
Brogi, M. & Line, M. R. (2019), AJ, 157, 114.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def bl19_normalise(
    wavelengths: np.ndarray,
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    n_brightest: int = 300,
    use_envelope_fit: bool = False,
    n_envelope_bins: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """BL19 normalisation: divide each spectrum by its brightness envelope.

    Two modes:

    ``use_envelope_fit=False`` (default)
        Divide each spectrum by the mean of its ``n_brightest`` pixels.
        Fast and robust for most cases.

    ``use_envelope_fit=True``
        Divide by a 2nd-order polynomial fit to the per-interval
        maximum values.  Better handles colour slopes across broad
        wavelength ranges.

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
    n_brightest:
        Number of brightest pixels to use in the mean normalisation
        (``use_envelope_fit=False`` only).
    use_envelope_fit:
        If True, use the polynomial envelope fit instead.
    n_envelope_bins:
        Number of wavelength bins for the envelope fit
        (``use_envelope_fit=True`` only).

    Returns
    -------
    data_normalised:
        Normalised spectra restricted to ``good_pixels``,
        shape ``(n_spectra, n_good_pixels)``.
    uncertainty_normalised:
        Propagated uncertainties, same shape.
    """
    wave = wavelengths[good_pixels]
    mat = data[:, good_pixels].copy().astype(float)
    noise = uncertainties[:, good_pixels].copy().astype(float)

    n_spectra, n_good = mat.shape

    if not use_envelope_fit:
        # Mean-of-brightest-pixels normalisation
        nb = min(n_brightest, n_good)
        brightest_idx = np.argsort(mat, axis=1)[:, -nb:]
        brightest_vals = mat[np.arange(n_spectra)[:, None], brightest_idx]
        mean_bright = np.mean(brightest_vals, axis=1)  # (n_spectra,)

        result = mat / mean_bright[:, None]
        error = result * np.sqrt((noise / mat) ** 2)

    else:
        # Polynomial envelope fit
        nb = min(n_envelope_bins, n_good)
        interval = n_good // nb
        mean_waves = np.array(
            [np.mean(wave[i * interval: (i + 1) * interval]) for i in range(nb)]
        )

        result = np.ones_like(mat)
        error = np.ones_like(noise)

        for n in range(n_spectra):
            max_vals = [
                np.max(mat[n, i * interval: (i + 1) * interval])
                for i in range(nb)
            ]
            coeff = np.polyfit(mean_waves, max_vals, deg=2)
            fit = np.polyval(coeff, wave)
            result[n] = mat[n] / fit
            error[n] = result[n] * np.sqrt((noise[n] / mat[n]) ** 2)

    return result, error


# ---------------------------------------------------------------------------
# Telluric correction
# ---------------------------------------------------------------------------


def bl19_telluric_correct(
    data_normalised: np.ndarray,
    uncertainties_normalised: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """BL19 telluric correction: two-stage polynomial subtraction.

    Stage 1, temporal correction:
        For each spectrum, fit a 2nd-order polynomial to the median
        spectrum and divide.

    Stage 2, pixel-by-pixel correction:
        For each spectral pixel, fit a 2nd-order polynomial to the
        time evolution and divide.

    Parameters
    ----------
    data_normalised:
        Normalised spectra, shape ``(n_spectra, n_good_pixels)``.
        Already restricted to good pixels (output of
        :func:`bl19_normalise`).
    uncertainties_normalised:
        Propagated uncertainties, same shape.

    Returns
    -------
    data_corrected:
        Telluric-corrected spectra, shape ``(n_spectra, n_good_pixels)``.
    uncertainties_corrected:
        Propagated uncertainties, same shape.
    """
    mat = data_normalised.copy().astype(float)
    noise = uncertainties_normalised.copy().astype(float)
    n_spectra, n_pixels = mat.shape

    # Stage 1: divide by polynomial fit to the median spectrum
    tel_spec = np.median(mat, axis=0)  # (n_pixels,)
    mat2 = np.empty_like(mat)
    noise2 = np.empty_like(noise)

    for n in range(n_spectra):
        coeff = np.polyfit(tel_spec, mat[n], deg=2)
        fit = np.polyval(coeff, tel_spec)
        mat2[n] = mat[n] / fit
        noise2[n] = mat2[n] * np.sqrt((noise[n] / mat[n]) ** 2)

    # Stage 2: divide by polynomial fit to each pixel's time evolution
    indices = np.arange(n_spectra)
    mat3 = np.empty_like(mat2)
    noise3 = np.empty_like(noise2)

    for k in range(n_pixels):
        coeff = np.polyfit(indices, mat2[:, k], deg=2)
        fit = np.polyval(coeff, indices)
        mat3[:, k] = mat2[:, k] / fit
        noise3[:, k] = mat3[:, k] * np.sqrt((noise2[:, k] / mat2[:, k]) ** 2)

    return mat3, noise3


# ---------------------------------------------------------------------------
# Convenience: run full BL19 pipeline in one call
# ---------------------------------------------------------------------------


def run_bl19_pipeline(
    wavelengths: np.ndarray,
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    apply_telluric_correction: bool = True,
    use_envelope_fit: bool = False,
    n_brightest: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full BL19 normalisation + telluric-correction pipeline.

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
    apply_telluric_correction:
        If True, run stage-2 telluric correction after normalisation.
    use_envelope_fit:
        Use polynomial envelope fit instead of mean-brightest.
    n_brightest:
        Number of brightest pixels used when ``use_envelope_fit=False``.

    Returns
    -------
    data_prepared:
        Prepared spectra, shape ``(n_spectra, n_good_pixels)``.
    uncertainties_prepared:
        Propagated uncertainties, same shape.
    """
    data_norm, unc_norm = bl19_normalise(
        wavelengths, data, uncertainties, good_pixels,
        n_brightest=n_brightest, use_envelope_fit=use_envelope_fit,
    )
    if apply_telluric_correction:
        return bl19_telluric_correct(data_norm, unc_norm)
    return data_norm, unc_norm


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def pipeline_BL19_norm(wave, mat, noise, good, max_fit=False):
    """BL19 normalisation.

    Dispatches to :func:`pipeline_pseudocontinuum_norm` when
    ``max_fit=True``, otherwise uses the BL19 brightest-pixels mean.

    Parameters
    ----------
    wave : ndarray
        Wavelength array, shape ``(n_pixels,)``.
    mat : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    noise : ndarray
        Noise matrix, same shape as ``mat``.
    good : ndarray
        Integer indices of unmasked columns.
    max_fit : bool
        If True, delegate to :func:`pipeline_pseudocontinuum_norm`
        (polynomial envelope fit to pseudo-continuum maxima, used by
        ASL19).  If False, use BL19 brightest-pixels mean.
    """
    if max_fit:
        return pipeline_pseudocontinuum_norm(wave, mat, noise, good)

    mat = mat[:, good]
    noise = noise[:, good]

    n_brightest = 300
    brightest_pixels = np.argsort(mat, axis=1)[:, -n_brightest:]
    brightest_values = mat[np.arange(mat.shape[0])[:, None],
                           brightest_pixels]
    mean_brightest_values = np.mean(brightest_values, axis=1)
    result1 = mat / mean_brightest_values[:, None]
    error1 = result1 * np.sqrt((noise / mat) ** 2.)
    return result1, error1


def pipeline_pseudocontinuum_norm(wave, mat, noise, good):
    """Pseudo-continuum normalisation for ASL19-style pipelines.

    Divides each spectrum by a degree-2 polynomial fitted to the
    maximum values in 80 equal-width wavelength bins, tracing the
    pseudo-continuum envelope.  Used by the ASL19 pipeline in place
    of the BL19 brightest-pixels mean.

    Parameters
    ----------
    wave : ndarray  shape (n_pixels,)
    mat  : ndarray  shape (n_spectra, n_pixels)
    noise: ndarray  same shape as mat
    good : ndarray  integer indices of unmasked columns

    Returns
    -------
    result1 : ndarray  shape (n_spectra, n_good)
    error1  : ndarray  same shape
    """
    mat   = mat[:, good]
    noise = noise[:, good]
    wave  = wave[good]

    n_bins        = 80
    interval_size = mat.shape[1] // n_bins
    bin_centres   = np.array(
        [np.mean(wave[i * interval_size: (i + 1) * interval_size])
         for i in range(n_bins)]
    )

    result1 = np.ones_like(mat)
    error1  = np.ones_like(noise)

    for n in range(mat.shape[0]):
        bin_maxima = [
            np.max(mat[n, i * interval_size: (i + 1) * interval_size])
            for i in range(n_bins)
        ]
        fit = np.polyval(np.polyfit(bin_centres, bin_maxima, deg=2), wave)
        result1[n, :] = mat[n, :] / fit
        error1[n, :]  = result1[n, :] * np.sqrt(
            (noise[n, :] / mat[n, :]) ** 2
        )

    return result1, error1


def pipeline_BL19_tellcorr(result1, error1, good):
    """BL19 two-stage telluric correction.

    Stage 1: for each spectrum, fit a 2nd-order polynomial to the
    median spectrum and divide.

    Stage 2: for each pixel, fit a 2nd-order polynomial to the
    time evolution and divide.

    Parameters
    ----------
    result1 : ndarray
        Normalised spectra, shape ``(n_spectra, n_pixels)``.
        (Full-grid matrix; ``good`` selects which columns to use.)
    error1 : ndarray
        Propagated noise, same shape.
    good : ndarray
        Integer indices of unmasked columns.

    Returns
    -------
    result3 : ndarray, shape ``(n_spectra, n_good)``
        Corrected spectra.
    error3 : ndarray, shape ``(n_spectra, n_good)``
        Propagated noise.
    """
    result1 = result1[:, good]
    error1 = error1[:, good]

    telluric_spec = np.median(result1, axis=0)
    result2 = np.empty_like(result1)
    error2 = np.empty_like(error1)

    for n in range(result1.shape[0]):
        c1 = np.polyfit(telluric_spec, result1[n, :], deg=2)
        telluric_fit_log = np.polyval(c1, telluric_spec)
        result2[n, :] = result1[n, :] / telluric_fit_log
        error2[n, :] = result2[n, :] * np.sqrt(
            (error1[n, :] / result1[n, :]) ** 2.
        )

    result3 = np.empty_like(result2)
    error3 = np.empty_like(error2)
    indices = np.arange(result1.shape[0])
    for k in range(result1.shape[1]):
        c1 = np.polyfit(indices, result2[:, k], deg=2)
        sec_telluric_fit_log = np.polyval(c1, indices)
        result3[:, k] = result2[:, k] / sec_telluric_fit_log
        error3[:, k] = result3[:, k] * np.sqrt(
            (error2[:, k] / result2[:, k]) ** 2.
        )

    return result3, error3
