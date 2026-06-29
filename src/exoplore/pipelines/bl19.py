"""
exoplore.pipelines.bl19
=======================

Brogi & Line (2019) data-preparation pipeline for high-resolution
spectroscopic time series.

The BL19 pipeline (Brogi & Line 2019, AJ, 157, 114) normalises each
spectrum by the mean of its brightest pixels (or by a polynomial fit
to the pseudo-continuum maxima when ``max_fit=True``), then performs
two rounds of telluric correction via polynomial fitting to the median
spectrum and to each pixel's time evolution.

This module provides ``pipeline_BL19_norm`` and
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
