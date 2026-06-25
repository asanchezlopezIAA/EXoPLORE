"""
exoplore.pipelines.tellurics
============================

Telluric transmittance loading and fixed-telluric pipeline.

This module provides:

- :func:`Load_Telluric_Transmittances`, loads or computes telluric
  transmittance spectra (from Skycalc FITS files or from a reference
  spectrum scaled by airmass).

- :func:`pipeline_fixedTellurics`, normalises a spectral time series
  using weighted-mean throughput correction and column-wise telluric
  removal (used when ``telluric_variation=False``).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Telluric transmittance loading
# ---------------------------------------------------------------------------


def Load_Telluric_Transmittances(snr, telluric_variation,
                                  Full_Skycalc, tell_ref_file, filepath,
                                  res, syn_jd, wave_ins, spec_mat, airmass):
    """Load or compute telluric transmittance spectra.

    Three modes depending on inputs:

    1. ``telluric_variation=False``: reference spectrum broadcast to all
       exposures.
    2. ``telluric_variation=True, Full_Skycalc=True``: loads per-exposure
       FITS files written by Skycalc CLI.
    3. ``telluric_variation=True, Full_Skycalc=False``: scales reference
       spectrum by geometric airmass.

    Parameters
    ----------
    snr : ndarray
        SNR array. Shape ``(n_pixels,)`` triggers 1-D branch;
        ``(n_spectra, n_pixels)`` triggers 2-D branch.
    telluric_variation : bool
        Whether to model time-varying telluric absorption.
    Full_Skycalc : bool
        If True, load per-exposure Skycalc FITS files from ``filepath``.
    tell_ref_file : str
        Path to a reference Skycalc FITS file containing columns
        ``'lam'`` (nm) and ``'trans'``.
    filepath : str
        Directory containing ``tell_spec_{n}.fits`` files.
    res : float
        Instrument resolving power R = λ / Δλ.
    syn_jd : array-like
        Julian dates of each synthetic exposure.
    wave_ins : ndarray
        Instrument wavelength grid in microns, shape ``(n_pixels,)``.
    spec_mat : ndarray
        Spectral matrix (used only for shape when
        ``telluric_variation=False``).
    airmass : ndarray
        Airmass for each exposure, shape ``(n_spectra,)``.

    Returns
    -------
    tell_ref : ndarray or None
        Reference telluric spectrum on ``wave_ins`` grid.
    tell_trans : ndarray
        Telluric transmittance for every exposure,
        shape ``(n_spectra, n_pixels)``.
    """
    from scipy import interpolate
    from astropy.io import fits
    from exoplore.atmosphere.prt import convolve

    if snr.ndim == 1 or not telluric_variation or not Full_Skycalc:
        with fits.open(tell_ref_file) as file:
            wvl_ref = file[1].data['lam'] * 1e-3
            tell_ref = file[1].data['trans']
        tell_ref = interpolate.interp1d(
            wvl_ref, tell_ref, bounds_error=False, fill_value=0.
        )(wave_ins)
        tell_ref = convolve(wave_ins, tell_ref, res)

    if telluric_variation:
        if Full_Skycalc:
            if snr.ndim != 1:
                tell_ref = None
            for n in range(len(syn_jd)):
                file_path = f'{filepath}tell_spec_{n}.fits'
                if n == 0:
                    with fits.open(file_path) as file:
                        dummy = file[1].data['trans']
                        wvl_trans = file[1].data['lam'] * 1e-3
                    tell_trans_temp = np.zeros((len(syn_jd), len(wvl_trans)))
                    tell_trans_temp[n, :] = dummy
                else:
                    with fits.open(file_path) as file:
                        tell_trans_temp[n, :] = file[1].data['trans']

            tell_trans = np.zeros((len(syn_jd), len(wave_ins)))
            tell_trans = interpolate.interp1d(
                wvl_trans, tell_trans_temp,
                bounds_error=False, fill_value=0.
            )(wave_ins)
            tell_trans = np.array([
                convolve(wave_ins, tell_trans[n, :], res)
                for n in range(len(syn_jd))
            ])
        else:
            tell_trans = np.exp(airmass.reshape(-1, 1) * np.log(tell_ref))
    else:
        tell_trans = np.empty_like(spec_mat)
        tell_trans[:] = tell_ref

    return tell_ref, tell_trans


# ---------------------------------------------------------------------------
# Fixed-telluric pipeline
# ---------------------------------------------------------------------------


def pipeline_fixedTellurics(phase, wave, mat, noise, good, mask, mask_snr):
    """Normalise and remove tellurics for a fixed-telluric observation.

    Step 1, throughput correction:
        For each spectrum, divide by the inverse-variance weighted mean
        flux over the good spectral points.

    Step 2, column-wise telluric removal:
        For each wavelength column, divide by the inverse-variance
        weighted mean over time.

    Parameters
    ----------
    phase : ndarray
        Orbital phase for each spectrum, shape ``(n_spectra,)``.
    wave : ndarray
        Wavelength array, shape ``(n_pixels,)``.
    mat : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    noise : ndarray
        Noise matrix, same shape as ``mat``.
    good : ndarray
        Integer indices of good spectral columns.
    mask : ndarray
        Integer mask (telluric or other; columns set to 1 after
        correction).
    mask_snr : ndarray
        Integer SNR mask (columns set to 1 after correction).

    Returns
    -------
    result : ndarray, shape ``(n_spectra, n_pixels)``
        Normalised and telluric-corrected spectral matrix.
    error : ndarray, shape ``(n_spectra, n_pixels)``
        Propagated noise matrix.
    """
    result1 = np.zeros_like(mat)
    error1 = np.zeros_like(noise)
    result = np.zeros_like(mat)
    error = np.zeros_like(noise)

    # Step 1: throughput and mean SNR correction
    for n in range(len(phase)):
        wa_wvl = (np.sum(mat[n, good] / noise[n, good])
                  / np.sum(1. / noise[n, good]))
        result1[n, :] = mat[n, :] / wa_wvl
        error1[n, :] = noise[n, :] / np.abs(wa_wvl)

    if mask.shape != (0,):
        result1[:, mask] = 1
        result1[:, mask_snr] = 1

    # Step 2: column-wise telluric removal
    for k in range(len(wave)):
        wa_t = (np.sum(result1[:, k] / error1[:, k])
                / np.sum(1. / error1[:, k]))
        result[:, k] = result1[:, k] / wa_t
        error[:, k] = error1[:, k] / np.abs(wa_t)

    if mask.shape != (0,):
        result[:, mask] = 1
        result[:, mask_snr] = 1

    return result, error
