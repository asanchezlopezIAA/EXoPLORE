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

# Reading this file
# Telluric transmittance is an input to synthetic observations, while
# pipeline_fixedTellurics is a preparation path for the configured fixed case.
# Loading a theoretical atmosphere and removing structure from observed spectra
# are different operations, even though both concern terrestrial absorption.


import numpy as np


def validate_telluric_coverage(wavelength_um, required_wave_um, *, source):
    """Reject insufficient coverage before interpolation can invent zero flux.

    Both arguments are in microns. The required array may contain several
    selected orders. Ascending and descending input grids remain supported by
    the existing interpolator; a malformed/duplicate grid is rejected.
    """
    wave = np.asarray(wavelength_um, dtype=float)
    required = np.asarray(required_wave_um, dtype=float)
    if (wave.ndim != 1 or wave.size < 2 or not np.isfinite(wave).all()
            or np.any(wave <= 0)
            or not (np.all(np.diff(wave) > 0) or np.all(np.diff(wave) < 0))):
        raise ValueError(f"Invalid telluric wavelength grid in {source}: expected finite, positive, strictly monotonic wavelengths.")
    if required.size == 0 or not np.isfinite(required).all():
        raise ValueError("Cannot check telluric coverage against an empty or non-finite instrument grid.")
    available_min, available_max = float(wave.min()), float(wave.max())
    needed_min, needed_max = float(required.min()), float(required.max())
    # Allow floating-point roundoff at identical endpoints, not missing pixels.
    tolerance = 64 * np.finfo(float).eps * max(1., abs(needed_min), abs(needed_max))
    if needed_min < available_min - tolerance or needed_max > available_max + tolerance:
        raise ValueError(
            f"Telluric wavelength coverage is insufficient in {source}. "
            f"File covers {available_min:.9g}–{available_max:.9g} microns; "
            f"selected instrument wavelengths require {needed_min:.9g}–{needed_max:.9g} microns. "
            "Set tellurics.skycalc_wavelength_limits_um in your simulation JSON to cover the selected orders "
            "with margin, then generate suitable inputs with scripts/generate_skycalc_inputs.py "
            "(--ref-only for a reference file; per-exposure files for use_full_skycalc=true). "
            "Point tellurics.reference_telluric_file/paths.inputs_dir at those files. "
            "Existing downloads are skipped unless explicitly replaced; out-of-range transmission will not be filled with zeros."
        )


def validate_telluric_file(path, required_wave_um):
    """Check a reference FITS table before expensive observation calculations."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        names = (hdul[1].columns.names if len(hdul) > 1
                 and isinstance(hdul[1], (fits.BinTableHDU, fits.TableHDU)) else [])
        if not {"lam", "trans"}.issubset(names or []):
            raise ValueError(f"Telluric file {path} must contain a FITS table with lam (nm) and trans columns.")
        validate_telluric_coverage(hdul[1].data["lam"] * 1e-3, required_wave_um, source=path)


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
        validate_telluric_coverage(wvl_ref, wave_ins, source=tell_ref_file)
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
                    validate_telluric_coverage(wvl_trans, wave_ins, source=file_path)
                    tell_trans_temp = np.zeros((len(syn_jd), len(wvl_trans)))
                    tell_trans_temp[n, :] = dummy
                else:
                    with fits.open(file_path) as file:
                        current_wave = file[1].data['lam'] * 1e-3
                        validate_telluric_coverage(current_wave, wave_ins, source=file_path)
                        # The array interpolation below uses a common grid.
                        # A different exposure grid cannot silently reuse it.
                        if not np.array_equal(current_wave, wvl_trans):
                            raise ValueError(f"Telluric wavelength grid in {file_path} differs from tell_spec_0.fits; per-exposure files must share one grid.")
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
