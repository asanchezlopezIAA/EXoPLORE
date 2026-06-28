"""
exoplore.pipelines.blain24
==========================

Blain, Sánchez-López & Mollière (2024) data-preparation pipeline for
high-resolution spectroscopic time series.

The Blain24 pipeline (Blain, Sánchez-López & Mollière 2024, AJ)
normalises and optionally telluric-corrects each spectrum using a
weighted polynomial fit.  Unlike BL19, it propagates uncertainties
through the fit, accounting for degrees of freedom lost in the
polynomial correction.

This module provides standalone functions: ``remove_throughput_fit`` for the
wavelength-axis normalisation and ``remove_telluric_lines_fit`` for the
airmass-axis telluric correction.

References
----------
Blain, D., Sánchez-López, A. & Mollière, P. (2024), AJ.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def blain24_normalise(
    wavelengths: np.ndarray,
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    polynomial_degree: int = 2,
    mask_threshold: float = 1e-16,
    propagate_uncertainties: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blain24 normalisation: divide each spectrum by a polynomial fit.

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


def blain24_telluric_correct(
    data: np.ndarray,
    uncertainties: np.ndarray,
    good_pixels: np.ndarray,
    airmass: np.ndarray,
    weights: Optional[np.ndarray] = None,
    mask_threshold: float = 0.8,
    polynomial_degree: int = 2,
    propagate_uncertainties: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blain24 telluric correction: airmass-based polynomial fit.

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
# Convenience: run full Blain24 pipeline in one call
# ---------------------------------------------------------------------------


def run_blain24_pipeline(
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
    """Run the full Blain24 normalisation (+ optional telluric) pipeline.

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
    data_norm, unc_norm, mask, gp = blain24_normalise(
        wavelengths, data, uncertainties, good_pixels,
        weights=weights,
        polynomial_degree=polynomial_degree,
    )
    if apply_telluric_correction:
        if airmass is None:
            raise ValueError(
                "airmass must be provided when apply_telluric_correction=True."
            )
        return blain24_telluric_correct(
            data_norm, unc_norm, gp, airmass,
            weights=weights,
            mask_threshold=telluric_mask_threshold,
            polynomial_degree=polynomial_degree,
        )
    return data_norm, unc_norm, mask, gp


# ---------------------------------------------------------------------------
# Pipeline normalisation and telluric-correction functions
# ---------------------------------------------------------------------------


def _merge_masks(mask1, mask2, n_pixels):
    """Return (combined_mask, useful_spectral_points)."""
    mask = np.unique(np.concatenate((mask1, mask2), axis=None))
    useful_spectral_points = np.setdiff1d(np.arange(n_pixels), mask)
    return mask, useful_spectral_points


def remove_telluric_lines_fit(
        data, airmass, mask, useful_spectral_points,
        correct_uncertainties, uncertainties=None,
        masking=True, mask_threshold=1e-16, polynomial_fit_degree=2,
        uncertainties_as_weights=False
):
    """Remove telluric lines via a log-space polynomial fit in airmass.

    Parameters
    ----------
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    airmass : ndarray
        Airmass per spectrum, shape ``(n_spectra,)``.
    mask : ndarray
        Integer mask of bad columns.
    useful_spectral_points : ndarray
        Integer indices of good columns.
    correct_uncertainties : bool
        If True, propagate uncertainties through the correction.
    uncertainties : ndarray or None
        Noise matrix, same shape as ``data``.
    masking : bool
    mask_threshold : float
    polynomial_fit_degree : int
    uncertainties_as_weights : bool

    Returns
    -------
    data_prepared, propag_uncertainties, mask, useful_spectral_points
    """
    degrees_of_freedom = polynomial_fit_degree + 1
    data_prepared = np.copy(data)
    propag_uncertainties = np.copy(uncertainties)

    if data.shape[1] <= degrees_of_freedom:
        raise Exception(
            f"not enough points in airmass axis ({data.shape[1]}) "
            f"for a meaningful correction with the requested fit degree "
            f"({polynomial_fit_degree})."
        )

    if uncertainties_as_weights:
        weights = 1. / np.abs(uncertainties / data)
    else:
        weights = np.ones_like(data)

    telluric_lines_fits = np.zeros(data.shape)

    if masking:
        mask_tel = list()

    mask_log = np.any(data <= 0, axis=0)
    mask_log = np.tile(mask_log, (data.shape[0], 1))
    mask_log = np.where(mask_log[0, :])[0]
    mask, useful_spectral_points = _merge_masks(
        mask, mask_log, data.shape[1]
    )

    if mask.shape != (0,):
        weights[:, mask] = 0

    data_log = np.log(data)

    for k, log_wavelength_column in enumerate(data_log.T):
        if k in mask:
            continue
        if np.sum(weights[:, k]) == 0.:
            raise Exception(
                "A useful spectral point has been masked? Check code."
            )

        fit_parameters = np.polyfit(
            x=airmass,
            y=log_wavelength_column,
            deg=polynomial_fit_degree,
            w=weights[:, k]
        )
        fit_function = np.poly1d(fit_parameters)
        telluric_lines_fits[:, k] = fit_function(airmass)
        telluric_lines_fits[:, k] = np.exp(telluric_lines_fits[:, k])

        if masking and np.where(
                telluric_lines_fits[:, k] < mask_threshold
        )[0].shape != (0,):
            mask_tel.append(k)

        data_prepared[:, k] /= telluric_lines_fits[:, k]

    if masking:
        mask, useful_spectral_points = _merge_masks(
            mask, mask_tel, data.shape[1]
        )

    if uncertainties is not None and correct_uncertainties:
        variance_corr_fac = (
            (int(data.shape[0]) - (polynomial_fit_degree + 1))
            / int(data.shape[0])
        )
        propag_uncertainties /= np.abs(telluric_lines_fits)
        # Multiply (not divide) by sqrt((N-d)/N): the fit reduces the residual
        # scatter, so the uncertainties are biased down to reflect it
        # (Blain et al. 2024; matches petitRADTRANS retrieval.preparing).
        propag_uncertainties *= np.sqrt(variance_corr_fac)

    return data_prepared, propag_uncertainties, mask, useful_spectral_points


def remove_throughput_fit(
        data, mask, useful_spectral_points, wavelengths,
        correct_uncertainties, uncertainties=None,
        mask_threshold=1e-16, polynomial_fit_degree=2,
        uncertainties_as_weights=False
):
    """Remove variable throughput via a per-exposure polynomial fit.

    Parameters
    ----------
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    mask : ndarray
        Integer mask of bad columns.
    useful_spectral_points : ndarray
        Integer indices of good columns.
    wavelengths : ndarray
        Wavelength array, shape ``(n_pixels,)``.
    correct_uncertainties : bool
    uncertainties : ndarray or None
    mask_threshold : float
    polynomial_fit_degree : int
    uncertainties_as_weights : bool

    Returns
    -------
    data_prepared : ndarray
    propag_uncertainties : ndarray
    """
    degrees_of_freedom = polynomial_fit_degree + 1
    data_prepared = np.copy(data)
    propag_uncertainties = np.copy(uncertainties)

    if data.shape[1] <= degrees_of_freedom:
        raise Exception(
            f"not enough points in wavelengths axis ({data.shape[1]}) "
            f"for a meaningful correction with the requested fit degree "
            f"({polynomial_fit_degree})."
        )

    if uncertainties_as_weights:
        weights = np.copy(uncertainties)
    else:
        weights = np.ones_like(data)

    if mask.shape != (0,):
        weights[:, mask] = 0

    throughput_fits = np.zeros_like(data)

    if np.ndim(wavelengths) == 3:
        print('Assuming same wavelength solution for each observation, '
              'taking wavelengths of observation 0')

    for j, exposure in enumerate(data):
        fit_parameters = np.polyfit(
            x=wavelengths,
            y=exposure,
            deg=polynomial_fit_degree,
            w=weights[j, :]
        )
        fit_function = np.poly1d(fit_parameters)
        throughput_fits[j, :] = fit_function(wavelengths)

    data_prepared /= throughput_fits

    if uncertainties is not None and correct_uncertainties:
        variance_corr_fac = (
            (len(useful_spectral_points) - degrees_of_freedom)
            / len(useful_spectral_points)
        )
        propag_uncertainties /= np.abs(throughput_fits)
        # Multiply (not divide) by sqrt((N-d)/N): the fit reduces the residual
        # scatter, so the uncertainties are biased down to reflect it
        # (Blain et al. 2024; matches petitRADTRANS retrieval.preparing).
        propag_uncertainties *= np.sqrt(variance_corr_fac)

    return data_prepared, propag_uncertainties


# ---------------------------------------------------------------------------
# v0.25 additions
# ---------------------------------------------------------------------------


def compute_k_sigma(
        data_dir: str,
        orig_orders: np.ndarray,
        order_selection: list,
        n_spectra_store: np.ndarray,
        pixel_mask_file: str,
        return_per_night: bool = False
        ):
    """Compute k_sigma as in Blain et al. (2024), with full diagnostics.


    Parameters
    ----------
    data_dir : str
        Path to folder containing mat_res_order_*.npz and
        std_noise_order_*.npz files.
    orig_orders : np.ndarray, shape (n_orders,)
        The ORIGINAL CARMENES order numbers you kept (e.g.
        ``np.delete(np.arange(28), [18,19,20])``).
    order_selection : list of np.ndarray
        Length = n_nights; each entry is an array of ORIGINAL order
        numbers to include that night.
    n_spectra_store : np.ndarray, shape (n_nights,)
        Number of valid exposures per night.
    pixel_mask_file : str
        Filename (inside data_dir) of your
        useful_spectral_points_*.npz mask.
    return_per_night : bool
        If True, also compute and return per-night k_sigma medians.

    Returns
    -------
    k_sigma : np.ndarray, shape (n_orders, n_nights, n_exposures_max)
        The per-(order, night, exposure) scaling factors.
    k_sigma_global : float
        Median over all valid k_sigma.
    k_sigma_per_night : np.ndarray, shape (n_nights,), optional
        If ``return_per_night=True``, the median k_sigma per night.
    """
    # Derived sizes
    n_nights        = len(order_selection)
    n_exposures_max = int(n_spectra_store.max())
    n_orders        = len(orig_orders)
    n_pixels        = 4080

    # Build exposure mask
    exposure_mask = np.zeros((n_nights, n_exposures_max), dtype=bool)
    for ni, nuse in enumerate(n_spectra_store):
        exposure_mask[ni, :nuse] = True

    # Helper to load a single array from .npz
    def load_npz_array(path):
        """Load the first named array from an NPZ file written by the simulator.

        Loads a named array from an NPZ archive produced by the simulator's
        Block 8 output routines (e.g., via ``np.savez_compressed``).  The
        function always returns the first key in the archive, by convention
        the simulator writes one array per file using keys such as
        ``'mat_res'``, ``'propag_noise'``, or ``'mat_star'``.  Typical
        array shape is ``(n_spectra, n_pixels)``.

        Parameters
        ----------
        path : str
            Absolute path to the NPZ file (e.g.,
            ``data_dir/mat_res_order_5_<sim_name>.npz``).

        Returns
        -------
        numpy.ndarray
            The stored array.  Returns ``None`` implicitly if the file
            cannot be opened (caller is responsible for handling missing
            files through the surrounding try/except in ``compute_k_sigma``).
        """
        with np.load(path) as d:
            return d[d.files[0]]

    # 1) Load data cubes
    mat_all   = np.full((n_orders, n_nights, n_exposures_max, n_pixels), np.nan)
    noise_all = mat_all.copy()
    for oi, orig in enumerate(orig_orders):
        mat_path   = os.path.join(data_dir, f"mat_res_order_{orig}_Gibson22_withsignal_5nights_SNR_comb1_realdata_noiseless_stdnoisex1.npz")
        noise_path = mat_path.replace("mat_res_order", "std_noise_order")
        mat_all[oi]   = load_npz_array(mat_path)
        noise_all[oi] = load_npz_array(noise_path)

    # 2) Mask out unused exposures
    for ni, nuse in enumerate(n_spectra_store):
        if nuse < n_exposures_max:
            mat_all[:, ni, nuse:, :]   = np.nan
            noise_all[:, ni, nuse:, :] = np.nan

    # 3) Load pixel mask
    usp = load_npz_array(os.path.join(data_dir, pixel_mask_file))  # shape (n_nights, n_orders, n_pixels)
    pixel_mask = np.transpose(usp, (1, 0, 2))                      # (orders, nights, pixels)
    pixel_mask = pixel_mask[:, :, None, :]                         # add exposure axis

    # 4) Allocate arrays for sigma_P and mean_sigma_U
    sigma_P      = np.full((n_orders, n_nights, n_exposures_max), np.nan)
    mean_sigma_U = np.full_like(sigma_P, np.nan)

    # 5) Compute per-(order, night, exposure)
    for oi, orig in enumerate(orig_orders):
        for ni in range(n_nights):
            if orig not in order_selection[ni]:
                continue
            for ei in range(n_exposures_max):
                if not exposure_mask[ni, ei]:
                    continue
                pix_good = pixel_mask[oi, ni, 0, :]
                if not pix_good.any():
                    continue

                F = mat_all[oi, ni, ei, pix_good]
                V = noise_all[oi, ni, ei, pix_good]
                U = np.sqrt(V)

                sigma_P[oi, ni, ei]      = np.nanstd(F, ddof=1)
                mean_sigma_U[oi, ni, ei] = np.nanmean(U)

    # 6) Build k_sigma and mask invalids
    k_sigma = mean_sigma_U / sigma_P
    k_sigma[~np.isfinite(k_sigma)] = np.nan

    # Diagnostics
    valid = ~np.isnan(k_sigma)
    print("Total valid points:", valid.sum())
    print("sigma_P:  min,50%,max =", np.nanpercentile(sigma_P[valid], [0,50,100]))
    print("meanU:    min,50%,max =", np.nanpercentile(mean_sigma_U[valid], [0,50,100]))

    k_sigma_global = float(np.nanmedian(k_sigma))
    print(f"\nDataset-wide k_sigma = {k_sigma_global:.3f}")
    print("And its inverse, beta≈1/k_sigma =", 1.0/k_sigma_global)

    if return_per_night:
        k_sigma_per_night = np.nanmedian(k_sigma, axis=(0,2))
        print("\nPer-night k_sigma:")
        for ni, ks in enumerate(k_sigma_per_night):
            print(f" Night {ni}: k_sigma = {ks:.3f}")
        return k_sigma, k_sigma_global, k_sigma_per_night

    else:
        print("\nPer-order k_sigma:")
        for oi, orig in enumerate(orig_orders):
            ks = np.nanmedian(k_sigma[oi])
            print(f" Order {orig:2d}: k_sigma = {ks:.3f}")
        return k_sigma, k_sigma_global, None
