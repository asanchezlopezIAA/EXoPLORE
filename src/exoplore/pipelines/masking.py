"""
exoplore.pipelines.masking
==========================

Spectral masking utilities for high-resolution time-series data.

Three mask sources are combined into a single boolean array of
*good* (unmasked) pixel indices:

Telluric mask
    Pixels where the telluric template transmittance falls below a
    threshold are masked, including an optional safety window around
    each bad pixel.

SNR mask
    Pixels whose time-averaged flux is below an SNR threshold are
    masked.

Column-scatter mask
    Pixels with a standard deviation more than 3σ above the matrix
    median are masked (``mask_noisy_columns``).

All mask arrays use **integer column indices** (not boolean arrays).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Telluric masking
# ---------------------------------------------------------------------------


def mask_telluric_columns(
    telluric_template: np.ndarray,
    threshold: float,
    existing_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mask entire spectral columns where telluric absorption is deep.

    A column is masked if *any* spectrum in ``telluric_template`` falls
    below ``threshold``.

    Parameters
    ----------
    telluric_template:
        Telluric transmittance array, shape ``(n_pixels,)`` or
        ``(n_spectra, n_pixels)``.  Values in [0, 1].
    threshold:
        Mask columns where transmittance < ``threshold``.
    existing_mask:
        Integer column indices already masked.  The new telluric mask is
        merged with this.

    Returns
    -------
    np.ndarray
        Integer array of masked column indices.
    """
    telluric_template = np.atleast_2d(np.asarray(telluric_template))
    bad = np.any(telluric_template < threshold, axis=0)  # (n_pixels,)
    new_mask = np.where(bad)[0]
    return merge_masks(existing_mask, new_mask)


def mask_telluric_columns_with_window(
    telluric_template: np.ndarray,
    threshold: float,
    safety_window: int = 1,
    existing_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mask telluric columns, plus a safety window around each bad pixel.

    Parameters
    ----------
    telluric_template:
        Telluric transmittance, shape ``(n_pixels,)`` or
        ``(n_spectra, n_pixels)``.
    threshold:
        Mask columns where transmittance < ``threshold``.
    safety_window:
        Number of pixels to mask on each side of a bad pixel.
        ``safety_window=1`` means only the bad pixel itself.
        ``safety_window=3`` means the bad pixel ±1 neighbours.
    existing_mask:
        Integer column indices already masked.

    Returns
    -------
    np.ndarray
        Integer array of masked column indices.
    """
    if safety_window <= 1:
        return mask_telluric_columns(
            telluric_template, threshold, existing_mask
        )

    telluric_template = np.atleast_2d(np.asarray(telluric_template))
    bad_pixels = np.any(telluric_template < threshold, axis=0)  # (n_pixels,)
    bad_indices = np.where(bad_pixels)[0]
    n_pixels = telluric_template.shape[-1]  # inferred for safety window expansion

    expanded = set(bad_indices.tolist())
    half = safety_window // 2
    for idx in bad_indices:
        for j in range(max(0, idx - half), min(n_pixels, idx + half + 1)):
            expanded.add(j)

    new_mask = np.array(sorted(expanded), dtype=int)
    return merge_masks(existing_mask, new_mask)


# ---------------------------------------------------------------------------
# SNR masking
# ---------------------------------------------------------------------------


def mask_low_snr_columns(
    snr_map: np.ndarray,
    snr_threshold: float,
    existing_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mask spectral columns with SNR below a threshold.

    Parameters
    ----------
    snr_map:
        SNR array, shape ``(n_pixels,)`` or ``(n_spectra, n_pixels)``.
        Columns where *any* SNR < ``snr_threshold`` are masked.
    snr_threshold:
        Minimum acceptable SNR.
    existing_mask:
        Integer column indices already masked.

    Returns
    -------
    np.ndarray
        Integer array of masked column indices.
    """
    snr_map = np.atleast_2d(np.asarray(snr_map))
    bad = np.any(snr_map < snr_threshold, axis=0)
    new_mask = np.where(bad)[0]
    return merge_masks(existing_mask, new_mask)


# ---------------------------------------------------------------------------
# Column-scatter masking
# ---------------------------------------------------------------------------


def mask_noisy_columns(
    data: np.ndarray,
    n_sigma: float = 3.0,
    existing_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mask spectral columns whose pixel scatter is unusually high.

    Computes the standard deviation along the time axis for each pixel
    and masks columns more than ``n_sigma`` × the matrix-wide standard
    deviation above the mean.

    Parameters
    ----------
    data:
        Time-series matrix, shape ``(n_spectra, n_pixels)``.
    n_sigma:
        Rejection threshold in units of the matrix standard deviation.
    existing_mask:
        Integer column indices already masked.

    Returns
    -------
    np.ndarray
        Integer array of masked column indices.
    """
    col_std = np.std(data, axis=0)
    bad = col_std > n_sigma * np.std(data)
    new_mask = np.where(bad)[0]
    return merge_masks(existing_mask, new_mask)


# ---------------------------------------------------------------------------
# Mask algebra
# ---------------------------------------------------------------------------


def merge_masks(
    mask1: Optional[np.ndarray],
    mask2: Optional[np.ndarray],
) -> np.ndarray:
    """Union of two integer mask arrays.

    Parameters
    ----------
    mask1, mask2:
        Integer arrays of masked column indices, or ``None``.

    Returns
    -------
    np.ndarray
        Sorted integer array of all masked column indices.
    """
    arrays = [m for m in (mask1, mask2) if m is not None and len(m) > 0]
    if not arrays:
        return np.array([], dtype=int)
    combined = np.concatenate(arrays, axis=None).astype(int)
    return np.unique(combined)


def good_pixel_indices(
    mask: np.ndarray,
    n_pixels: int,
) -> np.ndarray:
    """Return the unmasked column indices.

    Parameters
    ----------
    mask:
        Integer array of masked column indices.
    n_pixels:
        Total number of spectral pixels.

    Returns
    -------
    np.ndarray
        Integer array of indices *not* in ``mask``.
    """
    all_indices = np.arange(n_pixels, dtype=int)
    if mask is None or len(mask) == 0:
        return all_indices
    return np.setdiff1d(all_indices, mask)


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def _merge_masks(mask1, mask2, n_pixels):
    """Like merge_masks but also returns the good-pixel index array.

    Returns
    -------
    mask : ndarray
        Combined integer mask (sorted, unique).
    useful_spectral_points : ndarray
        Integer indices of unmasked columns.
    """
    mask = np.unique(np.concatenate((mask1, mask2), axis=None))
    useful_spectral_points = np.setdiff1d(np.arange(n_pixels), mask)
    return mask, useful_spectral_points


def mask_tellurics(inp_dat, data, mask_snr):
    """Mask spectral columns with flux below the telluric threshold.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Key used: ``"telluric_mask"``
        (float threshold, columns with flux < threshold are masked).
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    mask_snr : ndarray
        Integer mask from a prior SNR masking step.

    Returns
    -------
    mask : ndarray
        Combined integer mask (telluric + SNR).
    useful_spectral_points : ndarray
        Integer indices of unmasked columns.
    """
    mask_tel = data < inp_dat['telluric_mask']
    mask_tel_indices = np.argwhere(mask_tel)
    for j in mask_tel_indices[:, 1]:
        mask_tel[:, j] = True
    mask_tel = np.where(mask_tel[0, :])[0]
    return _merge_masks(mask_snr, mask_tel, data.shape[1])


def mask_tellurics_window(inp_dat, data, mask_snr):
    """Mask telluric columns with an optional safety window.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``"telluric_mask"``,
        ``"safety_window"``.
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    mask_snr : ndarray
        Integer mask from a prior SNR masking step.

    Returns
    -------
    mask : ndarray
        Combined integer mask (telluric + SNR + window).
    useful_spectral_points : ndarray
        Integer indices of unmasked columns.
    """
    if inp_dat["safety_window"] != 1:
        mask_tel = data < inp_dat['telluric_mask']
        mask_tel_indices = np.argwhere(mask_tel)

        for idx in mask_tel_indices:
            row_idx, col_idx = idx
            mask_tel[row_idx, col_idx] = True
            half_width = inp_dat["safety_window"] // 2
            start_col = max(0, col_idx - half_width)
            end_col = min(data.shape[1], col_idx + half_width + 1)
            mask_tel[row_idx, start_col:end_col] = True

        mask_tel_columns = np.where(mask_tel[0, :])[0]
        return _merge_masks(mask_snr, mask_tel_columns, data.shape[1])
    else:
        print("Your safety window is 1, which means NO window around the pixel that triggered the mask")
        return mask_tellurics(inp_dat, data, mask_snr)


def mask_columns(data, mask):
    """Mask columns whose standard deviation is > 3× the matrix std.

    Parameters
    ----------
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    mask : ndarray
        Existing integer mask to merge with.

    Returns
    -------
    mask : ndarray
        Updated integer mask.
    useful_spectral_points : ndarray
        Integer indices of unmasked columns.
    """
    bad_pixels = np.std(data, axis=0) > 3 * np.std(data)
    new_mask = np.where(bad_pixels)[0]
    return _merge_masks(mask, new_mask, data.shape[1])


# ---------------------------------------------------------------------------
# NaN / outlier correction
# ---------------------------------------------------------------------------


def Correct_NaN(spec, sig):
    """Replace NaN values with the median of finite values in the same row.

    Parameters
    ----------
    spec : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    sig : ndarray
        Uncertainty matrix, same shape.

    Returns
    -------
    spec, sig : ndarray
        Corrected matrices (in-place modification + return).
    """
    for i in range(spec.shape[0]):
        nans = np.where(~np.isfinite(spec[i, :]))[0]
        no_nans = np.where(np.isfinite(spec[i, :]))[0]
        if nans.shape != (0,):
            spec[i, nans] = np.median(spec[i, no_nans])
            sig[i, nans] = np.median(sig[i, no_nans])
    return spec, sig


def Remove_Outliers(spec, sig):
    """Replace 3-sigma outliers with the column mean.

    Parameters
    ----------
    spec : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    sig : ndarray
        Uncertainty matrix, same shape.

    Returns
    -------
    fixed_spec, fixed_sig : ndarray
        Matrices with outliers replaced by column mean values.
    """
    mean_values = np.mean(spec, axis=0)
    std_dev = np.std(spec, axis=0)
    mean_values_uncertainties = np.mean(sig, axis=0)
    threshold = 3 * std_dev
    outliers = np.abs(spec - mean_values) > threshold
    fixed_spec = np.where(outliers, mean_values, spec)
    fixed_sig = np.where(outliers, mean_values_uncertainties, sig)
    return fixed_spec, fixed_sig


def Robust_Outlier_Removal(data, noise, polynomial_degree=3, threshold=4,
                            pixel_window=1):
    """Detect and correct outliers using a robust polynomial fit over time.

    Uses statsmodels RLM (Tukey biweight) to fit each pixel's time curve,
    then replaces points exceeding ``threshold`` × residual std with the
    predicted fit value.

    Parameters
    ----------
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    noise : ndarray
        Uncertainty matrix, same shape.
    polynomial_degree : int
        Degree of the robust polynomial fit.  Default 3.
    threshold : float
        Sigma threshold for flagging outliers.  Default 4.
    pixel_window : int
        Half-window around each outlier to correct.  Default 1.

    Returns
    -------
    data_corrected, noise_corrected : ndarray
        Corrected matrices.
    """
    import statsmodels.api as sm

    data_corrected = np.copy(data)
    noise_corrected = np.copy(noise)
    n_spectra, n_pixels = data.shape

    for i in range(n_pixels):
        x = np.arange(n_spectra)
        X = np.vander(x, polynomial_degree + 1)

        rlm_model = sm.RLM(
            data[:, i], X,
            M=sm.robust.norms.TukeyBiweight()
        )
        rlm_results = rlm_model.fit()
        predicted_values = rlm_results.predict()

        residuals = data[:, i] - predicted_values
        outlier_indices = np.where(
            np.abs(residuals) > threshold * np.std(residuals)
        )[0]

        if len(outlier_indices) > 0:
            for idx in outlier_indices:
                start_idx = max(0, idx - pixel_window)
                end_idx = min(n_spectra - 1, idx + pixel_window)
                data_corrected[start_idx:end_idx + 1, i] = \
                    predicted_values[start_idx:end_idx + 1]
                noise_corrected[start_idx:end_idx + 1, i] = np.interp(
                    np.arange(start_idx, end_idx + 1),
                    [start_idx, end_idx],
                    [noise[start_idx, i], noise[end_idx, i]]
                )

    return data_corrected, noise_corrected
