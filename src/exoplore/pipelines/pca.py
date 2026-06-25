"""
exoplore.pipelines.pca
=======================

PCA-based systematic removal for high-resolution spectral time-series.

Principal Component Analysis detrending (de Kok et al. 2013; Giacobbe et al.
2021; Holmberg & Madhusudhan 2022; Cheverall et al. 2023) removes the dominant
*common modes* in the time-variation of each wavelength channel by subtracting
the leading singular components of the ``(n_spectra, n_pixels)`` matrix.

SYSREM (:mod:`exoplore.pipelines.sysrem`) is the inverse-variance-weighted
counterpart of the same idea: each SYSREM iteration removes one rank-1
outer-product ``a ⊗ c`` weighted by ``1/σ²``, whereas each PCA component removes
one unweighted rank-1 SVD mode ``σ_i u_i v_iᵀ``.  Cheverall et al. (2023) report
"minimal difference between residuals when detrending with each of PCA and
SYSREM".

This is a deliberately *pipeline-agnostic* operator: :func:`apply_pca` mirrors
the call signature of :func:`exoplore.pipelines.sysrem.apply_sysrem`, so any
preparing pipeline can switch between PCA and SYSREM through a single
``detrend_method`` configuration field.

References
----------
de Kok et al. (2013), A&A, 554, A82.
Cheverall, Madhusudhan & Holmberg (2023), MNRAS, 522, 661.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Single PCA component
# ---------------------------------------------------------------------------


def pca_iteration(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the single dominant principal component (leading SVD mode).

    Parameters
    ----------
    data:
        Input spectral matrix, shape ``(n_spectra, n_pixels)``.

    Returns
    -------
    data_corrected:
        Residual after removing the leading rank-1 component,
        shape ``(n_spectra, n_pixels)``.
    correction:
        The subtracted component ``σ₁ u₁ v₁ᵀ``, shape ``(n_spectra, n_pixels)``.
    a_vector:
        Leading left singular vector (time-evolution mode), shape ``(n_spectra,)``.
    """
    data = np.asarray(data, dtype=float)
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    correction = np.outer(U[:, 0] * S[0], Vt[0])
    return data - correction, correction, U[:, 0]


# ---------------------------------------------------------------------------
# Multi-component PCA
# ---------------------------------------------------------------------------


def apply_pca(
    data: np.ndarray,
    n_components: int,
    good_pixels: np.ndarray | None = None,
    uncertainties: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Remove the top ``n_components`` principal components from a matrix.

    The leading ``n_components`` singular modes (the dominant common modes in
    the time-variation of each wavelength channel) are subtracted from the
    good-column submatrix.  No separate mean subtraction is applied: the
    dominant continuum/common mode is captured by the first component, so
    ``n_components = 1`` removes one common mode, directly comparable to one
    SYSREM iteration.

    The signature mirrors :func:`exoplore.pipelines.sysrem.apply_sysrem` so the
    two are interchangeable behind a ``detrend_method`` switch.  ``uncertainties``
    is accepted for signature compatibility but unused (PCA is unweighted; use
    SYSREM for inverse-variance weighting).

    Parameters
    ----------
    data:
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    n_components:
        Number of principal components to remove.
    good_pixels:
        Integer index array of unmasked columns.  If ``None``, all columns used.
    uncertainties:
        Ignored (compatibility with the SYSREM signature).

    Returns
    -------
    data_cleaned:
        Residual over the good columns after removing ``n_components`` modes,
        shape ``(n_spectra, n_good_pixels)``.
    corrections:
        List of length ``n_components``; each element is the rank-1 component
        removed, shape ``(n_spectra, n_good_pixels)``.
    basis:
        Left singular vectors of the removed modes (time-evolution modes),
        shape ``(n_spectra, n_components)``, for model reprocessing/projection,
        analogous to the SYSREM ``U`` basis.
    """
    if good_pixels is None:
        good_pixels = np.arange(data.shape[1])

    d = np.asarray(data[:, good_pixels], dtype=float)
    n_spectra = d.shape[0]

    nan_rows = ~np.isfinite(d).all(axis=1)
    if nan_rows.any():
        raise ValueError(
            f"apply_pca received {nan_rows.sum()} NaN/Inf row(s). "
            "Slice padded multi-night arrays to [:n_spectra, :] before calling."
        )

    n_components = int(max(0, min(n_components, min(d.shape))))

    U, S, Vt = np.linalg.svd(d, full_matrices=False)

    corrections: List[np.ndarray] = []
    basis = np.zeros((n_spectra, n_components), dtype=float)
    for i in range(n_components):
        corrections.append(np.outer(U[:, i] * S[i], Vt[i]))
        basis[:, i] = U[:, i]

    s_kept = S.copy()
    s_kept[:n_components] = 0.0
    data_cleaned = (U * s_kept) @ Vt

    return data_cleaned, corrections, basis
