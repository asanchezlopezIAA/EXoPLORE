"""
exoplore.pipelines.sysrem
=========================

SYSREM systematic noise removal for high-resolution spectral
time-series.

SYSREM (Tamuz et al. 2005) iteratively fits and removes the dominant
linear systematic component from a spectral matrix.  Each iteration
removes one "mode"; the standard approach is to apply multiple
iterations in sequence.

This implementation of the ``sysrem`` algorithm provides:

- takes clearly named arguments,
- is a standalone function (not a class method),
- returns all intermediate products for diagnostic use, and
- is tested independently.

References
----------
Tamuz, Mazeh & Zucker (2005), MNRAS, 356, 1466.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Single SYSREM iteration
# ---------------------------------------------------------------------------


def sysrem_iteration(
    data: np.ndarray,
    uncertainties: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove one SYSREM component from a spectral matrix.

    The algorithm fits the outer product ``a ⊗ c`` that minimises the
    inverse-variance weighted residuals, where ``a`` is the
    time-evolution vector and ``c`` is the spectral profile vector.
    Convergence is declared when the relative change in the correction
    drops below 0.1 %.

    Parameters
    ----------
    data:
        Input spectral matrix, shape ``(n_spectra, n_pixels)``.
    uncertainties:
        Per-pixel uncertainties (σ), same shape as ``data``.

    Returns
    -------
    data_corrected:
        Residual matrix after removing the component,
        shape ``(n_spectra, n_pixels)``.
    correction:
        The subtracted systematic component ``a1 * c1``,
        shape ``(n_spectra, n_pixels)``.
    a_vector:
        Time-evolution vector, shape ``(n_spectra, n_pixels)``
        (tiled form for direct subtraction).
    c_vector:
        Spectral profile vector, shape ``(n_spectra, n_pixels)``
        (tiled form for direct subtraction).
    """
    data = np.asarray(data, dtype=float)
    unc = np.asarray(uncertainties, dtype=float)
    n_spectra, n_pixels = data.shape

    # Guard: NaN rows corrupt the iterative sums silently.
    # This can only happen if a caller passes un-sliced padded multi-night arrays.
    nan_rows = ~np.isfinite(data).all(axis=1)
    if nan_rows.any():
        raise ValueError(
            f"sysrem_iteration received {nan_rows.sum()} NaN/Inf row(s) in 'data'. "
            "When using multi-night padded arrays, slice to [:n_spectra, :] before calling."
        )

    inv_var = unc ** -2.0

    # Initialise 'a' from a representative column (n_pixels // 3)
    a = data[:, n_pixels // 3].copy()

    # Tile to matrix
    a1 = np.tile(a, (n_pixels, 1)).T  # (n_spectra, n_pixels)

    # Initialise 'c'
    c = np.sum(data * a1 * inv_var, axis=0) / np.sum(a1 ** 2 * inv_var, axis=0)
    c1 = np.tile(c, (n_spectra, 1))  # (n_spectra, n_pixels)

    cor1 = c1 * a1
    cor0 = np.zeros_like(cor1)

    while np.sum(np.abs(cor0 - cor1)) / (np.sum(np.abs(cor0)) + 1e-300) >= 1e-3:
        cor0 = cor1

        a = np.sum(data * c1 * inv_var, axis=1) / np.sum(c1 ** 2 * inv_var, axis=1)
        a1 = np.tile(a, (n_pixels, 1)).T

        c = np.sum(data * a1 * inv_var, axis=0) / np.sum(a1 ** 2 * inv_var, axis=0)
        c1 = np.tile(c, (n_spectra, 1))

        cor1 = a1 * c1

    data_corrected = data - cor1
    return data_corrected, cor1, a1, c1


# ---------------------------------------------------------------------------
# Multi-iteration SYSREM
# ---------------------------------------------------------------------------


def apply_sysrem(
    data: np.ndarray,
    uncertainties: np.ndarray,
    n_iterations: int,
    good_pixels: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Apply SYSREM for ``n_iterations`` and return the cleaned matrix.

    Each iteration removes the dominant residual systematic.  The
    output data covers only the *good* (unmasked) columns while
    corrections are a list of the ``(n_spectra, n_good_pixels)``
    systematic components removed in each iteration.

    Parameters
    ----------
    data:
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    uncertainties:
        Per-pixel σ, same shape as ``data``.
    n_iterations:
        Number of SYSREM components to remove.
    good_pixels:
        Integer index array of unmasked columns.  If ``None``, all
        columns are used.

    Returns
    -------
    data_cleaned:
        Residual matrix after ``n_iterations`` components are removed,
        shape ``(n_spectra, n_good_pixels)``.
    corrections:
        List of length ``n_iterations``; each element is the systematic
        component removed in that iteration,
        shape ``(n_spectra, n_good_pixels)``.
    """
    if good_pixels is None:
        good_pixels = np.arange(data.shape[1])

    d = data[:, good_pixels].copy()
    u = uncertainties[:, good_pixels].copy()

    corrections: List[np.ndarray] = []
    for _ in range(n_iterations):
        d, cor, _, _ = sysrem_iteration(d, u)
        corrections.append(cor)

    return d, corrections


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def sysrem(data_in, errors_in, useful_pts):
    """Single SYSREM iteration.

    Slices ``data_in`` and ``errors_in`` to ``useful_pts``, runs one
    SYSREM iteration, and returns the sliced residual matrix together
    with the correction, ``a`` (time) vector, and ``c`` (spectral)
    vector.

    Parameters
    ----------
    data_in : ndarray
        Full spectral matrix, shape ``(n_spectra, n_pixels)``.
    errors_in : ndarray
        Noise matrix, same shape.
    useful_pts : ndarray
        Integer indices of good columns.

    Returns
    -------
    data_out : ndarray, shape ``(n_spectra, n_good)``
        Residual after removing one SYSREM component.
    cor1 : ndarray, shape ``(n_spectra, n_good)``
        Systematic component removed.
    a1 : ndarray, shape ``(n_spectra, n_good)``
        Time-evolution vector (tiled).
    c1 : ndarray, shape ``(n_spectra, n_good)``
        Spectral profile vector (tiled).
    """
    d = data_in[:, useful_pts].copy()
    e = errors_in[:, useful_pts].copy()
    data_out, cor1, a1, c1 = sysrem_iteration(d, e)
    return data_out, cor1, a1, c1


def SYSREM_filtering_projector(inp_dat, n_spectra, propag_noise, U):
    """Build per-order SYSREM projection matrices from stored basis vectors.

    Constructs the weighted least-squares projector P for each spectral
    order and night, using the basis vectors U stored from previous
    SYSREM passes.  Supports both single-night and multi-night modes.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used:
        ``"Different_nights"``, ``"n_orders"``, ``"n_nights"``,
        ``"sysrem_its"``.
    n_spectra : int or array-like
        Number of spectra.  Scalar for single-night; array of length
        ``n_nights`` for multi-night.
    propag_noise : ndarray
        Propagated noise, shape
        ``(n_orders, n_nights, n_spectra, n_pixels)``.
    U : ndarray
        SYSREM basis vectors, shape
        ``(n_orders, n_nights, n_spectra, n_passes)``.

    Returns
    -------
    P : ndarray
        Projection matrices, shape
        ``(n_orders, n_nights, n_spectra, n_spectra)``.
    """
    reg = 1e-8
    if not inp_dat["Different_nights"]:
        P = np.zeros(
            (inp_dat["n_orders"], inp_dat["n_nights"], n_spectra, n_spectra),
            float
        )
    else:
        P = np.full(
            (inp_dat["n_orders"], inp_dat["n_nights"],
             int(max(n_spectra)), int(max(n_spectra))),
            np.nan, float
        )

    for i_ord in range(inp_dat["n_orders"]):
        for j_night in range(inp_dat["n_nights"]):
            if not inp_dat["Different_nights"]:
                sigma_hat = np.mean(
                    propag_noise[i_ord, j_night, :, :], axis=-1
                )
            else:
                sigma_hat = np.mean(
                    propag_noise[i_ord, j_night, :n_spectra[j_night], :],
                    axis=-1
                )

            lambda2 = 1.0 / sigma_hat ** 2

            if not inp_dat["Different_nights"]:
                U0 = U[i_ord, j_night, :, :]
                ones_col = np.ones((n_spectra, 1))
            else:
                U0 = U[i_ord, j_night, :n_spectra[j_night], :]
                ones_col = np.ones((n_spectra[j_night], 1))

            U_aug = np.concatenate([U0, ones_col], axis=1)
            n_bases = inp_dat["sysrem_its"] + 1

            Mmat = U_aug.T @ (lambda2[:, None] * U_aug)
            invM = np.linalg.inv(Mmat + reg * np.eye(n_bases))

            Ut_L2 = U_aug.T * lambda2[None, :]
            P_ij = U_aug @ (invM @ Ut_L2)

            if not inp_dat["Different_nights"]:
                P[i_ord, j_night, :, :] = P_ij
            else:
                P[i_ord, j_night,
                  :n_spectra[j_night], :n_spectra[j_night]] = P_ij

    return P


def filter_model_singleorder(P, model_mat, useful_spectral_points):
    """Apply SYSREM projector P to a normalised model matrix (single order).

    The model is first median-normalised over the good pixels, then the
    SYSREM projector is applied:

    .. code-block:: text

        model_filt = model_norm - P @ model_norm

    This yields the residual model after the linear systematic baseline
    described by ``P`` has been removed, matching the treatment applied
    to the data during the SYSREM projector pipeline.

    Parameters
    ----------
    P : ndarray
        SYSREM projection matrix, shape ``(n_spectra, n_spectra)``.
        Typically built by :func:`SYSREM_filtering_projector` or
        :func:`SYSREM_filtering_projector_singleorder`.
    model_mat : ndarray
        Input model matrix, shape ``(n_spectra, n_pixels)``.
    useful_spectral_points : ndarray
        Integer indices of the unmasked (good) columns.

    Returns
    -------
    model_mat_prepared : ndarray
        SYSREM-filtered model matrix, shape ``(n_spectra, n_pixels)``.
        Columns outside ``useful_spectral_points`` are set to 1
        (the normalisation baseline).
    """
    # Median-normalise the model over good pixels
    model_norm = np.ones_like(model_mat)
    for nn in range(model_mat.shape[0]):
        model_norm[nn, useful_spectral_points] = (
            model_mat[nn, useful_spectral_points]
            / np.median(model_mat[nn, useful_spectral_points])
        )

    # Apply the projector to each pixel's time series
    model_filt = P @ model_norm          # shape: (n_spectra, n_pixels)
    model_mat_prepared = model_norm - model_filt

    return model_mat_prepared


def SYSREM_filtering_projector_singleorder(
        inp_dat, n_spectra, propag_noise, U, no_night_loop=False
):
    """Build SYSREM projection matrices for a single spectral order.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``"n_nights"``,
        ``"sysrem_its"``.
    n_spectra : int
        Number of spectra per night.
    propag_noise : ndarray
        Propagated noise, shape ``(1, n_nights, n_spectra, n_pixels)``.
    U : ndarray
        SYSREM basis vectors, shape
        ``(1, n_nights, n_spectra, n_passes)``.
    no_night_loop : bool
        If True, compute a single projector for shape ``(1,1,...)``.

    Returns
    -------
    P : ndarray
        Projection matrices, shape ``(1, n_nights, n_spectra, n_spectra)``
        (or ``(1,1,...) if ``no_night_loop=True``).
    """
    reg = 1e-8

    if no_night_loop:
        P = np.zeros((1, 1, n_spectra, n_spectra), float)
        sigma_hat = np.mean(propag_noise[0, 0, :, :], axis=-1)
        lambda2 = 1.0 / sigma_hat ** 2
        U0 = U[0, 0, :, :]
        ones_col = np.ones((n_spectra, 1))
        U_aug = np.concatenate([U0, ones_col], axis=1)
        n_bases = inp_dat["sysrem_its"] + 1
        Mmat = U_aug.T @ (lambda2[:, None] * U_aug)
        invM = np.linalg.inv(Mmat + reg * np.eye(n_bases))
        Ut_L2 = U_aug.T * lambda2[None, :]
        P_ij = U_aug @ (invM @ Ut_L2)
        P[0, 0, :, :] = P_ij
    else:
        P = np.zeros((1, inp_dat["n_nights"], n_spectra, n_spectra), float)
        for j_night in range(inp_dat["n_nights"]):
            sigma_hat = np.mean(propag_noise[0, j_night, :, :], axis=-1)
            lambda2 = 1.0 / sigma_hat ** 2
            U0 = U[0, j_night, :, :]
            ones_col = np.ones((n_spectra, 1))
            U_aug = np.concatenate([U0, ones_col], axis=1)
            n_bases = inp_dat["sysrem_its"] + 1
            Mmat = U_aug.T @ (lambda2[:, None] * U_aug)
            invM = np.linalg.inv(Mmat + reg * np.eye(n_bases))
            Ut_L2 = U_aug.T * lambda2[None, :]
            P_ij = U_aug @ (invM @ Ut_L2)
            P[0, j_night, :, :] = P_ij

    return P


def get_SYSREM_its_ordbyord(
        inp_dat, ccf_store, v_rest, with_signal, phase, berv, v_sys,
        pixels_left_right, ccf_v_step, v_erf
        ):
    """Determine the optimal number of SYSREM iterations order-by-order.

    For each order and each night, evaluates which number of SYSREM
    iterations maximises the cross-correlation peak of an injected
    planetary signal.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``"n_orders"``,
        ``"n_nights"``, ``"sysrem_its"``, ``"Kp_Vrest_inj"``,
        ``"BERV"``, ``"V_sys"``, ``"V_wind"``.
    ccf_store : ndarray, shape (n_orders, n_nights, n_lags, n_spectra, 2, sysrem_its)
        CCF store for all orders and SYSREM iterations.
    v_rest : ndarray
        Rest-frame velocity grid (km/s).
    with_signal : ndarray
        Indices of in-transit (signal) exposures.
    phase : ndarray
        Orbital phases of all exposures.
    berv : float
        BERV correction (km/s).
    v_sys : float
        Systemic velocity (km/s).
    pixels_left_right : int
        Half-width in pixels of the extraction window.
    ccf_v_step : float
        CCF velocity step (km/s).
    v_erf : ndarray
        Earth rest-frame CCF velocity grid.

    Returns
    -------
    sysrem_opt : ndarray, shape (n_orders, n_nights, 2)
        Optimal SYSREM iteration index for each order and night.
        ``[:, :, 0]`` = index of maximum CCF peak;
        ``[:, :, 1]`` = index of maximum CCF difference (with - without injection).
    """
    from exoplore.observation.velocity import get_V
    from exoplore.atmosphere.winds import find_nearest

    ccf_values_shift = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"], len(v_rest), len(with_signal),
         2, inp_dat["sysrem_its"]), float
        )

    # Calculate injected-planetary velocities during the night
    vp = get_V(
        inp_dat["Kp_Vrest_inj"][0], phase, berv,
        v_sys, inp_dat["Kp_Vrest_inj"][1]
        )

    # Move all matrices to INJECTION REST-FRAME
    for idx, i in enumerate(with_signal):
        v_inj_prf = np.linspace(
            vp[i] - pixels_left_right * ccf_v_step,
            vp[i] + pixels_left_right * ccf_v_step,
            num=2 * pixels_left_right + 1
            )
        for b in range(inp_dat["n_nights"]):
            for h in range(inp_dat["n_orders"]):
                for n in range(2):
                    for l in range(inp_dat["sysrem_its"]):
                        ccf_values_shift[h, b, :, idx, n, l] = np.interp(
                            v_inj_prf, v_erf, ccf_store[h, b, :, idx, n, l]
                            )

    # Co-adding in time
    ccf_tot = np.sum(ccf_values_shift, axis=3)

    # Find the velocity index nearest to 0 (V_wind of the injected signal)
    injection_v = np.argwhere(v_rest == find_nearest(v_rest, 0))[0][0]
    ccf_maxinj_pos = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"], 2, inp_dat["sysrem_its"]), float
        )
    v_maxinj_pos = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"], inp_dat["sysrem_its"]), int
        )

    for b in range(inp_dat["n_nights"]):
        for h in range(inp_dat["n_orders"]):
            for l in range(inp_dat["sysrem_its"]):
                v_maxinj_pos[h, b, l] = np.where(
                    ccf_tot[h, b, :, 1, l] == np.amax(
                        ccf_tot[h, b, injection_v - 20:injection_v + 21, 1, l]
                        )
                    )[0][0]
                ccf_maxinj_pos[h, b, 1, l] = ccf_tot[h, b, v_maxinj_pos[h, b, l], 1, l]
                ccf_maxinj_pos[h, b, 0, l] = ccf_tot[h, b, v_maxinj_pos[h, b, l], 0, l]

    sysrem_opt = np.zeros((inp_dat["n_orders"], inp_dat["n_nights"], 2), float)
    for b in range(inp_dat["n_nights"]):
        for h in range(inp_dat["n_orders"]):
            diff = ccf_maxinj_pos[h, b, 1, :] - ccf_maxinj_pos[h, b, 0, :]
            # Exclude the first SYSREM iteration (index 0 and 1) as it often
            # has strong residuals that can artificially dominate the optimum.
            sysrem_opt[h, b, 0] = int(
                np.where(
                    ccf_maxinj_pos[h, b, 1, 2:] == np.amax(ccf_maxinj_pos[h, b, 1, 2:])
                    )[0][0] + 2
                )
            sysrem_opt[h, b, 1] = int(np.where(diff[2:] == np.amax(diff[2:]))[0] + 2)

    return sysrem_opt
