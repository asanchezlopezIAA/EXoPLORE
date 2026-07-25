"""
exoplore.pipelines.nortmann26
=============================

Faithful implementation of the CRIRES+ high resolution analysis pipeline of
**Nortmann et al. (2026)** ("Cloudy with a chance of metals: Indications of
CO2 in the atmosphere of GJ 1214 b from high resolution K band spectroscopy",
A&A, arXiv:2604.15292).

This module provides the Nortmann specific *preparation* primitives; the
driving branch lives in :mod:`exoplore.pipelines.prepare` (pipeline name
``"Nortmann26"``), mirroring the way :mod:`exoplore.pipelines.cheverall26`
backs the ``"Cheverall26"`` route.  Everything here is additive and gated to
that route; no other pipeline is affected.

The two methodological pieces implemented here, verbatim from the paper:

1. **Two step common blaze normalisation** (their Sec 3.1).  All A and B
   spectra are treated as one data set.  (i) A *master* spectrum is formed by
   summing the out of transit spectra; each individual spectrum is divided by
   a 2nd order polynomial fit to its ratio to master, with significant
   tellurics (theoretical transmittance < 96 % of continuum) masked *for the
   fit only*, plus NaN/>4 sigma outlier flagging.  (ii) The master itself is
   fitted with a 2nd order polynomial while iteratively masking stellar lines
   (strong negative outliers + neighbours, 5 passes) and every spectrum is
   divided by that common blaze.

2. **SYSREM in the division convention** (their Sec 3.2, following Gibson et
   al. 2020).  After shifting to the stellar rest frame and masking the
   deepest tellurics (< 20 % of continuum), SYSREM is run for N iterations and
   the *normalised uncorrected data are divided by the summed up correction
   matrices* (rank N reconstruction of the systematics), rather than having
   the components subtracted.  Uncertainties are divided through by the same
   model.  The per iteration time basis vectors ``U`` are retained for the
   Gibson 2022 model filtering used in the retrieval (their Appendix A.3).

References
----------
Nortmann et al. 2026, A&A (arXiv:2604.15292v1).
Gibson et al. 2020, MNRAS 493, 2215 (division convention SYSREM).
Gibson et al. 2022, MNRAS 512, 4618 (model filtering).
Tamuz, Mazeh & Zucker 2005, MNRAS 356, 1466 (SYSREM).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from exoplore.pipelines.sysrem import sysrem_iteration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_C_KMS = 299792.458  # speed of light, km/s


# ---------------------------------------------------------------------------
# Telluric masks (Sec 3.1 / 3.2)
# ---------------------------------------------------------------------------
def nortmann_telluric_mask(
    telluric_transmittance: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean column mask where a theoretical telluric spectrum is deep.

    Following Nortmann Sec 3.1 (threshold 0.96, normalisation only mask) and
    Sec 3.2 (threshold 0.20, pre SYSREM mask of the deepest cores whose
    time variation saturates).

    Parameters
    ----------
    telluric_transmittance:
        Theoretical telluric transmittance (1 = transparent), shape
        ``(n_pixels,)`` or ``(n_spectra, n_pixels)``.  A ``Molecfit`` model in
        practice; any per pixel transmittance works.
    threshold:
        Mask a column where the transmittance falls below this fraction of the
        continuum in *any* spectrum.

    Returns
    -------
    mask:
        Boolean array, shape ``(n_pixels,)``; ``True`` = masked.
    """
    t = np.atleast_2d(np.asarray(telluric_transmittance, dtype=float))
    return np.any(t < threshold, axis=0)


# ---------------------------------------------------------------------------
# Two step common blaze normalisation (Sec 3.1)
# ---------------------------------------------------------------------------
def _polyfit_divide(
    x: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    deg: int = 2,
) -> np.ndarray:
    """Fit a degree-``deg`` polynomial to ``y`` over ``~fit_mask`` and divide.

    ``fit_mask`` is ``True`` where a point is EXCLUDED from the fit (tellurics,
    stellar lines, outliers).  The polynomial is evaluated on the full grid so
    the returned division is defined everywhere.
    """
    good = ~fit_mask & np.isfinite(y)
    if good.sum() <= deg + 1:
        return np.ones_like(y)
    coef = np.polyfit(x[good], y[good], deg)
    trend = np.polyval(coef, x)
    trend[trend == 0.0] = 1.0
    return trend


def nortmann_normalise(
    wave: np.ndarray,
    data: np.ndarray,
    uncertainties: np.ndarray,
    without_signal: np.ndarray,
    telluric_transmittance: np.ndarray | None = None,
    tell_threshold: float = 0.96,
    n_stellar_iter: int = 5,
    outlier_sigma: float = 4.0,
    deg: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two step common blaze normalisation of one wavelength segment.

    Faithful to Nortmann Sec 3.1.  Operates on a single segment (order x
    detector) of the combined A+B time series.

    Parameters
    ----------
    wave:
        Wavelength grid, shape ``(n_pixels,)``.
    data:
        Flux time series, shape ``(n_spectra, n_pixels)``.
    uncertainties:
        Per pixel uncertainties, same shape.
    without_signal:
        Integer indices of out of transit spectra (used to build the master).
    telluric_transmittance:
        Theoretical telluric transmittance (``Molecfit``), shape
        ``(n_pixels,)`` or ``(n_spectra, n_pixels)``.  If ``None``, the
        telluric mask is empty (tellurics not excluded from the fits).
    tell_threshold:
        Telluric mask fraction for the normalisation fits (0.96 in the paper).
    n_stellar_iter:
        Stellar line masking iterations for the master fit (5 in the paper).
    outlier_sigma:
        Sigma for NaN/hot pixel flagging (4 in the paper).
    deg:
        Polynomial degree (2 in the paper).

    Returns
    -------
    data_norm:
        Common blaze normalised flux, shape ``(n_spectra, n_pixels)``.
    noise_norm:
        Uncertainties propagated through the same divisions.
    flagged:
        Boolean array of flagged (NaN / >outlier_sigma) pixels, same shape,
        for the downstream column masking step.
    """
    data = np.asarray(data, dtype=float)
    unc = np.asarray(uncertainties, dtype=float)
    n_spectra, n_pixels = data.shape
    x = np.asarray(wave, dtype=float)

    tell_mask = (
        nortmann_telluric_mask(telluric_transmittance, tell_threshold)
        if telluric_transmittance is not None
        else np.zeros(n_pixels, dtype=bool)
    )

    # ---- Step (i): per spectrum divide by 2nd order fit to ratio to master.
    master = np.nansum(data[without_signal], axis=0)
    master_safe = np.where(master == 0.0, np.nan, master)

    data_1 = np.empty_like(data)
    noise_1 = np.empty_like(unc)
    flagged = np.zeros_like(data, dtype=bool)

    for i in range(n_spectra):
        ratio = data[i] / master_safe
        # NaN + >outlier_sigma flagging on the ratio (bad/hot pixels).
        finite = np.isfinite(ratio)
        flag = ~finite
        if finite.sum() > deg + 2:
            med = np.median(ratio[finite])
            mad = np.median(np.abs(ratio[finite] - med)) * 1.4826 + 1e-300
            flag |= np.abs(ratio - med) > outlier_sigma * mad
        flagged[i] = flag
        fit_mask = tell_mask | flag
        trend = _polyfit_divide(x, ratio, fit_mask, deg)
        data_1[i] = data[i] / trend
        noise_1[i] = unc[i] / trend

    # ---- Step (ii): common blaze = 2nd order fit to the master with
    #      iterative stellar line masking; divide every spectrum by it.
    master_n = np.nansum(data_1[without_signal], axis=0)
    master_n /= np.nanmedian(master_n[np.isfinite(master_n)]) or 1.0
    stellar_mask = tell_mask.copy()
    for _ in range(n_stellar_iter):
        trend = _polyfit_divide(x, master_n, stellar_mask, deg)
        resid = master_n / trend - 1.0
        good = ~stellar_mask & np.isfinite(resid)
        if good.sum() <= deg + 2:
            break
        sd = np.std(resid[good])
        # Strong NEGATIVE outliers = stellar/telluric line cores; mask them
        # and their immediate neighbours.
        newly = (resid < -outlier_sigma * sd) & np.isfinite(resid)
        if not newly.any():
            break
        idx = np.where(newly)[0]
        stellar_mask[idx] = True
        stellar_mask[np.clip(idx - 1, 0, n_pixels - 1)] = True
        stellar_mask[np.clip(idx + 1, 0, n_pixels - 1)] = True

    common_blaze = _polyfit_divide(x, master_n, stellar_mask, deg)
    data_norm = data_1 / common_blaze[None, :]
    noise_norm = noise_1 / common_blaze[None, :]

    return data_norm, noise_norm, flagged


def nortmann_column_mask(
    flagged: np.ndarray,
    data: np.ndarray,
    max_flagged_per_column: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Column mask heavily flagged columns; interpolate the rest (Sec 3.1).

    A column with more than ``max_flagged_per_column`` flagged pixels is
    masked entirely; otherwise flagged pixels are linearly interpolated over
    in time.

    Returns
    -------
    data_out:
        Data with light flags interpolated (heavy columns left as is; caller
        applies the column mask).
    column_mask:
        Boolean array, shape ``(n_pixels,)``; ``True`` = column masked.
    """
    data_out = np.array(data, dtype=float, copy=True)
    n_spectra, n_pixels = data_out.shape
    per_col = flagged.sum(axis=0)
    column_mask = per_col > max_flagged_per_column
    # Light columns: interpolate flagged pixels along the time axis.
    for j in np.where((per_col > 0) & ~column_mask)[0]:
        rows = np.where(flagged[:, j])[0]
        good = np.where(~flagged[:, j])[0]
        if good.size >= 2:
            data_out[rows, j] = np.interp(rows, good, data_out[good, j])
    return data_out, column_mask


# ---------------------------------------------------------------------------
# Stellar rest frame alignment (Sec 3.2, Eq. 1)
# ---------------------------------------------------------------------------
def nortmann_shift_to_stellar_frame(
    wave: np.ndarray,
    data: np.ndarray,
    v_star: np.ndarray,
) -> np.ndarray:
    """Doppler shift each spectrum into the stellar rest frame (Eq. 1).

    ``v_star(t) = v_sys + v_bary(t) + K_star sin(2 pi phi(t))``.  Each exposure
    is interpolated onto the grid it would have at zero stellar velocity, so
    the (quasi static) stellar lines align across time for optimal SYSREM
    removal.  Flux is resampled by interpolation, faithful to this lineage.

    Parameters
    ----------
    wave:
        Common wavelength grid, shape ``(n_pixels,)``.
    data:
        Flux time series (already A/B aligned), shape ``(n_spectra, n_pixels)``.
    v_star:
        Stellar radial velocity per exposure (km/s), shape ``(n_spectra,)``.

    Returns
    -------
    shifted:
        Flux resampled into the stellar rest frame, same shape.
    """
    wave = np.asarray(wave, dtype=float)
    data = np.asarray(data, dtype=float)
    v_star = np.asarray(v_star, dtype=float)
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        # A feature at observed wavelength w sits at w*(1 - v/c) in the star
        # frame; sample the observed spectrum at the wavelengths that map onto
        # the common grid.
        src = wave * (1.0 + v_star[i] / _C_KMS)
        out[i] = np.interp(wave, src, data[i])
    return out


# ---------------------------------------------------------------------------
# SYSREM, division convention (Sec 3.2, Gibson et al. 2020)
# ---------------------------------------------------------------------------
def nortmann_sysrem_division(
    data: np.ndarray,
    uncertainties: np.ndarray,
    n_iterations: int,
    good_pixels: np.ndarray | None = None,
    return_all_iterations: bool = False,
) -> dict:
    """Run SYSREM and correct by DIVISION (Gibson 2020; Nortmann Sec 3.2).

    Standard SYSREM finds, at iteration k, a rank 1 systematic component
    ``C_k = a_k (x) c_k`` of the running residual.  The *summed* correction
    after N passes, ``M_N = sum_k C_k``, is the rank N reconstruction of the
    (quasi static) telluric+stellar systematics ,  which do NOT include the
    velocity shifting planet signal.  The division convention then sets

        R_N = D / M_N           (centred near 1; planet preserved),

    instead of the subtractive ``D - M_N``.  Uncertainties are divided through
    by the same model.  The per iteration time vectors ``U`` (the ``a_k``) are
    kept for the Gibson 2022 model filter used in the retrieval.

    Parameters
    ----------
    data:
        Common blaze normalised, stellar frame flux, shape
        ``(n_spectra, n_pixels)``; continuum near 1.
    uncertainties:
        Per pixel uncertainties, same shape.
    n_iterations:
        Number of SYSREM components (9 in the paper for GJ 1214 b).
    good_pixels:
        Integer indices of unmasked columns.  If ``None``, all are used.
    return_all_iterations:
        If ``True``, also return the division corrected residual after *each*
        iteration (for the iteration choice diagnostics of Appendix A.2).

    Returns
    -------
    dict with keys:
        ``residual``   Division corrected data at ``n_iterations`` (full width,
                       masked columns = 1), shape ``(n_spectra, n_pixels)``.
        ``noise``      Propagated uncertainties (masked columns = 1).
        ``model``      Summed correction ``M_N`` (systematics model).
        ``U``          Time basis vectors, shape ``(n_spectra, n_iterations)``.
        ``residual_per_iter`` (optional) list of length ``n_iterations``.
    """
    data = np.asarray(data, dtype=float)
    unc = np.asarray(uncertainties, dtype=float)
    n_spectra, n_pixels = data.shape
    if good_pixels is None:
        good_pixels = np.arange(n_pixels)

    d = data[:, good_pixels].copy()
    u = unc[:, good_pixels].copy()

    model = np.zeros_like(d)          # M_k = sum of corrections
    U = np.zeros((n_spectra, n_iterations))
    residual_per_iter: List[np.ndarray] = []

    r = d.copy()
    for k in range(n_iterations):
        r, cor, a1, _ = sysrem_iteration(r, u)
        model += cor
        U[:, k] = a1[:, 0]
        if return_all_iterations:
            m_safe = np.where(np.abs(model) < 1e-12, 1.0, model)
            residual_per_iter.append(d / m_safe)

    # Division correction with the final rank N model.
    m_safe = np.where(np.abs(model) < 1e-12, 1.0, model)
    div = d / m_safe
    noise_div = u / np.abs(m_safe)

    # Re embed on the full grid (masked columns -> 1).
    residual = np.ones((n_spectra, n_pixels), float)
    noise = np.ones((n_spectra, n_pixels), float)
    model_full = np.ones((n_spectra, n_pixels), float)
    residual[:, good_pixels] = div
    noise[:, good_pixels] = noise_div
    model_full[:, good_pixels] = model

    out = {
        "residual": residual,
        "noise": noise,
        "model": model_full,
        "U": U,
    }
    if return_all_iterations:
        full_iters = []
        for ri in residual_per_iter:
            fr = np.ones((n_spectra, n_pixels), float)
            fr[:, good_pixels] = ri
            full_iters.append(fr)
        out["residual_per_iter"] = full_iters
    return out


# ---------------------------------------------------------------------------
# Model filtering for the retrieval (Sec 3.3 / Appendix A.3, Eqs. A.1 to A.3)
# ---------------------------------------------------------------------------
def nortmann_model_filter(
    U: np.ndarray,
    sigma_per_spectrum: np.ndarray,
    model_dev: np.ndarray,
    good_pixels: np.ndarray | None = None,
) -> np.ndarray:
    """Filter a forward model to match the SYSREM division distortion.

    Faithful to Nortmann Appendix A.3 (Eqs. A.1 to A.3), which reuses the fixed
    time basis vectors ``U`` from the data's SYSREM run and refits only the
    wavelength weights per likelihood evaluation:

        F  = U (Lambda U)^dagger Lambda                          (A.1)
        C  = F M = U (Lambda U)^dagger (Lambda M)                (A.2)
        M' = (M + 1) / (C + 1) - 1                               (A.3)

    where ``Lambda = diag(1/sigma_av)`` (``sigma_av`` = mean uncertainty over
    wavelength per spectrum) and ``M`` is the normalised model minus 1.  Two
    deliberate deviations from Gibson et al. (2022), stated in the paper:
    **no ones column** ``u_0`` is appended to ``U`` (there is no master spectrum
    division before SYSREM in this analysis), and the correction is applied by
    **division** (A.3) rather than subtraction (the two differ negligibly, but
    division is the faithful form for the division convention SYSREM).

    Parameters
    ----------
    U:
        Time basis vectors from the data's SYSREM, shape ``(n_spectra,
        n_iterations)`` (the ``U`` returned by :func:`nortmann_sysrem_division`).
    sigma_per_spectrum:
        Mean uncertainty over wavelength for each spectrum, shape
        ``(n_spectra,)`` (``sigma_av``).
    model_dev:
        Normalised model matrix **minus 1** (continuum at 0), shape
        ``(n_spectra, n_pixels)``; already Doppler shifted to the trial planet
        velocity per exposure.
    good_pixels:
        Integer indices of unmasked columns.  If ``None``, all are used.

    Returns
    -------
    model_filtered:
        Filtered model deviation ``M'`` (continuum at 0), shape
        ``(n_spectra, n_pixels)``; masked columns left at 0.
    """
    U = np.asarray(U, dtype=float)
    n_spectra = U.shape[0]
    if good_pixels is None:
        good_pixels = np.arange(model_dev.shape[1])
    m = np.asarray(model_dev, dtype=float)[:, good_pixels]

    lam = 1.0 / np.asarray(sigma_per_spectrum, dtype=float)     # (L,)
    lam_U = lam[:, None] * U                                    # (L, N)
    pinv = np.linalg.pinv(lam_U)                               # (ΛU)^dagger, (N, L)
    C = U @ (pinv @ (lam[:, None] * m))                        # (L, n_good)

    m_prime = (m + 1.0) / (C + 1.0) - 1.0                       # Eq. A.3
    out = np.zeros_like(model_dev, dtype=float)
    out[:, good_pixels] = m_prime
    return out


# ---------------------------------------------------------------------------
# Kp v maps and trail statistics (Sec 3.4 / Appendix A.4)
# ---------------------------------------------------------------------------
def _shift_ccf_to_planet_frame(
    ccf: np.ndarray,
    v_grid: np.ndarray,
    vp_star: np.ndarray,
) -> np.ndarray:
    """Shift each exposure's CCF so the planet trail lands at rest velocity 0.

    A signal that sits at velocity ``vp_star[t]`` in the stellar frame CCF is
    moved to rest velocity 0 by evaluating the CCF at ``v_rest + vp_star[t]``.

    Parameters
    ----------
    ccf:
        CCF matrix, shape ``(n_v, n_spectra)``.
    v_grid:
        Velocity grid of the CCF (km/s), shape ``(n_v,)``.
    vp_star:
        Planet velocity per exposure in the CCF frame (km/s), shape
        ``(n_spectra,)`` (i.e. ``Kp sin 2 pi phi - v_star`` for a trial Kp).

    Returns
    -------
    shifted:
        CCF resampled onto the rest velocity grid, same shape as ``ccf``.
    """
    n_v, n_spectra = ccf.shape
    shifted = np.empty_like(ccf)
    for t in range(n_spectra):
        shifted[:, t] = np.interp(
            v_grid + vp_star[t], v_grid, ccf[:, t],
            left=np.nan, right=np.nan,
        )
    return shifted


def nortmann_kpv_map(
    ccf: np.ndarray,
    v_grid: np.ndarray,
    phase: np.ndarray,
    in_transit: np.ndarray,
    kp_grid: np.ndarray,
    v_star: np.ndarray | float = 0.0,
    noise_exclude_kms: float = 10.0,
    noise_max_kms: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the Kp v signal to noise map (Nortmann Sec 3.4.1).

    For each trial ``Kp`` the summed CCF is Doppler shifted from the stellar
    rest frame into the planet rest frame by ``vp*(t) = Kp sin 2 pi phi(t) -
    v_star`` (their Eq. 3) and co added over the fully in transit exposures
    (second to third contact).  The resulting map is normalised by the
    standard deviation of the no signal velocity regions
    ``noise_exclude_kms < |v_rest| < noise_max_kms`` over the full Kp range,
    giving a signal to noise map.

    Parameters
    ----------
    ccf:
        CCF summed over wavelength segments, shape ``(n_v, n_spectra)``, in the
        stellar rest frame.
    v_grid:
        CCF velocity grid (km/s), shape ``(n_v,)``.
    phase:
        Orbital phase per exposure, shape ``(n_spectra,)``.
    in_transit:
        Boolean mask of the fully in transit (t2 to t3) exposures to co add,
        shape ``(n_spectra,)``.
    kp_grid:
        Trial Kp values (km/s), shape ``(n_kp,)``.
    v_star:
        Stellar velocity per exposure (km/s) subtracted in Eq. 3.  Pass 0 when
        the CCF is already in the stellar rest frame.
    noise_exclude_kms, noise_max_kms:
        Inner and outer bounds of the no signal velocity band used for the
        noise normalisation (10 and 150 in the paper).

    Returns
    -------
    snr_map:
        Signal to noise map, shape ``(n_kp, n_v)`` (rows = Kp, columns =
        rest velocity ``v_grid``).
    v_grid:
        The rest velocity grid (returned for convenience; equals the input).
    """
    ccf = np.asarray(ccf, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)
    phase = np.asarray(phase, dtype=float)
    kp_grid = np.asarray(kp_grid, dtype=float)
    it = np.asarray(in_transit, dtype=bool)
    vst = (np.full(phase.shape, float(v_star))
           if np.ndim(v_star) == 0 else np.asarray(v_star, dtype=float))

    n_kp = kp_grid.size
    cmap = np.full((n_kp, v_grid.size), np.nan)
    for k, kp in enumerate(kp_grid):
        vp_star = kp * np.sin(2.0 * np.pi * phase) - vst
        shifted = _shift_ccf_to_planet_frame(ccf, v_grid, vp_star)
        cmap[k] = np.nansum(shifted[:, it], axis=1)

    # Noise from the no signal band over the whole map.
    band = (np.abs(v_grid) > noise_exclude_kms) & (np.abs(v_grid) < noise_max_kms)
    noise = np.nanstd(cmap[:, band])
    if noise == 0 or not np.isfinite(noise):
        noise = 1.0
    return cmap / noise, v_grid


def nortmann_in_out_trail(
    ccf: np.ndarray,
    v_grid: np.ndarray,
    phase: np.ndarray,
    in_transit: np.ndarray,
    kp: float,
    v_star: np.ndarray | float = 0.0,
    peak_v: float = 0.0,
    trail_halfwidth_kms: float = 0.5,
    out_min_kms: float = 10.0,
    out_max_kms: float = 250.0,
) -> dict:
    """In trail vs out of trail Welch t test for a CCF signal (Appendix A.4).

    The CCF is shifted to the planet rest frame at the given ``kp``.  The in
    trail distribution is the CCF values within ``+/- trail_halfwidth_kms`` of
    ``peak_v`` across the fully in transit exposures; the out of trail
    distribution is the values at ``out_min_kms < |v_rest| < out_max_kms`` over
    the same exposures.  A Welch t test (unequal variances) tests whether the
    two are drawn from the same parent distribution.

    Returns
    -------
    dict with keys ``t`` (Welch t), ``sigma`` (two sided Gaussian equivalent),
    ``mid_in``, ``mid_out`` (distribution means), ``n_in``, ``n_out``.
    """
    ccf = np.asarray(ccf, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)
    phase = np.asarray(phase, dtype=float)
    it = np.asarray(in_transit, dtype=bool)
    vst = (np.full(phase.shape, float(v_star))
           if np.ndim(v_star) == 0 else np.asarray(v_star, dtype=float))

    vp_star = kp * np.sin(2.0 * np.pi * phase) - vst
    shifted = _shift_ccf_to_planet_frame(ccf, v_grid, vp_star)[:, it]

    in_band = np.abs(v_grid - peak_v) <= trail_halfwidth_kms
    out_band = (np.abs(v_grid) > out_min_kms) & (np.abs(v_grid) < out_max_kms)
    a = shifted[in_band].ravel()
    b = shifted[out_band].ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / a.size + vb / b.size)
    t = (ma - mb) / se if se > 0 else 0.0
    # Two sided Gaussian equivalent (large sample limit of the t statistic).
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))
    from scipy.special import erfcinv  # local import; engine ships scipy
    sigma = float(sqrt(2.0) * erfcinv(p)) if p > 0 else float("inf")
    return {"t": float(t), "sigma": sigma, "mid_in": float(ma),
            "mid_out": float(mb), "n_in": int(a.size), "n_out": int(b.size)}


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Inject a moving planet into a synthetic telluric+stellar time series
    and confirm the division SYSREM chain recovers it while removing the
    quasi static systematics.
    """
    rng = np.random.default_rng(0)
    n_spectra, n_pixels = 40, 400
    wave = np.linspace(2000.0, 2010.0, n_pixels)          # nm
    airmass = np.linspace(1.15, 1.9, n_spectra)

    # Quasi static telluric lines (fixed wavelength, depth grows with airmass).
    tell_depth = np.zeros(n_pixels)
    for w0, d0 in [(2002.0, 0.4), (2004.5, 0.6), (2007.3, 0.3), (2008.9, 0.5)]:
        tell_depth += d0 * np.exp(-0.5 * ((wave - w0) / 0.05) ** 2)
    tell = np.exp(-np.outer(airmass / airmass.mean(), tell_depth))

    # Quasi static stellar lines (fixed in the stellar frame == here).
    star = np.ones(n_pixels)
    for w0, d0 in [(2003.2, 0.2), (2006.1, 0.25)]:
        star -= d0 * np.exp(-0.5 * ((wave - w0) / 0.04) ** 2)

    # Smooth blaze (per exposure throughput drift) + continuum.
    blaze = 1.0 + 0.05 * np.sin((wave - wave[0]) / 3.0)
    throughput = np.linspace(1.0, 0.9, n_spectra)

    # Moving planet: a FOREST of many weak lines (a realistic molecular
    # template) that Doppler shifts across the transit.  Each line is buried
    # in the noise (sub pixel S/N << 1, so it never trips the outlier flag,
    # exactly like a real planet); the signal only emerges after co adding
    # many lines and many frames in the CCF.
    phase = np.linspace(-0.03, 0.03, n_spectra)
    intransit = np.abs(phase) < 0.02
    kp, vsys = 150.0, 0.0
    vp = kp * np.sin(2 * np.pi * phase) + vsys
    n_lines = 30
    line_w0 = rng.uniform(2000.5, 2009.5, n_lines)
    line_d0 = rng.uniform(0.002, 0.004, n_lines)   # ~ noise level, per line
    line_sig = 0.03

    def _planet_template(vel):
        """Absorption template (1 - depth) at radial velocity ``vel`` km/s."""
        t = np.ones(n_pixels)
        for w0, d0 in zip(line_w0, line_d0):
            t -= d0 * np.exp(-0.5 * ((wave - w0 * (1.0 + vel / _C_KMS))
                                     / line_sig) ** 2)
        return t

    planet = np.ones((n_spectra, n_pixels))
    for i in range(n_spectra):
        if intransit[i]:
            planet[i] = _planet_template(vp[i])

    flux = (throughput[:, None] * blaze[None, :]
            * tell * star[None, :] * planet)
    flux *= 1.0 + rng.normal(0, 0.002, flux.shape)
    sigma = 0.002 * np.ones_like(flux)

    without_signal = np.where(~intransit)[0]

    dn, nn, flagged = nortmann_normalise(
        wave, flux, sigma, without_signal,
        telluric_transmittance=(tell.min(axis=0)),  # deepest transmittance
    )
    di, colmask = nortmann_column_mask(flagged, dn)
    good = np.where(~colmask)[0]
    res = nortmann_sysrem_division(di, nn, n_iterations=5, good_pixels=good)

    r = res["residual"]
    # The telluric/stellar structure should be gone: residual std at the deep
    # telluric columns should be small and comparable to elsewhere.
    resid_dev = np.nanstd(r[:, good] - 1.0)
    # Cross correlate the in transit residual with the moving planet template
    # (mean subtracted, matched filter) along the true track; compare against a
    # shuffled phase null.
    def _ccf_at_track(shift_phase):
        acc = 0.0
        for i in np.where(intransit)[0]:
            tmpl = _planet_template(kp * np.sin(2 * np.pi * shift_phase[i]) + vsys)
            tmpl = tmpl[good] - np.mean(tmpl[good])
            acc += np.nansum((r[i, good] - 1.0) * tmpl)
        return acc
    true_resp = _ccf_at_track(phase)
    null = np.array([_ccf_at_track(rng.permutation(phase)) for _ in range(300)])
    z = (true_resp - null.mean()) / (null.std() + 1e-300)

    print("nortmann26 self test 1: normalisation + SYSREM division")
    print(f"  residual std about 1 (systematics removed): {resid_dev:.4f}")
    print(f"  masked columns: {colmask.sum()}/{n_pixels}")
    print(f"  planet CCF response z vs shuffled phase null: {z:.2f}")
    assert resid_dev < 0.05, "systematics not removed"
    assert z > 3.0, "injected moving planet not recovered by division SYSREM"
    print("  PASS")


def _self_test_filter() -> None:
    """Validate the model filter (Appendix A.3) by the paper's own criterion:
    the filtered clean model must reproduce the distortion SYSREM imposes on
    the same model when it is injected into the data.

    The Gibson 2022 / Nortmann filter is a **first order** approximation of
    iterative SYSREM: it is essentially exact when SYSREM removes little of the
    (velocity shifting) signal ,  the physically relevant, signal preserving
    regime any real analysis operates in ,  and degrades only when SYSREM nearly
    annihilates the signal (a regime one avoids by choosing the iteration count
    on injection recovery, their Appendix A.2).  This test therefore uses a
    moving planet + modest iteration count (~20 % removed), where the filter is
    accurate; the amplitude/correlation degrade smoothly outside it, as
    documented.
    """
    rng = np.random.default_rng(1)
    n_spectra, n_pixels = 60, 400
    wave = np.linspace(2000.0, 2010.0, n_pixels)
    airmass = np.linspace(1.15, 1.9, n_spectra)

    # Systematics only data (telluric grows with airmass + stellar + blaze).
    td = np.zeros(n_pixels)
    for w0, d0 in [(2002.0, 0.4), (2004.5, 0.6), (2007.3, 0.3), (2008.9, 0.5)]:
        td += d0 * np.exp(-0.5 * ((wave - w0) / 0.05) ** 2)
    tell = np.exp(-np.outer(airmass / airmass.mean(), td))
    star = np.ones(n_pixels)
    for w0, d0 in [(2003.2, 0.2), (2006.1, 0.25)]:
        star -= d0 * np.exp(-0.5 * ((wave - w0) / 0.04) ** 2)
    blaze = 1.0 + 0.05 * np.sin((wave - wave[0]) / 3.0)
    d_sys = (np.linspace(1.0, 0.9, n_spectra)[:, None] * blaze[None, :]
             * tell * star[None, :])
    d_sys *= 1.0 + rng.normal(0, 0.001, d_sys.shape)
    sigma = 0.001 * np.ones_like(d_sys)

    # A moving planet MODEL matrix (continuum 1), same track for data & filter.
    # Fast trail + few iterations -> signal preserving regime.
    phase = np.linspace(-0.06, 0.06, n_spectra)
    intransit = np.abs(phase) < 0.048
    vp = 300.0 * np.sin(2 * np.pi * phase)
    n_lines = 30
    lw = rng.uniform(2000.5, 2009.5, n_lines)
    Mmat = np.ones((n_spectra, n_pixels))
    for i in range(n_spectra):
        if intransit[i]:
            for w0 in lw:
                Mmat[i] -= 0.01 * np.exp(
                    -0.5 * ((wave - w0 * (1.0 + vp[i] / _C_KMS)) / 0.03) ** 2)

    good = np.arange(n_pixels)
    n_it = 3
    # SYSREM on clean systematics -> get U and R_sys.
    out_sys = nortmann_sysrem_division(d_sys, sigma, n_it, good)
    R_sys, U = out_sys["residual"], out_sys["U"]
    # Inject the model multiplicatively and re run SYSREM.
    out_inj = nortmann_sysrem_division(d_sys * Mmat, sigma, n_it, good)
    R_inj = out_inj["residual"]
    # Planet as SYSREM leaves it, in the division convention (ratio - 1).
    distortion_actual = R_inj / np.where(R_sys == 0, 1.0, R_sys) - 1.0

    # Filter the CLEAN model with the data's U (Appendix A.3).
    sigma_av = np.mean(sigma, axis=1)
    Mp = nortmann_model_filter(U, sigma_av, Mmat - 1.0, good)

    it = np.where(intransit)[0]
    a = distortion_actual[it].ravel()
    b = Mp[it].ravel()
    r = np.corrcoef(a, b)[0, 1]
    removed = 1.0 - np.std(a) / np.std((Mmat - 1.0)[it])
    # Relative amplitude match (slope of actual vs filtered).
    slope = np.dot(a, b) / np.dot(b, b)

    print("nortmann26 self test 2: model filter (Appendix A.3)")
    print(f"  SYSREM removed {removed * 100:.0f}% of the signal "
          f"(signal preserving regime)")
    print(f"  corr(filtered model, actual SYSREM distortion) = {r:.3f}")
    print(f"  amplitude match (slope) = {slope:.2f}")
    assert r > 0.9, "model filter does not reproduce the SYSREM distortion"
    assert 0.75 < slope < 1.25, "model filter amplitude off"
    print("  PASS")


def _self_test_kpv() -> None:
    """Validate the Kp v map + trail test by injecting a synthetic CCF trail
    at a known (Kp, v) and confirming both recover it there.  This certifies
    the Eq. 3 shift convention empirically (no assumption about frames).
    """
    rng = np.random.default_rng(2)
    n_spectra = 60
    v_grid = np.arange(-150.0, 150.0 + 1.0, 1.0)
    phase = np.linspace(-0.05, 0.05, n_spectra)
    intransit = np.abs(phase) < 0.04

    kp_true, v_true, v_star = 102.5, 0.0, 0.0
    # A CCF that peaks along the planet trail vp*(t) = Kp sin 2 pi phi - v_star,
    # for the in transit exposures, on a noisy background.
    ccf = rng.normal(0.0, 1.0, (v_grid.size, n_spectra))
    amp = 6.0
    for t in np.where(intransit)[0]:
        vp = kp_true * np.sin(2 * np.pi * phase[t]) - v_star + v_true
        ccf[:, t] += amp * np.exp(-0.5 * ((v_grid - vp) / 3.0) ** 2)

    kp_grid = np.arange(0.0, 300.0 + 2.0, 2.0)
    snr, vgo = nortmann_kpv_map(ccf, v_grid, phase, intransit, kp_grid,
                                v_star=v_star)
    ik, iv = np.unravel_index(np.nanargmax(snr), snr.shape)
    kp_rec, v_rec, snr_pk = kp_grid[ik], vgo[iv], snr[ik, iv]

    tr = nortmann_in_out_trail(ccf, v_grid, phase, intransit, kp_true,
                               v_star=v_star, peak_v=v_true)

    print("nortmann26 self test 3: Kp v map + in/out trail (Sec 3.4 / A.4)")
    print(f"  peak at Kp={kp_rec:.0f} (true {kp_true}), "
          f"v={v_rec:+.0f} (true {v_true}), S/N={snr_pk:.1f}")
    print(f"  in/out trail Welch t={tr['t']:.1f} -> {tr['sigma']:.1f} sigma "
          f"(mid_in {tr['mid_in']:.2f} vs mid_out {tr['mid_out']:.2f})")
    assert abs(kp_rec - kp_true) <= 6.0, "Kp not recovered"
    assert abs(v_rec - v_true) <= 2.0, "v_rest not recovered"
    assert snr_pk > 4.0, "map S/N too low"
    assert tr["sigma"] > 4.0, "in/out trail test did not detect the trail"
    print("  PASS")


if __name__ == "__main__":
    _self_test()
    _self_test_filter()
    _self_test_kpv()
