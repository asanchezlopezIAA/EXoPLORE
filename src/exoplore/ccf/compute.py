"""
exoplore.ccf.compute
====================

High-level CCF wrapper.

This module sits between the low-level Numba kernels in
:mod:`exoplore.ccf.kernels` and the user-facing simulation pipeline.
It handles:

- Building the velocity lag grid from the config.
- Looping over spectral orders and stacking CCF results.
- Computing the Kp-Vsys map from per-frame CCFs.

Scientific parameters come from a
:class:`~exoplore.config.CrossCorrelationConfig` object, not from
hard-coded constants.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from exoplore.ccf.kernels import compute_inverse_variance_weighted_ccf


def build_velocity_grid(
    velocity_max_kms: float,
    velocity_step_kms: float,
) -> np.ndarray:
    """Build a symmetric velocity grid centred on zero.

    Parameters
    ----------
    velocity_max_kms:
        Half-width of the velocity range in km/s.
    velocity_step_kms:
        Step size in km/s.

    Returns
    -------
    numpy.ndarray
        Velocity array from ``-velocity_max`` to ``+velocity_max``.
    """
    return np.arange(-velocity_max_kms, velocity_max_kms + velocity_step_kms, velocity_step_kms)


def compute_ccf_timeseries(
    observed_spectra: np.ndarray,
    template_spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    template_wavelength_grid: np.ndarray,
    uncertainties: np.ndarray,
    in_transit_indices: np.ndarray,
    velocity_max_kms: float = 325.0,
    velocity_step_kms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the CCF time series for one spectral order.

    Parameters
    ----------
    observed_spectra:
        Residual spectra after pipeline cleaning. Shape (n_spectra, n_pixels).
    template_spectra:
        CCF template evaluated at each epoch. Shape (n_spectra, n_pixels).
    wavelength_grid:
        Wavelength axis for the observed spectra in microns.
    template_wavelength_grid:
        Wavelength axis for the template in microns.
    uncertainties:
        Per-pixel 1σ uncertainties. Shape (n_spectra, n_pixels).
    in_transit_indices:
        Indices of in-transit exposures.
    velocity_max_kms:
        Half-width of CCF velocity range in km/s.
    velocity_step_kms:
        CCF velocity step in km/s.

    Returns
    -------
    velocity_grid : numpy.ndarray
        Velocity axis, shape (n_lags,).
    ccf : numpy.ndarray
        CCF values, shape (n_lags, n_spectra).
    """
    velocity_grid = build_velocity_grid(velocity_max_kms, velocity_step_kms)

    ccf = compute_inverse_variance_weighted_ccf(
        lag_kms=velocity_grid,
        observed_spectra=observed_spectra,
        template_spectra=template_spectra,
        wavelength_grid=wavelength_grid,
        template_wavelength_grid=template_wavelength_grid,
        uncertainties=uncertainties,
        in_transit_indices=in_transit_indices,
    )

    return velocity_grid, ccf


def compute_kp_vsys_map(
    ccf_timeseries: np.ndarray,
    velocity_grid: np.ndarray,
    orbital_phase: np.ndarray,
    kp_grid: np.ndarray,
    systemic_velocity_kms: float = 0.0,
) -> np.ndarray:
    """Collapse a CCF time series into a Kp-Vsys detection map.

    For each trial Kp, the planetary signal is shifted to the planet rest
    frame and co-added over all in-transit frames.

    Parameters
    ----------
    ccf_timeseries:
        CCF values, shape (n_lags, n_spectra).
    velocity_grid:
        Velocity axis in km/s, shape (n_lags,).
    orbital_phase:
        Orbital phase of each spectrum (0 = mid-transit), shape (n_spectra,).
    kp_grid:
        Trial Kp values in km/s, shape (n_kp,).
    systemic_velocity_kms:
        Systemic velocity of the system in km/s.

    Returns
    -------
    numpy.ndarray, shape (n_kp, n_lags)
        Kp-Vsys map; rows index Kp, columns index Vsys.
    """
    n_lags = len(velocity_grid)
    n_kp = len(kp_grid)
    n_spectra = ccf_timeseries.shape[1]

    kp_map = np.zeros((n_kp, n_lags))

    for i, kp in enumerate(kp_grid):
        for j in range(n_spectra):
            # Planet velocity at this orbital phase
            v_planet = kp * np.sin(2.0 * np.pi * orbital_phase[j]) + systemic_velocity_kms
            # Shift the CCF to the planet rest frame
            ccf_shifted = np.interp(
                velocity_grid - v_planet,
                velocity_grid,
                ccf_timeseries[:, j],
                left=0.0,
                right=0.0,
            )
            kp_map[i] += ccf_shifted

    return kp_map


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def call_ccf_numba(lag, n_spectra, obs, ccf_iterations, wave, wave_CC, template):
    """CCF using the basic (non-parallel) Numba kernel.

    Parameters
    ----------
    lag : ndarray
        Trial velocity lags in km/s.
    n_spectra : int
        Number of spectra (time frames).
    obs : ndarray, shape (n_spectra, n_pixels)
        Observed residual spectra.
    ccf_iterations : int
        Number of CCF lag positions (== len(lag)).
    wave : ndarray
        Wavelength grid of the observed spectra.
    wave_CC : ndarray
        Wavelength grid of the CCF template.
    template : ndarray, shape (n_spectra, n_template_pixels)
        CCF template evaluated at each epoch.

    Returns
    -------
    ndarray, shape (ccf_iterations, n_spectra)
        CCF values.
    """
    from exoplore.ccf.kernels import ccf_numba as _ccf_numba
    ccf_values = np.zeros((ccf_iterations, n_spectra))
    return _ccf_numba(lag, n_spectra, obs, ccf_iterations, wave, wave_CC,
                      ccf_values, template)


def call_ccf_numba_par(lag, n_spectra, obs, ccf_iterations, wave,
                       wave_CC, template, uncertainties):
    """CCF using the parallel (non-weighted) Numba kernel.

    Parameters
    ----------
    lag : ndarray
        Trial velocity lags in km/s.
    n_spectra : int
        Number of spectra.
    obs : ndarray, shape (n_spectra, n_pixels)
        Observed residual spectra.
    ccf_iterations : int
        Number of CCF lag positions.
    wave : ndarray
        Wavelength grid of the observed spectra.
    wave_CC : ndarray
        Wavelength grid of the template.
    template : ndarray
        CCF template.
    uncertainties : ndarray, shape (n_spectra, n_pixels)
        Per-pixel uncertainties (used for telluric masking).

    Returns
    -------
    ndarray, shape (ccf_iterations, n_spectra)
        CCF values.
    """
    from exoplore.ccf.kernels import ccf_numba_par as _ccf_numba_par
    ccf_values = np.zeros((ccf_iterations, n_spectra))
    return _ccf_numba_par(lag, n_spectra, obs, ccf_iterations, wave,
                          wave_CC, ccf_values, template, uncertainties)


def call_ccf_numba_par_weighted(lag, n_spectra, obs, ccf_iterations, wave,
                                wave_CC, template, uncertainties, with_signal):
    """CCF using the inverse-variance-weighted parallel Numba kernel.

    Parameters
    ----------
    lag : ndarray
        Trial velocity lags in km/s.
    n_spectra : int
        Number of spectra.
    obs : ndarray, shape (n_spectra, n_pixels)
        Observed residual spectra.
    ccf_iterations : int
        Number of CCF lag positions.
    wave : ndarray
        Wavelength grid of the observed spectra.
    wave_CC : ndarray
        Wavelength grid of the template.
    template : ndarray
        CCF template.
    uncertainties : ndarray, shape (n_spectra, n_pixels)
        Per-pixel 1σ uncertainties for inverse-variance weighting.
    with_signal : ndarray
        Indices of in-transit (signal) frames used to set the telluric mask.

    Returns
    -------
    ndarray, shape (ccf_iterations, n_spectra)
        CCF values.
    """
    from exoplore.ccf.kernels import ccf_numba_par_weighted as _ccf_weighted
    ccf_values = np.zeros((ccf_iterations, n_spectra))
    return _ccf_weighted(lag, n_spectra, obs, ccf_iterations, wave,
                         wave_CC, ccf_values, template, uncertainties,
                         with_signal)


def call_ccf_numba_par_matched_filter(lag, n_spectra, obs, ccf_iterations, wave,
                                      wave_CC, template, uncertainties,
                                      with_signal):
    """Matched-filter CCF (Nortmann+24 Eq. 1): Σ R·M/E², no normalisation.

    Same signature as :func:`call_ccf_numba_par_weighted` but uses the
    un-normalised inverse-variance-weighted matched-filter kernel.
    """
    from exoplore.ccf.kernels import (
        ccf_numba_par_matched_filter as _ccf_mf)
    ccf_values = np.zeros((ccf_iterations, n_spectra))
    return _ccf_mf(lag, n_spectra, obs, ccf_iterations, wave,
                   wave_CC, ccf_values, template, uncertainties,
                   with_signal)


def call_ccf_literature(lag, n_spectra, obs, ccf_iterations, wave,
                        wave_CC, template, uncertainties, with_signal):
    """CCF using the literature-convention weighted Numba kernel.

    Parameters
    ----------
    lag : ndarray
        Trial velocity lags in km/s.
    n_spectra : int
        Number of spectra.
    obs : ndarray, shape (n_spectra, n_pixels)
        Observed residual spectra.
    ccf_iterations : int
        Number of CCF lag positions.
    wave : ndarray
        Wavelength grid of the observed spectra.
    wave_CC : ndarray
        Wavelength grid of the template.
    template : ndarray
        CCF template.
    uncertainties : ndarray, shape (n_spectra, n_pixels)
        Per-pixel 1σ uncertainties.
    with_signal : ndarray
        Indices of in-transit frames for telluric masking.

    Returns
    -------
    ndarray, shape (ccf_iterations, n_spectra)
        CCF values.
    """
    from exoplore.ccf.kernels import ccf_literature as _ccf_lit
    ccf_values = np.zeros((ccf_iterations, n_spectra))
    return _ccf_lit(lag, n_spectra, obs, ccf_iterations, wave,
                    wave_CC, ccf_values, template, uncertainties, with_signal)


def get_shifted_ccf_matrix(inp_dat, with_signal, v_rest, v_erf, kp_range,
                           phase, v_sys, berv, pixels_left_right,
                           ccf_v_step, ccf_complete, sysrem_opt=False):
    """Shift the CCF time series to the planet rest frame.

    For each in-transit exposure and each trial Kp, the CCF is
    interpolated onto a velocity grid centred on the expected planet
    velocity, producing a 3-D (or 5-D with SYSREM) array.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used:
        ``"significant_eccentricity"``, ``"eccentricity"``,
        ``"arg_periastron_w"``, ``"sysrem_its"`` (if sysrem_opt).
    with_signal : array
        Indices of in-transit exposures.
    v_rest : array
        Rest-frame velocity grid, shape (n_v,).
    v_erf : array
        Earth-frame CCF velocity grid, shape (n_v,).
    kp_range : array
        Trial Kp values in km/s, shape (n_kp,).
    phase : array
        Orbital phases for all exposures.
    v_sys : float
        Systemic velocity (km/s).
    berv : float or array
        BERV correction (km/s).
    pixels_left_right : int
        Half-width in pixels of the extraction window.
    ccf_v_step : float
        Velocity step size of the CCF (km/s).
    ccf_complete : ndarray
        Full CCF matrix.  Shape (n_v, n_exp) normally, or
        (n_v, n_exp, 2, sysrem_its) if ``sysrem_opt=True``.
    sysrem_opt : bool
        If True, the output has extra SYSREM dimensions.

    Returns
    -------
    ndarray
        Shifted CCF matrix.  Shape (n_v, n_transit, n_kp) normally, or
        (n_v, n_transit, n_kp, 2, sysrem_its) if ``sysrem_opt=True``.
    """
    from exoplore.observation.velocity import get_V, get_V_eccentric

    if not sysrem_opt:
        shape = (len(v_rest), len(with_signal), len(kp_range))
    else:
        shape = (len(v_rest), len(with_signal), len(kp_range),
                 2, inp_dat["sysrem_its"])

    ccf_values_shift = np.zeros(shape, float)

    if not inp_dat["significant_eccentricity"]:
        vp_all = get_V(kp_range[:, np.newaxis], phase, berv, v_sys, 0)
    else:
        vp_all = get_V_eccentric(
            kp_range[:, np.newaxis], phase,
            inp_dat["eccentricity"], inp_dat["arg_periastron_w"],
            berv, v_sys, 0
        )

    for idx, i in enumerate(with_signal):
        for k_idx, kp in enumerate(kp_range):
            v_prf = np.linspace(
                vp_all[k_idx, i] - pixels_left_right * ccf_v_step,
                vp_all[k_idx, i] + pixels_left_right * ccf_v_step,
                num=2 * pixels_left_right + 1
            )
            if not sysrem_opt:
                ccf_values_shift[:, idx, k_idx] = np.interp(
                    v_prf, v_erf, ccf_complete[:, i]
                )
            else:
                for n in range(2):
                    for l in range(inp_dat["sysrem_its"]):
                        ccf_values_shift[:, idx, k_idx, n, l] = np.interp(
                            v_prf, v_erf, ccf_complete[:, i, n, l]
                        )

    return ccf_values_shift


def get_max_CCF_peak(inp_dat, ccf_tot, v_rest, kp_range,
                     b=None, stats=None, sysrem_opt=False, CCF_Noise=False):
    """Find the maximum CCF peak and compute its S/N significance.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Key used: ``"CCF_SNR_exclude"``
        (km/s exclusion window around the peak), ``"sysrem_its"``.
    ccf_tot : ndarray
        Kp-Vsys CCF map.  Shape (n_v, n_kp) normally, or
        (n_v, n_kp, 2, sysrem_its) if ``sysrem_opt=True``.
    v_rest : array
        Rest-frame velocity grid (km/s).
    kp_range : array
        Trial Kp values (km/s).
    b : int or None
        Bootstrap index (used only when ``CCF_Noise=True``).
    stats : array or None
        Bootstrap statistics array (used only when ``CCF_Noise=True``).
    sysrem_opt : bool
        If True, treat extra SYSREM dimensions in ``ccf_tot``.
    CCF_Noise : bool
        If True, report S/N at the position given by ``stats[b]``
        rather than at the true maximum.

    Returns
    -------
    ccf_tot_sig : ndarray
        CCF normalised to significance (same shape as ``ccf_tot``).
    max_sig : float or ndarray
        Maximum significance value.
    max_kp_idx : int or ndarray
        Index of the Kp row with the maximum.
    max_v_rest : float or ndarray
        Rest-frame velocity at the maximum.
    cc_values_std : ndarray
        Standard deviation of the noise region (same shape as ``ccf_tot``).
    """
    if not sysrem_opt:
        max_sig = 0
        max_kp_idx = 0
        max_v_rest = 0
        ccf_tot_sig = np.zeros(ccf_tot.shape)
        cc_values_std = np.zeros(ccf_tot.shape)

        # Noise-exclusion convention.  "peak" (default): per Kp row,
        # exclude +-CCF_SNR_exclude km/s around the row's maximum.  "point"
        # (Cheverall+26 Sec 2.4): for each evaluated velocity, the noise is the
        # std of CCF values >CCF_SNR_exclude km/s from THAT velocity (the trail
        # under test).  Computed per row with a sliding-window via cumulative
        # sums (assumes a uniform v_rest grid).
        _excl_mode = inp_dat.get('ccf_snr_exclude_around', 'peak')
        _excl = inp_dat['CCF_SNR_exclude']

        # Noise-source convention.  "peak_row" (default): the noise std is
        # measured within each Kp row (below).  "signal_free_rows": one
        # global std from the Kp rows more than CCF_SNR_kp_exclude away
        # from the detected peak's Kp and beyond CCF_SNR_exclude of the
        # peak velocity.  For strong detections the peak row's own noise
        # region contains the signal's correlation wings, which repeat
        # identically in every night and saturate the reported S/N; the
        # signal-free rows share the map's noise statistics without that
        # structure, so co-added significances recover the expected
        # sqrt(N) scaling when the per-night noise is uncorrelated.
        if inp_dat.get('CCF_SNR_noise_source', 'peak_row') == 'signal_free_rows':
            _ipk = np.unravel_index(np.argmax(ccf_tot), ccf_tot.shape)
            _kp_excl = float(inp_dat.get('CCF_SNR_kp_exclude', 120.0))
            _rows = np.abs(kp_range - kp_range[_ipk[1]]) > _kp_excl
            _vsel = np.abs(v_rest - v_rest[_ipk[0]]) > _excl
            _reg = ccf_tot[np.ix_(_vsel, _rows)]
            _mu, _sd = float(np.nanmean(_reg)), float(np.nanstd(_reg))
            if _sd > 0:
                ccf_tot_sig = (ccf_tot - _mu) / _sd
                cc_values_std = np.full(ccf_tot.shape, _sd)
                if not CCF_Noise:
                    max_sig = float(np.nanmax(ccf_tot_sig))
                    max_kp_idx = int(_ipk[1])
                    max_v_rest = v_rest[_ipk[0]]
                else:
                    max_kp_idx = int(stats[b, 1] + len(kp_range) // 2)
                    max_v_rest = stats[b, 2]
                    max_sig = ccf_tot_sig[
                        np.argwhere(v_rest == stats[b, 2])[0][0], max_kp_idx
                    ]
                return (ccf_tot_sig, max_sig, max_kp_idx, max_v_rest,
                        cc_values_std)
        if _excl_mode == 'point':
            _dv = np.median(np.diff(v_rest))
            _W = int(round(_excl / _dv)) if _dv > 0 else 0
            _n = len(v_rest)
            _idx = np.arange(_n)
            _lo = np.maximum(0, _idx - _W)
            _hi = np.minimum(_n, _idx + _W + 1)

        for k in range(len(kp_range)):
            if _excl_mode == 'point':
                _col = ccf_tot[:, k]
                _cs = np.concatenate(([0.0], np.cumsum(_col)))
                _css = np.concatenate(([0.0], np.cumsum(_col ** 2)))
                _win_n = _hi - _lo
                _nn = _n - _win_n
                _ns = _cs[-1] - (_cs[_hi] - _cs[_lo])
                _nss = _css[-1] - (_css[_hi] - _css[_lo])
                _nmean = _ns / _nn
                _nstd = np.sqrt(np.maximum(_nss / _nn - _nmean ** 2, 0.0))
                ccf_tot_sig[:, k] = np.where(
                    _nstd > 0, (_col - _nmean) / _nstd, 0.0)
                cc_values_std[:, k] = _nstd
                if not CCF_Noise:
                    max_ccf_sn = np.max(ccf_tot_sig[:, k])
                    if max_ccf_sn > max_sig:
                        max_sig = max_ccf_sn
                        max_kp_idx = int(k)
                        max_v_rest = v_rest[ccf_tot_sig[:, max_kp_idx] == max_sig]
                continue

            max_index = np.argmax(ccf_tot[:, k])
            std_pts = (
                (v_rest < (v_rest[max_index] - inp_dat['CCF_SNR_exclude'])) |
                (v_rest > (v_rest[max_index] + inp_dat['CCF_SNR_exclude']))
            )
            ccf_tot_sig[:, k] = (
                (ccf_tot[:, k] - np.mean(ccf_tot[std_pts, k]))
                / np.std(ccf_tot[std_pts, k])
            )
            cc_values_std[:, k] = np.std(ccf_tot[std_pts, k])

            if not CCF_Noise:
                max_ccf_sn = np.max(ccf_tot_sig[:, k])
                if max_ccf_sn > max_sig:
                    max_sig = max_ccf_sn
                    max_kp_idx = int(k)
                    max_v_rest = v_rest[ccf_tot_sig[:, max_kp_idx] == max_sig]

        if CCF_Noise:
            max_kp_idx = int(stats[b, 1] + len(kp_range) // 2)
            max_v_rest = stats[b, 2]
            max_sig = ccf_tot_sig[
                np.argwhere(v_rest == stats[b, 2])[0][0], max_kp_idx
            ]
    else:
        max_sig = np.zeros((2, inp_dat["sysrem_its"]))
        max_kp_idx = np.zeros((2, inp_dat["sysrem_its"]))
        max_v_rest = np.zeros((2, inp_dat["sysrem_its"]))
        ccf_tot_sig = np.zeros(ccf_tot.shape)
        cc_values_std = np.zeros(ccf_tot.shape)

        for k in range(len(kp_range)):
            for n in range(2):
                for l in range(inp_dat["sysrem_its"]):
                    max_index = np.argmax(ccf_tot[:, k, n, l])
                    std_pts = (
                        (v_rest < (v_rest[max_index] - inp_dat['CCF_SNR_exclude'])) |
                        (v_rest > (v_rest[max_index] + inp_dat['CCF_SNR_exclude']))
                    )
                    ccf_tot_sig[:, k, n, l] = (
                        (ccf_tot[:, k, n, l] - np.mean(ccf_tot[std_pts, k, n, l]))
                        / np.std(ccf_tot[std_pts, k, n, l])
                    )
                    cc_values_std[:, k, n, l] = np.std(ccf_tot[std_pts, k, n, l])
                    max_ccf_sn = np.max(ccf_tot_sig[:, k, n, l])
                    if not CCF_Noise:
                        if max_ccf_sn > max_sig[n, l]:
                            max_sig[n, l] = max_ccf_sn
                            max_kp_idx[n, l] = int(k)
                            max_v_rest[n, l] = v_rest[
                                ccf_tot_sig[:, int(max_kp_idx[n, l]), n, l] == max_sig[n, l]
                            ]

        if CCF_Noise:
            max_kp_idx = int(stats[b, 1] + len(kp_range) // 2) - 1
            max_v_rest = stats[b, 2]
            max_sig = ccf_tot_sig[
                np.argwhere(v_rest == stats[b, 2])[0][0], max_kp_idx
            ]

    return ccf_tot_sig, max_sig, max_kp_idx, max_v_rest, cc_values_std


def call_ccf_numba_par_weighted_ordbord_opt(
        sysrem_its, lag, n_spectra, obs, ccf_iterations, wave,
        wave_CC, template, uncertainties
        ):
    """Allocate and call the order-by-order weighted CCF kernel.

    Wrapper that pre-allocates the 4-D ``ccf_values`` array and delegates
    to :func:`~exoplore.ccf.kernels.ccf_numba_par_weighted_ordbord_opt`.

    Parameters
    ----------
    sysrem_its : int
        Number of SYSREM iterations (size of the last two axes of ``obs``).
    lag : ndarray, shape (ccf_iterations,)
        Trial velocities in km/s.
    n_spectra : int
        Number of spectra.
    obs : ndarray, shape (n_spectra, n_pixels, 2, sysrem_its)
        Observed residual spectra cube.
    ccf_iterations : int
        Number of lag steps.
    wave : ndarray, shape (n_pixels,)
        Observed wavelength grid.
    wave_CC : ndarray, shape (n_template_pixels,)
        Template wavelength grid.
    template : ndarray, shape (n_spectra, n_template_pixels)
        CCF template spectra.
    uncertainties : ndarray, shape (n_spectra, n_pixels)
        Per-pixel uncertainties.

    Returns
    -------
    ccf_values : ndarray, shape (ccf_iterations, n_spectra, 2, sysrem_its)
    """
    from exoplore.ccf.kernels import ccf_numba_par_weighted_ordbord_opt
    ccf_values = np.zeros((ccf_iterations, n_spectra, 2, sysrem_its))
    ccf_values = ccf_numba_par_weighted_ordbord_opt(
        sysrem_its,
        lag, n_spectra, obs, ccf_iterations, wave,
        wave_CC, ccf_values, template, uncertainties
    )
    return ccf_values


def quick_CCF(
        inp_dat, ccf_iterations, n_spectra, data,
        propag_noise, model, wave, v_ccf, night_max, night_min,
        with_signal,
        min_max=False, verbose=False):
    """Run the per-order CCF loop and return the order-stacked CCF cube.

    Convenience wrapper that iterates over orders and nights, calling
    :func:`call_ccf_numba_par_weighted` for each slice, then subtracts
    the per-row median.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``'n_orders'``,
        ``'n_nights'``, and ``'order_selection'``.
    ccf_iterations : int
        Number of CCF velocity steps.
    n_spectra : int
        Number of spectra per night.
    data : ndarray, shape (n_orders, n_nights, n_spectra, n_pixels)
        Residual spectral cube.
    propag_noise : ndarray, same shape as *data*
        Per-pixel propagated noise.
    model : ndarray, shape (n_orders, n_spectra, n_pixels)
        CCF template cube (one template per order per spectrum).
    wave : ndarray, shape (n_orders, n_pixels)
        Wavelength grid per order.
    v_ccf : ndarray, shape (ccf_iterations,)
        Trial velocity grid in km/s.
    night_max, night_min : int
        Indices of the best and worst S/N nights (used only when
        ``min_max=True``).
    with_signal : ndarray of int
        In-transit frame indices.
    min_max : bool, optional
        If True run only three nights: co-added, best, and worst.
    verbose : bool, optional
        If True print progress per order.

    Returns
    -------
    ccf_store : ndarray, shape (n_orders, n_nights, ccf_iterations, n_spectra)
    """
    if min_max:
        n_nights = 3
        ccf_store = np.zeros((inp_dat['n_orders'], n_nights,
                              ccf_iterations, n_spectra), float)
        for h in range(inp_dat['n_orders']):
            for b in range(n_nights):
                if b == 0:
                    ccf_store[h, b, :] = call_ccf_numba_par_weighted(
                        lag=v_ccf, n_spectra=n_spectra, obs=data[h, b, :],
                        ccf_iterations=ccf_iterations, wave=wave,
                        wave_CC=wave, template=model[h, :],
                        uncertainties=propag_noise[h, b, :])
                elif b == 1:
                    ccf_store[h, b, :] = call_ccf_numba_par_weighted(
                        lag=v_ccf, n_spectra=n_spectra, obs=data[h, night_max, :],
                        ccf_iterations=ccf_iterations, wave=wave,
                        wave_CC=wave, template=model[h, :, :],
                        uncertainties=propag_noise[h, night_max, :])
                elif b == 2:
                    ccf_store[h, b, :] = call_ccf_numba_par_weighted(
                        lag=v_ccf, n_spectra=n_spectra, obs=data[h, night_min, :],
                        ccf_iterations=ccf_iterations, wave=wave,
                        wave_CC=wave, template=model[h, :],
                        uncertainties=propag_noise[h, night_min, :])
    else:
        ccf_store = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'],
             ccf_iterations, n_spectra), float
        )
        for h in range(inp_dat['n_orders']):
            if verbose:
                print(f"Order {h}")
            for b in range(inp_dat['n_nights']):
                ccf_store[h, b, :] = call_ccf_numba_par_weighted(
                    lag=v_ccf, n_spectra=n_spectra, obs=data[h, b, :],
                    ccf_iterations=ccf_iterations,
                    wave=wave[inp_dat['order_selection'][h], :],
                    wave_CC=wave[inp_dat['order_selection'][h], :],
                    template=model[h, :],
                    uncertainties=propag_noise[h, b, :],
                    with_signal=with_signal)
    ccf_store[h, b, :] -= np.median(ccf_store[h, b, :], axis=0)
    return ccf_store
