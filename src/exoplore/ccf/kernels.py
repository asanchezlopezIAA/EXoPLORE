"""
exoplore.ccf.kernels
====================

Low-level, Numba-accelerated cross-correlation kernels.

The primary function is :func:`compute_inverse_variance_weighted_ccf`,
which computes a CCF using inverse-variance weighting of the spectral
uncertainties.  This is the main numerical kernel used throughout EXoPLORE.

The implementation uses Numba JIT compilation with parallelisation
(``nb.prange``) for performance.  The scientific interface is intentionally
separated from implementation details: callers do not need to know that
Numba is used internally.

References
----------
Brogi & Line 2019 (BL19), likelihood-based CCF framework
Blain et al. 2024 (Blain24), improved CCF weighting scheme
"""

from __future__ import annotations

import numpy as np

try:
    import numba as nb
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Speed of light in km/s, used by all CCF kernels.
# Module-level float scalars are safely captured by Numba nopython JIT.
_C_KMS = 2.998e5


# ---------------------------------------------------------------------------
# Numba kernel (defined at module level so Numba can JIT-compile it)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _ccf_iv_weighted_kernel(
        lag,
        n_spectra,
        obs,
        ccf_iterations,
        wave,
        wave_CC,
        ccf_values,
        template,
        uncertainties,
        with_signal,
        c_kms,
    ):
        """Inner Numba kernel, do not call directly.  Use
        :func:`compute_inverse_variance_weighted_ccf`.
        """
        # Telluric mask: determine valid pixels from the first in-transit frame
        valid_tellurics = np.where((obs[with_signal[0], :] != 1))[0]
        # Safety: if all pixels look "valid" (no mask set), fall back to
        # pixels where uncertainty is non-zero
        if obs[0, :].shape == valid_tellurics.shape:
            valid_tellurics = np.where((uncertainties[0, :] != 0))[0]

        obs_masked = obs[:, valid_tellurics]
        unc_masked = uncertainties[:, valid_tellurics]

        for m in nb.prange(ccf_iterations):
            for i in range(n_spectra):
                # Doppler-shift the template to the current trial velocity
                syn_shifted = np.interp(
                    wave,
                    wave_CC * (1.0 + lag[m] / c_kms),
                    template[i, :],
                )
                syn_shifted = syn_shifted[valid_tellurics]

                obs_i = obs_masked[i, :]
                unc_i = unc_masked[i, :]

                # Remove interpolation edge effects: pixels where consecutive Doppler-shifted
                # template values are identical (difference = 0) are at the grid boundary or
                # in a masked/telluric region where the interpolation has plateaued. Excluding
                # them prevents spurious weight accumulation at those positions.
                valid_interp = np.where(np.diff(syn_shifted) != 0)[0]
                obs_i = obs_i[valid_interp]
                syn_shifted = syn_shifted[valid_interp]
                unc_i = unc_i[valid_interp]

                # Inverse-variance weights
                weights = 1.0 / (unc_i ** 2 + 1e-300)

                # Mean-subtract (weighted)
                w_sum = np.sum(weights)
                x_mean = np.sum(weights * syn_shifted) / w_sum
                y_mean = np.sum(weights * obs_i) / w_sum

                xd = syn_shifted - x_mean
                yd = obs_i - y_mean

                num = np.sum(weights * yd * xd)
                den = np.sqrt(
                    np.sum(weights * xd**2) * np.sum(weights * yd**2)
                )
                ccf_values[m, i] = num / den if den > 0.0 else 0.0

        return ccf_values

else:
    def _ccf_iv_weighted_kernel(*args, **kwargs):
        raise ImportError(
            "Numba is required for the CCF kernel.  Install it with: "
            "pip install numba"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_inverse_variance_weighted_ccf(
    lag_kms: np.ndarray,
    observed_spectra: np.ndarray,
    template_spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    template_wavelength_grid: np.ndarray,
    uncertainties: np.ndarray,
    in_transit_indices: np.ndarray,
) -> np.ndarray:
    """Compute an inverse-variance weighted cross-correlation function.

    For each trial velocity in ``lag_kms``, the template is Doppler-shifted
    and cross-correlated with the observed spectra.  Each pixel is weighted
    by the inverse of its variance.  Telluric-masked pixels (where the
    observed spectrum equals exactly 1.0) are excluded.

    Parameters
    ----------
    lag_kms:
        1-D array of trial radial velocities in km/s.
        Shape: (n_lags,).
    observed_spectra:
        2-D array of observed (residual) spectra.
        Shape: (n_spectra, n_pixels).
    template_spectra:
        2-D array of the CCF template at each exposure epoch.
        Shape: (n_spectra, n_template_pixels).
    wavelength_grid:
        1-D wavelength array for the observed spectra in microns (or nm).
        Shape: (n_pixels,).
    template_wavelength_grid:
        1-D wavelength array for the template.
        Shape: (n_template_pixels,).
    uncertainties:
        2-D array of per-pixel uncertainties (1σ) for each spectrum.
        Shape: (n_spectra, n_pixels).
    in_transit_indices:
        1-D array of integer indices indicating which spectra are
        in-transit.  Used to determine the telluric mask from a
        representative in-transit frame.
        Shape: (n_in_transit,).

    Returns
    -------
    numpy.ndarray, shape (n_lags, n_spectra)
        CCF values.  Rows correspond to trial velocities; columns to
        individual spectra (time frames).

    Notes
    -----
    Parallelisation is via ``numba.prange`` over the velocity axis.
    For large grids (many orders × many velocities) this is the dominant
    computational cost.

    Examples
    --------
    >>> import numpy as np
    >>> from exoplore.ccf import compute_inverse_variance_weighted_ccf
    >>> n_spec, n_pix = 30, 1000
    >>> obs = np.random.normal(0, 0.01, (n_spec, n_pix)).astype(np.float64)
    >>> tmpl = np.random.normal(0, 1, (n_spec, n_pix)).astype(np.float64)
    >>> wave = np.linspace(1.0, 1.1, n_pix)
    >>> unc = np.full((n_spec, n_pix), 0.01)
    >>> lag = np.linspace(-50, 50, 101)
    >>> intrans = np.arange(10, 20)
    >>> ccf = compute_inverse_variance_weighted_ccf(
    ...     lag, obs, tmpl, wave, wave, unc, intrans
    ... )
    >>> ccf.shape
    (101, 30)
    """
    lag = np.asarray(lag_kms, dtype=np.float64)
    obs = np.asarray(observed_spectra, dtype=np.float64)
    tmpl = np.asarray(template_spectra, dtype=np.float64)
    wave = np.asarray(wavelength_grid, dtype=np.float64)
    wave_cc = np.asarray(template_wavelength_grid, dtype=np.float64)
    unc = np.asarray(uncertainties, dtype=np.float64)
    intrans = np.asarray(in_transit_indices, dtype=np.int64)

    # Guard: the Numba kernel cannot handle NaN rows, they produce silent
    # corruption.  This fires if a caller forgets to slice multi-night arrays.
    nan_rows = ~np.isfinite(obs).all(axis=1)
    if nan_rows.any():
        raise ValueError(
            f"compute_inverse_variance_weighted_ccf received {nan_rows.sum()} "
            "NaN/Inf row(s) in 'observed_spectra'. "
            "When using multi-night padded arrays, slice to [:n_spectra, :] before calling."
        )

    n_spectra = obs.shape[0]
    n_lags = len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)

    return _ccf_iv_weighted_kernel(
        lag,
        n_spectra,
        obs,
        n_lags,
        wave,
        wave_cc,
        ccf_values,
        tmpl,
        unc,
        intrans,
        _C_KMS,
    )


# Weighted CCF
# Has a different interface from compute_inverse_variance_weighted_ccf:
# takes (lag, n_spectra, obs, ccf_iterations, wave, wave_CC,
#         ccf_values, template, uncertainties, with_signal)
try:
    import numba as nb
    from numba import jit

    @jit(nopython=True, parallel=True)
    def ccf_numba_par_weighted(
            lag, n_spectra, obs, ccf_iterations,
            wave, wave_CC, ccf_values, template, uncertainties, with_signal):
        """Inverse-variance weighted CCF with positional parameter order.

        Computes the same inverse-variance weighted normalised cross-correlation
        function as :func:`compute_inverse_variance_weighted_ccf`, but takes its
        arguments in positional rather than keyword order.

        For new code, prefer :func:`compute_inverse_variance_weighted_ccf`
        which has descriptive keyword arguments.

        Parameters
        ----------
        lag : ndarray, shape (ccf_iterations,)
            Trial radial velocities in km/s.
        n_spectra : int
            Number of observed spectra (exposures).
        obs : ndarray, shape (n_spectra, n_pixels)
            Observed (residual) spectra.  Telluric-masked pixels should be
            set to 1.0.
        ccf_iterations : int
            Number of lag steps (= ``len(lag)``).
        wave : ndarray, shape (n_pixels,)
            Wavelength grid of the observed spectra (μm or nm; consistent
            with ``wave_CC``).
        wave_CC : ndarray, shape (n_template_pixels,)
            Wavelength grid of the template.
        ccf_values : ndarray, shape (ccf_iterations, n_spectra)
            Pre-allocated output array; modified in-place and returned.
        template : ndarray, shape (n_spectra, n_template_pixels)
            CCF template spectra (one per exposure epoch).
        uncertainties : ndarray, shape (n_spectra, n_pixels)
            Per-pixel 1σ uncertainties.
        with_signal : ndarray of int
            Indices of in-transit exposures, used to select a representative
            frame for the telluric mask.

        Returns
        -------
        ndarray, shape (ccf_iterations, n_spectra)
            Inverse-variance weighted CCF values.

        Notes
        -----
        This is a Numba JIT-compiled, parallel function.  It is available
        only when Numba is installed; otherwise the stub below raises
        ``ImportError``.
        """
        valid_tellurics = np.where((obs[with_signal[0], :] != 1))[0]
        if obs[0, :].shape == valid_tellurics.shape:
            valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
        obs = obs[:, valid_tellurics]
        uncertainties = uncertainties[:, valid_tellurics]
        for m in nb.prange(ccf_iterations):
            for i in range(n_spectra):
                syn_spec_shifted = np.interp(
                    wave,
                    wave_CC * (1. + lag[m] / (2.998e5)),
                    template[i, :])
                syn_spec_shifted = syn_spec_shifted[valid_tellurics]
                obs_i = obs[i, :]
                uncertainties_i = uncertainties[i, :]
                valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
                if valid_interp.shape != (0,):
                    obs_i = obs_i[valid_interp]
                    uncertainties_i = uncertainties_i[valid_interp]
                    syn_spec_shifted = syn_spec_shifted[valid_interp]
                    weighted_mean1 = np.average(
                        syn_spec_shifted, weights=(1 / uncertainties_i ** 2.))
                    syn_spec_shifted -= weighted_mean1
                    weighted_mean2 = np.average(
                        obs_i, weights=(1 / uncertainties_i ** 2.))
                    obs_i -= weighted_mean2
                    cross = np.sum(obs_i * syn_spec_shifted / uncertainties_i ** 2)
                    norm = np.sqrt(
                        np.sum(syn_spec_shifted ** 2 / uncertainties_i ** 2) *
                        np.sum(obs_i ** 2 / uncertainties_i ** 2))
                    ccf_values[m, i] = cross / norm
        return ccf_values

    @jit(nopython=True, parallel=True)
    def ccf_numba_par_matched_filter(
            lag, n_spectra, obs, ccf_iterations,
            wave, wave_CC, ccf_values, template, uncertainties, with_signal):
        """Inverse-variance-weighted matched-filter CCF (Nortmann+24 Eq. 1).

        Computes the noise-weighted projection of the residual spectra onto the
        Doppler-shifted model, ``CCF(v,t) = sum_j R_j(t) M_j(v) / E_j^2``, with
        no Pearson normalisation (the defining difference from the normalised
        kernel above) and no mean subtraction, following Nortmann et al. (2024,
        A&A 693, A213) Eq. 1.  Named for its form (a matched filter), not its
        author.  Same masking/interpolation as the normalised kernel.
        """
        valid_tellurics = np.where((obs[with_signal[0], :] != 1))[0]
        if obs[0, :].shape == valid_tellurics.shape:
            valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
        obs = obs[:, valid_tellurics]
        uncertainties = uncertainties[:, valid_tellurics]
        for m in nb.prange(ccf_iterations):
            for i in range(n_spectra):
                syn_spec_shifted = np.interp(
                    wave,
                    wave_CC * (1. + lag[m] / (2.998e5)),
                    template[i, :])
                syn_spec_shifted = syn_spec_shifted[valid_tellurics]
                obs_i = obs[i, :]
                uncertainties_i = uncertainties[i, :]
                valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
                if valid_interp.shape != (0,):
                    obs_i = obs_i[valid_interp]
                    uncertainties_i = uncertainties_i[valid_interp]
                    syn_spec_shifted = syn_spec_shifted[valid_interp]
                    # Weighted-mean-subtract both (continuum/DC removal) so the
                    # un-normalised weighted covariance captures the line
                    # correlation rather than the constant continuum term.
                    weighted_mean1 = np.average(
                        syn_spec_shifted, weights=(1 / uncertainties_i ** 2.))
                    syn_spec_shifted -= weighted_mean1
                    weighted_mean2 = np.average(
                        obs_i, weights=(1 / uncertainties_i ** 2.))
                    obs_i -= weighted_mean2
                    ccf_values[m, i] = np.sum(
                        obs_i * syn_spec_shifted / uncertainties_i ** 2)
        return ccf_values

except ImportError:
    def ccf_numba_par_weighted(*args, **kwargs):  # type: ignore[misc]
        """Raises ImportError, Numba is unavailable on this system."""
        raise ImportError(
            "Numba is required for ccf_numba_par_weighted. "
            "Install with: pip install numba")

    def ccf_numba_par_matched_filter(*args, **kwargs):  # type: ignore[misc]
        """Raises ImportError, Numba is unavailable on this system."""
        raise ImportError(
            "Numba is required for ccf_numba_par_matched_filter. "
            "Install with: pip install numba")


# ---------------------------------------------------------------------------
# Additional CCF kernels
# ---------------------------------------------------------------------------


def ccf_numba(lag, n_spectra, obs, ccf_iterations, wave, wave_CC,
              ccf_values, template):
    """Compute the CCF between observed spectra and a template (non-parallel).

    Basic non-Numba CCF kernel.  Iterates over trial velocities and spectra,
    Doppler-shifts the template to each lag, removes edge effects and telluric
    pixels, then computes the normalised cross-correlation coefficient.

    Parameters
    ----------
    lag : array-like, shape (ccf_iterations,)
        Trial radial velocities in km/s.
    n_spectra : int
        Number of observed spectra.
    obs : ndarray, shape (n_spectra, n_pixels)
        Observed (residual) spectra.
    ccf_iterations : int
        Number of lag values.
    wave : ndarray, shape (n_pixels,)
        Wavelength grid of the observed spectra.
    wave_CC : ndarray, shape (n_template_pixels,)
        Wavelength grid of the template.
    ccf_values : ndarray, shape (ccf_iterations, n_spectra)
        Pre-allocated output array (modified in-place and returned).
    template : ndarray, shape (n_spectra, n_template_pixels)
        CCF template spectra.

    Returns
    -------
    ccf_values : ndarray, shape (ccf_iterations, n_spectra)
    """
    for m in range(ccf_iterations):
        for i in range(n_spectra):
            syn_spec_shifted = np.interp(wave,
                                         wave_CC * (1. + lag[m] / _C_KMS),
                                         template[i, :])
            obs_i = obs[i, :]
            diff_arr = np.diff(syn_spec_shifted)
            non_interp_issue = np.where(diff_arr != 0)[0]
            syn_spec_shifted = syn_spec_shifted[non_interp_issue]
            obs_i = obs_i[non_interp_issue]
            non_tell_mask = np.where(obs_i != 1)[0]
            syn_spec_shifted = syn_spec_shifted[non_tell_mask]
            obs_i = obs_i[non_tell_mask]
            xd = syn_spec_shifted - np.mean(syn_spec_shifted)
            yd = obs_i - np.mean(obs_i)
            cross = np.sum(yd * xd)
            ccf_values[m, i] = cross / np.sqrt(np.sum(xd**2) * np.sum(yd**2))
    return ccf_values


if _NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def ccf_numba_par(
            lag, n_spectra, obs, ccf_iterations, wave, wave_CC,
            ccf_values, template, uncertainties
            ):
        """Parallel CCF kernel (unweighted normalised cross-correlation).

        Numba-parallelised version of :func:`ccf_numba`.  Telluric-masked
        pixels (obs == 1) are excluded; interpolation edge effects are
        removed.  No uncertainty weighting, each pixel contributes equally
        to the cross-correlation.

        Parameters
        ----------
        lag : ndarray, shape (ccf_iterations,)
            Trial velocities in km/s.
        n_spectra : int
            Number of spectra.
        obs : ndarray, shape (n_spectra, n_pixels)
            Observed residual spectra.
        ccf_iterations : int
            Number of lag steps.
        wave : ndarray, shape (n_pixels,)
            Observed wavelength grid.
        wave_CC : ndarray, shape (n_template_pixels,)
            Template wavelength grid.
        ccf_values : ndarray, shape (ccf_iterations, n_spectra)
            Pre-allocated output array.
        template : ndarray, shape (n_spectra, n_template_pixels)
            CCF template.
        uncertainties : ndarray, shape (n_spectra, n_pixels)
            Per-pixel uncertainties (used only for the fallback telluric mask).

        Returns
        -------
        ccf_values : ndarray, shape (ccf_iterations, n_spectra)
        """
        valid_tellurics = np.where((obs[0, :] != 1))[0]
        if obs[0, :].shape == valid_tellurics.shape:
            valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
        obs = obs[:, valid_tellurics]
        uncertainties = uncertainties[:, valid_tellurics]
        for m in nb.prange(ccf_iterations):
            for i in range(n_spectra):
                syn_spec_shifted = np.interp(wave,
                                             wave_CC * (1. + lag[m] / _C_KMS),
                                             template[i, :])
                syn_spec_shifted = syn_spec_shifted[valid_tellurics]
                obs_i = obs[i, :]
                valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
                obs_i = obs_i[valid_interp]
                syn_spec_shifted = syn_spec_shifted[valid_interp]
                xd = syn_spec_shifted - np.mean(syn_spec_shifted)
                yd = obs_i - np.mean(obs_i)
                cross = np.sum(yd * xd)
                ccf_values[m, i] = cross / np.sqrt(np.sum(xd**2) * np.sum(yd**2))
        return ccf_values

    @jit(nopython=True, parallel=True)
    def ccf_numba_par_weighted_ordbord_opt(
            sysrem_its, lag, n_spectra, obs, ccf_iterations,
            wave, wave_CC, ccf_values, template, uncertainties
            ):
        """Parallel weighted CCF for the order-by-order SYSREM optimisation.

        Variant of the inverse-variance weighted CCF designed for the
        order-by-order SYSREM iteration search.  The ``obs`` array has shape
        ``(n_spectra, n_pixels, 2, sysrem_its)``, the two extra axes hold
        the two detector halves and the SYSREM iteration index.

        Parameters
        ----------
        sysrem_its : int
            Number of SYSREM iterations (size of last axis of ``obs``).
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
        ccf_values : ndarray, shape (ccf_iterations, n_spectra, 2, sysrem_its)
            Pre-allocated output array.
        template : ndarray, shape (n_spectra, n_template_pixels)
            CCF template.
        uncertainties : ndarray, shape (n_spectra, n_pixels)
            Per-pixel uncertainties.

        Returns
        -------
        ccf_values : ndarray, shape (ccf_iterations, n_spectra, 2, sysrem_its)
        """
        valid_tellurics = np.where((obs[0, :, 0, 0] != 1))[0]
        if obs[0, :, 0, 0].shape == valid_tellurics.shape:
            valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
        obs = obs[:, valid_tellurics, :, :]
        uncertainties = uncertainties[:, valid_tellurics]
        for m in nb.prange(ccf_iterations):
            for i in range(n_spectra):
                syn_spec_shifted = np.interp(wave,
                                             wave_CC * (1. + lag[m] / _C_KMS),
                                             template[i, :])
                syn_spec_shifted = syn_spec_shifted[valid_tellurics]
                uncertainties_i = uncertainties[i, :]
                valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
                if valid_interp.shape != (0,):
                    obs_i = obs[i, valid_interp, :, :]
                    uncertainties_i = uncertainties[i, valid_interp]
                    syn_spec_shifted = syn_spec_shifted[valid_interp]
                    for k in range(2):
                        for l in range(sysrem_its):
                            weighted_mean1 = np.average(syn_spec_shifted,
                                                        weights=(1 / uncertainties_i**2.))
                            syn_spec_shifted -= weighted_mean1
                            weighted_mean2 = np.average(obs_i[:, k, l],
                                                        weights=(1 / uncertainties_i**2.))
                            obs_i[:, k, l] -= weighted_mean2
                            cross = np.sum(obs_i[:, k, l] * syn_spec_shifted / uncertainties_i**2)
                            norm = np.sqrt(np.sum(syn_spec_shifted**2 / uncertainties_i**2) *
                                           np.sum(obs_i[:, k, l]**2 / uncertainties_i**2))
                            ccf_values[m, i, k, l] = cross / norm
        return ccf_values

    @jit(nopython=True, parallel=True)
    def ccf_literature(
            lag, n_spectra, obs, ccf_iterations,
            wave, wave_CC, ccf_values, template, uncertainties, with_signal
            ):
        """Parallel CCF matching the unweighted literature definition.

        Same structure as :func:`ccf_numba_par` but uses simple (unweighted)
        mean subtraction, matching the standard CCF definition in the
        high-resolution spectroscopy literature.

        Parameters
        ----------
        lag : ndarray, shape (ccf_iterations,)
            Trial velocities in km/s.
        n_spectra : int
            Number of spectra.
        obs : ndarray, shape (n_spectra, n_pixels)
            Observed residual spectra.
        ccf_iterations : int
            Number of lag steps.
        wave : ndarray, shape (n_pixels,)
            Observed wavelength grid.
        wave_CC : ndarray, shape (n_template_pixels,)
            Template wavelength grid.
        ccf_values : ndarray, shape (ccf_iterations, n_spectra)
            Pre-allocated output array.
        template : ndarray, shape (n_spectra, n_template_pixels)
            CCF template.
        uncertainties : ndarray, shape (n_spectra, n_pixels)
            Per-pixel uncertainties (used only for the fallback telluric mask).
        with_signal : ndarray of int
            Indices of in-signal (in-transit) frames for telluric mask selection.

        Returns
        -------
        ccf_values : ndarray, shape (ccf_iterations, n_spectra)
        """
        valid_tellurics = np.where((obs[with_signal[0], :] != 1))[0]
        if obs[0, :].shape == valid_tellurics.shape:
            valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
        obs = obs[:, valid_tellurics]
        uncertainties = uncertainties[:, valid_tellurics]
        for m in nb.prange(ccf_iterations):
            for i in range(n_spectra):
                syn_spec_shifted = np.interp(wave,
                                             wave_CC * (1. + lag[m] / _C_KMS),
                                             template[i, :])
                syn_spec_shifted = syn_spec_shifted[valid_tellurics]
                obs_i = obs[i, :]
                uncertainties_i = uncertainties[i, :]
                valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
                if valid_interp.shape != (0,):
                    obs_i = obs_i[valid_interp]
                    uncertainties_i = uncertainties_i[valid_interp]
                    syn_spec_shifted = syn_spec_shifted[valid_interp]
                    nx = float(len(syn_spec_shifted))
                    xd = syn_spec_shifted - np.sum(syn_spec_shifted) / nx
                    yd = obs_i - np.sum(obs_i) / nx
                    cross = np.sum(yd * xd)
                    ccf_values[m, i] = cross / np.sqrt(np.sum(xd**2) * np.sum(yd**2))
        return ccf_values

else:
    def ccf_numba_par(*args, **kwargs):  # type: ignore[misc]
        """Raises ImportError, Numba is unavailable on this system."""
        raise ImportError("Numba is required for ccf_numba_par. Install with: pip install numba")

    def ccf_numba_par_weighted_ordbord_opt(*args, **kwargs):  # type: ignore[misc]
        """Raises ImportError, Numba is unavailable on this system."""
        raise ImportError("Numba is required for ccf_numba_par_weighted_ordbord_opt. Install with: pip install numba")

    def ccf_literature(*args, **kwargs):  # type: ignore[misc]
        """Raises ImportError, Numba is unavailable on this system."""
        raise ImportError("Numba is required for ccf_literature. Install with: pip install numba")
