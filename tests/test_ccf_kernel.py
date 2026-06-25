"""Tests for the CCF kernels, clean API and positional-argument variants."""

import numpy as np
import pytest

from exoplore.ccf.kernels import (
    compute_inverse_variance_weighted_ccf,
    ccf_numba_par_weighted,
    ccf_numba,
    ccf_numba_par,
    ccf_numba_par_weighted_ordbord_opt,
    ccf_literature,
)


def test_compute_inverse_variance_weighted_ccf_import_and_run():
    """Smoke test: kernel runs and returns the right shape."""
    n_spectra = 2
    n_wave = 100

    wave = np.linspace(1.0, 2.0, n_wave)
    line = np.exp(-0.5 * ((wave - 1.5) / 0.03) ** 2)

    obs = np.vstack([line, line]).astype(np.float64)
    template = np.vstack([line, line]).astype(np.float64)
    uncertainties = np.ones((n_spectra, n_wave), dtype=np.float64)
    lag = np.array([-10.0, 0.0, 10.0])
    in_transit = np.array([0], dtype=np.int64)

    result = compute_inverse_variance_weighted_ccf(
        lag_kms=lag,
        observed_spectra=obs,
        template_spectra=template,
        wavelength_grid=wave,
        template_wavelength_grid=wave,
        uncertainties=uncertainties,
        in_transit_indices=in_transit,
    )
    assert result.shape == (len(lag), n_spectra)


def test_positional_ccf_name_available():
    """ccf_numba_par_weighted must still be importable (separate from clean API).

    These are two distinct functions with different signatures:
    - compute_inverse_variance_weighted_ccf: clean 7-arg API
    - ccf_numba_par_weighted: 10-arg positional API
    Both must be importable; they are intentionally not the same object.
    """
    assert callable(ccf_numba_par_weighted)
    assert callable(compute_inverse_variance_weighted_ccf)
    assert ccf_numba_par_weighted is not compute_inverse_variance_weighted_ccf


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_ccf_inputs(n_spectra=4, n_wave=120, n_lags=11):
    """Return (lag, obs, template, wave, uncertainties, in_transit)."""
    rng = np.random.default_rng(42)
    wave = np.linspace(1.0, 1.1, n_wave)
    line = np.exp(-0.5 * ((wave - 1.05) / 0.005) ** 2)
    obs = np.tile(line, (n_spectra, 1)).astype(np.float64)
    obs += rng.normal(0, 0.01, obs.shape)
    template = np.tile(line, (n_spectra, 1)).astype(np.float64)
    uncertainties = np.full_like(obs, 0.01)
    lag = np.linspace(-50.0, 50.0, n_lags)
    in_transit = np.array([0, 1], dtype=np.int64)
    return lag, obs, template, wave, uncertainties, in_transit


# ---------------------------------------------------------------------------
# ccf_numba (plain Python, no Numba)
# ---------------------------------------------------------------------------

def test_ccf_numba_shape():
    lag, obs, tmpl, wave, _, _ = _make_ccf_inputs()
    n_spectra = obs.shape[0]
    n_lags = len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)
    result = ccf_numba(lag, n_spectra, obs, n_lags, wave, wave, ccf_values, tmpl)
    assert result.shape == (n_lags, n_spectra)


def test_ccf_numba_peak_at_zero_lag():
    """When template == obs the CCF should peak at zero velocity."""
    lag, obs, tmpl, wave, _, _ = _make_ccf_inputs()
    n_spectra, n_lags = obs.shape[0], len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)
    # Use obs as template so peak is at 0 km/s
    result = ccf_numba(lag, n_spectra, obs, n_lags, wave, wave, ccf_values, obs.copy())
    zero_idx = np.argmin(np.abs(lag))
    for i in range(n_spectra):
        assert result[zero_idx, i] == result[:, i].max()


# ---------------------------------------------------------------------------
# ccf_numba_par (Numba parallel, unweighted)
# ---------------------------------------------------------------------------

def test_ccf_numba_par_shape():
    lag, obs, tmpl, wave, unc, _ = _make_ccf_inputs()
    n_spectra, n_lags = obs.shape[0], len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)
    result = ccf_numba_par(lag, n_spectra, obs, n_lags, wave, wave,
                           ccf_values, tmpl, unc)
    assert result.shape == (n_lags, n_spectra)


def test_ccf_numba_par_values_in_range():
    """Normalised CCF values must be in [-1, 1]."""
    lag, obs, tmpl, wave, unc, _ = _make_ccf_inputs()
    n_spectra, n_lags = obs.shape[0], len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)
    result = ccf_numba_par(lag, n_spectra, obs, n_lags, wave, wave,
                           ccf_values, tmpl, unc)
    assert np.all(result >= -1.0 - 1e-10)
    assert np.all(result <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# ccf_numba_par_weighted_ordbord_opt (4-D obs array)
# ---------------------------------------------------------------------------

def test_ccf_numba_par_weighted_ordbord_opt_shape():
    rng = np.random.default_rng(0)
    n_spectra, n_wave, n_lags, sysrem_its = 3, 80, 7, 2
    wave = np.linspace(1.0, 1.1, n_wave)
    # obs shape: (n_spectra, n_wave, 2, sysrem_its)
    obs_4d = rng.normal(0, 0.01, (n_spectra, n_wave, 2, sysrem_its)).astype(np.float64)
    tmpl = rng.normal(0, 1, (n_spectra, n_wave)).astype(np.float64)
    unc = np.full((n_spectra, n_wave), 0.01)
    lag = np.linspace(-30.0, 30.0, n_lags)
    ccf_values = np.zeros((n_lags, n_spectra, 2, sysrem_its), dtype=np.float64)
    result = ccf_numba_par_weighted_ordbord_opt(
        sysrem_its, lag, n_spectra, obs_4d, n_lags,
        wave, wave, ccf_values, tmpl, unc
    )
    assert result.shape == (n_lags, n_spectra, 2, sysrem_its)


# ---------------------------------------------------------------------------
# ccf_literature (parallel, unweighted mean subtraction)
# ---------------------------------------------------------------------------

def test_ccf_literature_shape():
    lag, obs, tmpl, wave, unc, in_transit = _make_ccf_inputs()
    n_spectra, n_lags = obs.shape[0], len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)
    result = ccf_literature(lag, n_spectra, obs, n_lags, wave, wave,
                            ccf_values, tmpl, unc, in_transit)
    assert result.shape == (n_lags, n_spectra)


def test_ccf_literature_values_in_range():
    lag, obs, tmpl, wave, unc, in_transit = _make_ccf_inputs()
    n_spectra, n_lags = obs.shape[0], len(lag)
    ccf_values = np.zeros((n_lags, n_spectra), dtype=np.float64)
    result = ccf_literature(lag, n_spectra, obs, n_lags, wave, wave,
                            ccf_values, tmpl, unc, in_transit)
    assert np.all(result >= -1.0 - 1e-10)
    assert np.all(result <= 1.0 + 1e-10)
