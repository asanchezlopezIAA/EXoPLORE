"""Tests for exoplore.instruments, ANDES instrument functions."""

import numpy as np
import pytest

from exoplore.instruments.wavegrid import (
    make_log_wave_grid,
    compute_pixel_velocity_scale,
    FromOrdersToDetectors,
    Interp_Uniform_Wvl_Grid,
    pixel_snr_one_order,
)


# ---------------------------------------------------------------------------
# make_log_wave_grid
# ---------------------------------------------------------------------------

def test_make_log_wave_grid_bounds():
    grid = make_log_wave_grid(1.0, 2.0, R=1e5, oversample=2)
    assert grid[0] == pytest.approx(1.0, rel=1e-6)
    assert grid[-1] >= 2.0 - 1e-8  # last point >= lambda_max


def test_make_log_wave_grid_constant_velocity_spacing():
    """Log spacing → constant dv = c / (R * oversample)."""
    R = 1e5
    oversample = 2
    grid = make_log_wave_grid(1.0, 1.1, R=R, oversample=oversample)
    # d(ln λ) should be constant
    dlnlam = np.diff(np.log(grid))
    expected = 1.0 / (R * oversample)
    np.testing.assert_allclose(dlnlam, expected, rtol=1e-10)


def test_make_log_wave_grid_higher_oversample_more_pixels():
    g1 = make_log_wave_grid(1.0, 1.1, R=1e5, oversample=2)
    g2 = make_log_wave_grid(1.0, 1.1, R=1e5, oversample=4)
    assert len(g2) > len(g1)


# ---------------------------------------------------------------------------
# compute_pixel_velocity_scale
# ---------------------------------------------------------------------------

def test_compute_pixel_velocity_scale_positive():
    v = compute_pixel_velocity_scale(R=1e5, pixels_per_res=2.5)
    assert v > 0


def test_compute_pixel_velocity_scale_proportional_to_1_over_R():
    v1 = compute_pixel_velocity_scale(R=1e5, pixels_per_res=2.5)
    v2 = compute_pixel_velocity_scale(R=2e5, pixels_per_res=2.5)
    assert v1 == pytest.approx(2 * v2, rel=1e-6)


def test_compute_pixel_velocity_scale_typical_value():
    # c / (1e5 * 2.5) ≈ 1.2 km/s per pixel
    v = compute_pixel_velocity_scale(R=1e5, pixels_per_res=2.5)
    assert 1.0 < v < 1.5


# ---------------------------------------------------------------------------
# FromOrdersToDetectors
# ---------------------------------------------------------------------------

def test_from_orders_to_detectors_shape():
    n_spec, n_orders, n_pixels = 10, 38, 2048
    arr = np.random.rand(n_spec, n_orders, n_pixels)
    out = FromOrdersToDetectors(arr, n_orders, n_pixels)
    assert out.shape == (n_spec, 2 * n_orders, n_pixels // 2)


def test_from_orders_to_detectors_values():
    """Left half of order 0 goes to row 0; right half goes to row 1."""
    arr = np.ones((2, 3, 4))  # n_spec=2, n_orders=3, n_pixels=4
    arr[:, 0, :2] = 7.0   # left half of order 0
    arr[:, 0, 2:] = 9.0   # right half of order 0
    out = FromOrdersToDetectors(arr, 3, 4)
    np.testing.assert_array_equal(out[:, 0, :], 7.0)  # even row = left
    np.testing.assert_array_equal(out[:, 1, :], 9.0)  # odd row = right


# ---------------------------------------------------------------------------
# Interp_Uniform_Wvl_Grid
# ---------------------------------------------------------------------------

def test_interp_uniform_wvl_grid_output_shapes():
    n_spec, n_orders, n_pix = 3, 2, 50
    wave = np.zeros((n_spec, n_orders, n_pix))
    spec = np.zeros((n_spec, n_orders, n_pix))
    sig = np.zeros((n_spec, n_orders, n_pix))
    for s in range(n_spec):
        for o in range(n_orders):
            wave[s, o, :] = np.linspace(1.0 + o * 0.1, 1.09 + o * 0.1, n_pix)
            spec[s, o, :] = np.ones(n_pix)
            sig[s, o, :]  = np.ones(n_pix)
    new_n = 30
    nw, ns, nsg = Interp_Uniform_Wvl_Grid(wave, spec, sig, new_n)
    assert nw.shape  == (n_spec, n_orders, new_n)
    assert ns.shape  == (n_spec, n_orders, new_n)
    assert nsg.shape == (n_spec, n_orders, new_n)


def test_interp_uniform_wvl_grid_uniform_spacing():
    """Output wavelength grid for each exposure/order must be uniformly spaced."""
    n_spec, n_orders, n_pix = 2, 1, 40
    wave = np.zeros((n_spec, n_orders, n_pix))
    spec = np.ones((n_spec, n_orders, n_pix))
    sig  = np.ones((n_spec, n_orders, n_pix))
    for s in range(n_spec):
        wave[s, 0, :] = np.linspace(1.0, 1.1, n_pix)
    nw, _, _ = Interp_Uniform_Wvl_Grid(wave, spec, sig, 20)
    diffs = np.diff(nw[0, 0, :])
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# pixel_snr_one_order
# ---------------------------------------------------------------------------

class TestPixelSnrOneOrder:
    """Tests for the standalone pixel_snr_one_order function."""

    def _make_inputs(self, n_times=5, n_pix=100):
        wave = np.linspace(1.0, 1.1, n_pix)
        tellurics = np.ones((n_times, n_pix))
        snr_center = 200.0
        return wave, tellurics, snr_center

    def test_output_shape(self):
        n_times, n_pix = 5, 100
        wave, tellurics, snr_center = self._make_inputs(n_times, n_pix)
        out = pixel_snr_one_order(wave, tellurics, snr_center)
        assert out.shape == (n_times, n_pix)

    def test_positive_values(self):
        wave, tellurics, snr_center = self._make_inputs()
        out = pixel_snr_one_order(wave, tellurics, snr_center)
        assert np.all(out > 0)

    def test_center_snr_scaled_by_px_per_resel(self):
        """Peak pixel SNR ≈ snr_center / sqrt(px_per_resel) for flat tellurics."""
        n_times, n_pix = 1, 200
        wave = np.linspace(1.0, 1.1, n_pix)
        tellurics = np.ones((n_times, n_pix))
        snr_center = 100.0
        px_per_resel = 4.0
        out = pixel_snr_one_order(
            wave, tellurics, snr_center, px_per_resel=px_per_resel
        )
        expected_peak = snr_center / np.sqrt(px_per_resel)
        assert out.max() == pytest.approx(expected_peak, rel=1e-3)

    def test_telluric_absorption_reduces_snr(self):
        """Pixels with low telluric transmission should have lower SNR.

        The function normalises relative to T at the blaze peak, so the
        absorption must be placed away from that peak to show a reduction.
        With 100 pixels the sinc^2 blaze peaks near the centre; pixel 10 is
        safely off-centre.
        """
        n_times, n_pix = 3, 100
        wave = np.linspace(1.0, 1.1, n_pix)
        tellurics_flat = np.ones((n_times, n_pix))
        tellurics_abs = np.ones((n_times, n_pix))
        tellurics_abs[:, 10] = 0.1  # strong absorption away from blaze peak
        out_flat = pixel_snr_one_order(wave, tellurics_flat, 100.0)
        out_abs = pixel_snr_one_order(wave, tellurics_abs, 100.0)
        assert out_abs[:, 10].mean() < out_flat[:, 10].mean()

    def test_snr_center_array(self):
        """snr_center can be a per-exposure array."""
        n_times, n_pix = 4, 80
        wave = np.linspace(1.0, 1.05, n_pix)
        tellurics = np.ones((n_times, n_pix))
        snr_center = np.array([100.0, 120.0, 140.0, 160.0])
        out = pixel_snr_one_order(wave, tellurics, snr_center)
        assert out.shape == (n_times, n_pix)
        # Higher snr_center → higher peak SNR
        peaks = out.max(axis=1)
        assert np.all(np.diff(peaks) > 0)

    def test_importable_from_top_level(self):
        from exoplore.instruments import pixel_snr_one_order as fn
        assert callable(fn)
