"""
Tests for exoplore.pipelines, masking, SYSREM, BL19, BLASP24.
"""

import numpy as np
import pytest

from exoplore.pipelines.masking import (
    mask_telluric_columns,
    mask_telluric_columns_with_window,
    mask_noisy_columns,
    merge_masks,
    good_pixel_indices,
)
from exoplore.pipelines.sysrem import sysrem_iteration, apply_sysrem
from exoplore.pipelines.bl19 import bl19_normalise, bl19_telluric_correct, run_bl19_pipeline
from exoplore.pipelines.blasp24 import blasp24_normalise, run_blasp24_pipeline


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

class TestMasking:
    def test_mask_telluric_columns_basic(self):
        tel = np.array([1.0, 0.5, 0.2, 1.0, 0.9])
        mask = mask_telluric_columns(tel, threshold=0.3)
        assert 2 in mask  # 0.2 < 0.3
        assert 0 not in mask
        assert 1 not in mask  # 0.5 > 0.3

    def test_mask_telluric_with_window(self):
        tel = np.array([1.0, 1.0, 0.1, 1.0, 1.0])
        mask = mask_telluric_columns_with_window(tel, threshold=0.3, safety_window=3)
        # pixel 2 is bad → window ±1 → pixels 1,2,3 masked
        assert 2 in mask
        assert 1 in mask
        assert 3 in mask
        assert 0 not in mask
        assert 4 not in mask

    def test_mask_noisy_columns(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (20, 50))
        # Alternating ±100 gives a large std (~100), vs ~1 for normal columns
        data[::2, 25] = 100.0
        data[1::2, 25] = -100.0
        mask = mask_noisy_columns(data)
        assert 25 in mask

    def test_merge_masks(self):
        m1 = np.array([1, 3, 5])
        m2 = np.array([3, 7])
        merged = merge_masks(m1, m2)
        assert set(merged) == {1, 3, 5, 7}
        assert len(merged) == len(np.unique(merged))  # no duplicates

    def test_good_pixel_indices(self):
        mask = np.array([0, 2, 4])
        good = good_pixel_indices(mask, n_pixels=5)
        assert set(good) == {1, 3}

    def test_merge_with_none(self):
        m = np.array([1, 2, 3])
        merged = merge_masks(None, m)
        np.testing.assert_array_equal(merged, [1, 2, 3])


# ---------------------------------------------------------------------------
# SYSREM
# ---------------------------------------------------------------------------

class TestSysrem:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n_spec, self.n_pix = 30, 80
        # Add a known systematic: airmass-like trend
        time = np.linspace(0, 1, self.n_spec)
        airmass = 1.0 + 0.5 * np.sin(np.pi * time)
        spectral = rng.uniform(0.8, 1.2, self.n_pix)
        systematic = np.outer(airmass, spectral)
        noise = rng.normal(0, 0.01, (self.n_spec, self.n_pix))
        self.data = systematic + noise
        self.unc = 0.01 * np.ones_like(self.data)

    def test_sysrem_iteration_returns_correct_shape(self):
        out, cor, a1, c1 = sysrem_iteration(self.data, self.unc)
        assert out.shape == self.data.shape
        assert cor.shape == self.data.shape

    def test_sysrem_reduces_variance(self):
        out, _, _, _ = sysrem_iteration(self.data, self.unc)
        assert np.var(out) < np.var(self.data)

    def test_apply_sysrem_shape(self):
        good = np.arange(self.n_pix)
        cleaned, corrections = apply_sysrem(self.data, self.unc, n_iterations=2, good_pixels=good)
        assert cleaned.shape == (self.n_spec, self.n_pix)
        assert len(corrections) == 2

    def test_apply_sysrem_with_good_pixels(self):
        good = np.arange(10, 70)
        cleaned, _ = apply_sysrem(self.data, self.unc, n_iterations=1, good_pixels=good)
        assert cleaned.shape[1] == len(good)


# ---------------------------------------------------------------------------
# BL19 pipeline
# ---------------------------------------------------------------------------

class TestBL19:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.n_spec, self.n_pix = 20, 100
        self.wave = np.linspace(1.0, 2.5, self.n_pix)
        self.data = rng.uniform(0.8, 1.2, (self.n_spec, self.n_pix))
        self.unc = 0.05 * np.ones_like(self.data)
        self.good = np.arange(self.n_pix)

    def test_normalise_output_shape(self):
        out, err = bl19_normalise(self.wave, self.data, self.unc, self.good)
        assert out.shape == (self.n_spec, self.n_pix)
        assert err.shape == out.shape

    def test_normalise_envelope_fit(self):
        out, err = bl19_normalise(self.wave, self.data, self.unc, self.good, use_envelope_fit=True)
        assert out.shape == (self.n_spec, self.n_pix)

    def test_telluric_correct_shape(self):
        norm_data, norm_unc = bl19_normalise(self.wave, self.data, self.unc, self.good)
        corr_data, corr_unc = bl19_telluric_correct(norm_data, norm_unc)
        assert corr_data.shape == norm_data.shape

    def test_run_pipeline_shape(self):
        out, err = run_bl19_pipeline(self.wave, self.data, self.unc, self.good)
        assert out.shape == (self.n_spec, self.n_pix)


# ---------------------------------------------------------------------------
# BLASP24 pipeline
# ---------------------------------------------------------------------------

class TestBLASP24:
    def setup_method(self):
        rng = np.random.default_rng(99)
        self.n_spec, self.n_pix = 15, 60
        self.wave = np.linspace(1.0, 2.5, self.n_pix)
        self.data = rng.uniform(0.9, 1.1, (self.n_spec, self.n_pix))
        self.unc = 0.02 * np.ones_like(self.data)
        self.good = np.arange(self.n_pix)

    def test_normalise_output_shape(self):
        out, err, mask, gp = blasp24_normalise(self.wave, self.data, self.unc, self.good)
        assert out.shape == self.data.shape
        assert err.shape == self.data.shape

    def test_normalise_with_mask(self):
        # Mark first 5 pixels as bad
        good = np.arange(5, self.n_pix)
        out, err, mask, gp = blasp24_normalise(self.wave, self.data, self.unc, good)
        assert 0 in mask  # masked pixels still masked

    def test_run_pipeline_shape(self):
        out, err, mask, gp = run_blasp24_pipeline(self.wave, self.data, self.unc, self.good)
        assert out.shape == self.data.shape


# ---------------------------------------------------------------------------
# filter_model_singleorder
# ---------------------------------------------------------------------------

class TestFilterModelSingleorder:
    """Tests for the SYSREM projector model filter."""

    def setup_method(self):
        from exoplore.pipelines.sysrem import filter_model_singleorder
        self.fn = filter_model_singleorder
        rng = np.random.default_rng(42)
        self.n_spec, self.n_pix = 10, 30
        self.good = np.arange(self.n_pix)
        # Build a trivial projector (identity → residual is zero)
        self.P_identity = np.eye(self.n_spec)
        # Build a zero projector (residual equals normalised model)
        self.P_zero = np.zeros((self.n_spec, self.n_spec))
        # Simple model matrix (all ones → median = 1 → norm = 1)
        self.model_ones = np.ones((self.n_spec, self.n_pix))
        # Random positive model
        self.model_random = np.abs(rng.normal(1.0, 0.1, (self.n_spec, self.n_pix))) + 0.5

    def test_output_shape(self):
        out = self.fn(self.P_zero, self.model_ones, self.good)
        assert out.shape == (self.n_spec, self.n_pix)

    def test_zero_projector_returns_normalised_model(self):
        """With P=0, output should equal the median-normalised model."""
        out = self.fn(self.P_zero, self.model_ones, self.good)
        # model_ones median-normalised is all-ones; P@ones=0; result=ones
        np.testing.assert_allclose(out[:, self.good], 1.0, atol=1e-12)

    def test_identity_projector_returns_zero(self):
        """With P=I, model_norm - P@model_norm = 0."""
        out = self.fn(self.P_identity, self.model_ones, self.good)
        np.testing.assert_allclose(out[:, self.good], 0.0, atol=1e-12)

    def test_good_pixels_only_normalised(self):
        """Columns outside useful_spectral_points stay at 1 (baseline)."""
        good_partial = np.arange(5, self.n_pix)
        out = self.fn(self.P_zero, self.model_random, good_partial)
        # Outside good region: model_norm initialised to 1, P@1-col = 0 → stays 1
        np.testing.assert_allclose(out[:, :5], 1.0, atol=1e-12)

    def test_median_normalisation_applied(self):
        """Each row of the normalised model should have median ≈ 1."""
        out = self.fn(self.P_zero, self.model_random, self.good)
        for row in out[:, self.good]:
            assert np.median(row) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# remove_throughput_fit_og  (v0.24)
# ---------------------------------------------------------------------------

class TestRemoveThroughputFitOg:
    """Tests for exoplore.pipelines.prepare.remove_throughput_fit_og."""

    def setup_method(self):
        from exoplore.pipelines.prepare import remove_throughput_fit_og
        self.fn = remove_throughput_fit_og
        np.random.seed(0)
        self.n_spec, self.n_pix = 10, 60
        self.wavelengths = np.linspace(2.3, 2.4, self.n_pix)
        # create spectrum with a simple polynomial throughput variation
        throughput = 1.0 + 0.05 * (self.wavelengths - 2.35)**2
        self.spectrum = np.outer(np.ones(self.n_spec), throughput)
        self.reduction_matrix = np.ones_like(self.spectrum)
        self.mask = np.array([], dtype=int)

    def test_output_shapes(self):
        corrected, rm, unc, mask_out, useful = self.fn(
            self.spectrum.copy(), self.reduction_matrix.copy(),
            self.wavelengths, self.mask.copy()
        )
        assert corrected.shape == self.spectrum.shape
        assert rm.shape == self.reduction_matrix.shape
        assert unc is None  # no uncertainties supplied

    def test_correction_flattens_polynomial(self):
        corrected, _, _, _, _ = self.fn(
            self.spectrum.copy(), self.reduction_matrix.copy(),
            self.wavelengths, self.mask.copy(), polynomial_fit_degree=2
        )
        # after polynomial correction the std per row should be very small
        for row in corrected:
            assert row.std() < 0.05

    def test_with_uncertainties(self):
        unc_in = np.ones_like(self.spectrum) * 0.01
        corrected, rm, unc_out, mask_out, useful = self.fn(
            self.spectrum.copy(), self.reduction_matrix.copy(),
            self.wavelengths, self.mask.copy(),
            uncertainties=unc_in.copy()
        )
        assert unc_out is not None
        assert unc_out.shape == unc_in.shape


# ---------------------------------------------------------------------------
# remove_telluric_lines_fit_og  (v0.25)
# ---------------------------------------------------------------------------

class TestRemoveTelluricLinesFitOg:
    """Tests for exoplore.pipelines.prepare.remove_telluric_lines_fit_og."""

    def setup_method(self):
        from exoplore.pipelines.prepare import remove_telluric_lines_fit_og
        self.fn = remove_telluric_lines_fit_og
        np.random.seed(1)
        self.n_spec, self.n_pix = 8, 40
        # airmass vector: monotonically increasing
        self.airmass = np.linspace(1.0, 2.0, self.n_spec)
        # build a spectrum with a simple airmass-dependent telluric
        # T = exp(-airmass * tau) with tau per pixel ~ 0.1
        tau = 0.1 * np.ones(self.n_pix)
        self.spectrum = np.exp(-np.outer(self.airmass, tau))
        self.reduction_matrix = np.ones_like(self.spectrum)
        self.mask = np.array([], dtype=int)

    def test_output_shapes(self):
        corrected, rm, unc, mask_out, useful = self.fn(
            self.spectrum.copy(), self.reduction_matrix.copy(),
            self.airmass, self.mask.copy()
        )
        assert corrected.shape == self.spectrum.shape
        assert rm.shape == self.reduction_matrix.shape
        assert unc is None  # no uncertainties supplied

    def test_returns_five_values(self):
        result = self.fn(
            self.spectrum.copy(), self.reduction_matrix.copy(),
            self.airmass, self.mask.copy()
        )
        assert len(result) == 5

    def test_with_uncertainties(self):
        unc_in = 0.01 * np.ones_like(self.spectrum)
        corrected, rm, unc_out, mask_out, useful = self.fn(
            self.spectrum.copy(), self.reduction_matrix.copy(),
            self.airmass, self.mask.copy(),
            uncertainties=unc_in.copy()
        )
        assert unc_out is not None
        assert unc_out.shape == unc_in.shape


# ---------------------------------------------------------------------------
# compute_k_sigma  (v0.25)
# ---------------------------------------------------------------------------

class TestComputeKSigma:
    """Smoke tests for exoplore.pipelines.blasp24.compute_k_sigma.

    The function requires .npz files on disk, so we only test that it
    imports and has the correct signature; the actual computation is
    tested via integration when real data are available.
    """

    def test_importable(self):
        from exoplore.pipelines.blasp24 import compute_k_sigma
        assert callable(compute_k_sigma)

    def test_signature(self):
        import inspect
        from exoplore.pipelines.blasp24 import compute_k_sigma
        sig = inspect.signature(compute_k_sigma)
        params = list(sig.parameters.keys())
        assert "data_dir" in params
        assert "orig_orders" in params
        assert "order_selection" in params
        assert "n_spectra_store" in params
        assert "pixel_mask_file" in params
        assert "return_per_night" in params

    def test_missing_file_raises(self, tmp_path):
        from exoplore.pipelines.blasp24 import compute_k_sigma
        orig_orders = np.array([0])
        order_selection = [np.array([0])]
        n_spectra_store = np.array([3])
        with pytest.raises(Exception):
            compute_k_sigma(
                str(tmp_path), orig_orders, order_selection,
                n_spectra_store, "nonexistent_mask.npz"
            )
