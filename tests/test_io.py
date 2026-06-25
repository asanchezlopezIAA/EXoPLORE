"""Tests for exoplore.io, output path construction and utility functions."""

from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

from exoplore.config import SimulationConfig
from exoplore.io.paths import build_output_tree, simulation_output_dir
from exoplore.io.utils import (
    weighted_quantile,
    check_consistent_wavelengths,
    convert_masked_arrays,
    find_nearest,
    convert_vega_to_ab,
    bootstrap_corrcoeffs,
    Utils_permute_nights_indices,
)
from exoplore.core.observation import spec_to_mat_fraction


def test_output_dir_structure():
    cfg = SimulationConfig()
    dirs = build_output_tree(cfg, "test_run", create=False)
    assert "root" in dirs
    assert "matrices" in dirs
    assert "plots" in dirs
    assert "correlations" in dirs


def test_output_dir_path_components():
    cfg = SimulationConfig()
    cfg.planet.name = "WASP-76b"
    cfg.instrument.name = "CRIRES"
    cfg.observation.event_type = "dayside"
    d = simulation_output_dir(cfg, "myrun")
    parts = d.parts
    assert "WASP-76b" in parts
    assert "CRIRES" in parts
    assert "dayside" in parts
    assert "myrun" in parts


def test_matrices_dir_name():
    cfg = SimulationConfig()
    dirs = build_output_tree(cfg, "run1")
    assert dirs["matrices"].name == "matrices"
    assert dirs["plots"].name == "plots"


# ---------------------------------------------------------------------------
# weighted_quantile
# ---------------------------------------------------------------------------

def test_weighted_quantile_uniform_weights():
    # CDF steps at [0.2, 0.4, 0.6, 0.8, 1.0] for 5 uniform-weight points.
    # np.interp(0.5, cumw, values) falls between the 2nd (2.0) and 3rd (3.0)
    # CDF steps → returns 2.5 by linear interpolation.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5)
    result = weighted_quantile(values, weights, [0.0, 0.5, 1.0])
    assert result[0] == pytest.approx(1.0, abs=0.1)
    assert result[1] == pytest.approx(2.5, abs=1e-10)
    assert result[2] == pytest.approx(5.0, abs=0.1)


def test_weighted_quantile_empty_returns_nan():
    result = weighted_quantile(np.array([]), np.array([]), [0.5])
    assert np.isnan(result[0])


def test_weighted_quantile_all_nan_returns_nan():
    values = np.array([np.nan, np.nan])
    weights = np.array([1.0, 1.0])
    result = weighted_quantile(values, weights, [0.5])
    assert np.isnan(result[0])


def test_weighted_quantile_concentrated_weight():
    # CDF steps: cumw_norm ≈ [0.0001, 0.9999, 1.0] for weights [0.01, 100, 0.01].
    # np.interp(0.5, cumw_norm, [1, 10, 100]) interpolates linearly between
    # the first two CDF steps → ≈ 1 + (0.5/0.9998)*9 ≈ 5.5.
    # Test verifies the function returns a value between the first two data points.
    values = np.array([1.0, 10.0, 100.0])
    weights = np.array([0.01, 100.0, 0.01])
    result = weighted_quantile(values, weights, [0.5])
    assert 1.0 < result[0] < 10.0


# ---------------------------------------------------------------------------
# check_consistent_wavelengths
# ---------------------------------------------------------------------------

def test_check_consistent_wavelengths_identical():
    wave = np.tile(np.linspace(1.0, 2.0, 100), (5, 1))
    assert check_consistent_wavelengths(wave) is True


def test_check_consistent_wavelengths_different():
    wave = np.tile(np.linspace(1.0, 2.0, 100), (5, 1))
    wave[2, 50] += 0.001   # perturb one spectrum
    assert check_consistent_wavelengths(wave) is False


def test_check_consistent_wavelengths_single_spectrum():
    wave = np.linspace(1.0, 2.0, 100).reshape(1, -1)
    assert check_consistent_wavelengths(wave) is True


# ---------------------------------------------------------------------------
# convert_masked_arrays
# ---------------------------------------------------------------------------

def test_convert_masked_arrays_basic():
    arr1 = ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    arr2 = ma.array([5.0, 6.0, 7.0, 8.0], mask=[False, False, False, True])
    d1, d2, idx1, idx2 = convert_masked_arrays(arr1, arr2)
    assert isinstance(d1, np.ndarray) and not isinstance(d1, ma.MaskedArray)
    assert isinstance(d2, np.ndarray) and not isinstance(d2, ma.MaskedArray)
    assert list(idx1) == [1]
    assert list(idx2) == [3]


def test_convert_masked_arrays_no_mask():
    arr1 = ma.array([1.0, 2.0, 3.0], mask=False)
    arr2 = ma.array([4.0, 5.0, 6.0], mask=False)
    d1, d2, idx1, idx2 = convert_masked_arrays(arr1, arr2)
    assert len(idx1) == 0
    assert len(idx2) == 0
    np.testing.assert_array_equal(d1, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# find_nearest
# ---------------------------------------------------------------------------

class TestFindNearest:
    def test_exact_match(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_nearest(arr, 3.0) == 3.0

    def test_nearest_rounds_down(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_nearest(arr, 2.4) == 2.0

    def test_nearest_rounds_up(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_nearest(arr, 2.7) == 3.0

    def test_list_input(self):
        result = find_nearest([10, 20, 30], 22)
        assert result == 20

    def test_single_element(self):
        assert find_nearest(np.array([42.0]), 0.0) == 42.0


# ---------------------------------------------------------------------------
# convert_vega_to_ab
# ---------------------------------------------------------------------------

class TestConvertVegaToAb:
    def test_V_band(self):
        # V offset is 0.02
        result = convert_vega_to_ab(8.0, 'V')
        assert abs(result - 8.02) < 1e-10

    def test_K_band(self):
        # K offset is 1.85
        result = convert_vega_to_ab(5.0, 'K')
        assert abs(result - 6.85) < 1e-10

    def test_invalid_band(self):
        with pytest.raises(ValueError, match="Invalid band"):
            convert_vega_to_ab(8.0, 'Z')

    def test_all_bands_defined(self):
        for band in ('U', 'B', 'V', 'R', 'I', 'J', 'H', 'K'):
            result = convert_vega_to_ab(0.0, band)
            assert isinstance(result, float)


# ---------------------------------------------------------------------------
# bootstrap_corrcoeffs
# ---------------------------------------------------------------------------

class TestBootstrapCorrcoeffs:
    def test_returns_float(self):
        np.random.seed(42)
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = bootstrap_corrcoeffs(X, Y, samples=100)
        assert isinstance(result, float)

    def test_perfect_anticorr_small_std(self):
        # perfectly anti-correlated data, std should be small
        np.random.seed(0)
        X = np.arange(1, 21, dtype=float)
        Y = -X
        std = bootstrap_corrcoeffs(X, Y, samples=200)
        assert std < 0.3

    def test_random_data_positive_std(self):
        np.random.seed(7)
        X = np.random.randn(50)
        Y = np.random.randn(50)
        std = bootstrap_corrcoeffs(X, Y, samples=200)
        assert std >= 0.0


# ---------------------------------------------------------------------------
# Utils_permute_nights_indices
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# spec_to_mat_fraction  (v0.25)
# ---------------------------------------------------------------------------

class TestSpecToMatFraction:
    """Smoke tests for exoplore.io.stellar.spec_to_mat_fraction."""

    def _make_inputs(self, n_spec=20, n_pix=100, n_prt=120):
        rng = np.random.default_rng(42)
        wave = np.linspace(2.3, 2.4, n_pix)
        wave_prt = np.linspace(2.29, 2.41, n_prt)
        spec = np.abs(rng.normal(0.01, 0.002, n_prt))
        v = rng.normal(0, 100.0, n_spec)
        mat_stellar = np.ones((n_spec, n_pix))
        with_signal = np.arange(5, 15, dtype=int)
        without_signal = np.concatenate([np.arange(5), np.arange(15, n_spec)])
        fraction = np.where(
            np.isin(np.arange(n_spec), with_signal), 0.01, 0.0
        )
        inp_dat = {
            "event": "transit",
            "Limb_asymmetries": False,
            "Scale_inj": 1.0,
            "Inject_Scale_Factor": 1.0,
        }
        return inp_dat, v, wave, wave_prt, spec, mat_stellar, with_signal, without_signal, fraction

    def test_transit_shape(self):
        inp_dat, v, wave, wave_prt, spec, mat_stellar, with_signal, without_signal, fraction = (
            self._make_inputs()
        )
        mat, mat_shift = spec_to_mat_fraction(
            inp_dat, None, None, v, wave, wave_prt,
            spec, mat_stellar, with_signal, without_signal, fraction
        )
        assert mat.shape == (len(v), len(wave))
        assert mat_shift.shape == mat.shape

    def test_dayside_shape(self):
        inp_dat, v, wave, wave_prt, spec, mat_stellar, with_signal, without_signal, fraction = (
            self._make_inputs()
        )
        inp_dat["event"] = "dayside"
        mat, mat_shift = spec_to_mat_fraction(
            inp_dat, None, None, v, wave, wave_prt,
            spec, mat_stellar, with_signal, without_signal, fraction
        )
        assert mat.shape == mat_stellar.shape

    def test_transit_no_signal_outside_transit(self):
        """Out-of-transit rows with include_star=True should equal mat_stellar."""
        inp_dat, v, wave, wave_prt, spec, mat_stellar, with_signal, without_signal, fraction = (
            self._make_inputs()
        )
        mat, _ = spec_to_mat_fraction(
            inp_dat, None, None, v, wave, wave_prt,
            spec, mat_stellar, with_signal, without_signal, fraction,
            include_star=True
        )
        for i in range(len(v)):
            if i not in with_signal:
                np.testing.assert_allclose(mat[i], mat_stellar[i], rtol=1e-10)

    def test_no_intransit_raises(self):
        inp_dat, v, wave, wave_prt, spec, mat_stellar, with_signal, without_signal, fraction = (
            self._make_inputs()
        )
        with pytest.raises(Exception, match="No spectra in-transit"):
            spec_to_mat_fraction(
                inp_dat, None, None, v, wave, wave_prt,
                spec, mat_stellar,
                np.array([], dtype=int),   # empty with_signal
                without_signal, fraction
            )


class TestUtilsPermuteNightsIndices:
    def test_shape_preserved(self):
        arr = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
        result = Utils_permute_nights_indices(arr)
        assert result.shape == arr.shape

    def test_values_preserved_per_slice(self):
        np.random.seed(0)
        arr = np.arange(1 * 1 * 1 * 6, dtype=float).reshape(1, 1, 1, 6)
        result = Utils_permute_nights_indices(arr)
        # same set of values, just reordered
        np.testing.assert_array_equal(
            np.sort(result[0, 0, 0, :]),
            np.sort(arr[0, 0, 0, :])
        )

    def test_output_is_new_array(self):
        arr = np.ones((2, 2, 2, 4))
        result = Utils_permute_nights_indices(arr)
        assert result is not arr
