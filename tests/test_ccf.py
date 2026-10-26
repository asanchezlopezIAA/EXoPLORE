"""Tests for exoplore.ccf, velocity grids and CCF computation."""

import numpy as np
import pytest

from exoplore.ccf.compute import build_velocity_grid, compute_ccf_timeseries, compute_kp_vsys_map


class TestVelocityGrid:
    def test_symmetric(self):
        v = build_velocity_grid(100.0, 1.0)
        assert v[0] == pytest.approx(-100.0)
        assert v[-1] == pytest.approx(100.0)

    def test_step(self):
        v = build_velocity_grid(50.0, 2.0)
        diffs = np.diff(v)
        assert np.allclose(diffs, 2.0)

    def test_length(self):
        v = build_velocity_grid(10.0, 1.0)
        # -10, -9, ..., 0, ..., 9, 10 → 21 elements
        assert len(v) == 21


class TestCCFTimeseries:
    """Smoke tests: verify shapes and that no exceptions are raised."""

    def _make_data(self, n_spec=20, n_pix=500):
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, (n_spec, n_pix)).astype(np.float64)
        tmpl = rng.normal(0, 1, (n_spec, n_pix)).astype(np.float64)
        wave = np.linspace(1.0, 1.05, n_pix)
        unc = np.full((n_spec, n_pix), 0.01)
        intrans = np.arange(5, 15, dtype=np.int64)
        return obs, tmpl, wave, wave, unc, intrans

    def test_output_shape(self):
        obs, tmpl, wave, wave_cc, unc, intrans = self._make_data()
        v_max, v_step = 50.0, 1.0
        v, ccf = compute_ccf_timeseries(
            obs, tmpl, wave, wave_cc, unc, intrans,
            velocity_max_kms=v_max,
            velocity_step_kms=v_step,
        )
        n_lags = len(build_velocity_grid(v_max, v_step))
        assert ccf.shape == (n_lags, obs.shape[0])

    def test_velocity_axis_matches(self):
        obs, tmpl, wave, wave_cc, unc, intrans = self._make_data()
        v, ccf = compute_ccf_timeseries(
            obs, tmpl, wave, wave_cc, unc, intrans,
            velocity_max_kms=30.0,
            velocity_step_kms=1.0,
        )
        assert v[0] == pytest.approx(-30.0)
        assert v[-1] == pytest.approx(30.0)


class TestKpVsysMap:
    def test_shape(self):
        n_spec, n_lags = 20, 101
        ccf = np.random.default_rng(0).normal(0, 1, (n_lags, n_spec))
        v = np.linspace(-50, 50, n_lags)
        phase = np.linspace(-0.05, 0.05, n_spec)
        kp_grid = np.linspace(50, 250, 10)
        kp_map = compute_kp_vsys_map(ccf, v, phase, kp_grid)
        assert kp_map.shape == (len(kp_grid), n_lags)


# ---------------------------------------------------------------------------
# get_corr_coeff  (v0.24)
# ---------------------------------------------------------------------------

class TestGetCorrCoeff:
    """Tests for exoplore.ccf.statistics.get_corr_coeff."""

    def setup_method(self):
        from exoplore.ccf.statistics import get_corr_coeff
        self.fn = get_corr_coeff
        np.random.seed(42)
        self.n_orders = 2
        self.n_nights = 4
        self.n_spec = 10
        self.n_pix = 50
        self.data = np.random.rand(self.n_orders, self.n_nights,
                                   self.n_spec, self.n_pix)
        self.model = np.random.rand(self.n_orders, self.n_spec, self.n_pix)
        self.with_signal = np.arange(3, 7)
        # stats must have n_nights rows. With first_night_noiseless=True,
        # stats_0 = stats[1:, 0] has shape (n_nights-1,) matching corr_coeff
        self.stats = np.random.rand(self.n_nights, 3)

    def test_2d_returns_two_values(self):
        result = self.fn(
            self.with_signal, self.data, self.model,
            np.arange(self.n_nights - 1), h=0,
            stats=self.stats, title='t',
            plotname='p', CC_2D=True, show_plot=False,
            first_night_noiseless=True, n_nights=self.n_nights, plots_dir='/tmp/'
        )
        assert len(result) == 2  # pearson_coeff, standard_error

    def test_pearson_is_float(self):
        pearson_coeff, std_err = self.fn(
            self.with_signal, self.data, self.model,
            np.arange(self.n_nights - 1), h=0,
            stats=self.stats, title='t',
            plotname='p', CC_2D=True, show_plot=False,
            first_night_noiseless=True, n_nights=self.n_nights, plots_dir='/tmp/'
        )
        assert isinstance(float(pearson_coeff), float)
        assert isinstance(float(std_err), float)

    def test_1d_returns_array(self):
        result = self.fn(
            self.with_signal, self.data, self.model,
            np.arange(self.n_nights - 1), h=0,
            stats=self.stats, title='t',
            plotname='p', CC_2D=False, show_plot=False,
            first_night_noiseless=True, n_nights=self.n_nights, plots_dir='/tmp/'
        )
        assert hasattr(result, '__len__')
        assert len(result) == self.n_nights - 1


def test_orbital_alignment_recovers_a_known_ccf_track():
    """A moving line pattern adds coherently at its injected orbital motion."""
    from exoplore.ccf.compute import get_shifted_ccf_matrix

    reference_kp = 120.0
    exposure_velocities = np.array([-20., -10., 0., 10., 20.])
    phase = np.arcsin(exposure_velocities / reference_kp) / (2 * np.pi)
    velocity = np.arange(-100., 101.)
    v_rest = np.arange(-40., 41.)
    kp_grid = np.arange(80., 161., 10.)
    injected_offset = 3.0
    ccf = np.exp(-0.5 * (
        (velocity[:, None] - exposure_velocities - injected_offset) / 2.5
    ) ** 2)

    shifted = get_shifted_ccf_matrix(
        np.arange(len(phase)),
        v_rest,
        velocity,
        kp_grid,
        phase,
        0.0,
        0.0,
        40,
        1.0,
        ccf,
        significant_eccentricity=False,
        eccentricity=0.0,
        argument_of_periastron_deg=0.0,
        sysrem_iterations=0,
    )
    coadded = np.sum(shifted, axis=1)
    velocity_index, kp_index = np.unravel_index(coadded.argmax(), coadded.shape)
    assert kp_grid[kp_index] == reference_kp
    assert v_rest[velocity_index] == injected_offset


def test_map_normalisation_uses_the_selected_off_peak_samples():
    """The map statistic subtracts the off-peak mean and scales by its scatter."""
    from exoplore.ccf.compute import get_max_CCF_peak

    values = np.array([[1.], [2.], [10.], [3.], [4.]])
    velocity = np.array([-20., -10., 0., 10., 20.])
    normalised, peak, kp_index, peak_velocity, scatter = get_max_CCF_peak(
        values,
        velocity,
        np.array([75.0]),
        snr_exclude_kms=5.0,
        sysrem_iterations=0,
    )
    expected_scatter = np.sqrt(1.25)
    np.testing.assert_allclose(normalised[:, 0], (values[:, 0] - 2.5) / expected_scatter)
    np.testing.assert_allclose(scatter, expected_scatter)
    assert float(peak) == pytest.approx(7.5 / expected_scatter)
    assert kp_index == 0
    np.testing.assert_array_equal(peak_velocity, [0.])
