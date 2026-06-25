"""
Tests for exoplore.observation, timing, velocity, airmass, noise.
"""

import math

import numpy as np
import pytest

from exoplore.observation.timing import (
    orbital_phase,
    in_transit_indices,
    observation_julian_dates,
    transit_contact_times,
)
from exoplore.observation.velocity import (
    planet_radial_velocity,
    stellar_radial_velocity,
)
from exoplore.observation.airmass import synthetic_airmass
from exoplore.observation.noise import photon_noise, snr_per_pixel


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class TestOrbitalPhase:
    def test_midtransit_is_zero(self):
        jd = np.array([2459000.0])
        phase = orbital_phase(jd, transit_epoch_bjd=2459000.0, orbital_period_days=3.0)
        assert phase[0] == pytest.approx(0.0)

    def test_half_period_is_half_phase(self):
        jd = np.array([2459001.5])
        phase = orbital_phase(jd, transit_epoch_bjd=2459000.0, orbital_period_days=3.0)
        assert abs(phase[0]) == pytest.approx(0.5)

    def test_phase_range(self):
        jd = np.linspace(2459000.0, 2459009.0, 100)
        phases = orbital_phase(jd, transit_epoch_bjd=2459000.0, orbital_period_days=3.0)
        assert np.all(phases >= -0.5)
        assert np.all(phases < 0.5)


class TestInTransitIndices:
    def test_midtransit_included(self):
        phases = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
        idx = in_transit_indices(phases, transit_duration_hours=2.4, orbital_period_days=3.0)
        assert 2 in idx  # phase=0

    def test_out_of_transit_excluded(self):
        phases = np.array([0.4, -0.4])
        idx = in_transit_indices(phases, transit_duration_hours=2.0, orbital_period_days=3.0)
        assert len(idx) == 0


class TestObservationJD:
    def test_output_length(self):
        jd = observation_julian_dates(
            transit_epoch_bjd=2459000.0,
            exposure_time_seconds=300.0,
            readout_time_seconds=30.0,
            overhead_time_seconds=10.0,
            n_exposures=20,
        )
        assert len(jd) == 20

    def test_monotonically_increasing(self):
        jd = observation_julian_dates(
            transit_epoch_bjd=2459000.0,
            exposure_time_seconds=300.0,
            readout_time_seconds=10.0,
            overhead_time_seconds=0.0,
            n_exposures=10,
        )
        assert np.all(np.diff(jd) > 0)


class TestContactTimes:
    def test_ordering(self):
        T1, T2, T3, T4 = transit_contact_times(2459000.0, transit_duration_hours=2.0)
        assert T1 < T2 < T3 < T4

    def test_symmetric_about_midtransit(self):
        T1, T2, T3, T4 = transit_contact_times(2459000.0, transit_duration_hours=2.0)
        mid = 2459000.0
        assert T4 - mid == pytest.approx(mid - T1)
        assert T3 - mid == pytest.approx(mid - T2)


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

class TestVelocity:
    def test_midtransit_zero_rv(self):
        """At phase=0 (mid-transit), circular-orbit planet RV is 0 (no winds/BERV/vsys)."""
        rv = planet_radial_velocity(np.array([0.0]), kp_kms=150.0)
        assert rv[0] == pytest.approx(0.0)

    def test_quadrature_is_kp(self):
        """At phase=0.25 (quadrature), RV = Kp."""
        rv = planet_radial_velocity(np.array([0.25]), kp_kms=150.0)
        assert rv[0] == pytest.approx(150.0)

    def test_vsys_offset(self):
        rv = planet_radial_velocity(np.array([0.0]), kp_kms=150.0, systemic_velocity_kms=10.0)
        assert rv[0] == pytest.approx(10.0)

    def test_stellar_rv_opposite_sign(self):
        """Stellar reflex velocity is opposite to planet at quadrature."""
        rv_pl = planet_radial_velocity(np.array([0.25]), kp_kms=150.0)
        rv_st = stellar_radial_velocity(np.array([0.25]), kstar_kms=1.0)
        assert np.sign(rv_pl[0]) != np.sign(rv_st[0])


# ---------------------------------------------------------------------------
# Airmass
# ---------------------------------------------------------------------------

class TestAirmass:
    def setup_method(self):
        self.jd = np.linspace(0.0, 1.0, 50)

    def test_up_and_down_min_at_midpoint(self):
        am = synthetic_airmass(self.jd, airmass_min=1.0, airmass_max=2.5, pattern="up_and_down")
        mid_idx = len(am) // 2
        assert am[mid_idx] == pytest.approx(1.0, abs=0.05)

    def test_up_pattern_decreasing(self):
        am = synthetic_airmass(self.jd, airmass_min=1.0, airmass_max=2.5, pattern="up")
        assert am[0] >= am[-1]

    def test_down_pattern_increasing(self):
        am = synthetic_airmass(self.jd, airmass_min=1.0, airmass_max=2.5, pattern="down")
        assert am[-1] >= am[0]

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="pattern must be one of"):
            synthetic_airmass(self.jd, 1.0, 2.0, pattern="sideways")

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="airmass_min"):
            synthetic_airmass(self.jd, airmass_min=3.0, airmass_max=1.0)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

class TestNoise:
    def test_photon_noise_shape(self):
        snr = 100.0 * np.ones(50)
        noise = photon_noise(snr, noise_scaling_factor=1.0)
        assert noise.shape == (50,)

    def test_photon_noise_scaling(self):
        snr = 100.0 * np.ones(500)
        noise_1x = photon_noise(snr, noise_scaling_factor=1.0, rng=np.random.default_rng(0))
        noise_2x = photon_noise(snr, noise_scaling_factor=2.0, rng=np.random.default_rng(0))
        assert np.std(noise_2x) > np.std(noise_1x)

    def test_snr_per_pixel_positive(self):
        flux = 1e-13 * np.ones(30)  # erg/s/cm²/Å
        snr = snr_per_pixel(flux, exposure_time_seconds=300.0, telescope_area_cm2=1e6)
        assert np.all(snr > 0)


# ---------------------------------------------------------------------------
# PWV generation
# ---------------------------------------------------------------------------

class TestPwvGenSkycalc:
    """Tests for pwv_gen_skycalc."""

    def setup_method(self):
        from exoplore.observation.airmass import pwv_gen_skycalc
        self.fn = pwv_gen_skycalc

    def test_output_length(self):
        pwv = self.fn(n_spectra=20)
        assert len(pwv) == 20

    def test_output_dtype(self):
        pwv = self.fn(n_spectra=5)
        assert pwv.dtype == np.float64

    def test_all_values_in_skycalc_grid(self):
        """Every returned value must belong to the Skycalc-accepted PWV grid."""
        grid = {0.05, 0.01, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0}
        pwv = self.fn(n_spectra=50)
        for val in pwv:
            assert val in grid, f"{val} not in Skycalc PWV grid"

    def test_ref_pwv_respected(self):
        """With ref_pwv=1.0, values should come from {0.5, 1.0, 1.5}."""
        allowed = {0.5, 1.0, 1.5}
        import random as _random
        _random.seed(0)
        pwv = self.fn(n_spectra=100, ref_pwv=1.0)
        for val in pwv:
            assert val in allowed, f"Unexpected PWV {val} for ref=1.0"

    def test_no_ref_pwv_random(self):
        """Without ref_pwv, function should run without error and return an array."""
        import random as _random
        _random.seed(7)
        pwv = self.fn(n_spectra=10, ref_pwv=None)
        assert len(pwv) == 10


# ---------------------------------------------------------------------------
# dayside_fraction  (v0.23)
# ---------------------------------------------------------------------------

class TestDaysideFraction:
    """Tests for exoplore.observation.timing.dayside_fraction."""

    def setup_method(self):
        from exoplore.observation.timing import dayside_fraction
        self.fn = dayside_fraction

    def test_shape(self):
        syn_jd = np.linspace(0, 1, 20)
        without_signal = np.arange(8, 13)
        result = self.fn(syn_jd, without_signal)
        assert result.shape == syn_jd.shape

    def test_transit_set_to_zero(self):
        syn_jd = np.linspace(0, 1, 20)
        without_signal = np.arange(8, 13)
        result = self.fn(syn_jd, without_signal)
        np.testing.assert_array_equal(result[without_signal], 0.0)

    def test_pre_transit_range(self):
        syn_jd = np.linspace(0, 1, 20)
        without_signal = np.arange(8, 13)
        result = self.fn(syn_jd, without_signal)
        # values before transit run from 0.5 to 1.0
        assert result[0] == pytest.approx(0.5, abs=1e-6)
        assert result[without_signal[0] - 1] == pytest.approx(1.0, abs=1e-6)

    def test_post_transit_range(self):
        syn_jd = np.linspace(0, 1, 20)
        without_signal = np.arange(8, 13)
        result = self.fn(syn_jd, without_signal)
        # values after transit run from 1.0 down to 0.65
        assert result[without_signal[-1] + 1] == pytest.approx(1.0, abs=1e-6)
        assert result[-1] == pytest.approx(0.65, abs=1e-6)


# ---------------------------------------------------------------------------
# UTC_to_TDB_CARMENES  (v0.23), smoke test only (requires network)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires astropy site lookup (network or cache)")
class TestUTCtoTDB:
    def test_returns_array(self):
        from exoplore.observation.timing import UTC_to_TDB_CARMENES
        inp_dat = {"RA": 268.03, "Dec": 22.09}
        utc = np.array([2459000.5, 2459000.6])
        result = UTC_to_TDB_CARMENES(inp_dat, utc)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# add_throughput  (v0.23)
# ---------------------------------------------------------------------------

class TestAddThroughput:
    """Tests for exoplore.observation.noise.add_throughput."""

    def setup_method(self):
        from exoplore.observation.noise import add_throughput
        self.fn = add_throughput

    def test_shape_preserved(self):
        F = np.ones((10, 50))
        result = self.fn(F, seed=0)
        assert result.shape == F.shape

    def test_white_mode_mean_near_unity(self):
        np.random.seed(1)
        F = np.ones((1000, 10))
        result = self.fn(F, jitter_frac=0.02, mode="white", seed=1)
        # mean throughput across exposures should be close to 1.0
        assert abs(result[:, 0].mean() - 1.0) < 0.05

    def test_red_mode_runs(self):
        F = np.ones((20, 50))
        result = self.fn(F, mode="red", red_smooth_sigma=2.0, seed=42)
        assert result.shape == F.shape

    def test_invalid_mode_raises(self):
        F = np.ones((5, 10))
        with pytest.raises(ValueError, match="mode must be"):
            self.fn(F, mode="green")

    def test_seed_reproducible(self):
        F = np.ones((10, 20))
        r1 = self.fn(F, seed=7)
        r2 = self.fn(F, seed=7)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        F = np.ones((10, 20))
        r1 = self.fn(F, seed=1)
        r2 = self.fn(F, seed=2)
        assert not np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# block_parameter  (v0.24)
# ---------------------------------------------------------------------------

class TestBlockParameter:
    """Tests for exoplore.observation.timing.block_parameter."""

    def setup_method(self):
        try:
            import batman  # noqa: F401
            self._batman_ok = True
        except ImportError:
            self._batman_ok = False

    def test_returns_array(self):
        if not self._batman_ok:
            pytest.skip("batman not installed")
        from exoplore.observation.timing import block_parameter
        JD = np.linspace(2450000.0, 2450000.2, 50)
        block = block_parameter(
            JD, T_0=2450000.1, P=1.0, R_P=1.0, a=10.0, R_s=1.0,
            i=90.0, uu=[0.1, 0.3]
        )
        assert block.shape == JD.shape

    def test_max_is_unity(self):
        if not self._batman_ok:
            pytest.skip("batman not installed")
        from exoplore.observation.timing import block_parameter
        JD = np.linspace(2450000.0, 2450000.2, 200)
        block = block_parameter(
            JD, T_0=2450000.1, P=1.0, R_P=1.0, a=10.0, R_s=1.0,
            i=90.0, uu=[0.1, 0.3]
        )
        assert abs(block.max() - 1.0) < 1e-6

    def test_nonnegative(self):
        if not self._batman_ok:
            pytest.skip("batman not installed")
        from exoplore.observation.timing import block_parameter
        JD = np.linspace(2450000.0, 2450000.2, 100)
        block = block_parameter(
            JD, T_0=2450000.1, P=1.0, R_P=1.0, a=10.0, R_s=1.0,
            i=90.0, uu=[0.1, 0.3]
        )
        assert np.all(block >= 0.0)
