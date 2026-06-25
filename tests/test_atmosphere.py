"""Tests for exoplore.atmosphere, pressure grids and temperature profiles."""

import numpy as np
import pytest

from exoplore.atmosphere.pressure import create_log_pressure_grid
from exoplore.atmosphere.temperature import (
    isothermal_profile,
    two_point_profile,
    guillot_profile,
)


# ---------------------------------------------------------------------------
# Pressure grid
# ---------------------------------------------------------------------------

class TestPressureGrid:
    def test_shape(self):
        p = create_log_pressure_grid(1e-6, 1e2, 100)
        assert p.shape == (100,)

    def test_endpoints(self):
        p = create_log_pressure_grid(1e-6, 1e2, 100)
        assert abs(p[0] - 1e-6) < 1e-12
        assert abs(p[-1] - 1e2) < 1e-6

    def test_monotonically_increasing(self):
        p = create_log_pressure_grid(1e-6, 1e2, 100)
        assert np.all(np.diff(p) > 0)

    def test_logarithmic_spacing(self):
        p = create_log_pressure_grid(1e-6, 1e2, 100)
        log_diffs = np.diff(np.log10(p))
        assert np.allclose(log_diffs, log_diffs[0], rtol=1e-6)

    def test_bad_min_raises(self):
        with pytest.raises(ValueError):
            create_log_pressure_grid(-1.0, 1e2, 100)

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError):
            create_log_pressure_grid(1e2, 1e-6, 100)

    def test_size_too_small_raises(self):
        with pytest.raises(ValueError):
            create_log_pressure_grid(1e-6, 1e2, 1)

    def test_custom_size(self):
        p = create_log_pressure_grid(1e-3, 10.0, 50)
        assert len(p) == 50


# ---------------------------------------------------------------------------
# Temperature profiles
# ---------------------------------------------------------------------------

class TestIsothermalProfile:
    def test_uniform(self):
        p = create_log_pressure_grid(1e-6, 1e2, 100)
        T = isothermal_profile(p, 1200.0)
        assert np.all(T == 1200.0)

    def test_shape(self):
        p = create_log_pressure_grid(1e-6, 1e2, 100)
        T = isothermal_profile(p, 800.0)
        assert T.shape == p.shape


class TestTwoPointProfile:
    def setup_method(self):
        self.p = create_log_pressure_grid(1e-6, 1e2, 100)
        self.p_top = 10**(-2.75)
        self.p_bot = 10**(0.1)
        self.T_top = 520.0
        self.T_bot = 1750.0

    def test_shape(self):
        T = two_point_profile(self.p, self.p_top, self.T_top, self.p_bot, self.T_bot)
        assert T.shape == self.p.shape

    def test_clamps_at_top(self):
        T = two_point_profile(self.p, self.p_top, self.T_top, self.p_bot, self.T_bot)
        # Pressures below p_top should be clamped to T_top
        assert T[0] == pytest.approx(self.T_top, abs=1.0)

    def test_clamps_at_bottom(self):
        T = two_point_profile(self.p, self.p_top, self.T_top, self.p_bot, self.T_bot)
        # Pressures above p_bot should be clamped to T_bot
        assert T[-1] == pytest.approx(self.T_bot, abs=1.0)

    def test_monotone_increasing(self):
        T = two_point_profile(self.p, self.p_top, self.T_top, self.p_bot, self.T_bot)
        # Temperature should increase with pressure (T_top < T_bot)
        interior = T[1:-1]
        assert np.all(np.diff(interior) >= 0)


class TestGuillotProfile:
    def setup_method(self):
        self.p = create_log_pressure_grid(1e-6, 1e2, 100)

    def test_shape(self):
        T = guillot_profile(
            self.p,
            equilibrium_temperature_K=1200.0,
            t_int_K=200.0,
            stellar_gravity_cgs=1e3,
            kappa_ir=0.01,
            gamma=0.4,
        )
        assert T.shape == self.p.shape

    def test_positive_temperatures(self):
        T = guillot_profile(
            self.p,
            equilibrium_temperature_K=1200.0,
            t_int_K=200.0,
            stellar_gravity_cgs=1e3,
            kappa_ir=0.01,
            gamma=0.4,
        )
        assert np.all(T > 0)
