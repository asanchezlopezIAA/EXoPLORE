"""
Tests for the new atmosphere modules: chemistry.py and prt.py.

petitRADTRANS and easychem are optional dependencies; tests that
require them are skipped when the packages are not installed.
"""

import numpy as np
import pytest

from exoplore.atmosphere.chemistry import manual_mass_fractions
from exoplore.atmosphere.prt import convolve, surface_gravity_cgs


# ---------------------------------------------------------------------------
# chemistry.py, manual_mass_fractions (no easychem required)
# ---------------------------------------------------------------------------

class TestManualMassFractions:
    def test_output_shapes(self):
        mf, mmw = manual_mass_fractions(
            species=["H2O", "CO"],
            vmr=[1e-4, 1e-3],
            mean_molecular_weight=2.3,
            n_layers=50,
        )
        assert set(mf.keys()) == {"H2O", "CO"}
        assert mf["H2O"].shape == (50,)
        assert mmw.shape == (50,)

    def test_vmr_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"len\(vmr\)"):
            manual_mass_fractions(["H2O", "CO"], [1e-4], 2.3, 50)

    def test_constant_values(self):
        mf, mmw = manual_mass_fractions(["H2O"], [5e-4], 2.35, 30)
        np.testing.assert_allclose(mf["H2O"], 5e-4)
        np.testing.assert_allclose(mmw, 2.35)


# ---------------------------------------------------------------------------
# chemistry.py, compute_equilibrium_chemistry (easychem optional)
# ---------------------------------------------------------------------------

class TestEquilibriumChemistry:
    def test_import_error_without_easychem(self, monkeypatch):
        """Should raise ImportError with helpful message if easychem absent."""
        import sys
        # Temporarily hide easychem from imports
        monkeypatch.setitem(sys.modules, "easychem", None)
        monkeypatch.setitem(sys.modules, "easychem.easychem", None)
        from exoplore.atmosphere.chemistry import compute_equilibrium_chemistry
        pressures = np.logspace(-6, 2, 20)
        temps = 1200 * np.ones(20)
        with pytest.raises(ImportError, match="easychem"):
            compute_equilibrium_chemistry(pressures, temps, ["H2O"])

    def test_shape_mismatch_raises(self):
        from exoplore.atmosphere.chemistry import compute_equilibrium_chemistry
        p = np.logspace(-6, 2, 20)
        t = np.ones(25)  # wrong length
        with pytest.raises(ValueError, match="same shape"):
            compute_equilibrium_chemistry(p, t, ["H2O"])


# ---------------------------------------------------------------------------
# prt.py, convolve (single instrument-LSF convolution; no pRT required)
# ---------------------------------------------------------------------------

class TestConvolve:
    def setup_method(self):
        self.wave = np.linspace(1.0, 2.5, 200)
        self.spec = np.ones(200)
        self.spec[100] = 10.0  # sharp spike

    def test_output_shape(self):
        out = convolve(self.wave, self.spec, 100000)
        assert out.shape == self.spec.shape

    def test_convolution_broadens_spike(self):
        # Use low R so the LSF FWHM (wave_mean/R) spans several pixels
        out = convolve(self.wave, self.spec, 50)
        # After convolution the peak should be lower and power spread to neighbours
        assert out[100] < 10.0
        assert out[99] > 1.0 or out[101] > 1.0

    def test_lsf_width_in_pixels(self):
        # A delta spike convolved to resolving power R should have an LSF whose
        # FWHM equals c/R, i.e. (wave.mean()/R) in wavelength -> the correct
        # pixel sigma divides by (step*wave.mean()).  Check the broadened width
        # matches the resolution element (within sampling), not lambda_mean times it.
        import numpy as _np
        c = 299792.458
        R = 45000.0
        lam0, dv, n = 2.2, 0.4, 4001          # K-band, where the old bug was ~2.2x
        v = (_np.arange(n) - n // 2) * dv
        wave = lam0 * _np.exp(v / c)
        spec = _np.ones(n); spec[n // 2] -= 1.0
        out = convolve(wave, spec, R)
        dip = 1.0 - out; dip /= dip.max()
        idx = _np.where(dip >= 0.5)[0]
        fwhm = (idx[-1] - idx[0]) * dv
        assert abs(fwhm - c / R) < 2.0   # ~6.7 km/s, not ~14.7 km/s

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Dimensions for wave and spec"):
            convolve(self.wave, np.ones(50), 100000)


# ---------------------------------------------------------------------------
# prt.py, surface_gravity_cgs
# ---------------------------------------------------------------------------

class TestSurfaceGravity:
    def test_earth_gravity(self):
        """Cross-check with Earth: M=6e24 kg, R=6.371e8 cm → g ≈ 981 cm/s²."""
        g = surface_gravity_cgs(
            planet_mass_kg=5.972e24,
            planet_radius_cm=6.371e8,
        )
        assert g == pytest.approx(981.0, rel=0.05)

    def test_jupiter_gravity(self):
        """Jupiter: M≈1.898e27 kg, R≈7.15e9 cm → g≈2479 cm/s²."""
        g = surface_gravity_cgs(
            planet_mass_kg=1.898e27,
            planet_radius_cm=7.149e9,
        )
        assert g == pytest.approx(2479.0, rel=0.10)
