"""Tests for exoplore.planets, PlanetParameters and catalog loading."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from exoplore.planets.models import PlanetParameters
from exoplore.planets.catalog import load_planet


# ---------------------------------------------------------------------------
# PlanetParameters model
# ---------------------------------------------------------------------------

def test_planet_kp_computed():
    """Kp should be computed from orbital parameters when not supplied."""
    planet = PlanetParameters(
        name="test",
        orbital_period_days=2.218575,
        semi_major_axis_au=0.03099,
        inclination_deg=85.71,
        eccentricity=0.0,
    )
    assert planet.kp_kms is not None
    assert planet.kp_kms > 0


def test_planet_transit_duration_computed():
    """Transit duration should be computed from orbital geometry."""
    planet = PlanetParameters(
        name="test",
        orbital_period_days=2.218575,
        semi_major_axis_au=0.03099,
        inclination_deg=85.71,
        planet_radius_rjup=1.138,
        stellar_radius_rsun=0.756,
        eccentricity=0.0,
    )
    assert planet.transit_duration_hours is not None
    # HD209458b transit is ~3 hours
    assert 1.0 < planet.transit_duration_hours < 5.0


def test_planet_kp_explicit():
    """Explicit Kp should override computed value."""
    planet = PlanetParameters(
        name="test",
        orbital_period_days=3.524,
        semi_major_axis_au=0.04707,
        inclination_deg=85.68,
        kp_kms=152.5,
    )
    assert planet.kp_kms == 152.5


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def _write_si_unit_json(path: Path) -> None:
    """Write a minimal SI-unit-format planet JSON for testing."""
    data = {
        "Exoplanet_name": "TestPlanet",
        "Period": 2.21857567,
        "T_0": 2454279.436714 + 2450000.0,
        "a": 1.496e11 * 0.03099,      # in metres
        "incl": math.radians(85.71),   # in radians
        "eccentricity": 0.0,
        "long_periastron_w": 0.0,
        "R_pl": 7.1492e7 * 1.138,      # in metres
        "R_star": 6.957e8 * 0.756,     # in metres
        "M_pl": 1.898e27 * 0.714,      # in kg
        "M_star": 1.989e30 * 1.01,     # in kg
        "T_star": 6065.0,
        "logg": 4.36,
        "met": 0.02,
        "V_sys": -14.7,
        "T_equ": 1450.0,
        "T_int": 200.0,
        "Kappa_IR": 0.01,
        "Gamma": 0.4,
        "RA": 330.795,
        "Dec": 18.884,
    }
    with open(path, "w") as fh:
        json.dump(data, fh)


def test_load_planet_si_unit_format():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "TestPlanet.json"
        _write_si_unit_json(p)
        planet = load_planet(p)

    assert planet.name == "TestPlanet"
    assert abs(planet.orbital_period_days - 2.21857567) < 1e-6
    assert planet.stellar_teff_K == 6065.0
    assert planet.systemic_velocity_kms == -14.7
    assert planet.eccentricity == 0.0


def test_load_planet_kp_computed():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "TestPlanet.json"
        _write_si_unit_json(p)
        planet = load_planet(p)

    # Kp should be computed from a, incl, Period when not in JSON
    assert planet.kp_kms is not None
    assert planet.kp_kms > 100.0


def test_load_planet_missing_file():
    with pytest.raises(FileNotFoundError):
        load_planet("nonexistent/planet.json")


# ---------------------------------------------------------------------------
# New clean-format JSON loading
# ---------------------------------------------------------------------------

def _write_new_format_json(path: Path) -> None:
    """Write a minimal new-format (clean unit-annotated) planet JSON."""
    data = {
        "name": "NewFormatPlanet",
        "orbital_period_days": 2.21857567,
        "transit_epoch_bjd": 2454279.436714,
        "semi_major_axis_au": 0.03099,
        "inclination_deg": 85.710,
        "eccentricity": 0.0,
        "argument_of_periastron_deg": 90.0,
        "planet_radius_rjup": 1.138,
        "planet_mass_mjup": 1.138,
        "stellar_radius_rsun": 0.756,
        "stellar_mass_msun": 0.806,
        "stellar_teff_K": 5052.0,
        "stellar_logg": 4.587,
        "stellar_metallicity": -0.03,
        "v_rotsini_kms": 3.5,
        "systemic_velocity_kms": -2.361,
        "equilibrium_temperature_K": 1200.0,
        "t_int_K": 200.0,
        "kappa_ir": 0.01,
        "gamma_guillot": 0.4,
        "limb_darkening_coeffs": [0.3808, 0.2814],
        "ra_deg": 300.1788,
        "dec_deg": 22.7105,
    }
    with open(path, "w") as fh:
        json.dump(data, fh)


def test_load_planet_new_format():
    """New-format JSON (clean unit-annotated keys) should load correctly."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "NewFormatPlanet.json"
        _write_new_format_json(p)
        planet = load_planet(p)

    assert planet.name == "NewFormatPlanet"
    assert abs(planet.orbital_period_days - 2.21857567) < 1e-6
    assert abs(planet.inclination_deg - 85.710) < 1e-3
    assert abs(planet.semi_major_axis_au - 0.03099) < 1e-6
    assert planet.stellar_teff_K == 5052.0
    assert abs(planet.systemic_velocity_kms - (-2.361)) < 1e-6
    assert planet.limb_darkening_coeffs == [0.3808, 0.2814]
    assert planet.v_rotsini_kms == 3.5
    assert planet.ra_deg is not None
    assert planet.dec_deg is not None


def test_new_format_kp_auto_computed():
    """Kp should be auto-computed from orbital params in new format."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "NewFormatPlanet.json"
        _write_new_format_json(p)
        planet = load_planet(p)

    assert planet.kp_kms is not None
    assert planet.kp_kms > 100.0


def test_new_format_stellar_ks_computed():
    """Stellar RV semi-amplitude K_s should be auto-computed."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "NewFormatPlanet.json"
        _write_new_format_json(p)
        planet = load_planet(p)

    assert planet.stellar_rv_semiamplitude_kms is not None
    assert planet.stellar_rv_semiamplitude_kms > 0.0
    # HD189733 K_s ~ 0.21 km/s
    assert planet.stellar_rv_semiamplitude_kms < 5.0


# ---------------------------------------------------------------------------
# to_inp_dat()
# ---------------------------------------------------------------------------

def test_to_inp_dat_keys():
    """to_inp_dat() should produce all expected inp_dat keys."""
    planet = PlanetParameters(
        name="TestPlanet",
        orbital_period_days=2.218575,
        semi_major_axis_au=0.03099,
        inclination_deg=85.71,
        planet_radius_rjup=1.138,
        stellar_radius_rsun=0.756,
        planet_mass_mjup=1.138,
        stellar_mass_msun=0.806,
        equilibrium_temperature_K=1200.0,
        systemic_velocity_kms=-2.361,
    )
    d = planet.to_inp_dat()

    for key in ("Exoplanet_name", "Period", "T_0", "a", "incl",
                "eccentricity", "R_pl", "R_star", "M_pl", "M_star",
                "T_star", "V_sys", "K_p", "T_equ", "Kappa_IR",
                "Gamma", "T_int", "limb_darkening_coeffs"):
        assert key in d, f"Missing key: {key}"


def test_to_inp_dat_units():
    """to_inp_dat() should return CGS units for sizes (matching pRT convention).

    to_inp_dat() uses CGS because call_pRT computes surface gravity as
    cst.G * M_pl / R_pl**2 where cst.G = 6.674e-8 cm³ g⁻¹ s⁻².
    Semi-major axis is in km (a_au * 1.496e8 km/AU).
    """
    planet = PlanetParameters(
        name="TestPlanet",
        orbital_period_days=2.218575,
        semi_major_axis_au=0.03099,
        inclination_deg=85.71,
        planet_radius_rjup=1.0,   # exactly 1 R_Jup
        stellar_radius_rsun=1.0,  # exactly 1 R_sun
        planet_mass_mjup=1.0,
        stellar_mass_msun=1.0,
    )
    d = planet.to_inp_dat()

    # R_pl should be in cm (1 R_Jup ~ 7.15e9 cm)
    assert 7.0e9 < d["R_pl"] < 7.3e9
    # R_star should be in cm (1 R_sun ~ 6.96e10 cm)
    assert 6.8e10 < d["R_star"] < 7.1e10
    # a should be in km (0.03099 AU * 1.496e8 km/AU ~ 4.64e6 km)
    assert d["a"] > 1e6
    # incl should be in radians
    assert 0 < d["incl"] < math.pi


def test_to_inp_dat_roundtrip():
    """Loading then converting back should preserve key values."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "NewFormatPlanet.json"
        _write_new_format_json(p)
        planet = load_planet(p)

    d = planet.to_inp_dat()
    assert d["Exoplanet_name"] == "NewFormatPlanet"
    assert abs(d["Period"] - 2.21857567) < 1e-6
    assert abs(d["V_sys"] - (-2.361)) < 1e-6


# ---------------------------------------------------------------------------
# Bundled planet_params/ JSON files
# ---------------------------------------------------------------------------

def test_load_hd189733b_bundled():
    """The bundled HD189733b.json should load without errors."""
    import importlib
    import os
    # Find the repo root relative to this test file
    repo_root = Path(__file__).parent.parent
    json_path = repo_root / "planet_params" / "HD189733b.json"
    if not json_path.exists():
        pytest.skip("planet_params/HD189733b.json not found, run from repo root")

    planet = load_planet(json_path)
    assert planet.name == "HD189733b"
    assert abs(planet.orbital_period_days - 2.21857567) < 1e-4
    # stellar_teff_K uses 5400 K (the reference value), not the
    # published observational value (5052 K), to reproduce the reference results.
    assert planet.stellar_teff_K == 5400.0
    assert planet.kp_kms is not None
    assert planet.transit_duration_hours is not None
    assert planet.limb_darkening_coeffs == [0.5079, -0.2239]


def test_load_hd209458b_bundled():
    """The bundled HD209458b.json should load without errors."""
    repo_root = Path(__file__).parent.parent
    json_path = repo_root / "planet_params" / "HD209458b.json"
    if not json_path.exists():
        pytest.skip("planet_params/HD209458b.json not found, run from repo root")

    planet = load_planet(json_path)
    assert planet.name == "HD209458b"
    assert planet.stellar_teff_K == 6117.0
    assert planet.kp_kms is not None
    assert planet.transit_duration_hours is not None
