"""
exoplore.planets.catalog
========================

Load a :class:`~exoplore.planets.models.PlanetParameters` object from a
JSON file on disk.

Two JSON formats are supported transparently:

**Named-unit format** (recommended)
    Uses the same field names as :class:`PlanetParameters`, i.e. with
    explicit units in the key names (``orbital_period_days``,
    ``planet_radius_rjup``, etc.).  This is what the bundled
    ``planet_params/`` files use.  Auto-detected when the key
    ``"orbital_period_days"`` is present.

**SI-unit format**
    Uses short keys (``Period``, ``R_pl`` in metres, ``incl`` in radians).
    Auto-detected when the key ``"Period"`` is present.

Example
-------
>>> from exoplore.planets import load_planet
>>> planet = load_planet("planet_params/HD189733b.json")
>>> print(f"Kp = {planet.kp_kms:.1f} km/s")
Kp = 149.4 km/s
>>> print(f"T₁₄ = {planet.transit_duration_hours:.2f} h")
T₁₄ = 1.94 h
"""

from __future__ import annotations

# Role in the calculation
# Read planetary-system JSON files into PlanetParameters. Format-specific branches
# convert input names and units to the fields used by the atmosphere and observing
# code.


import json
import math
from pathlib import Path

from exoplore.planets.models import PlanetParameters

# ─── Unit conversion constants ────────────────────────────────────────────────
_AU_m = 1.496e11        # metres per AU
_R_jup_m = 7.1492e7    # metres per Jupiter radius
_R_sun_m = 6.957e8     # metres per solar radius
_M_jup_kg = 1.898e27   # kg per Jupiter mass
_M_sun_kg = 1.989e30   # kg per solar mass


def _required_number(raw: dict, key: str) -> float:
    """Read a finite physical value and identify missing or invalid fields."""
    if key not in raw or raw[key] is None:
        raise ValueError(f"Required planet parameter {key!r} is missing or null.")
    if isinstance(raw[key], bool):
        raise ValueError(f"Planet parameter {key!r} must be a finite number.")
    try:
        value = float(raw[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Planet parameter {key!r} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Planet parameter {key!r} must be a finite number.")
    return value


def _required_limb_darkening(raw: dict) -> list:
    """Read the supplied coefficients for the configured limb-darkening model."""
    key = "limb_darkening_coeffs"
    values = raw.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Required planet parameter {key!r} must be a list.")
    return [_required_number({key: value}, key) for value in values]


def load_planet(path: str | Path) -> PlanetParameters:
    """Load planet parameters from a JSON file.

    Supports both the new clean format (field names with units) and the
    SI-unit format with short keys.

    Parameters
    ----------
    path : str or Path
        Path to the planet JSON file.

    Returns
    -------
    PlanetParameters
        Fully populated object. derived quantities (Kp, transit duration,
        K_s) are computed automatically if not present in the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required physical parameter is absent, null or non-finite.
        Kp, stellar RV amplitude and transit duration can be calculated from
        the supplied system parameters. Sky coordinates may be omitted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Planet parameter file not found: {path}")

    with open(path) as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Planet parameter file {path} must contain a JSON object.")
    # Report the source file together with the offending physical parameter.
    try:
        if "orbital_period_days" in raw:
            return _load_new_format(raw, path.stem)
        if "Period" in raw:
            return _load_si_unit_format(raw, path.stem)
        raise ValueError("Required orbital_period_days (or SI-format Period) is missing.")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid planet parameter file {path}: {exc}") from exc


# ─── New format loader ────────────────────────────────────────────────────────

def _load_new_format(raw: dict, stem: str) -> PlanetParameters:
    """Load from the clean, unit-annotated JSON format."""

    def _get(key, default=None):
        return raw.get(key, default)

    def _getf(key):
        return _required_number(raw, key)

    name = _get("name", stem)

    kp = raw.get("kp_kms", None)
    if kp is not None:
        kp = float(kp)

    ks = raw.get("stellar_rv_semiamplitude_kms", None)
    if ks is not None:
        ks = float(ks)

    t_dur = raw.get("transit_duration_hours", None)
    if t_dur is not None:
        t_dur = float(t_dur)

    ld = _required_limb_darkening(raw)

    ra = raw.get("ra_deg", None)
    dec = raw.get("dec_deg", None)

    return PlanetParameters(
        name=name,
        # Orbital
        orbital_period_days=_getf("orbital_period_days"),
        transit_epoch_bjd=_getf("transit_epoch_bjd"),
        semi_major_axis_au=_getf("semi_major_axis_au"),
        inclination_deg=_getf("inclination_deg"),
        eccentricity=_getf("eccentricity"),
        argument_of_periastron_deg=_getf("argument_of_periastron_deg"),
        # Planet
        planet_radius_rjup=_getf("planet_radius_rjup"),
        planet_mass_mjup=_getf("planet_mass_mjup"),
        # Star
        stellar_radius_rsun=_getf("stellar_radius_rsun"),
        stellar_mass_msun=_getf("stellar_mass_msun"),
        stellar_teff_K=_getf("stellar_teff_K"),
        stellar_logg=_getf("stellar_logg"),
        stellar_metallicity=_getf("stellar_metallicity"),
        v_rotsini_kms=_getf("v_rotsini_kms"),
        # Velocities
        systemic_velocity_kms=_getf("systemic_velocity_kms"),
        kp_kms=kp,
        stellar_rv_semiamplitude_kms=ks,
        # Atmosphere
        equilibrium_temperature_K=_getf("equilibrium_temperature_K"),
        t_int_K=_getf("t_int_K"),
        kappa_ir=_getf("kappa_ir"),
        gamma_guillot=_getf("gamma_guillot"),
        # Limb darkening
        limb_darkening_coeffs=list(ld),
        # Sky
        ra_deg=float(ra) if ra is not None else None,
        dec_deg=float(dec) if dec is not None else None,
        # Derived
        transit_duration_hours=t_dur,
    )


# --- SI-unit format loader ---

def _load_si_unit_format(raw: dict, stem: str) -> PlanetParameters:
    """Load the SI-unit planet JSON format."""

    name = raw.get("Exoplanet_name", stem)

    # ── Orbital ──
    period = _required_number(raw, "Period")
    epoch = _required_number(raw, "T_0")

    # Semi-major axis: this format stores metres (e.g. 1.496e8 * 0.03099)
    a_raw = _required_number(raw, "a")
    a_au = a_raw / _AU_m if a_raw > 1e6 else a_raw   # heuristic: >1e6 ⇒ metres

    # Inclination: this format stores radians
    incl_raw = _required_number(raw, "incl")
    incl_deg = math.degrees(incl_raw) if incl_raw <= 2 * math.pi else incl_raw

    ecc = _required_number(raw, "eccentricity")
    omega = _required_number(raw, "long_periastron_w")

    # ── Sizes ──
    R_pl_raw = _required_number(raw, "R_pl")
    R_pl_rjup = R_pl_raw / _R_jup_m if R_pl_raw > 1e4 else R_pl_raw

    R_s_raw = _required_number(raw, "R_star")
    R_s_rsun = R_s_raw / _R_sun_m if R_s_raw > 1e4 else R_s_raw

    M_pl_raw = _required_number(raw, "M_pl")
    M_pl_mjup = M_pl_raw / _M_jup_kg if M_pl_raw > 1e20 else M_pl_raw

    M_s_raw = _required_number(raw, "M_star")
    M_s_msun = M_s_raw / _M_sun_kg if M_s_raw > 1e25 else M_s_raw

    # ── Stellar ──
    t_star = _required_number(raw, "T_star")
    logg = _required_number(raw, "logg")
    met = _required_number(raw, "met")
    v_rot = _required_number(raw, "v_rotsini")

    # ── Velocities ──
    v_sys = _required_number(raw, "V_sys")
    kp = raw.get("K_p", None)
    if kp is not None:
        kp = float(kp)

    # ── Atmosphere ──
    t_equ = _required_number(raw, "T_equ")
    kappa_ir = _required_number(raw, "Kappa_IR")
    gamma = _required_number(raw, "Gamma")
    t_int = _required_number(raw, "T_int")

    # ── Limb darkening ──
    ld = _required_limb_darkening(raw)

    # ── Sky coords ──
    ra = raw.get("RA", None)
    dec = raw.get("Dec", None)

    # ── Transit duration ──
    t_dur = raw.get("T_duration", None)
    if t_dur is not None:
        t_dur = float(t_dur) * 24.0   # days → hours

    return PlanetParameters(
        name=name,
        orbital_period_days=period,
        transit_epoch_bjd=epoch,
        semi_major_axis_au=a_au,
        inclination_deg=incl_deg,
        eccentricity=ecc,
        argument_of_periastron_deg=omega,
        planet_radius_rjup=R_pl_rjup,
        planet_mass_mjup=M_pl_mjup,
        stellar_radius_rsun=R_s_rsun,
        stellar_mass_msun=M_s_msun,
        stellar_teff_K=t_star,
        stellar_logg=logg,
        stellar_metallicity=met,
        v_rotsini_kms=v_rot,
        systemic_velocity_kms=v_sys,
        kp_kms=kp,
        equilibrium_temperature_K=t_equ,
        t_int_K=t_int,
        kappa_ir=kappa_ir,
        gamma_guillot=gamma,
        limb_darkening_coeffs=list(ld),
        transit_duration_hours=t_dur,
        ra_deg=float(ra) if ra is not None else None,
        dec_deg=float(dec) if dec is not None else None,
    )
