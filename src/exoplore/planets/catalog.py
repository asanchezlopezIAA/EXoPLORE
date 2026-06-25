"""
exoplore.planets.catalog
========================

Load a :class:`~exoplore.planets.models.PlanetParameters` object from a
JSON file on disk.

Two JSON formats are supported transparently:

**New format** (recommended)
    Uses the same field names as :class:`PlanetParameters`, i.e. with
    explicit units in the key names (``orbital_period_days``,
    ``planet_radius_rjup``, etc.).  This is what the bundled
    ``planet_params/`` files use.  Auto-detected when the key
    ``"orbital_period_days"`` is present.

**SI-unit (inp_dat) format**
    Uses the short key names from the original ``inp_dat`` dict
    (``Period``, ``R_pl`` in metres, ``incl`` in radians, etc.).
    Auto-detected when the key ``"Period"`` is present.
    Existing planet JSON files continue to work with no changes.

Example
-------
>>> from exoplore.planets import load_planet
>>> planet = load_planet("planet_params/HD189733b.json")
>>> print(f"Kp = {planet.kp_kms:.1f} km/s")
Kp = 152.8 km/s
>>> print(f"T₁₄ = {planet.transit_duration_hours:.2f} h")
T₁₄ = 1.83 h
"""

from __future__ import annotations

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


def load_planet(path: str | Path) -> PlanetParameters:
    """Load planet parameters from a JSON file.

    Supports both the new clean format (field names with units) and the
    SI-unit format (short keys from ``inp_dat``).

    Parameters
    ----------
    path : str or Path
        Path to the planet JSON file.

    Returns
    -------
    PlanetParameters
        Fully populated object; derived quantities (Kp, transit duration,
        K_s) are computed automatically if not present in the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Planet parameter file not found: {path}")

    with open(path) as fh:
        raw = json.load(fh)

    # Auto-detect format
    if "orbital_period_days" in raw:
        return _load_new_format(raw, path.stem)
    else:
        return _load_si_unit_format(raw, path.stem)


# ─── New format loader ────────────────────────────────────────────────────────

def _load_new_format(raw: dict, stem: str) -> PlanetParameters:
    """Load from the clean, unit-annotated JSON format."""

    def _get(key, default=None):
        return raw.get(key, default)

    def _getf(key, default=0.0):
        v = raw.get(key, default)
        return float(v) if v is not None else default

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

    ld = raw.get("limb_darkening_coeffs", [0.3, 0.1])

    ra = raw.get("ra_deg", None)
    dec = raw.get("dec_deg", None)

    return PlanetParameters(
        name=name,
        # Orbital
        orbital_period_days=_getf("orbital_period_days"),
        transit_epoch_bjd=_getf("transit_epoch_bjd"),
        semi_major_axis_au=_getf("semi_major_axis_au"),
        inclination_deg=_getf("inclination_deg", 90.0),
        eccentricity=_getf("eccentricity", 0.0),
        argument_of_periastron_deg=_getf("argument_of_periastron_deg", 0.0),
        # Planet
        planet_radius_rjup=_getf("planet_radius_rjup"),
        planet_mass_mjup=_getf("planet_mass_mjup"),
        # Star
        stellar_radius_rsun=_getf("stellar_radius_rsun"),
        stellar_mass_msun=_getf("stellar_mass_msun"),
        stellar_teff_K=_getf("stellar_teff_K", 5778.0),
        stellar_logg=_getf("stellar_logg", 4.44),
        stellar_metallicity=_getf("stellar_metallicity", 0.0),
        v_rotsini_kms=_getf("v_rotsini_kms", 0.0),
        # Velocities
        systemic_velocity_kms=_getf("systemic_velocity_kms", 0.0),
        kp_kms=kp,
        stellar_rv_semiamplitude_kms=ks,
        # Atmosphere
        equilibrium_temperature_K=_getf("equilibrium_temperature_K"),
        t_int_K=_getf("t_int_K", 200.0),
        kappa_ir=_getf("kappa_ir", 0.01),
        gamma_guillot=_getf("gamma_guillot", 0.4),
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
    """Load from the inp_dat SI-unit JSON format."""

    name = raw.get("Exoplanet_name", stem)

    # ── Orbital ──
    period = float(raw.get("Period", 0.0))
    epoch = float(raw.get("T_0", 0.0))

    # Semi-major axis: this format stores metres (e.g. 1.496e8 * 0.03099)
    a_raw = float(raw.get("a", 0.0))
    a_au = a_raw / _AU_m if a_raw > 1e6 else a_raw   # heuristic: >1e6 ⇒ metres

    # Inclination: this format stores radians
    incl_raw = float(raw.get("incl", math.pi / 2))
    incl_deg = math.degrees(incl_raw) if incl_raw <= 2 * math.pi else incl_raw

    ecc = float(raw.get("eccentricity", 0.0))
    omega = float(raw.get("long_periastron_w", 0.0))

    # ── Sizes ──
    R_pl_raw = float(raw.get("R_pl", 0.0))
    R_pl_rjup = R_pl_raw / _R_jup_m if R_pl_raw > 1e4 else R_pl_raw

    R_s_raw = float(raw.get("R_star", 0.0))
    R_s_rsun = R_s_raw / _R_sun_m if R_s_raw > 1e4 else R_s_raw

    M_pl_raw = float(raw.get("M_pl", 0.0))
    M_pl_mjup = M_pl_raw / _M_jup_kg if M_pl_raw > 1e20 else M_pl_raw

    M_s_raw = float(raw.get("M_star", 0.0))
    M_s_msun = M_s_raw / _M_sun_kg if M_s_raw > 1e25 else M_s_raw

    # ── Stellar ──
    t_star = float(raw.get("T_star", 5778.0))
    logg = float(raw.get("logg", 4.44))
    met = float(raw.get("met", 0.0))
    v_rot = float(raw.get("v_rotsini", 0.0))

    # ── Velocities ──
    v_sys = float(raw.get("V_sys", 0.0))
    kp = raw.get("K_p", None)
    if kp is not None:
        kp = float(kp)

    # ── Atmosphere ──
    t_equ = float(raw.get("T_equ", 0.0))
    kappa_ir = float(raw.get("Kappa_IR", 0.01))
    gamma = float(raw.get("Gamma", 0.4))
    t_int = float(raw.get("T_int", 200.0))

    # ── Limb darkening ──
    ld = raw.get("limb_darkening_coeffs", [0.3, 0.1])

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
