#!/usr/bin/env python3
"""
make_planet_params.py
=====================

Template script for generating a planet parameter JSON file for EXoPLORE.

Edit the values in the "USER INPUT" section below, then run:

    python scripts/make_planet_params.py

The script computes derived quantities (Kp, T_duration) from the inputs
and writes a ready-to-use JSON to planet_params/<name>.json.

Units expected here:
    Masses       : Jupiter masses (M_jup) and Solar masses (M_sun)
    Radii        : Jupiter radii (R_jup) and Solar radii (R_sun)
    Period       : days
    a            : AU
    incl         : degrees
    V_sys, K_s   : km/s
    T_0          : BJD
    RA, Dec      : degrees
    Temperatures : K
"""

import json
import math
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AU_cm        = 1.496e13          # 1 AU in cm
R_sun_cm     = 6.957e10          # 1 R_sun in cm
R_jup_cm     = 7.1492e9          # 1 R_jup in cm
M_sun_g      = 1.989e33          # 1 M_sun in g
M_jup_g      = 1.898e30          # 1 M_jup in g
DAY_s        = 86400.0           # 1 day in seconds


# ---------------------------------------------------------------------------
# USER INPUT, edit everything in this block
# ---------------------------------------------------------------------------

name             = "WASP-127b"

# Orbital
period_days      = 4.17806203       # orbital period [days]
T_0_bjd          = 2456776.62124    # mid-transit epoch [BJD]
a_au             = 0.0495           # semi-major axis [AU]
incl_deg         = 87.84            # orbital inclination [degrees]
eccentricity     = 0.0              # orbital eccentricity
arg_periastron_deg = 0.0            # argument of periastron [degrees]
long_periastron_deg = 90.0          # longitude of periastron [degrees]

# Planet
R_pl_rjup        = 1.311            # planet radius [R_jup]
M_pl_mjup        = 0.1647           # planet mass [M_jup]

# Star
R_star_rsun      = 1.33             # stellar radius [R_sun]
M_star_msun      = 0.949            # stellar mass [M_sun]
T_star_K         = 5828.0           # stellar effective temperature [K]
stellar_logg     = 4.18             # stellar log g [cgs], stored but not yet used in computations; set to None if unknown
stellar_metallicity = 0.0           # stellar [Fe/H], stored but not yet used in computations; set to None if unknown
v_rotsini_kms    = None             # stellar v sin i [km/s]; None if unknown

# Velocities
V_sys_kms        = -8.25            # systemic velocity [km/s]
K_s_kms          = 0.022            # stellar RV semi-amplitude [km/s]
# Kp will be computed automatically from orbital geometry below

# Atmosphere / thermal
T_equ_K          = 1400.0           # equilibrium temperature [K]
T_int_K          = 200.0            # internal temperature (Guillot profile) [K]
kappa_ir         = 0.01             # IR opacity (Guillot profile)
gamma_guillot    = 0.4              # gamma (Guillot profile)

# Limb darkening (quadratic batman [u1, u2])
limb_darkening_coeffs = [0.35, 0.25]

# Sky coordinates
RA_deg           = 160.55868204     # right ascension [degrees]
Dec_deg          = -3.83507226      # declination [degrees]

# References (optional, good practice to record them)
references = [
    "Lam et al. 2017, ...",         # replace with real references
]

# ---------------------------------------------------------------------------
# Output path, writes to planet_params/ relative to repo root
# ---------------------------------------------------------------------------
output_dir  = os.path.join(os.path.dirname(__file__), "..", "planet_params")
output_file = os.path.join(output_dir, f"{name.replace(' ', '-')}.json")


# ---------------------------------------------------------------------------
# Derived quantities (do not edit)
# ---------------------------------------------------------------------------

# Convert to SI/CGS for intermediate calculations
a_cm     = a_au * AU_cm
R_pl_cm  = R_pl_rjup * R_jup_cm
R_star_cm = R_star_rsun * R_sun_cm
incl_rad = math.radians(incl_deg)
period_s = period_days * DAY_s

# Kp from circular orbit geometry
Kp_kms = (
    (2.0 * math.pi * a_cm)
    / (period_s * math.sqrt(1.0 - eccentricity**2))
    * math.sin(incl_rad)
    / 1e5   # cm/s → km/s
)

# Transit duration (first to fourth contact)
b = (a_cm / R_star_cm) * math.cos(incl_rad)   # impact parameter [R_star]
num = math.sqrt(max((1.0 + R_pl_cm / R_star_cm)**2 - b**2, 0.0))
den = (a_cm / R_star_cm) * math.sin(incl_rad)
T_duration_days = (period_days / math.pi) * math.asin(num / den)

print(f"Derived Kp         : {Kp_kms:.4f} km/s")
print(f"Derived T_duration : {T_duration_days * 24:.4f} hours")


# ---------------------------------------------------------------------------
# Build JSON
# ---------------------------------------------------------------------------

planet = {
    "_references": references,

    "name": name,

    "_section_orbital": "--- Orbital parameters ---",
    "orbital_period_days":        period_days,
    "transit_epoch_bjd":          T_0_bjd,
    "semi_major_axis_au":         a_au,
    "inclination_deg":            incl_deg,
    "eccentricity":               eccentricity,
    "argument_of_periastron_deg": arg_periastron_deg,
    "longitude_of_periastron_deg": long_periastron_deg,

    "_section_planet": "--- Planet parameters ---",
    "planet_radius_rjup":  R_pl_rjup,
    "planet_mass_mjup":    M_pl_mjup,

    "_section_star": "--- Stellar parameters ---",
    "stellar_radius_rsun":    R_star_rsun,
    "stellar_mass_msun":      M_star_msun,
    "stellar_teff_K":         T_star_K,
    "stellar_logg":           stellar_logg,
    "stellar_metallicity":    stellar_metallicity,
    "v_rotsini_kms":          v_rotsini_kms,

    "_section_velocities": "--- Velocities ---",
    "systemic_velocity_kms":          V_sys_kms,
    "stellar_rv_semiamplitude_kms":   K_s_kms,
    "kp_kms":                         Kp_kms,

    "_section_atmosphere": "--- Atmospheric / thermal ---",
    "equilibrium_temperature_K": T_equ_K,
    "t_int_K":                   T_int_K,
    "kappa_ir":                  kappa_ir,
    "gamma_guillot":             gamma_guillot,

    "_section_limbdark": "--- Limb darkening (quadratic [u1, u2] for batman) ---",
    "limb_darkening_coeffs": limb_darkening_coeffs,

    "_section_coords": "--- Sky coordinates (degrees) ---",
    "ra_deg":  RA_deg,
    "dec_deg": Dec_deg,
}

os.makedirs(output_dir, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(planet, f, indent=2)

print(f"\nWritten to: {output_file}")
