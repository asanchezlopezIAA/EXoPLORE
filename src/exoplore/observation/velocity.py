"""
exoplore.observation.velocity
==============================

Radial velocity calculations for planet and host star.

All velocities are in **km/s**.

For circular orbits use :func:`planet_radial_velocity`.
For eccentric orbits use :func:`planet_radial_velocity_eccentric`
(requires the ``kepler`` package, available via pip).
"""

from __future__ import annotations

import math

import numpy as np

# Function arguments specify orbital phase and the velocity-frame
# convention used in the calculation.

def planet_radial_velocity(
    phases: np.ndarray,
    kp_kms: float,
    systemic_velocity_kms: float = 0.0,
    berv_kms: np.ndarray | float = 0.0,
    wind_velocity_kms: float = 0.0,
) -> np.ndarray:
    """Planet radial velocity for a circular orbit.

    .. math::

        V_p = K_p \\sin(2\\pi\\phi) + V_{\\rm sys} - V_{\\rm BERV} + V_{\\rm wind}

    Parameters
    ----------
    phases:
        Orbital phase array (φ = 0 at mid-transit).
    kp_kms:
        Planet radial velocity semi-amplitude Kp in km/s.
    systemic_velocity_kms:
        Systemic (barycentric) RV of the star in km/s.
    berv_kms:
        Barycentric Earth radial velocity correction in km/s.
        Can be a scalar or an array of the same length as ``phases``.
    wind_velocity_kms:
        Assumed atmospheric wind velocity in km/s.

    Returns
    -------
    np.ndarray
        Planet RV in km/s, shape matching ``phases``.
    """
    phases = np.asarray(phases, dtype=float)
    berv = np.asarray(berv_kms, dtype=float)
    return (
        kp_kms * np.sin(2.0 * math.pi * phases)
        + systemic_velocity_kms
        - berv
        + wind_velocity_kms
    )


def planet_radial_velocity_eccentric(
    phases: np.ndarray,
    kp_kms: float,
    eccentricity: float,
    argument_of_periastron_rad: float,
    systemic_velocity_kms: float = 0.0,
    berv_kms: np.ndarray | float = 0.0,
    wind_velocity_kms: float = 0.0,
) -> np.ndarray:
    """Planet radial velocity for an eccentric orbit.

    Uses the Kepler equation solver from the ``kepler`` package.

    .. math::

        V_p = K_p \\bigl[\\sin(\\nu+\\omega)
              + e\\sin\\omega\\bigr] - V_{\\rm BERV} + V_{\\rm sys} + V_{\\rm wind}

    where :math:`\\nu` is the true anomaly and :math:`\\omega` is the
    argument of periastron.

    Parameters
    ----------
    phases:
        Orbital phase array (φ = 0 at transit centre).
    kp_kms:
        Semi-amplitude Kp in km/s.
    eccentricity:
        Orbital eccentricity.
    argument_of_periastron_rad:
        Argument of periastron in radians.
    systemic_velocity_kms, berv_kms, wind_velocity_kms:
        Same as :func:`planet_radial_velocity`.

    Returns
    -------
    np.ndarray
        Planet RV in km/s.

    Raises
    ------
    ImportError
        If the ``kepler`` package is not installed.
    """
    try:
        import kepler
    except ImportError as exc:
        raise ImportError(
            "The 'kepler' package is required for eccentric RV calculations.\n"
            "Install with:  pip install kepler.py"
        ) from exc

    phases = np.asarray(phases, dtype=float)
    berv = np.asarray(berv_kms, dtype=float)
    omega = argument_of_periastron_rad

    # Mean anomaly: shift so φ=0 corresponds to transit (ν = π/2 - ω)
    M = 2.0 * math.pi * phases - omega
    _, cos_nu, sin_nu = kepler.kepler(M, eccentricity)

    bracket = (
        sin_nu * math.cos(omega)
        + cos_nu * math.sin(omega)
        + eccentricity * math.sin(omega)
    )
    return kp_kms * bracket - berv + systemic_velocity_kms + wind_velocity_kms


def stellar_radial_velocity(
    phases: np.ndarray,
    kstar_kms: float,
    systemic_velocity_kms: float = 0.0,
    berv_kms: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Stellar reflex radial velocity (circular orbit).

    .. math::

        V_* = -K_* \\sin(2\\pi\\phi) + V_{\\rm sys} - V_{\\rm BERV}

    Parameters
    ----------
    phases:
        Orbital phase array.
    kstar_kms:
        Stellar RV semi-amplitude K* in km/s.
    systemic_velocity_kms:
        Systemic velocity in km/s.
    berv_kms:
        BERV correction in km/s.

    Returns
    -------
    np.ndarray
        Stellar RV in km/s.
    """
    phases = np.asarray(phases, dtype=float)
    berv = np.asarray(berv_kms, dtype=float)
    return (
        -kstar_kms * np.sin(2.0 * math.pi * phases)
        + systemic_velocity_kms
        - berv
    )


# ---------------------------------------------------------------------------
# Configuration-facing velocity functions
# get_V_eccentric takes omega in degrees and anchors the orbital phase at
# transit. Each function documents its angle units and velocity convention.
# ---------------------------------------------------------------------------

def get_V(sem_amp, phase, berv, v_sys, v_wind):
    """Planet radial velocity wrt Earth (circular orbit).

    Parameters
    ----------
    sem_amp : float or array  Planet RV semi-amplitude Kp (km/s).
    phase : array  Orbital phase (φ = 0 at transit mid).
    berv : float  Barycentric Earth RV correction (km/s).
    v_sys : float  Systemic velocity (km/s).
    v_wind : float  Atmospheric wind velocity (km/s).

    Returns
    -------
    float or numpy.ndarray
    """
    return sem_amp * np.sin(2.0 * np.pi * phase) + v_sys - berv + v_wind


def get_V_eccentric(Kp, phases, e, omega, berv, v_sys, v_wind=0.0):
    """Planet radial velocity wrt Earth (eccentric orbit).

    Parameters
    ----------
    Kp : float  Semi-amplitude (km/s).
    phases : array  Orbital phases (phi = 0 at mid-transit).
    e : float  Eccentricity.
    omega : float  Argument of periastron (degrees).
    berv : float  BERV correction (km/s).
    v_sys : float  Systemic velocity (km/s).
    v_wind : float  Wind velocity (km/s, default 0).
    """
    import kepler
    phases = np.asarray(phases)
    omega = np.deg2rad(omega)
    # Anchor phi = 0 at mid-transit. In this convention the true anomaly at
    # transit is nu_t = -omega. convert it to the mean anomaly M_t so the phase
    # offset is exact at any eccentricity (it reduces to -omega when e = 0).
    nu_t = -omega
    E_t = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu_t / 2.0),
                           np.sqrt(1.0 + e) * np.cos(nu_t / 2.0))
    M_t = E_t - e * np.sin(E_t)
    M = 2.0 * np.pi * phases + M_t
    _, cos_nu, sin_nu = kepler.kepler(M, e)
    bracket = sin_nu * np.cos(omega) + cos_nu * np.sin(omega) + e * np.sin(omega)
    return Kp * bracket - berv + v_sys + v_wind


def compute_Kp(*, semi_major_axis_km, orbital_period_days, eccentricity, inclination_rad):
    """Calculate the orbital radial-velocity amplitude in km/s.

    Supply the semi-major axis in km, orbital period in days, eccentricity
    and inclination in radians.
    """
    return (
        (2.0 * np.pi * semi_major_axis_km)
        / (orbital_period_days * 24.0 * 3600.0 * np.sqrt(1.0 - eccentricity**2))
        * np.sin(inclination_rad)
    )


def get_phase(julian_date, orbital_period_days, transit_mid_JD):
    """Calculate orbital phase relative to the supplied event midpoint.

    julian_date and transit_mid_JD share the input time convention.
    orbital_period_days is measured in days.
    """
    return (julian_date - transit_mid_JD) / orbital_period_days
