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
