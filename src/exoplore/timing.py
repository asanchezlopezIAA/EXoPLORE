"""
exoplore.observation.timing
============================

Orbital phase and in-transit timing utilities.

All functions work with Julian Dates (JD or BJD) and orbital periods
in days.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def orbital_phase(
    julian_dates: np.ndarray,
    transit_epoch_bjd: float,
    orbital_period_days: float,
) -> np.ndarray:
    """Compute orbital phase relative to mid-transit.

    Phase is defined so that ``φ = 0`` at mid-transit, and runs in
    ``[-0.5, 0.5)`` for a full orbit.

    Parameters
    ----------
    julian_dates:
        Array of observation times in BJD (or JD), shape ``(n,)``.
    transit_epoch_bjd:
        Reference mid-transit time in BJD.
    orbital_period_days:
        Orbital period in days.

    Returns
    -------
    np.ndarray
        Orbital phase array, shape ``(n,)``.
    """
    raw = (np.asarray(julian_dates) - transit_epoch_bjd) / orbital_period_days
    # Wrap to [-0.5, 0.5)
    return raw - np.floor(raw + 0.5)


def in_transit_indices(
    phases: np.ndarray,
    transit_duration_hours: float,
    orbital_period_days: float,
) -> np.ndarray:
    """Return integer indices of in-transit frames.

    Parameters
    ----------
    phases:
        Orbital phase array (output of :func:`orbital_phase`).
    transit_duration_hours:
        Full transit duration (T14) in hours.
    orbital_period_days:
        Orbital period in days.

    Returns
    -------
    np.ndarray
        Integer array of in-transit frame indices.
    """
    half_duration = (transit_duration_hours / 24.0) / orbital_period_days / 2.0
    return np.where(np.abs(phases) <= half_duration)[0]


def observation_julian_dates(
    transit_epoch_bjd: float,
    exposure_time_seconds: float,
    readout_time_seconds: float,
    overhead_time_seconds: float,
    n_exposures: int,
    start_offset_hours: float = 0.0,
) -> np.ndarray:
    """Build a uniform JD grid for a sequence of exposures.

    Parameters
    ----------
    transit_epoch_bjd:
        Mid-transit time in BJD (used as reference; the grid starts at
        ``transit_epoch_bjd - start_offset_hours / 24``).
    exposure_time_seconds:
        Duration of each exposure in seconds.
    readout_time_seconds:
        CCD readout time in seconds.
    overhead_time_seconds:
        Per-exposure overhead (telescope moves, etc.) in seconds.
    n_exposures:
        Number of exposures.
    start_offset_hours:
        How many hours before mid-transit to start observing.

    Returns
    -------
    np.ndarray
        Array of mid-exposure BJD times, shape ``(n_exposures,)``.
    """
    dt_days = (
        exposure_time_seconds + readout_time_seconds + overhead_time_seconds
    ) / 86400.0  # seconds → days

    t_start = transit_epoch_bjd - start_offset_hours / 24.0
    # Mid-exposure times
    return t_start + (np.arange(n_exposures) + 0.5) * dt_days


def transit_contact_times(
    transit_epoch_bjd: float,
    transit_duration_hours: float,
) -> Tuple[float, float, float, float]:
    """Return the four contact times T1, T4 for a full transit.

    T1 = first external contact (ingress start)
    T2 = second contact (ingress end / full transit start)
    T3 = third contact (egress start / full transit end)
    T4 = fourth external contact (egress end)

    This simplified version assumes equal ingress/egress duration
    (circular orbit, no limb-darkening correction).

    Parameters
    ----------
    transit_epoch_bjd:
        Mid-transit BJD.
    transit_duration_hours:
        Total transit duration T14 in hours.

    Returns
    -------
    T1, T2, T3, T4:
        Four contact times in BJD.  T2, T3 (full transit) is assumed
        to be 80 % of T14 (rough approximation).
    """
    half14 = transit_duration_hours / 24.0 / 2.0
    half23 = 0.8 * half14  # approximate inner duration

    T1 = transit_epoch_bjd - half14
    T2 = transit_epoch_bjd - half23
    T3 = transit_epoch_bjd + half23
    T4 = transit_epoch_bjd + half14
    return T1, T2, T3, T4
