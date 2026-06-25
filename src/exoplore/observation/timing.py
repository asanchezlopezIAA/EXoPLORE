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


# ---------------------------------------------------------------------------
# get_event_v2
# ---------------------------------------------------------------------------

def get_event_v2(cfg, planet, JD_og):
    """Compute event timing directly from a config.

    Takes SimulationConfig + PlanetData instead of an inp_dat dict.
    """
    _pre = (cfg.observation.pre_event_hours
            if cfg.observation.pre_event_hours > 0
            else planet.transit_duration_hours / 2.0)
    _post = (cfg.observation.post_event_hours
             if cfg.observation.post_event_hours > 0
             else planet.transit_duration_hours / 2.0)
    _mini = {
        "event":            cfg.observation.event_type,
        "T_0":              planet.transit_epoch_bjd,
        "Period":           planet.orbital_period_days,
        "T_duration":       (planet.transit_duration_hours / 24.0
                             if planet.transit_duration_hours is not None
                             else None),
        "DIT":              cfg.observation.exposure_time_seconds,
        "readout":          cfg.observation.readout_time_seconds,
        "overheads":        cfg.observation.overhead_time_seconds,
        "flag_event":       cfg.observation.flag_event,
        "pre_event":        _pre,
        "post_event":       _post,
        "specific_event":   cfg.observation.specific_event,
        "specific_T_0":     cfg.observation.specific_T0_bjd,
        "Different_nights": cfg.observation.different_nights,
        "n_nights":         cfg.observation.n_nights,
        "RA":               planet.ra_deg,
        "Dec":              planet.dec_deg,
    }
    return get_event(_mini, JD_og)


# ---------------------------------------------------------------------------
# get_event
# ---------------------------------------------------------------------------

def get_event(inp_dat, JD_og):
    """Build synthetic JD grid and find in/out-of-event indices.

    Parameters
    ----------
    inp_dat : dict
        Full simulation input dictionary.  The following keys are used:
        ``event``, ``T_0``, ``Period``, ``T_duration``, ``DIT``,
        ``readout``, ``overheads``, ``flag_event``, ``pre_event``,
        ``post_event``, ``specific_event``, ``specific_T_0``,
        ``Different_nights``, ``n_nights``.
    JD_og : array or list of arrays
        Original Julian dates.  Used as-is when simulating specific events
        or multiple different nights.

    Returns
    -------
    syn_jd : numpy.ndarray (or list for Different_nights)
        Synthetic Julian dates of each exposure.
    with_signal : numpy.ndarray (or list)
        Indices of in-transit (or out-of-eclipse) exposures.
    without_signal : numpy.ndarray (or list)
        Indices of out-of-transit (or in-eclipse) exposures.
    transit_mid_JD : float or list
        Mid-transit (or mid-eclipse) time(s) in JD.
    """
    event = inp_dat['event']
    t_0 = inp_dat['T_0']
    period = inp_dat['Period']
    transit_duration = inp_dat['T_duration']
    DIT = inp_dat['DIT']
    readout = inp_dat['readout']
    overheads = inp_dat['overheads']
    flag_event = inp_dat['flag_event']
    pre_time = inp_dat['pre_event']
    post_time = inp_dat['post_event']

    if inp_dat["Different_nights"]:
        in_transit = []
        out_transit = []
        transit_mid_JD = []
        # Use specific_T_0 when available (specific_event=True); fall back
        # to T_0 (the reference epoch) for fully synthetic nights so that
        # the nearest transit to each night's JD array can still be found.
        _t0_ref = (inp_dat["specific_T_0"]
                   if inp_dat["specific_T_0"] is not None
                   else inp_dat["T_0"])
        for n in range(inp_dat["n_nights"]):
            transit_mid_JD.append(
                _t0_ref
                + inp_dat["Period"] * int(
                    ((JD_og[n] - _t0_ref) / inp_dat["Period"])[-1]
                )
            )
            transit_begin_JD = transit_mid_JD[n] - inp_dat["T_duration"] / 2.0
            transit_end_JD = transit_mid_JD[n] + inp_dat["T_duration"] / 2.0
            in_transit.append(
                np.where(
                    np.logical_and(JD_og[n] > transit_begin_JD,
                                   JD_og[n] < transit_end_JD)
                )[0]
            )
            out_transit.append(
                np.where(
                    np.logical_or(JD_og[n] < transit_begin_JD,
                                  JD_og[n] > transit_end_JD)
                )[0]
            )
        return JD_og, in_transit, out_transit, transit_mid_JD

    # Validate specific-event inputs
    if inp_dat['specific_T_0'] is not None and inp_dat['specific_event']:
        t_0 = inp_dat['specific_T_0']
    elif inp_dat['specific_T_0'] is not None and not inp_dat['specific_event']:
        raise ValueError(
            "You must switch inp_dat['specific_event'] to True. "
            "Or rather put inp_dat['specific_T_0'] to None."
        )
    elif inp_dat['specific_event'] and inp_dat['specific_T_0'] is None:
        raise ValueError("Please provide the T_0 of your event.")

    # Compute start/end JDs
    if event == 'transit' and not inp_dat['specific_event']:
        jd_ini = t_0 - transit_duration / 2.0 - pre_time / 24.0
        jd_fin = t_0 + transit_duration / 2.0 + post_time / 24.0
    elif event == 'dayside' and not inp_dat['specific_event']:
        eclip_mid = t_0 + period / 2.0
        if flag_event == 'full_event':
            jd_ini = eclip_mid - transit_duration / 2.0 - pre_time / 24.0
            jd_fin = eclip_mid + transit_duration / 2.0 + post_time / 24.0
        elif flag_event == 'pre':
            jd_ini = eclip_mid - transit_duration / 2.0 - pre_time / 24.0
            jd_fin = eclip_mid - transit_duration / 2.0 - 1.0 / 60.0 / 24.0
        elif flag_event == 'post':
            jd_ini = eclip_mid + transit_duration / 2.0
            jd_fin = eclip_mid + transit_duration / 2.0 + post_time / 24.0

    # Build synthetic JD grid (or use provided one)
    if not inp_dat['specific_event']:
        jd_step = (DIT + overheads + readout) / (3600.0 * 24.0)
        syn_jd = np.arange(jd_ini, jd_fin + jd_step, jd_step)
    else:
        syn_jd = JD_og

    # Find in/out-of-event indices
    if event == 'transit':
        transit_mid_JD = t_0
        transit_begin_JD = transit_mid_JD - transit_duration / 2.0
        transit_end_JD = transit_mid_JD + transit_duration / 2.0
        in_transit = np.where(
            np.logical_and(syn_jd > transit_begin_JD, syn_jd < transit_end_JD)
        )[0]
        out_transit = np.where(
            np.logical_or(syn_jd < transit_begin_JD, syn_jd > transit_end_JD)
        )[0]
        return syn_jd, in_transit, out_transit, transit_mid_JD

    elif event == 'dayside':
        eclipse_mid_JD = t_0 + period / 2.0
        eclipse_begin_JD = eclipse_mid_JD - transit_duration / 2.0
        eclipse_end_JD = eclipse_mid_JD + transit_duration / 2.0
        in_eclipse = np.where(
            np.logical_and(syn_jd > eclipse_begin_JD, syn_jd < eclipse_end_JD)
        )[0]
        out_eclipse = np.where(
            np.logical_or(syn_jd < eclipse_begin_JD, syn_jd > eclipse_end_JD)
        )[0]
        # Note: for dayside, in_eclipse is the "without_signal" (in-eclipse = no planet)
        return syn_jd, out_eclipse, in_eclipse, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_nights_with_extrema(stats, first_night_noiseless):
    """Determine the night indices with minimum and maximum detection significance.

    Parameters
    ----------
    stats : ndarray, shape (n_nights, 3)
        Statistics array where ``stats[:, 0]`` holds the detection S/N
        for each simulation night.
    first_night_noiseless : bool
        If ``True``, the first element of ``stats`` corresponds to a
        noiseless reference night and is excluded from the maximum search.

    Returns
    -------
    night_min : int
        Index of the night with the lowest S/N.
    night_max : int
        Index of the night with the highest S/N.
    """
    night_min = np.where(stats[:, 0] == stats[:, 0].min())[0][0]
    if first_night_noiseless:
        night_max = np.where(stats[1:, 0] == stats[1:, 0].max())[0][0] + 1
    else:
        night_max = np.where(stats[:, 0] == stats[:, 0].max())[0][0]
    return night_min, night_max


def get_transit(inp_dat, julian_date):
    """Return in-transit and out-of-transit frame indices from Julian dates.

    Computes the mid-transit JD nearest to the last observation, then
    selects frames inside and outside the transit window defined by
    ``inp_dat['T_duration']``.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``'T_0'`` (transit epoch
        in JD), ``'Period'`` (orbital period in days), and ``'T_duration'``
        (transit duration in days).
    julian_date : numpy.ndarray
        1-D array of Julian dates for all observed frames.

    Returns
    -------
    in_transit : numpy.ndarray of int
        Indices of frames inside the transit window.
    out_transit : numpy.ndarray of int
        Indices of frames outside the transit window.
    """
    transit_mid_JD = inp_dat['T_0'] + inp_dat['Period'] * (
        int(((julian_date - inp_dat['T_0']) / inp_dat['Period'])[-1])
    )
    transit_begin_JD = transit_mid_JD - inp_dat['T_duration'] / 2.
    transit_end_JD   = transit_mid_JD + inp_dat['T_duration'] / 2.
    in_transit  = np.where(np.logical_and(julian_date > transit_begin_JD,
                                          julian_date < transit_end_JD))[0]
    out_transit = np.where(np.logical_or(julian_date < transit_begin_JD,
                                         julian_date > transit_end_JD))[0]
    return in_transit, out_transit


# ---------------------------------------------------------------------------
# v0.23 additions
# ---------------------------------------------------------------------------

def dayside_fraction(syn_jd, without_signal):
    """
    Calculate the fraction of the exoplanet's dayside facing Earth along an orbit.

    The array increases linearly from 0.5 to 1 before the transit
    (``without_signal``), is set to 0 during the transit, and decreases
    linearly from 1 to 0.65 after the transit.

    Parameters
    ----------
    syn_jd : numpy.ndarray
        Synthetic Julian dates; only ``len(syn_jd)`` is used to size the output.
    without_signal : numpy.ndarray of int
        Indices of the out-of-transit frames.

    Returns
    -------
    fraction : numpy.ndarray
        Array of dayside fractions, same length as *syn_jd*.
    """
    fraction = np.empty_like(syn_jd)
    fraction[0:without_signal[0]] = np.linspace(0.5, 1, without_signal[0])
    fraction[without_signal] = 0
    fraction[without_signal[-1]+1:] = np.linspace(1, 0.65, len(syn_jd)-without_signal[-1]-1)
    return fraction


def UTC_to_TDB_CARMENES(inp_dat, utc):
    """
    Convert UTC Julian dates to TDB (barycentric dynamical time) for CARMENES.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``'RA'`` and ``'Dec'``
        keys (decimal degrees).
    utc : array_like
        1-D array of observation times in UTC Julian dates.

    Returns
    -------
    times_tdb : numpy.ndarray
        Barycentric dynamical time values (JD_TDB) as a plain NumPy array.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    site_name = "CAHA"  # Calar Alto astropy site name
    ra = inp_dat["RA"] * u.deg
    dec = inp_dat["Dec"] * u.deg
    times_utc = utc  # load MJD_UTC times
    observer_location = EarthLocation.of_site(site_name)
    target_coordinates = SkyCoord(ra=ra, dec=dec)
    times_utc = Time(times_utc, format="jd", scale="utc")
    times_tdb = (
        times_utc.tdb + times_utc.light_travel_time(
            target_coordinates, location=observer_location
            )
        )
    return times_tdb.value


def block_parameter(JD, T_0, P, R_P, a, R_s, i, uu, e=0, omega=90,
                    limb_dark_mode='quadratic'):
    """
    Compute the transit blocking factor of a planet crossing its host star.

    Parameters
    ----------
    JD : array_like
        Array of Julian dates at which to evaluate the blocking factor.
    T_0 : float
        Time of inferior conjunction (same units as *JD*).
    P : float
        Orbital period (same units as *JD*).
    R_P : float
        Planet radius (same physical units as *R_s*).
    a : float
        Semi-major axis (same physical units as *R_s*).
    R_s : float
        Stellar radius.
    i : float
        Orbital inclination in degrees.
    uu : sequence of float
        Limb-darkening coefficients.
    e : float, optional
        Orbital eccentricity.  Default 0.
    omega : float, optional
        Longitude of periastron in degrees.  Default 90.
    limb_dark_mode : str, optional
        Limb-darkening model name understood by batman.
        Default ``'quadratic'``.

    Returns
    -------
    block : numpy.ndarray
        Normalised blocking factor at each time in *JD*.
        Values range from 0 (out of transit) to 1 (centre of transit).
    """
    import batman as _batman

    # Define transit parameters
    params = _batman.TransitParams()
    params.t0 = T_0                       # time of inferior conjunction
    params.per = P                        # orbital period
    params.rp = R_P / R_s                 # planet radius (in units of stellar radius)
    params.a = a / R_s                    # semi-major axis (in units of stellar radius)
    params.inc = i                        # orbital inclination (in degrees)
    params.ecc = e                        # eccentricity
    params.w = omega                      # longitude of periastron (in degrees)
    params.u = uu                         # limb darkening coefficients
    params.limb_dark = limb_dark_mode     # limb darkening model

    # Define time array
    t = JD
    # Initialize transit model
    m = _batman.TransitModel(params, t)

    # Generate light curve
    flux = m.light_curve(params)

    # Get the blocking factor
    block = -(flux - 1)
    block /= block.max()

    return block
