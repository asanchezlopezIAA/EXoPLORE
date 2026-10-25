"""
exoplore.observation.timing
============================

Orbital phase and in-transit timing utilities.

All functions work with Julian Dates (JD or BJD) and orbital periods
in days.
"""

from __future__ import annotations

# Role in the calculation
# Calculate exposure times, event membership and light-curve weights. Exposure-index
# arrays select the spectra used to construct and combine the planetary signal.


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
        Mid-transit time in BJD (used as reference, the grid starts at
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
# get_event
# ---------------------------------------------------------------------------

def get_event(observation, planet, JD_og):
    """Build event times and select exposures from the observing configuration.

    observation is ObservationConfig and planet is PlanetParameters. JD_og
    supplies the measured dates for a specific event or separate nights.
    Return dates, signal-exposure indices, other-exposure indices and event
    midpoints using the input time convention. For dayside spectra, the
    signal selection contains the exposures outside secondary eclipse.
    """
    event = observation.event_type
    t_0 = planet.transit_epoch_bjd
    period = planet.orbital_period_days
    transit_duration = (planet.transit_duration_hours / 24.0
                        if planet.transit_duration_hours is not None else None)
    DIT = observation.exposure_time_seconds
    readout = observation.readout_time_seconds
    overheads = observation.overhead_time_seconds
    flag_event = observation.flag_event
    pre_time = (observation.pre_event_hours if observation.pre_event_hours > 0
                else planet.transit_duration_hours / 2.0)
    post_time = (observation.post_event_hours if observation.post_event_hours > 0
                 else planet.transit_duration_hours / 2.0)

    if observation.different_nights:
        in_transit = []
        out_transit = []
        transit_mid_JD = []
        # Use specific_T_0 when available (specific_event=True). fall back
        # to T_0 (the reference epoch) for fully synthetic nights so that
        # the nearest transit to each night's JD array can still be found.
        _t0_ref = (observation.specific_T0_bjd
                   if observation.specific_T0_bjd is not None
                   else planet.transit_epoch_bjd)
        for n in range(observation.n_nights):
            transit_mid_JD.append(
                _t0_ref
                + planet.orbital_period_days * int(
                    ((JD_og[n] - _t0_ref) / planet.orbital_period_days)[-1]
                )
            )
            transit_begin_JD = transit_mid_JD[n] - transit_duration / 2.0
            transit_end_JD = transit_mid_JD[n] + transit_duration / 2.0
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
    if observation.specific_T0_bjd is not None and observation.specific_event:
        t_0 = observation.specific_T0_bjd
    elif observation.specific_T0_bjd is not None and not observation.specific_event:
        raise ValueError(
            "Set observation.specific_event=true to use specific_T0_bjd, "
            "or set observation.specific_T0_bjd=null."
        )
    elif observation.specific_event and observation.specific_T0_bjd is None:
        raise ValueError("Please provide the T_0 of your event.")

    # Compute start/end JDs
    if event == 'transit' and not observation.specific_event:
        jd_ini = t_0 - transit_duration / 2.0 - pre_time / 24.0
        jd_fin = t_0 + transit_duration / 2.0 + post_time / 24.0
    elif event == 'dayside' and not observation.specific_event:
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
    if not observation.specific_event:
        jd_step = (DIT + overheads + readout) / (3600.0 * 24.0)
        syn_jd = np.arange(jd_ini, jd_fin + jd_step, jd_step)
    else:
        syn_jd = JD_og

    # Find in/out-of-event indices
    if event == 'transit':
        # Fold the reference epoch to the transit nearest the observed data.
        # A transit ephemeris is periodic, so the event that matters is the one
        # closest to the night, regardless of whether T_0 precedes or follows it
        # (e.g. a JWST T_0 measured years after a ground-based run).  Using T_0
        # verbatim would place the window far from the data and flag nothing.
        transit_mid_JD = t_0 + period * int(
            np.round((np.median(syn_jd) - t_0) / period))
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
        # Fold to the secondary eclipse nearest the data (see transit note).
        eclipse_mid_JD = t_0 + period / 2.0
        eclipse_mid_JD = eclipse_mid_JD + period * int(
            np.round((np.median(syn_jd) - eclipse_mid_JD) / period))
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


def get_transit(julian_date, *, transit_epoch_bjd, orbital_period_days, transit_duration_days):
    """Return in-transit and out-of-transit exposure indices.

    julian_date and transit_epoch_bjd share the input time convention.
    orbital_period_days and transit_duration_days are measured in days.
    The event epoch is selected using the final observation time.
    """
    transit_mid_JD = transit_epoch_bjd + orbital_period_days * (
        int(((julian_date - transit_epoch_bjd) / orbital_period_days)[-1])
    )
    transit_begin_JD = transit_mid_JD - transit_duration_days / 2.
    transit_end_JD   = transit_mid_JD + transit_duration_days / 2.
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
        Synthetic Julian dates. Only ``len(syn_jd)`` is used to size the output.
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


def UTC_to_TDB_CARMENES(utc, *, ra_deg, dec_deg):
    """Convert UTC Julian dates to barycentric TDB Julian dates for CAHA.

    utc contains UTC Julian dates. ra_deg and dec_deg locate the target in
    decimal degrees. Astropy supplies the CAHA observatory coordinates.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    site_name = "CAHA"  # Calar Alto astropy site name
    ra = ra_deg * u.deg
    dec = dec_deg * u.deg
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
