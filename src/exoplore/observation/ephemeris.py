"""
exoplore.observation.ephemeris
==============================

Geometry-based per-night ephemerides for fully synthetic
``different_nights`` simulations.

Real observing campaigns spread the transits of a target over weeks to
months: each epoch has its own airmass evolution and its own Barycentric
Earth Radial Velocity (BERV).  The functions here derive all three
quantities from the target's sky coordinates and the observatory
location alone, so a synthetic multi-night simulation can reproduce that
night-to-night diversity without any input files:

``find_observable_transit_epochs``
    Mid-transit times of the first ``n_nights`` transits that are
    observable from the site (target above a minimum altitude at night).

``accurate_airmass``
    Airmass (sec z) of the target at each exposure time.

``accurate_berv``
    Barycentric correction (km/s) at each exposure time, with the same
    sign convention as ``observation.berv_kms`` (the term entering the
    Earth-frame velocities as ``v_sys - berv``).

All functions accept the observatory name used in the instrument config
(e.g. ``"paranal"``); resolution to an Astropy ``EarthLocation`` uses
the built-in offline site registry.
"""

from __future__ import annotations

import numpy as np

# Config observatory name -> astropy site-registry name.  Unknown names
# are passed through unchanged so any registry entry can be used
# directly in the config.
_SITE_NAMES = {
    "lasilla":  "La Silla Observatory",
    "paranal":  "Paranal Observatory",
    "caha":     "CAHA",
    "cfht":     "CFHT",
    "tng":      "TNG",
}


def _site_location(observatory: str):
    """Return the astropy EarthLocation for a config observatory name."""
    from astropy.coordinates import EarthLocation
    name = _SITE_NAMES.get(str(observatory).lower(), observatory)
    return EarthLocation.of_site(name)


def find_observable_transit_epochs(
    t0_bjd: float,
    period_days: float,
    ra_deg: float,
    dec_deg: float,
    observatory: str,
    n_nights: int,
    min_alt_deg: float = 30.0,
    max_sun_alt_deg: float = 0.0,
    max_epochs: int = 500,
) -> np.ndarray:
    """Find the first ``n_nights`` observable mid-transit epochs after T0.

    A transit is considered observable when, at mid-transit, the target
    is above ``min_alt_deg`` and the Sun is below ``max_sun_alt_deg``
    (the same criteria as ``scripts/generate_skycalc_inputs.py``).  The
    search starts at the first transit after the reference epoch, so the
    result is fully deterministic for a given ephemeris and site.

    Parameters
    ----------
    t0_bjd:
        Reference mid-transit epoch (JD).
    period_days:
        Orbital period in days.
    ra_deg, dec_deg:
        Target ICRS coordinates in degrees.
    observatory:
        Observatory name as given in the instrument config.
    n_nights:
        Number of distinct observable epochs to return.
    min_alt_deg:
        Minimum target altitude at mid-transit (degrees).
    max_sun_alt_deg:
        Maximum Sun altitude at mid-transit (degrees); 0 = below the
        horizon.
    max_epochs:
        Maximum number of consecutive transits to test.

    Returns
    -------
    np.ndarray, shape (n_nights,)
        Mid-transit times (JD) of the selected epochs.

    Raises
    ------
    RuntimeError
        If fewer than ``n_nights`` observable transits are found within
        ``max_epochs`` periods.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, SkyCoord, get_sun
    from astropy.time import Time

    location = _site_location(observatory)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    k = np.arange(1, max_epochs + 1)
    mids = Time(t0_bjd + k * period_days, format="jd", scale="utc")
    frame = AltAz(obstime=mids, location=location)
    target_alt = target.transform_to(frame).alt.deg
    sun_alt = get_sun(mids).transform_to(frame).alt.deg

    good = np.where((target_alt > min_alt_deg)
                    & (sun_alt < max_sun_alt_deg))[0]
    if len(good) < n_nights:
        raise RuntimeError(
            f"Only {len(good)} observable transits found in {max_epochs} "
            f"periods from T0; need {n_nights}.  Relax min_alt_deg / "
            f"max_sun_alt_deg or increase max_epochs."
        )
    return np.asarray(mids.jd[good[:n_nights]], dtype=float)


def accurate_airmass(
    julian_dates: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    observatory: str,
) -> np.ndarray:
    """Airmass (sec z, clipped to [1, 10]) of the target at each JD."""
    import astropy.units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    location = _site_location(observatory)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    times = Time(np.asarray(julian_dates, dtype=float), format="jd",
                 scale="utc")
    altaz = target.transform_to(AltAz(obstime=times, location=location))
    return np.clip(altaz.secz.value, 1.0, 10.0)


def accurate_berv(
    julian_dates: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    observatory: str,
) -> np.ndarray:
    """Barycentric correction (km/s) toward the target at each JD.

    Uses ``SkyCoord.radial_velocity_correction("barycentric")``, whose
    sign convention matches ``observation.berv_kms``: the correction is
    added to a topocentric radial velocity to obtain the barycentric
    one, so Earth-frame velocities in the simulator carry ``-berv``.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    location = _site_location(observatory)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    times = Time(np.asarray(julian_dates, dtype=float), format="jd",
                 scale="utc")
    corr = target.radial_velocity_correction(
        "barycentric", obstime=times, location=location)
    return np.atleast_1d(corr.to(u.km / u.s).value)
