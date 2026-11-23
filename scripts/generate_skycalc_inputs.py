"""Generate SkyCalc telluric FITS files for EXoPLORE simulations.

Replaces the ``skycalc_model()`` shell-script workflow with a direct
call to the ESO SkyCalc REST API via the ``skycalc_ipy`` Python package.
No command-line installation of ``skycalc_cli`` is required.

The output files are functionally identical to those produced by the
Skycalc CLI, each FITS file has a binary-table extension with columns ``lam`` (nm)
and ``trans``, and are read by the simulator without modification.

Prerequisites
-------------
An internet connection is required.  Install the ESO package once with::

    pip install skycalc_ipy

Usage
-----
Point the script at an EXoPLORE config JSON and it reads all required
parameters (instrument, wavelength limits, observatory, airmass evolution,
PWV, flag_event) directly from the config::

    python scripts/generate_skycalc_inputs.py \\
        configs/hd189733b_carmenes_transit.json

The output files are written into the directory that the simulator expects::

    {inputs_dir}/Skycalc_{flag_event}/{Fixed|Variable}_PWV/
        tell_spec_0.fits
        tell_spec_1.fits
        ...
        airmass.fits
    {inputs_dir}/tellurics/
        tell_ref_airmass_{X.X}.fits

Outputs
-------
tell_spec_{n}.fits
    One FITS file per synthetic exposure.  Extension 1: binary table with
    columns ``lam`` (nm) and ``trans`` (dimensionless transmittance).

tell_ref_airmass_{X.X}.fits
    Telluric spectrum at the reference airmass (used when
    ``tellurics.use_full_skycalc = false`` for a single-airmass reference).
    Same format as the per-exposure files.

airmass.fits
    The rounded/clipped airmass values used for the exposure queries.

Notes
-----
- The wavelength interval is read from tellurics.skycalc_wavelength_limits_um
  in the simulation JSON. Other sky-model defaults are defined in _query_skycalc.
- The wavelength grid is queried at R = 150 000 (fixed_spectral_resolution)
  before interpolation/convolution onto the selected instrument grid. Wavelength
  coverage is checked separately. high resolving power cannot supply missing bands.
- ``vacair = vac``, vacuum wavelengths, consistent with petitRADTRANS output.
- The observatory code is read from the instrument config
  (e.g. ``lasilla`` for CARMENES, ``paranal`` for CRIRES+, ANDES).
"""

from __future__ import annotations

# Role in the calculation
# Download telluric transmission spectra from SkyCalc for the simulation's wavelength
# range, observatory and water-vapour settings. The simulator reads these files when
# building the observations.


import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

# Validate wavelength settings with the configuration class used by the
# simulator. Add src so the script can also run from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exoplore.config.models import TelluricConfig
from exoplore.observation.skycalc_inputs import (
    SkyCalcCache, exposure_pwv, pwv_bracket, skycalc_directory, configured_exposure_dates,
)

# ---------------------------------------------------------------------------
# SkyCalc query
# ---------------------------------------------------------------------------

def _query_skycalc(
    airmass: float,
    pwv: float,
    observatory: str,
    wmin_nm: float,
    wmax_nm: float,
    wres: float = 150_000.0,
) -> fits.HDUList:
    """Query ESO SkyCalc REST API for one (airmass, PWV) pair.

    Parameters match the Skycalc CLI parameter file exactly.  Returns an
    open astropy HDUList. Caller is responsible for closing it.
    """
    try:
        from skycalc_ipy import SkyCalc
        from skycalc_ipy.core import SkyModel
    except ImportError:
        print("ERROR: skycalc_ipy is not installed.  Run: pip install skycalc_ipy")
        sys.exit(1)

    sky = SkyCalc()
    sky["airmass"]          = round(float(airmass), 1)
    sky["pwv_mode"]         = "pwv"
    sky["season"]           = 0
    sky["time"]             = 0
    sky["pwv"]              = float(pwv)
    sky["msolflux"]         = 130.0
    sky["incl_moon"]        = "N"
    sky["moon_sun_sep"]     = 90.0
    sky["moon_target_sep"]  = 45.0
    sky["moon_alt"]         = 45.0
    sky["moon_earth_dist"]  = 1.0
    sky["incl_starlight"]   = "N"
    sky["incl_zodiacal"]    = "N"
    sky["ecl_lon"]          = 135.0
    sky["ecl_lat"]          = 90.0
    sky["incl_loweratm"]    = "Y"
    sky["incl_upperatm"]    = "Y"
    sky["incl_airglow"]     = "Y"
    sky["incl_therm"]       = "N"
    sky["vacair"]           = "vac"
    sky["wmin"]             = float(wmin_nm)
    sky["wmax"]             = float(wmax_nm)
    sky["wgrid_mode"]       = "fixed_spectral_resolution"
    sky["wdelta"]           = 0.01
    sky["wres"]             = float(wres)
    sky["lsf_type"]         = "none"
    sky["lsf_gauss_fwhm"]   = 5.0
    sky["lsf_boxcar_fwhm"]  = 5.0
    # ESO SkyCalc only models its own sites. Map other observatory codes to
    # the nearest ESO site: Calar Alto (CARMENES, 2168 m) -> La Silla (2400 m).
    _skycalc_obs = {"caha": "lasilla"}
    sky["observatory"]      = _skycalc_obs.get(observatory.lower(), observatory)

    skm = SkyModel()
    try:
        skm(**{k: sky[k] for k in sky.defaults})
        return skm.data   # astropy HDUList
    except Exception:
        return None   # SkyCalc rejected parameters (e.g. airmass > 3)


def _save_hdul(hdul: fits.HDUList, out_path: str) -> None:
    """Write HDUList to *out_path*, creating parent directories."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hdul.writeto(out_path, overwrite=True)


# ---------------------------------------------------------------------------
# Exposure airmass
# ---------------------------------------------------------------------------

def _build_airmass(cfg: dict, n_spectra: int = None) -> np.ndarray:
    """Compute the per-exposure airmass array using the same logic as Block 2.

    Calls :func:`exoplore.observation.timing.observation_julian_dates` to
    build the JD grid from transit epoch and exposure cadence, then
    :func:`exoplore.observation.airmass.synthetic_airmass`, the identical
    parabolic model the simulator uses for airmass-scaled tellurics.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from exoplore.observation.timing import observation_julian_dates
    from exoplore.observation.airmass import synthetic_airmass

    obs   = cfg.get("observation", {})
    tell  = cfg.get("tellurics", {})
    planet = cfg.get("planet", {})

    # Load planet params to get transit epoch
    p_file = planet.get("parameter_file", "")
    if p_file and os.path.exists(p_file):
        with open(p_file) as f:
            import json as _j
            p = _j.load(f)
        t0_bjd = p.get("transit_epoch_bjd", 0.0)
    else:
        t0_bjd = obs.get("specific_T0_bjd", 0.0)

    # n_spectra must match what the simulator builds. The caller (generate())
    # already derives it from the event duration, so use that. Only fall back
    # to a config value if not passed. Read the real config keys
    # (exposure_time_seconds and readout_time_seconds).
    if n_spectra is None:
        n_spectra = int(obs.get("n_spectra", 45))
    exp_s        = float(obs.get("exposure_time_seconds",
                                 obs.get("exposure_time_s", 198.0)))
    readout_s    = float(obs.get("readout_time_seconds",
                                 obs.get("readout_time_s", 0.0)))
    overhead_s   = float(obs.get("overhead_time_seconds",
                                 obs.get("overhead_time_s", 0.0)))
    start_offset = obs.get("start_offset_hours",
                           obs.get("pre_event_hours", 1.0))

    jd = observation_julian_dates(
        transit_epoch_bjd=t0_bjd,
        exposure_time_seconds=exp_s,
        readout_time_seconds=readout_s,
        overhead_time_seconds=overhead_s,
        n_exposures=n_spectra,
        start_offset_hours=start_offset,
    )

    am_limits = tell.get("airmass_limits", [1.4, 1.7])
    pattern   = tell.get("airmass_evolution", "up_and_down")
    return synthetic_airmass(jd, am_limits[0], am_limits[1], pattern)


# ---------------------------------------------------------------------------
# Astronomical airmass, real dates via astroplan (mirrors Ratri's approach)
# ---------------------------------------------------------------------------

# Map EXoPLORE observatory codes to astropy/astroplan site names
_OBS_SITE = {
    "lasilla":  "La Silla Observatory",
    "paranal":  "Paranal Observatory",
    "caha":     "CAHA",
    "cfht":     "CFHT",
    "tng":      "TNG",
}


def _transit_duration_hours(p: dict) -> float:
    """Compute transit duration from orbital elements if not in params."""
    if "transit_duration_hours" in p:
        return float(p["transit_duration_hours"])
    Rsun_au = 0.00465047
    Rs  = p["stellar_radius_rsun"] * Rsun_au
    Rp  = p["planet_radius_rjup"] * 0.10045 * Rsun_au
    a   = p["semi_major_axis_au"]
    inc = np.radians(p["inclination_deg"])
    P   = p["orbital_period_days"]
    arg = np.clip(np.sqrt((Rs + Rp)**2 - (a * np.cos(inc))**2) / a, -1.0, 1.0)
    return (P / np.pi) * np.arcsin(arg) * 24.0


def _build_airmass_astro(cfg: dict, search_from: str = None,
                         skip_transits: int = 0, return_dates: bool = False) -> np.ndarray:
    """Compute real per-exposure airmass from sky geometry via astroplan.

    Two behaviours depending on ``observation.specific_event``:

    specific_event = true (real observed night)
        The transit T0 is already known (``observation.specific_T0_bjd``).
        Airmass is computed at the JDs the simulator would build for that
        night using the config exposure time and cadence.
        WARNING: the proper approach is to load the real airmass from
        ``reference_night/airmass_0.fits``, this mode is a fallback.

    specific_event = false (fully synthetic simulation)
        Uses astroplan's EclipsingSystem to find the next observable transit
        from the site starting at ``search_from`` (or today).  JDs are
        built with ``observation_julian_dates()``, the same function the
        simulator calls, so the airmass array matches the synthetic JD grid
        exactly.

    Parameters
    ----------
    cfg : dict
        EXoPLORE config dict (JSON-loaded).
    search_from : str or None
        UTC date to start the transit search, e.g. ``"2030-01-01"``.
        Used only for ``specific_event=false``.  Defaults to today.
    skip_transits : int
        Skip this many observable transits before selecting one.
        Night 0 → skip=0 (first), night 1 → skip=1 (second), etc.
        Each night therefore gets a distinct real transit epoch.

    Returns
    -------
    ndarray, shape (n_spectra,)
        Airmass at each mid-exposure time (clipped to [1, 10]).
    """
    try:
        from astroplan import Observer, FixedTarget, EclipsingSystem
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u
        import warnings
    except ImportError:
        raise ImportError(
            "astroplan is required for --mode astro.\n"
            "Install with: pip install astroplan"
        )

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from exoplore.observation.timing import observation_julian_dates

    obs_cfg  = cfg.get("observation", {})
    inst_cfg = cfg.get("instrument", {})
    planet_s = cfg.get("planet", {})

    # --- Load planet params ---
    p_file = planet_s.get("parameter_file", "")
    if not (p_file and os.path.exists(p_file)):
        raise ValueError("Planet parameter file not found.")
    with open(p_file) as f:
        import json as _j
        p = _j.load(f)
    for req in ("ra_deg", "dec_deg", "transit_epoch_bjd", "orbital_period_days"):
        if req not in p:
            raise ValueError(f"Planet parameter file must contain '{req}'.")

    # --- Observatory and target ---
    site_name = _OBS_SITE.get(inst_cfg.get("observatory", "paranal").lower(),
                               inst_cfg.get("observatory", "paranal"))
    observer = Observer.at_site(site_name)
    target   = FixedTarget(
        SkyCoord(ra=p["ra_deg"] * u.deg, dec=p["dec_deg"] * u.deg),
        name=p.get("name", "target"),
    )

    # Use PlanetParameters for transit duration so it matches the simulator
    # exactly (same formula used at simulator.py lines 388-392).
    try:
        # Use load_planet (full unit handling) so transit_duration_hours is
        # computed exactly as the simulator does. constructing PlanetParameters
        # directly from the raw JSON skips the conversions and yields 0.
        from exoplore.planets import load_planet as _load_planet
        _pp = _load_planet(p_file)
        dur_h = _pp.transit_duration_hours
    except Exception:
        dur_h = _transit_duration_hours(p)  # fallback

    # Per-night cadence: with exposure_time_seconds_per_night set, each
    # night has its own DIT and therefore its own number of exposures.
    # skip_transits doubles as the night index here.
    _exp_pn = obs_cfg.get("exposure_time_seconds_per_night")
    if _exp_pn and skip_transits is not None and skip_transits < len(_exp_pn):
        exp_s = float(_exp_pn[skip_transits])
    else:
        exp_s = float(obs_cfg.get("exposure_time_seconds", 198.0) or 198.0)
    readout_s = float(obs_cfg.get("readout_time_seconds",    0.0) or 0.0)
    overhead_s = float(obs_cfg.get("overhead_time_seconds",  0.0) or 0.0)
    cadence_s  = exp_s + readout_s + overhead_s
    _pre_h_raw  = float(obs_cfg.get("pre_event_hours",  0.0) or 0.0)
    _post_h_raw = float(obs_cfg.get("post_event_hours", 0.0) or 0.0)
    # Mirror simulator Block 2 auto-substitution: pre/post_event_hours=0.0
    # means "use half the transit duration as baseline", not "no baseline".
    pre_h  = _pre_h_raw  if _pre_h_raw  > 0.0 else dur_h / 2.0
    post_h = _post_h_raw if _post_h_raw > 0.0 else dur_h / 2.0

    # n_spectra: use config value if set, else derive using the same
    # np.arange formula as get_event() in the simulator, so the number
    # of SkyCalc files generated exactly matches the number of exposures
    # the simulator will build.
    n_spectra_cfg = obs_cfg.get("n_spectra")
    if n_spectra_cfg:
        n_spectra = int(n_spectra_cfg)
    else:
        _jd_step = cadence_s / 86400.0
        _total_d = (dur_h + pre_h + post_h) / 24.0
        # Mirror np.arange(0, _total_d + _jd_step, _jd_step)
        n_spectra = len(np.arange(0, _total_d + _jd_step, _jd_step))

    # start_offset: how many hours before mid-transit observation begins
    start_offset_h = dur_h / 2.0 + pre_h

    specific_event = bool(obs_cfg.get("specific_event", False))

    # ── Case 1: specific_event = true ────────────────────────────────────────
    # The simulator uses the real JDs from julian_date_0.fits verbatim
    # (get_event returns syn_jd = JD_og directly).  We do the same here.
    if specific_event:
        inputs_dir = cfg.get("paths", {}).get("inputs_dir", "inputs/")
        jd_fits    = os.path.join(inputs_dir, "reference_night",
                                  "julian_date_0.fits")
        if not os.path.exists(jd_fits):
            raise FileNotFoundError(
                f"specific_event=true but real JD file not found: {jd_fits}\n"
                f"The simulator uses the real julian_date_0.fits timestamps.  "
                f"Run with --mode synthetic or provide the reference night files."
            )
        from astropy.io import fits as _fits_jd
        jd_array = _fits_jd.open(jd_fits)[0].data.astype(float)
        print(f"\n  [astro] WARNING: specific_event=true, you should use the "
              f"real airmass from reference_night/airmass_0.fits.\n"
              f"          Computing astronomical airmass at the {len(jd_array)} "
              f"real JDs from {jd_fits} as a fallback.")

    # ── Case 2: specific_event = false, find next observable transit ─────────
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = EclipsingSystem(
                primary_eclipse_time=Time(p["transit_epoch_bjd"],
                                          format="jd", scale="tdb"),
                orbital_period=p["orbital_period_days"] * u.day,
                duration=dur_h * u.hour,
                name=p.get("name", "target"),
            )
            t_from = Time(search_from or Time.now().iso[:10], scale="utc")
            mids   = system.next_primary_eclipse_time(t_from, n_eclipses=120)

            t_mid_obj = None
            found = 0
            for t in mids:
                alt    = observer.altaz(t, target).alt.deg
                is_ngt = bool(observer.is_night(t))
                if is_ngt and alt > 30.0:
                    if found >= skip_transits:
                        t_mid_obj = t
                        break
                    found += 1

        if t_mid_obj is None:
            raise RuntimeError(
                f"No observable transit found in 120 periods from "
                f"{t_from.iso[:10]}.  Try --search-from with a later date."
            )
        t_mid_bjd = float(t_mid_obj.jd)
        print(f"  [astro] Next observable transit: "
              f"{t_mid_obj.iso[:10]}  (mid {t_mid_obj.iso[11:16]} UTC)")

        # Build JDs with the same function the simulator uses in Block 2
        jd_array = observation_julian_dates(
            transit_epoch_bjd=t_mid_bjd,
            exposure_time_seconds=exp_s,
            readout_time_seconds=readout_s,
            overhead_time_seconds=overhead_s,
            n_exposures=n_spectra,
            start_offset_hours=start_offset_h,
        )

    # ── Compute airmass at those exact JDs ────────────────────────────────────
    times = Time(jd_array, format="jd", scale="utc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        altaz   = observer.altaz(times, target)
        airmass = np.clip(altaz.secz.value, 1.0, 10.0)

    n_computed = len(jd_array)
    t_start_iso = times[0].iso[11:16]
    print(f"  [astro] {site_name}  →  {target.name}")
    if specific_event:
        print(f"  [astro] {n_computed} real JDs from julian_date_0.fits")
    else:
        print(f"  [astro] Duration {dur_h:.2f} h | pre {pre_h:.1f} h | post {post_h:.1f} h "
              f"| {n_computed} exp × {cadence_s:.0f} s")
    print(f"  [astro] Obs start UTC : {t_start_iso}")
    print(f"  [astro] Airmass       : {airmass.min():.3f} → {airmass.max():.3f}")
    below = int((altaz.alt.deg < 0).sum())
    if below:
        print(f"  [astro] WARNING: {below}/{n_spectra} exposures below horizon")

    return (airmass, jd_array) if return_dates else airmass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(config_path: str, overwrite: bool = False,
             mode: str = "config", search_from: str = None,
             night_idx: int = None, ref_only: bool = False,
             plan_only: bool = False) -> None:
    """Generate SkyCalc telluric FITS files for one EXoPLORE config.

    Parameters
    ----------
    config_path : str
        Path to EXoPLORE JSON config.
    overwrite : bool
        Overwrite existing output files (default: skip existing).
    mode : str
        ``"config"`` (default) uses the simulator's observing dates and
        airmass choices. ``"astro"`` plans a future observable transit
        using target coordinates and the observatory. Downloaded date
        records are checked against the simulation when they are loaded.

        ``"synthetic"``, parabolic airmass from ``tellurics.airmass_limits``
        (same model the simulator uses internally in Block 2).  Fast fallback
        when coordinates are unavailable.
    search_from : str or None
        UTC date from which to start the transit search for ``mode="astro"``,
        e.g. ``"2030-01-01"``.  Defaults to today.  For ``--night N > 0``
        the search continues from the transit found for night N-1.
    night_idx : int or None
        Night index for ``different_nights=true`` simulations.  When set:

        - Output goes to ``Skycalc_{event}/night_{N}/Fixed_PWV/`` so each
          night has its own set of telluric files with independent PWV and
          airmass.
        - PWV is taken from ``tellurics.pwv_mm_per_night[N]`` if set,
          otherwise falls back to ``tellurics.pwv_mm``.
        - In ``mode="astro"``, the (N+1)-th observable transit is used,
          so night 0 gets the first transit, night 1 gets the second, etc.

        When ``None`` (default), a single shared directory is used
        (``Skycalc_{event}/Fixed_PWV/``), appropriate for single-night
        simulations or ``different_nights=false``.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    obs       = cfg.get("observation", {})
    tell      = cfg.get("tellurics",   {})
    paths     = cfg.get("paths",       {})
    inst_cfg  = cfg.get("instrument",  {})

    # Read the download interval from tellurics. The instrument key is also
    # accepted and produces a message directing users to the tellurics field.
    limits = tell.get("skycalc_wavelength_limits_um")
    if limits is None and inst_cfg.get("wavelength_limits_um") is not None:
        import warnings
        warnings.warn(
            "Move instrument.wavelength_limits_um to tellurics.skycalc_wavelength_limits_um "
            "in the same simulation JSON.", FutureWarning,
        )
        limits = inst_cfg["wavelength_limits_um"]
    wave_limits = TelluricConfig(skycalc_wavelength_limits_um=limits).skycalc_wavelength_limits_um

    tell_cfg = TelluricConfig(**tell)
    inputs_dir   = paths.get("inputs_dir", "inputs/")
    flag_event   = obs.get("flag_event", "full_event")
    constant_pwv = tell.get("constant_pwv", True)
    ref_airmass  = tell.get("reference_airmass", 1.0)
    am_limits    = tell.get("airmass_limits", [1.4, 1.7])

    # Per-night PWV: use pwv_mm_per_night[night_idx] when set.
    pwv_mm_per_night = tell.get("pwv_mm_per_night")
    if night_idx is not None and pwv_mm_per_night is not None:
        pwv_mm = float(pwv_mm_per_night[night_idx])
        print(f"  [night {night_idx}] Using pwv_mm_per_night[{night_idx}] = {pwv_mm} mm")
    else:
        pwv_mm = float(tell.get("pwv_mm", 10.0))

    observatory = inst_cfg.get("observatory", "paranal")

    # Wavelength limits
    wmin_um = 0.95
    wmax_um = 2.45
    if wave_limits is not None:
        wmin_um, wmax_um = wave_limits
    wmin_nm = wmin_um * 1000.0
    wmax_nm = wmax_um * 1000.0

    # A reference-only request needs no exposure calendar.
    if ref_only and not plan_only and tell_cfg.skycalc_source != "external":
        ref_path = os.path.join(inputs_dir, "tellurics", f"tell_ref_airmass_{ref_airmass:.1f}.fits")
        if os.path.exists(ref_path) and not overwrite:
            return
        hdul = _query_skycalc(ref_airmass, pwv_mm, observatory, wmin_nm, wmax_nm)
        if hdul is None:
            raise RuntimeError("SkyCalc returned no reference spectrum. No file was written.")
        with hdul:
            _save_hdul(hdul, ref_path)
        return

    out_dir = str(skycalc_directory(tell_cfg, inputs_dir, flag_event, night_idx))
    os.makedirs(out_dir, exist_ok=True)

    # Plan the same samples used by the molecular runner before querying ESO.
    from exoplore.config.models import ObservationConfig
    from exoplore.planets import load_planet
    planet = load_planet(cfg["planet"]["parameter_file"])
    observation = ObservationConfig(**obs)
    jd_array = configured_exposure_dates(observation, planet, inputs_dir,
                                         night_idx, tell_cfg, observatory)
    n_spectra = len(jd_array)

    # Airmass array
    _specific = cfg.get("observation", {}).get("specific_event", False)
    _night_number = night_idx if night_idx is not None else 0
    _am_fits = os.path.join(inputs_dir, "reference_night", f"airmass_{_night_number}.fits")
    if _specific and os.path.exists(_am_fits):
        # Observed night: the real per-exposure airmass is already known
        # (the simulator itself reads it from airmass_0.fits), so use it
        # directly. This branch reads the observing arrays from the input files. No astroplan
        # is needed in this case.
        from astropy.io import fits as _fits_am
        with _fits_am.open(_am_fits) as hdul:
            airmass = hdul[0].data.astype(float)
        print(f"\n  Mode: specific_event (real airmass from airmass_0.fits, "
              f"{airmass.min():.2f} to {airmass.max():.2f})")
    elif _specific:
        # Observed night but no airmass_0.fits: recover the airmass from the
        # target coordinates and the real BJDs (julian_date_0.fits) via
        # astroplan using the target coordinates and observatory location.
        print("\n  Mode: specific_event (no airmass_0.fits; computing airmass "
              "from target + real BJDs via astroplan)")
        airmass, jd_array = _build_airmass_astro(cfg, search_from,
                                       skip_transits=night_idx or 0, return_dates=True)
    elif mode == "config":
        from exoplore.observation.airmass import synthetic_airmass
        if (observation.different_nights and tell_cfg.use_accurate_airmass
                and planet.ra_deg is not None and planet.dec_deg is not None):
            from exoplore.observation.ephemeris import accurate_airmass
            airmass = accurate_airmass(jd_array, planet.ra_deg, planet.dec_deg, observatory)
        else:
            night_limits = tell_cfg.airmass_limits_per_night
            limits = night_limits[night_idx] if night_limits and night_idx is not None else am_limits
            airmass = synthetic_airmass(jd_array, limits[0], limits[1], tell_cfg.airmass_evolution)
    elif mode == "astro":
        print(f"\n  Mode: astro (real sky geometry via astroplan)")
        # For night N, skip the first N observable transits so each night
        # gets its own distinct transit epoch.
        airmass, jd_array = _build_airmass_astro(cfg, search_from,
                                       skip_transits=night_idx or 0, return_dates=True)
    elif am_limits[0] == am_limits[1]:
        airmass = np.full(n_spectra, am_limits[0])
        print(f"\n  Mode: synthetic (fixed airmass {am_limits[0]:.1f})")
    else:
        print(f"\n  Mode: synthetic (parabolic airmass model)")
        if jd_array is not None:
            from exoplore.observation.airmass import synthetic_airmass
            airmass = synthetic_airmass(jd_array, am_limits[0], am_limits[1],
                                        tell.get("airmass_evolution", "up_and_down"))
        else:
            airmass = _build_airmass(cfg, n_spectra)

    n_spectra = len(airmass)
    if len(jd_array) != n_spectra:
        raise ValueError("Exposure dates and airmass have different lengths.")
    # Times follow the simulator's exposure samples. A small rounding removes
    # JD subtraction noise at the endpoints of user-supplied time intervals.
    hours = np.round((jd_array - jd_array[0])*24., 8)
    event_midpoint = planet.transit_epoch_bjd + planet.orbital_period_days * round(
        (float(np.median(jd_array))-planet.transit_epoch_bjd)/planet.orbital_period_days)
    phases = (jd_array-event_midpoint)/planet.orbital_period_days
    if tell_cfg.constant_pwv:
        pwv_arr = np.full(n_spectra, pwv_mm)
    elif tell_cfg.pwv_time_series is not None:
        pwv_arr = exposure_pwv(hours, tell_cfg.pwv_time_series)
    elif plan_only or tell_cfg.skycalc_source == "external":
        pwv_arr = np.full(n_spectra, np.nan)
    else:
        raise ValueError("Variable PWV downloads require tellurics.pwv_time_series. Export --plan-only to prepare external models.")

    night_label = f" night {night_idx}" if night_idx is not None else ""
    print(f"\nGenerating {n_spectra} tell_spec files{night_label}")
    print(f"  Observatory : {observatory}")
    print(f"  λ range     : {wmin_nm:.0f} to {wmax_nm:.0f} nm")
    print(f"  PWV         : {pwv_mm} mm" if tell_cfg.constant_pwv else "  PWV         : per exposure")
    print(f"  Output      : {out_dir}\n")

    # Preserve the computed airmass in the table alongside the service value.
    # Existing downloads round to 0.1 and clip to the SkyCalc interval.
    query_airmass = np.round(np.clip(airmass, 1., 3.), 1)
    import csv
    with open(os.path.join(out_dir, "exposure_plan.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["night", "exposure", "jd", "phase", "hours_from_observation_start",
                         "exposure_seconds", "airmass", "skycalc_airmass", "pwv_mm",
                         "pwv_lower_mm", "pwv_upper_mm", "upper_weight", "filename"])
        exp_pn = obs.get("exposure_time_seconds_per_night")
        exp_seconds = (exp_pn[night_idx] if exp_pn and night_idx is not None
                       else obs.get("exposure_time_seconds", 198.))
        for n, (jd, am, query_am, pwv) in enumerate(zip(jd_array, airmass, query_airmass, pwv_arr)):
            bracket = pwv_bracket(pwv) if np.isfinite(pwv) else ("", "", "")
            writer.writerow([night_idx if night_idx is not None else 0, n, jd, phases[n], hours[n],
                             exp_seconds, am, query_am, pwv if np.isfinite(pwv) else "",
                             *bracket, f"tell_spec_{n}.fits"])
    Path(out_dir, "skycalc_request.json").write_text(json.dumps({
        "observatory": observatory, "wmin_nm": wmin_nm, "wmax_nm": wmax_nm,
        "wres": 150000., "vacair": "vac", "wgrid_mode": "fixed_spectral_resolution",
        "lsf_type": "none", "source": tell_cfg.skycalc_source,
        "note": "Exposure dates use the simulator sampling convention. PWV interpolation applies to trans only."
    }, indent=2) + "\n")
    print(f"  Exposure plan: {out_dir}/exposure_plan.csv")
    if plan_only or tell_cfg.skycalc_source == "external":
        print("  Planning complete. No SkyCalc queries sent.")
        return
    cache = SkyCalcCache(Path(inputs_dir) / "skycalc_cache", _query_skycalc, {
        "recipe": "exoplore-skycalc-v1",
        "arguments": dict(observatory=observatory, wmin_nm=wmin_nm,
                          wmax_nm=wmax_nm, wres=150000.)})

    # --- Per-exposure files (skipped when --ref-only) ---
    if ref_only:
        print("  --ref-only: skipping per-exposure files, generating reference only.")
    for n in range(n_spectra if not ref_only else 0):
        out_path = os.path.join(out_dir, f"tell_spec_{n}.fits")
        am = float(query_airmass[n])
        pwv = float(pwv_arr[n])
        signature = dict(cache.settings, airmass=am, pwv=pwv)
        signature_path = Path(out_path).with_suffix(".json")
        if os.path.exists(out_path) and not overwrite:
            if signature_path.exists() and json.loads(signature_path.read_text()) == signature:
                print(f"  [{n+1:3d}/{n_spectra}] tell_spec_{n}.fits, matching request, skip")
                continue
            raise ValueError(f"{out_path} has no matching request record. Use --overwrite after checking the exposure plan.")
        print(f"  [{n+1:3d}/{n_spectra}] tell_spec_{n}.fits  "
              f"(airmass={am:.1f}, pwv={pwv:.1f} mm) ...", end=" ", flush=True)
        t0   = time.time()
        hdul = cache.transmission(am, pwv)
        # Record the request next to each final exposure product. Existing
        # products are reusable only when this signature still matches.
        _save_hdul(hdul, out_path)
        hdul.close()
        signature_path.write_text(json.dumps(signature, sort_keys=True) + "\n")
        print(f"{time.time()-t0:.1f}s")

    fits.PrimaryHDU(jd_array).writeto(os.path.join(out_dir, "exposure_jd.fits"), overwrite=True)
    fits.PrimaryHDU(pwv_arr).writeto(os.path.join(out_dir, "pwv_values.fits"), overwrite=True)
    # --- Per-night airmass array, as queried (rounded/clipped) ---
    # Saved next to the tell_spec files so the simulator can use the exact
    # airmass the telluric spectra were generated with (pipelines that
    # regress against airmass then see a consistent regressor).
    _am_path = os.path.join(out_dir, "airmass.fits")
    fits.PrimaryHDU(query_airmass).writeto(_am_path, overwrite=True)
    print(f"  airmass.fits written ({len(airmass)} exposures)")

    # --- Reference airmass file (shared, written only for night 0 or single) ---
    if night_idx is None or night_idx == 0:
        ref_path = os.path.join(
            inputs_dir, "tellurics",
            f"tell_ref_airmass_{ref_airmass:.1f}.fits",
        )
        if os.path.exists(ref_path) and not overwrite:
            print(f"\n  tell_ref_airmass_{ref_airmass:.1f}.fits, exists, skip")
        else:
            print(f"\n  Generating reference: airmass={ref_airmass:.1f}, "
                  f"pwv={pwv_mm:.1f} mm ...", end=" ", flush=True)
            t0   = time.time()
            hdul = cache.transmission(round(float(ref_airmass), 1), pwv_mm)
            if hdul is None:
                raise RuntimeError(
                    f"SkyCalc returned no reference spectrum for {wmin_nm:g}–{wmax_nm:g} nm. "
                    "No reference file was written; check the service response and your simulation JSON."
                )
            _save_hdul(hdul, ref_path)
            hdul.close()
            print(f"{time.time()-t0:.1f}s")

    print(f"\nDone. Files written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SkyCalc telluric FITS files for EXoPLORE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  config (default)
      Match the simulator's exposure dates and configured airmass treatment.
      --plan-only writes the exposure table without sending SkyCalc queries.

  astro
      Real airmass from true sky geometry via astroplan (Dash et al.,
      arXiv:2602.22830, 2026).  Automatically finds the next observable
      transit from the site, no date input required.  Requires ra_deg
      and dec_deg in the planet parameter JSON file.

  synthetic
      Parabolic airmass from tellurics.airmass_limits (same model as the
      simulator's Block 2).  Use when coordinates are unavailable.

Per-night usage (different_nights=true)
-----------------------------------------
  Generate one set of files per night with --night N.  Each night gets
  its own directory (Skycalc_{event}/night_N/Fixed_PWV/), its own PWV
  from tellurics.pwv_mm_per_night[N], and its own transit epoch (the
  (N+1)-th observable transit in astro mode):

    python scripts/generate_skycalc_inputs.py config.json --night 0
    python scripts/generate_skycalc_inputs.py config.json --night 1

Examples
--------
  # Single night, configured observing samples:
  python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_transit.json

  # Single night, search from a specific date:
  python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_transit.json \\
      --mode astro --search-from 2030-01-01

  # Two different nights:
  python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_dn.json --night 0
  python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_dn.json --night 1
""")
    parser.add_argument("config", help="Path to EXoPLORE JSON config file")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files (default: skip existing)"
    )
    parser.add_argument(
        "--mode", choices=["config", "synthetic", "astro"], default="config",
        help="Airmass mode (default: config, match the simulator dates and airmass)"
    )
    parser.add_argument(
        "--search-from", metavar="YYYY-MM-DD", default=None,
        help="Search for observable transit from this date (default: today). "
             "Only used with --mode astro."
    )
    parser.add_argument(
        "--night", metavar="N", type=int, default=None,
        help="Night index for different_nights=true simulations (0-based). "
             "Outputs to night_N/ subdirectory, uses pwv_mm_per_night[N], "
             "and selects the (N+1)-th observable transit in astro mode."
    )
    parser.add_argument(
        "--ref-only", action="store_true",
        help="Generate only the reference telluric file "
             "(tell_ref_airmass_X.X.fits) without per-exposure files. "
             "Useful when tellurics.use_full_skycalc is false and only "
             "the Mode 1 reference file is needed."
    )
    parser.add_argument("--plan-only", action="store_true",
                        help="Export exposure dates, airmasses and expected filenames without downloading models.")
    args = parser.parse_args()
    generate(args.config, overwrite=args.overwrite,
             mode=args.mode, search_from=args.search_from,
             night_idx=args.night, ref_only=getattr(args, 'ref_only', False),
             plan_only=args.plan_only)
