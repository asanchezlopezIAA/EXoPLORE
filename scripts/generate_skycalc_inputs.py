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
        tell_ref_airmass_{X.X}.fits
        pwv_values.fits

Outputs
-------
tell_spec_{n}.fits
    One FITS file per synthetic exposure.  Extension 1: binary table with
    columns ``lam`` (nm) and ``trans`` (dimensionless transmittance).

tell_ref_airmass_{X.X}.fits
    Telluric spectrum at the reference airmass (used when
    ``tellurics.use_full_skycalc = false`` for a single-airmass reference).
    Same format as the per-exposure files.

pwv_values.fits
    1-D FITS array of PWV values (mm) assigned to each exposure.
    Written only when ``constant_pwv = false``.

Notes
-----
- All SkyCalc parameters (resolution, wavelength grid, moon/star/zodiacal
  components, etc.) are set to match the ``skycalc_model()`` defaults.
- The wavelength grid is queried at R = 150 000 (fixed_spectral_resolution)
  to ensure the simulator can interpolate onto any instrument wavelength grid.
- ``vacair = vac``, vacuum wavelengths, consistent with petitRADTRANS output.
- The observatory code is read from the instrument config
  (e.g. ``lasilla`` for CARMENES, ``paranal`` for CRIRES+, ANDES).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits


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
    open astropy HDUList; caller is responsible for closing it.
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
# PWV generation
# ---------------------------------------------------------------------------

def _gen_pwv(n_spectra: int, ref_pwv: float) -> np.ndarray:
    """Assign per-exposure PWV values from the discrete SkyCalc grid.

    The grid accepted by SkyCalc is:
    0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0 mm.
    Each exposure gets ref_pwv, one step above, or one step below.
    """
    grid = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0])
    idx = int(np.argmin(np.abs(grid - ref_pwv)))
    choices = grid[max(0, idx - 1):min(len(grid), idx + 2)]
    rng = np.random.default_rng(42)
    return rng.choice(choices, size=n_spectra)


# ---------------------------------------------------------------------------
# Airmass computation, uses the same function as the simulator
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

    # n_spectra must match what the simulator builds; the caller (generate())
    # already derives it from the event duration, so use that. Only fall back
    # to a config value if not passed. Read the real config keys
    # (exposure_time_seconds, not the old CARMENES-default _s aliases).
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
                         skip_transits: int = 0) -> np.ndarray:
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
        # computed exactly as the simulator does; constructing PlanetParameters
        # directly from the raw JSON skips the conversions and yields 0.
        from exoplore.planets import load_planet as _load_planet
        _pp = _load_planet(p_file)
        dur_h = _pp.transit_duration_hours
    except Exception:
        dur_h = _transit_duration_hours(p)  # fallback

    exp_s     = float(obs_cfg.get("exposure_time_seconds", 198.0) or 198.0)
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

    return airmass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(config_path: str, overwrite: bool = False,
             mode: str = "astro", search_from: str = None,
             night_idx: int = None, ref_only: bool = False) -> None:
    """Generate SkyCalc telluric FITS files for one EXoPLORE config.

    Parameters
    ----------
    config_path : str
        Path to EXoPLORE JSON config.
    overwrite : bool
        Overwrite existing output files (default: skip existing).
    mode : str
        ``"astro"`` (**default**), real airmass from sky geometry via
        astroplan.  Finds the next observable transit automatically; no
        date input required.  Recommended for all simulations because the
        telluric spectra are generated at the correct per-exposure airmass
        for the actual sky geometry.  Requires ``ra_deg`` and ``dec_deg``
        in the planet parameter file.

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

    inputs_dir   = paths.get("inputs_dir", "inputs/")
    flag_event   = obs.get("flag_event", "full_event")
    # Number of exposures the simulator will build: from the transit duration
    # + pre/post baseline and the exposure cadence (mirrors get_event). The old
    # default of 1 silently produced a single SkyCalc file for synthetic modes.
    n_spectra    = obs.get("n_spectra")
    if n_spectra:
        n_spectra = int(n_spectra)
    else:
        import sys as _sy7, os as _os7, json as _jn7
        _sy7.path.insert(0, _os7.path.join(_os7.path.dirname(__file__),
                                           "..", "src"))
        _pfile = cfg.get("planet", {}).get("parameter_file", "")
        try:
            # load_planet computes transit_duration_hours exactly as the
            # simulator does, so the file count matches the simulator's
            # n_spectra (the raw _transit_duration_hours fallback is ~3 short).
            from exoplore.planets import load_planet as _lp7
            _durh = _lp7(_pfile).transit_duration_hours
        except Exception:
            try:
                with open(_pfile) as _pf7:
                    _durh = _transit_duration_hours(_jn7.load(_pf7))
            except Exception:
                _durh = 2.0
        _exp = float(obs.get("exposure_time_seconds", 198.0) or 198.0)
        _cad = _exp + float(obs.get("readout_time_seconds", 0.0) or 0.0) \
             + float(obs.get("overhead_time_seconds", 0.0) or 0.0)
        _pre = float(obs.get("pre_event_hours", 0.0) or 0.0) or _durh / 2.0
        _post = float(obs.get("post_event_hours", 0.0) or 0.0) or _durh / 2.0
        _step = _cad / 86400.0
        _tot = (_durh + _pre + _post) / 24.0
        n_spectra = len(np.arange(0, _tot + _step, _step))
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
    wave_limits = inst_cfg.get("wavelength_limits_um", None)
    if wave_limits:
        wmin_um, wmax_um = wave_limits
    wmin_nm = wmin_um * 1000.0
    wmax_nm = wmax_um * 1000.0

    # Output directory: per-night when night_idx is set and use_full_skycalc
    pwv_subdir = "Fixed_PWV" if constant_pwv else "Variable_PWV"
    if night_idx is not None:
        out_dir = os.path.join(inputs_dir,
                               f"Skycalc_{flag_event}",
                               f"night_{night_idx}", pwv_subdir)
    else:
        out_dir = os.path.join(inputs_dir,
                               f"Skycalc_{flag_event}", pwv_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Airmass array
    _specific = cfg.get("observation", {}).get("specific_event", False)
    _am_fits = os.path.join(inputs_dir, "reference_night", "airmass_0.fits")
    if _specific and os.path.exists(_am_fits):
        # Observed night: the real per-exposure airmass is already known
        # (the simulator itself reads it from airmass_0.fits), so use it
        # directly rather than recomputing the sky geometry. No astroplan
        # is needed in this case.
        from astropy.io import fits as _fits_am
        airmass = _fits_am.open(_am_fits)[0].data.astype(float)
        print(f"\n  Mode: specific_event (real airmass from airmass_0.fits, "
              f"{airmass.min():.2f} to {airmass.max():.2f})")
    elif _specific:
        # Observed night but no airmass_0.fits: recover the airmass from the
        # target coordinates and the real BJDs (julian_date_0.fits) via
        # astroplan, rather than a synthetic model.
        print("\n  Mode: specific_event (no airmass_0.fits; computing airmass "
              "from target + real BJDs via astroplan)")
        airmass = _build_airmass_astro(cfg, search_from,
                                       skip_transits=night_idx or 0)
    elif mode == "astro":
        print(f"\n  Mode: astro (real sky geometry via astroplan)")
        # For night N, skip the first N observable transits so each night
        # gets its own distinct transit epoch.
        airmass = _build_airmass_astro(cfg, search_from,
                                       skip_transits=night_idx or 0)
    elif am_limits[0] == am_limits[1]:
        airmass = np.full(n_spectra, am_limits[0])
        print(f"\n  Mode: synthetic (fixed airmass {am_limits[0]:.1f})")
    else:
        print(f"\n  Mode: synthetic (parabolic airmass model)")
        airmass = _build_airmass(cfg, n_spectra)

    # PWV array (always constant within a night)
    pwv_arr = np.full(len(airmass), pwv_mm)
    n_spectra = len(airmass)

    night_label = f" night {night_idx}" if night_idx is not None else ""
    print(f"\nGenerating {n_spectra} tell_spec files{night_label}")
    print(f"  Observatory : {observatory}")
    print(f"  λ range     : {wmin_nm:.0f} to {wmax_nm:.0f} nm")
    print(f"  PWV         : {pwv_mm} mm (constant)")
    print(f"  Output      : {out_dir}\n")

    # SkyCalc maximum supported airmass is 3.0.  Clip and warn.
    _SKYCALC_AM_MAX = 3.0
    _n_clipped = int((airmass > _SKYCALC_AM_MAX).sum())
    if _n_clipped:
        print(f"  WARNING: {_n_clipped}/{n_spectra} exposures have airmass > "
              f"{_SKYCALC_AM_MAX} (SkyCalc limit).  Clipping to {_SKYCALC_AM_MAX}.")
        airmass = np.clip(airmass, 1.0, _SKYCALC_AM_MAX)

    # --- Per-exposure files (skipped when --ref-only) ---
    if ref_only:
        print("  --ref-only: skipping per-exposure files, generating reference only.")
    for n in range(n_spectra if not ref_only else 0):
        out_path = os.path.join(out_dir, f"tell_spec_{n}.fits")
        if os.path.exists(out_path) and not overwrite:
            print(f"  [{n+1:3d}/{n_spectra}] tell_spec_{n}.fits, exists, skip")
            continue
        am  = round(float(airmass[n]), 1)
        pwv = float(pwv_arr[n])
        print(f"  [{n+1:3d}/{n_spectra}] tell_spec_{n}.fits  "
              f"(airmass={am:.1f}, pwv={pwv:.1f} mm) ...", end=" ", flush=True)
        t0   = time.time()
        hdul = _query_skycalc(am, pwv, observatory, wmin_nm, wmax_nm)
        if hdul is None:
            print(f"FAILED (SkyCalc returned no data), skipping")
            continue
        _save_hdul(hdul, out_path)
        hdul.close()
        print(f"{time.time()-t0:.1f}s")

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
            hdul = _query_skycalc(ref_airmass, pwv_mm, observatory,
                                   wmin_nm, wmax_nm)
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
  astro (default)
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
  # Single night, astro mode (default):
  python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_transit.json

  # Single night, search from a specific date:
  python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_transit.json \\
      --search-from 2030-01-01

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
        "--mode", choices=["synthetic", "astro"], default="astro",
        help="Airmass mode (default: astro, real sky geometry via astroplan)"
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
    args = parser.parse_args()
    generate(args.config, overwrite=args.overwrite,
             mode=args.mode, search_from=args.search_from,
             night_idx=args.night, ref_only=getattr(args, 'ref_only', False))
