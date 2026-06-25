"""Prepare IGRINS PLP-processed spectra for EXoPLORE.

Reads all IGRINS Pipeline Package (PLP) output files for one transit night
and writes them in the EXoPLORE reference-night format used by the
IGRINS instrument class.

IGRINS data structure
---------------------
The PLP produces one pair of FITS files per A/B nodding pair:

    SDCH_YYYYMMDD_NNNN.spec_a0v.fits[.bz2]   H-band (28 orders x 2048 px)
    SDCK_YYYYMMDD_NNNN.spec_a0v.fits[.bz2]   K-band (25 orders x 2048 px)

Each file has the same structure:

    ext 1  SCI   flux after A0V telluric correction  (n_orders, n_pixels)
    ext 2  VAR   variance of the flux                (n_orders, n_pixels)
    ext 3  SCI   wavelength grid in microns           (n_orders, n_pixels)
    ext 11 DQ    data quality flags (0=good, 1=bad)  (n_orders, n_pixels)

With ``--no-a0v`` the script reads ``.spec.fits`` instead, which has only
three extensions (SCI, VAR, wavelength) without the DQ array:

    ext 1  SCI   raw extracted flux                  (n_orders, n_pixels)
    ext 2  VAR   variance                            (n_orders, n_pixels)
    ext 3  SCI   wavelength grid in microns           (n_orders, n_pixels)

Order convention
----------------
Within each band the PLP stores orders from red to blue (index 0 = reddest).
This script reverses each band and places H before K so that the combined
53-order array increases in wavelength:

    Combined orders 0 to 27  : H-band reversed (1.43 to 1.83 µm)
    Combined orders 28 to 52 : K-band reversed (1.87 to 2.52 µm)

BERV
----
IGRINS PLP does not compute the Barycentric Earth Radial Velocity (BERV).
This script computes it at mid-exposure from the header keywords JD-OBS,
OBJRA, OBJDEC, and the known Gemini South coordinates using
``astropy.coordinates.SkyCoord.radial_velocity_correction``.  The
barycentric convention is used (Solar System barycentre reference frame),
which is the modern standard in high-resolution spectroscopy.  Timestamps
are reported as BJD_TDB.

Outputs (written to ``output_dir``)
------------------------------------
julian_date.fits / julian_date_0.fits
    BJD_TDB at mid-exposure, shape (n_spectra,).

airmass.fits / airmass_0.fits
    Mean of AMSTART and AMEND per exposure, shape (n_spectra,).

observations_berv.fits / observations_berv_0.fits
    Barycentric BERV in km/s at mid-exposure, shape (n_spectra,).

sig.fits / sig_0.fits
    Per-pixel noise (sqrt of variance from ext 2), shape
    (n_spectra, n_orders, n_pixels).

snr.fits / snr_0.fits
    Per-pixel S/N (flux / sigma), shape (n_spectra, n_orders, n_pixels).

observations_night_{n}_order_{h}.fits
    Flux matrix for order h across all spectra, shape (n_spectra, n_pixels).

wave.fits / wave_0.fits
    Wavelength grid in microns from ext 3, shape
    (n_spectra, n_orders, n_pixels).

Usage
-----
Single night (no index suffix on output files)::

    python scripts/prepare_igrins_night.py \\
        /path/to/IGRINS1-PLP_3.2/ \\
        /path/to/inputs/IGRINS/L9859d/reference_night/ \\
        --night-index 0

Multiple nights (output files suffixed with night index)::

    python scripts/prepare_igrins_night.py \\
        /path/to/IGRINS1-PLP_3.2/ \\
        /path/to/inputs/IGRINS/L9859d/reference_night/ \\
        --night-index 1

Use raw extracted flux without A0V telluric pre-correction::

    python scripts/prepare_igrins_night.py \\
        /path/to/IGRINS1-PLP_3.2/ \\
        /path/to/inputs/ \\
        --no-a0v

References
----------
Cheverall et al. 2026, used .spec_a0v.fits (A0V-corrected) as the default;
    tested without it in their Section 3.4.
Wright & Eastman 2014, BERV computation algorithm (implemented in
    astropy.coordinates.radial_velocity_correction).
"""

from __future__ import annotations

import argparse
import bz2
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

# ---------------------------------------------------------------------------
# Gemini South coordinates (Cerro Pachón, Chile)
# Source: Gemini Observatory technical documentation
# ---------------------------------------------------------------------------
_GEMINI_SOUTH = EarthLocation(
    lat=-30.240741 * u.deg,
    lon=-70.736683 * u.deg,
    height=2749.0 * u.m,
)

# IGRINS order layout after reversing each band for ascending wavelength
# H-band: 28 orders (PLP indices 0 to 27, reversed → ascending 1.43 to 1.83 µm)
# K-band: 25 orders (PLP indices 0 to 24, reversed → ascending 1.87 to 2.52 µm)
_N_ORDERS_H = 28
_N_ORDERS_K = 25
_N_ORDERS   = _N_ORDERS_H + _N_ORDERS_K   # 53
_N_PIXELS   = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decompress_if_needed(path: str, tmpdir: str) -> str:
    """Return path to an uncompressed FITS file.

    If *path* ends in ``.bz2`` the file is decompressed into *tmpdir* and
    the temporary path is returned.  Otherwise *path* is returned unchanged.
    """
    if path.endswith(".bz2"):
        dest = os.path.join(tmpdir, os.path.basename(path)[:-4])
        with bz2.open(path, "rb") as fi, open(dest, "wb") as fo:
            fo.write(fi.read())
        return dest
    return path


def _read_spec_file(
    fits_path: str,
    use_a0v: bool,
    tmpdir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read one PLP FITS file (either .spec_a0v or .spec).

    Parameters
    ----------
    fits_path:
        Path to the compressed or uncompressed FITS file.
    use_a0v:
        If True, read from ext 1 (A0V-corrected flux) and ext 11 (DQ).
        If False, read from ext 1 (raw flux); DQ array is all zeros.
    tmpdir:
        Temporary directory for decompression.

    Returns
    -------
    flux : ndarray, shape (n_orders, n_pixels)
    sigma : ndarray, shape (n_orders, n_pixels)
    wave : ndarray, shape (n_orders, n_pixels), wavelength in µm
    dq : ndarray, shape (n_orders, n_pixels), 0=good 1=bad
    hdr : dict-like primary header
    """
    fpath = _decompress_if_needed(fits_path, tmpdir)
    hdul  = fits.open(fpath)

    flux  = hdul[1].data.astype(np.float64)
    var   = hdul[2].data.astype(np.float64)
    wave  = hdul[3].data.astype(np.float64)

    if use_a0v and len(hdul) > 11:
        dq = hdul[11].data.astype(np.int32)
    else:
        dq = np.zeros(flux.shape, dtype=np.int32)

    hdr = hdul[0].header
    hdul.close()

    # Flag DQ-masked pixels as NaN
    bad = dq > 0
    flux[bad] = np.nan
    var[bad]  = np.nan

    # sigma = sqrt(variance); guard against zero/negative variance
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(np.abs(var))
    sigma[var <= 0] = np.nan

    return flux, sigma, wave, dq, hdr


def _compute_berv_bjd(hdr: dict) -> tuple[float, float]:
    """Compute BERV (km/s) and BJD_TDB at mid-exposure from FITS header.

    Uses the barycentric convention (Solar System barycentre) via
    ``astropy.coordinates.SkyCoord.radial_velocity_correction``.

    Parameters
    ----------
    hdr :
        Primary FITS header containing JD-OBS, EXPTIME, OBJRA, OBJDEC.

    Returns
    -------
    berv_kms : float
        Barycentric Earth Radial Velocity in km/s at mid-exposure.
    bjd_tdb : float
        Barycentric Julian Date (TDB timescale) at mid-exposure.
    """
    jd_obs  = float(hdr["JD-OBS"])
    exptime = float(hdr["EXPTIME"])
    ra_deg  = float(hdr["OBJRA"])
    dec_deg = float(hdr["OBJDEC"])

    # Mid-exposure time in JD_UTC
    jd_mid = jd_obs + exptime / 2.0 / 86400.0

    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    # Barycentric correction (do not attach location to Time here to avoid
    # duplicate-location error in astropy)
    t_utc = Time(jd_mid, format="jd", scale="utc")
    berv = target.radial_velocity_correction(
        "barycentric", obstime=t_utc, location=_GEMINI_SOUTH
    )
    berv_kms = berv.to(u.km / u.s).value

    # BJD_TDB: TDB timescale conversion *plus* the barycentric light-travel-time
    # (Roemer) correction.  ``.tdb.jd`` alone only converts the timescale; the
    # spatial light-travel term (up to ~8.3 min; ~98 s for this target/night)
    # must be added explicitly to obtain a true BJD_TDB.
    t_loc    = Time(jd_mid, format="jd", scale="utc", location=_GEMINI_SOUTH)
    ltt_bary = t_loc.light_travel_time(target, kind="barycentric")
    bjd_tdb  = (t_loc.tdb + ltt_bary).jd

    return berv_kms, bjd_tdb


def _airmass_checked(hdr: dict, tol: float = 0.05) -> tuple[float, bool, float, float]:
    """Mean airmass from the header, sanity-checked against astropy.

    The airmass is taken from the data (mean of AMSTART and AMEND).  It is then
    validated against the airmass computed from the target altitude (astropy
    AltAz) at mid-exposure.  If the header value is non-finite or disagrees with
    the astropy value by more than *tol*, the header keyword is treated as
    corrupt (e.g. the AMEND=1.0 placeholder seen in the first frame) and the
    astropy value is returned instead.

    Returns
    -------
    airmass : float
        Header value if consistent, else the astropy value.
    corrupt : bool
        True if the header value failed the sanity check.
    am_hdr, am_astro : float
        The header and astropy values, for logging.
    """
    am_hdr  = 0.5 * (float(hdr["AMSTART"]) + float(hdr["AMEND"]))
    jd      = float(hdr["JD-OBS"]); exptime = float(hdr["EXPTIME"])
    jd_mid  = jd + exptime / 2.0 / 86400.0
    target  = SkyCoord(ra=float(hdr["OBJRA"]) * u.deg,
                       dec=float(hdr["OBJDEC"]) * u.deg, frame="icrs")
    alt     = target.transform_to(
        AltAz(obstime=Time(jd_mid, format="jd", scale="utc"),
              location=_GEMINI_SOUTH)).alt
    am_astro = float(1.0 / np.cos((90.0 * u.deg - alt).to(u.rad).value))
    corrupt  = (not np.isfinite(am_hdr)) or (abs(am_hdr - am_astro) > tol)
    return (am_astro if corrupt else am_hdr), corrupt, am_hdr, am_astro


def _secondary_wavelength_calibration(
    all_flux: np.ndarray,
    all_sigma: np.ndarray,
    all_wave: np.ndarray,
    n_segments: int = 10,
    poly_deg: int = 2,
    max_lag: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Secondary wavelength calibration: 2nd-order stretch+shift, resampled.

    Following Cheverall et al. (2026) Section 2.3 and the methods of
    Line et al. (2021); Brogi et al. (2023); Smith et al. (2024): each spectrum
    is aligned to the spectrum at the END of the observing night (the last
    exposure, closest in time to the PLP wavelength-calibration reference) via a
    per-order, per-exposure SECOND-ORDER stretch-and-shift transform, and the
    flux (and sigma) are RESAMPLED onto the reference exposure's wavelength grid.

    Procedure, per exposure i (≠ reference) and per order h:
      1. Split the order into ``n_segments`` windows; cross-correlate the
         exposure flux against the reference flux in each window to measure a
         sub-pixel pixel shift at each window centre (parabolic peak interp).
      2. Fit a ``poly_deg``-order polynomial Δx(p) = c₀ + c₁p + c₂p² to the
         per-window shifts as a function of detector pixel p (the 2nd-order
         stretch+shift of that spectrum).
      3. Resample the exposure's flux and sigma onto the reference pixel grid by
         interpolating at source pixels p - Δx(p).  Bad pixels (NaN) are kept
         out of the interpolation source and re-imposed on the output (the mask
         is carried through the sub-pixel shift), so masked columns stay masked.

    The reference exposure (last) is returned unchanged.  Because all flux is
    resampled onto the single reference grid, that grid is returned for saving
    (it is now valid for every exposure, no per-exposure grid is discarded).

    Parameters
    ----------
    all_flux, all_sigma : ndarray, shape (n_spectra, n_orders, n_pixels)
        Flux and sigma, with bad/edge pixels as NaN.
    all_wave : ndarray, shape (n_spectra, n_orders, n_pixels)
        PLP wavelength grids (µm).

    Returns
    -------
    flux_out, sigma_out : ndarray
        Flux and sigma resampled onto the reference (last-exposure) grid.
    wave_ref : ndarray, shape (n_orders, n_pixels)
        The reference (last-exposure) wavelength grid.
    """
    n_spectra, n_orders, n_pixels = all_flux.shape
    ref_idx  = n_spectra - 1
    wave_ref = all_wave[ref_idx].copy()
    px       = np.arange(n_pixels, dtype=float)

    flux_out  = all_flux.copy()
    sigma_out = all_sigma.copy()

    centres   = np.linspace(0.08 * n_pixels, 0.92 * n_pixels, n_segments).astype(int)
    half      = max(64, n_pixels // (2 * n_segments))
    lags      = np.arange(-max_lag, max_lag + 1)

    drift_kms = []   # diagnostic: max |Δv| applied

    for i in range(n_spectra):
        if i == ref_idx:
            continue
        for h in range(n_orders):
            f_ref = all_flux[ref_idx, h]
            f_exp = all_flux[i, h]
            good_exp = np.isfinite(f_exp)
            if good_exp.sum() < 0.3 * n_pixels:
                continue                                  # too little signal; leave as-is

            seg_px, seg_shift = [], []
            for c in centres:
                a, b = max(0, c - half), min(n_pixels, c + half)
                sr, se = f_ref[a:b].copy(), f_exp[a:b].copy()
                mr, me = np.isfinite(sr), np.isfinite(se)
                if mr.sum() < 0.5 * (b - a) or me.sum() < 0.5 * (b - a):
                    continue
                sr[~mr] = np.nanmedian(sr[mr]); se[~me] = np.nanmedian(se[me])
                sr -= sr.mean(); se -= se.mean()
                if sr.std() < 1e-12 or se.std() < 1e-12:
                    continue
                # cc(L) = sum( roll(exp,L) * ref ); peak L => shift to apply to exp
                cc = np.array([np.dot(np.roll(se, L), sr) for L in lags])
                k  = int(np.argmax(cc))
                if k == 0 or k == len(cc) - 1:
                    continue                              # peak at the rail: unreliable
                y0, y1, y2 = cc[k-1], cc[k], cc[k+1]
                den = 2.0 * y1 - y0 - y2
                sub = 0.5 * (y2 - y0) / den if abs(den) > 1e-12 else 0.0
                seg_px.append(c); seg_shift.append(lags[k] + sub)

            if len(seg_shift) < poly_deg + 1:
                continue
            seg_px_a, seg_shift_a = np.array(seg_px, float), np.array(seg_shift, float)
            # reject segments far from the median shift (spurious CC locks)
            med = np.median(seg_shift_a)
            keep_seg = np.abs(seg_shift_a - med) < 1.0    # within 1 px of median
            if keep_seg.sum() < poly_deg + 1:
                continue
            seg_px_a, seg_shift_a = seg_px_a[keep_seg], seg_shift_a[keep_seg]
            deg   = min(poly_deg, len(seg_shift_a) - 1)
            coeff = np.polyfit(seg_px_a, seg_shift_a, deg)
            shift_px = np.polyval(coeff, px)              # 2nd-order stretch+shift
            # physical sanity clip: drift is expected <~0.2 px (Cheverall+26)
            shift_px = np.clip(shift_px, -1.0, 1.0)

            # resample exposure flux/sigma onto reference grid: sample exp at p - shift
            src = px - shift_px
            gx  = px[good_exp]
            flux_out[i, h]  = np.interp(src, gx, f_exp[good_exp])
            gs  = np.isfinite(all_sigma[i, h])
            sigma_out[i, h] = (np.interp(src, px[gs], all_sigma[i, h][gs])
                               if gs.sum() > 1 else all_sigma[i, h])
            # re-impose bad-pixel mask, carried through the sub-pixel shift
            nearest = np.clip(np.round(src).astype(int), 0, n_pixels - 1)
            bad_out = ~good_exp[nearest]
            flux_out[i, h, bad_out]  = np.nan
            sigma_out[i, h, bad_out] = np.nan

            dlam = np.nanmean(np.diff(wave_ref[h]))
            drift_kms.append(np.nanmax(np.abs(shift_px)) * dlam
                             / np.nanmean(wave_ref[h]) * 299792.458)

    if drift_kms:
        print(f"  secondary wavecal: max |Δv| applied = {np.nanmax(drift_kms):.3f} km/s "
              f"(median {np.nanmedian(drift_kms):.3f}); reference = last exposure")
    return flux_out, sigma_out, wave_ref


def _wavecal_doppler(
    all_flux: np.ndarray,
    all_sigma: np.ndarray,
    all_wave: np.ndarray,
    vmax: float = 5.0,
    vstep: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-order single-velocity drift correction, applied as a Doppler shift.

    Simpler alternative to the per-segment 2nd-order stretch
    (:func:`_secondary_wavelength_calibration`): for each exposure and order,
    cross-correlate the *whole* order against the last (reference) exposure over
    a velocity grid, find the single best-fit velocity, and shift the
    exposure's flux with the standard Doppler relation
    ``wave_shift = wave*(1 + v/c)``; ``flux_aligned = interp(wave, wave_shift, flux)``.

    One velocity per order per exposure (no stretch).  Self-consistent in sign:
    the velocity that maximises alignment is the one applied.
    """
    c_kms = 299792.458
    n_spectra, n_orders, n_pixels = all_flux.shape
    ref_idx = n_spectra - 1
    vg = np.arange(-vmax, vmax + vstep, vstep)
    flux_out  = all_flux.copy()
    sigma_out = all_sigma.copy()
    drift = []
    for i in range(n_spectra):
        if i == ref_idx:
            continue
        for h in range(n_orders):
            wv   = all_wave[i, h]
            fref = all_flux[ref_idx, h]
            fexp = all_flux[i, h]
            good = np.isfinite(fref) & np.isfinite(fexp)
            if good.sum() < 0.3 * n_pixels:
                continue
            rg = fref[good] - np.mean(fref[good])
            rnorm = np.sqrt(np.sum(rg ** 2))
            fe = np.where(np.isfinite(fexp), fexp, np.nanmedian(fexp[good]))
            cc = np.empty(vg.size)
            for j, v in enumerate(vg):
                es = np.interp(wv, wv * (1.0 + v / c_kms), fe)
                eg = es[good] - np.mean(es[good])
                den = np.sqrt(np.sum(eg ** 2)) * rnorm
                cc[j] = np.sum(eg * rg) / den if den > 0 else 0.0
            k = int(np.argmax(cc))
            if 1 <= k <= len(cc) - 2:
                y0, y1, y2 = cc[k-1], cc[k], cc[k+1]
                d = 2.0 * y1 - y0 - y2
                sub = 0.5 * (y2 - y0) / d if abs(d) > 1e-12 else 0.0
            else:
                sub = 0.0
            vbest = vg[k] + sub * vstep
            ws = wv * (1.0 + vbest / c_kms)
            flux_out[i, h]  = np.interp(wv, ws, fe)
            sigma_out[i, h] = np.interp(wv, ws, all_sigma[i, h])
            badf = (~np.isfinite(fexp)).astype(float)
            bad_shifted = np.interp(wv, ws, badf) > 0.5
            flux_out[i, h, bad_shifted]  = np.nan
            sigma_out[i, h, bad_shifted] = np.nan
            drift.append(abs(vbest))
    if drift:
        print(f"  doppler wavecal: max |v| = {np.max(drift):.3f} km/s "
              f"(median {np.median(drift):.3f}); reference = last exposure")
    return flux_out, sigma_out, all_wave[ref_idx]


def _correct_nan(flux: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set bad/edge pixels to a masking sentinel (0.0) instead of fabricating data.

    DQ-flagged bad pixels and edge-trimmed columns arrive here as NaN.  Rather
    than filling them with the time-median (which injects fabricated values into
    the cross-correlation), they are set to 0.0 so the EXoPLORE pipeline masks
    those columns: any column containing a value <= 0 in any exposure is masked
    (simulator Block 5d.8).  This mirrors the paper's "cleaned of bad pixels and
    outliers" rather than interpolating over them.  The corresponding sigma is
    set to a large value.

    Parameters
    ----------
    flux : ndarray, shape (n_spectra, n_pixels)
    sigma : ndarray, shape (n_spectra, n_pixels)

    Returns
    -------
    flux, sigma : ndarrays with bad pixels set to the masking sentinel.
    """
    bad_f = ~np.isfinite(flux)
    flux[bad_f] = 0.0                                   # -> column masked downstream

    bad_s = ~np.isfinite(sigma) | (sigma < 1e-7)
    sigma[bad_s] = 1e10                                 # large uncertainty

    return flux, sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    plp_dir: str,
    output_dir: str,
    night_index: int | None,
    use_a0v: bool,
    edge_trim_pixels: int = 100,
    apply_wavecal: bool = True,
    wavecal_method: str = "stretch",
    wavecal_poly_deg: int = 2,
) -> None:
    """Read all PLP spec files in *plp_dir* and write EXoPLORE inputs.

    Parameters
    ----------
    plp_dir : str
        Directory containing PLP output files.
    output_dir : str
        Directory where EXoPLORE reference-night files are written.
    night_index : int or None
        Suffix index for multi-night runs; None for single-night.
    use_a0v : bool
        If True, read .spec_a0v.fits (A0V telluric pre-corrected).
        If False, read .spec.fits (raw extracted flux).
    edge_trim_pixels : int
        Number of pixels to mask at each edge of every spectral order to
        remove low-throughput regions due to the instrumental blaze function.
        Brogi et al. (2023) use 100; Smith et al. (2024) use 200.
        Default 100 following Brogi et al. (2023), as cited by Cheverall
        et al. (2026) Section 2.3.
    """
    suffix = "_a0v" if use_a0v else ""
    pattern_k = os.path.join(plp_dir, f"SDCK_*.spec{suffix}.fits*")
    pattern_h = os.path.join(plp_dir, f"SDCH_*.spec{suffix}.fits*")

    files_k = sorted(glob.glob(pattern_k))
    files_h = sorted(glob.glob(pattern_h))

    if not files_k or not files_h:
        print(
            f"ERROR: no files matching:\n  {pattern_k}\n  {pattern_h}\n"
            "Check --no-a0v flag and directory path."
        )
        sys.exit(1)

    if len(files_k) != len(files_h):
        print(
            f"WARNING: unequal number of K ({len(files_k)}) and "
            f"H ({len(files_h)}) files, using minimum."
        )

    n_exp = min(len(files_k), len(files_h))
    print(f"Found {n_exp} exposure pairs (H+K) in {plp_dir}")
    print(f"  Telluric pre-correction: {'A0V (spec_a0v)' if use_a0v else 'none (spec)'}")
    print(f"  Combined orders: {_N_ORDERS} (H {_N_ORDERS_H} + K {_N_ORDERS_K})")
    print(f"  Pixels per order: {_N_PIXELS}")
    print(f"  Edge trim: {edge_trim_pixels} pixels each side (Brogi+2023; Cheverall+2026 Sec. 2.3)")

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ------------------------------------------------------------------
        # First pass: read all exposures
        # ------------------------------------------------------------------
        all_flux  = np.full((n_exp, _N_ORDERS, _N_PIXELS), np.nan)
        all_sigma = np.full((n_exp, _N_ORDERS, _N_PIXELS), np.nan)
        all_wave  = np.full((n_exp, _N_ORDERS, _N_PIXELS), np.nan)
        bjd_arr   = np.zeros(n_exp)
        berv_arr  = np.zeros(n_exp)
        airmass   = np.zeros(n_exp)
        expt      = np.zeros(n_exp)

        for i, (fk, fh) in enumerate(zip(files_k, files_h)):
            print(f"  [{i+1:2d}/{n_exp}] {os.path.basename(fk)}", end=" ... ")

            # Read K-band (25 orders, red→blue)
            flux_k, sig_k, wave_k, _, hdr_k = _read_spec_file(fk, use_a0v, tmpdir)
            # Read H-band (28 orders, red→blue)
            flux_h, sig_h, wave_h, _, _     = _read_spec_file(fh, use_a0v, tmpdir)

            # Reverse each band so wavelength increases within the band,
            # then concatenate: H (ascending) first, K (ascending) second
            # H reversed: PLP index 27→0 maps to combined index 0→27
            # K reversed: PLP index 24→0 maps to combined index 28→52
            flux_h_r = flux_h[::-1, :]   # (28, 2048)
            sig_h_r  = sig_h[::-1,  :]
            wave_h_r = wave_h[::-1, :]

            flux_k_r = flux_k[::-1, :]   # (25, 2048)
            sig_k_r  = sig_k[::-1,  :]
            wave_k_r = wave_k[::-1, :]

            combined_flux  = np.vstack([flux_h_r, flux_k_r])
            combined_sigma = np.vstack([sig_h_r,  sig_k_r])
            combined_wave  = np.vstack([wave_h_r, wave_k_r])

            # Edge trim: mask first and last edge_trim_pixels of every order
            # to remove low-throughput blaze regions (Brogi+2023: 100 px;
            # Smith+2024: 200 px; Cheverall+2026 Section 2.3).
            if edge_trim_pixels > 0:
                combined_flux[:,  :edge_trim_pixels] = np.nan
                combined_flux[:,  -edge_trim_pixels:] = np.nan
                combined_sigma[:, :edge_trim_pixels] = np.nan
                combined_sigma[:, -edge_trim_pixels:] = np.nan

            all_flux[i]  = combined_flux
            all_sigma[i] = combined_sigma
            all_wave[i]  = combined_wave

            # Timing and ancillary, read from K-band header (H is identical
            # for JD-OBS, OBJRA/DEC; use K as authoritative)
            berv_kms, bjd = _compute_berv_bjd(hdr_k)
            bjd_arr[i]  = bjd
            berv_arr[i] = berv_kms
            airmass[i], _am_bad, _am_h, _am_a = _airmass_checked(hdr_k)
            if _am_bad:
                print(f"\n      [airmass] exp {i}: header {_am_h:.3f} inconsistent "
                      f"with astropy {_am_a:.3f} -> using astropy value", end="")
            expt[i]     = float(hdr_k["EXPTIME"])

            sn_mean = np.nanmean(all_flux[i] / all_sigma[i])
            print(f"BJD={bjd:.6f}  BERV={berv_kms:+.3f} km/s  AM={airmass[i]:.3f}  S/N={sn_mean:.0f}")

        # ------------------------------------------------------------------
        # No exposure-level S/N cut.  Cheverall et al. (2026) §2.3 keep every
        # exposure of the night (14 in-transit + 16 out-of-transit = 30); they
        # describe no exposure S/N selection.  The earlier mean-S/N<20 discard
        # here was inherited from the CARMENES reader and is removed for
        # faithfulness to the paper.
        # ------------------------------------------------------------------
        n_spectra = all_flux.shape[0]
        print(f"\n{n_spectra} exposures (no exposure S/N cut; Cheverall+2026 §2.3).")

        # ------------------------------------------------------------------
        # Secondary wavelength calibration (Cheverall+26 Section 2.3):
        # per-exposure 2nd-order stretch+shift aligning each spectrum to the LAST
        # exposure, applied by resampling flux & sigma onto the reference grid
        # (following Line+2021; Brogi+2023; Smith+2024).
        # ------------------------------------------------------------------
        if apply_wavecal and wavecal_method == "doppler":
            print("Applying secondary wavelength calibration (per-order single "
                  "Doppler velocity, wave*(1+v/c)) ...")
            all_flux, all_sigma, wave_ref = _wavecal_doppler(
                all_flux, all_sigma, all_wave
            )
        elif apply_wavecal:
            _ord_name = {1: "linear (1st-order, Line+2021)",
                         2: "2nd-order (Cheverall+2026)"}.get(
                            wavecal_poly_deg, f"{wavecal_poly_deg}-order")
            print(f"Applying secondary wavelength calibration ({_ord_name} stretch+shift, "
                  "resampled to last-exposure grid) ...")
            all_flux, all_sigma, wave_ref = _secondary_wavelength_calibration(
                all_flux, all_sigma, all_wave, poly_deg=wavecal_poly_deg
            )
        else:
            print("Secondary wavelength calibration: SKIPPED (--no-wavecal); "
                  "using PLP wavelength solution directly.")
            wave_ref = all_wave[0]

        # ------------------------------------------------------------------
        # NaN correction per order
        # ------------------------------------------------------------------
        print("Correcting NaN values ...")
        for h in range(_N_ORDERS):
            all_flux[:, h, :], all_sigma[:, h, :] = _correct_nan(
                all_flux[:, h, :].copy(),
                all_sigma[:, h, :].copy(),
            )

        # ------------------------------------------------------------------
        # Write outputs
        # ------------------------------------------------------------------
        n = night_index
        multi = n is not None

        def _name(base):
            return f"{base}_{n}.fits" if multi else f"{base}.fits"

        print(f"\nWriting outputs to {output_dir}")
        fits.writeto(os.path.join(output_dir, _name("julian_date")),
                     bjd_arr, overwrite=True)
        fits.writeto(os.path.join(output_dir, _name("airmass")),
                     airmass, overwrite=True)
        fits.writeto(os.path.join(output_dir, _name("observations_berv")),
                     berv_arr, overwrite=True)
        fits.writeto(os.path.join(output_dir, _name("sig")),
                     all_sigma, overwrite=True)   # (n_spectra, n_orders, n_pixels)
        fits.writeto(os.path.join(output_dir, _name("snr")),
                     all_flux / all_sigma, overwrite=True)
        fits.writeto(os.path.join(output_dir, _name("wave")),
                     wave_ref, overwrite=True)  # reference (last-exposure) grid; all flux resampled onto it

        # Per-order observation files (only needed for Mode C real-data analysis)
        n_idx = n if multi else 0
        for h in range(_N_ORDERS):
            fname = f"observations_night_{n_idx}_order_{h}.fits"
            fits.writeto(
                os.path.join(output_dir, fname),
                all_flux[:, h, :],   # (n_spectra, n_pixels)
                overwrite=True,
            )

        print(f"Done. Files written:")
        print(f"  {_name('julian_date')}  ({n_spectra} BJD_TDB timestamps)")
        print(f"  {_name('airmass')}      ({n_spectra} values)")
        print(f"  {_name('observations_berv')}  ({n_spectra} BERV values, km/s)")
        print(f"  {_name('sig')}          (shape {all_sigma.shape})")
        print(f"  {_name('snr')}          (shape {all_flux.shape})")
        print(f"  {_name('wave')}         (shape {wave_ref.shape}, µm)")
        print(f"  observations_night_{n_idx}_order_{{0..{_N_ORDERS-1}}}.fits")
        print(f"\nBERV range: {berv_arr.min():.4f} to {berv_arr.max():.4f} km/s")
        print(f"BJD range:  {bjd_arr[0]:.6f} to {bjd_arr[-1]:.6f}")
        print(f"Airmass range: {airmass.min():.3f} to {airmass.max():.3f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare IGRINS PLP spectra for EXoPLORE (reference-night format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("plp_dir",    help="Directory containing PLP .spec_a0v.fits[.bz2] files")
    p.add_argument("output_dir", help="Output directory for EXoPLORE reference-night files")
    p.add_argument(
        "--night-index", "-n", type=int, default=None,
        help="Night index for multi-night runs (adds _N suffix to output files). "
             "Omit for single-night."
    )
    p.add_argument(
        "--no-a0v", action="store_true",
        help="Use .spec.fits (raw extracted, no A0V telluric pre-correction) "
             "instead of .spec_a0v.fits. Corresponds to the alternative reduction "
             "tested in Cheverall et al. (2026) Section 3.4."
    )
    p.add_argument(
        "--edge-trim-pixels", type=int, default=100,
        help="Pixels to mask at each order edge (default: 100; Brogi+2023)."
    )
    p.add_argument(
        "--no-wavecal", action="store_true",
        help="Skip the secondary wavelength calibration (use the PLP wavelength "
             "solution directly), as in analyses that omit it."
    )
    p.add_argument(
        "--wavecal-method", choices=["stretch", "doppler"], default="stretch",
        help="Drift-correction method: 'stretch' (per-segment polynomial pixel "
             "stretch+shift, default) or 'doppler' (per-order single velocity, "
             "wave*(1+v/c))."
    )
    p.add_argument(
        "--wavecal-poly-deg", type=int, default=2,
        help="Polynomial order of the 'stretch' wavecal: 2 = 2nd-order "
             "(Cheverall+2026, default), 1 = linear (Line+2021 'linear stretch "
             "re-alignment'). Ignored for --wavecal-method doppler."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        plp_dir=args.plp_dir,
        output_dir=args.output_dir,
        night_index=args.night_index,
        use_a0v=not args.no_a0v,
        edge_trim_pixels=args.edge_trim_pixels,
        apply_wavecal=not args.no_wavecal,
        wavecal_method=args.wavecal_method,
        wavecal_poly_deg=args.wavecal_poly_deg,
    )
