#!/usr/bin/env python3
"""
prepare_crires_night.py
=======================

Ingest an externally reduced CRIRES+ nodding time series into the EXoPLORE
``reference_night`` input format, following the data handling of Nortmann et
al. 2026.

The upstream reduction (ESO cr2res pipeline: dark, flat, wavelength solution
from a uranium neon lamp and a Fabry Perot etalon, per A/B pair nodding
extraction) produces one extracted 1D spectrum per nodding exposure, each with
the CRIRES+ layout of 3 detectors times 6 echelle orders = 18 wavelength
segments (columns ``<order>_01_SPEC``, ``<order>_01_ERR``, ``<order>_01_WL``).
A per segment molecfit fit (Nortmann Appendix A.1) provides a refined
wavelength solution and a theoretical telluric transmittance per nodding
position.  This script assembles those products, computes the barycentric
correction and barycentric Julian date per exposure, and writes the standard
reference night files consumed by the simulator.

Flux is kept in native pixel space and never resampled here; the per exposure
wavelength solution is stored in ``wave_perframe`` so the analysis pipeline can
align the nodding positions and shift to the stellar rest frame itself.

Inputs
------
A reduction directory containing::

    reduced/timeseries_manifest.txt   (mjd  extracted_spectrum_path per line)
    reduced/pair_*/cr2res_obs_nodding_extracted[AB].fits
    molecfit/molecfit_nod[AB].npz     (optional: refined wave + transmittance)

Usage
-----
    python scripts/prepare_crires_night.py <reduction_dir> <output_dir>

References
----------
Nortmann et al. 2026, A&A (arXiv:2604.15292v1).
Wright & Eastman 2014, PASP 126, 838 (barycentric correction).
"""
import argparse
import glob
import os

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

# The molecfit wrapper (per segment wavelength refinement, plausibility gate,
# intra order fallback and theoretical telluric model) lives in the package.
# Make it importable when this script is run from a source checkout.
try:
    from exoplore.instruments import crires_molecfit
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from exoplore.instruments import crires_molecfit

# Cerro Paranal (VLT UT3), where CRIRES+ is mounted.
_PARANAL = EarthLocation.from_geodetic(
    lon=-70.4045 * u.deg, lat=-24.6268 * u.deg, height=2635.0 * u.m)
_C_KMS = 299792.458


def read_segments(path):
    """Return an ordered list of (name, wave_nm, flux, err) for the 18
    segments of one extracted spectrum, sorted by ascending wavelength."""
    h = fits.open(path)
    segs = []
    for ext in h[1:]:
        if not hasattr(ext, "columns"):
            continue
        orders = sorted(set(c.split("_")[0] for c in ext.columns.names
                            if c.endswith("_WL")))
        for o in orders:
            segs.append((f"{ext.name}_{o}",
                         np.asarray(ext.data[f"{o}_01_WL"], float),
                         np.asarray(ext.data[f"{o}_01_SPEC"], float),
                         np.asarray(ext.data[f"{o}_01_ERR"], float)))
    return sorted(segs, key=lambda s: np.nanmin(s[1]))


def barycentric(hdr):
    """Return (BJD_TDB, BERV_kms) at mid exposure from an extracted spectrum
    header, using astropy.  BERV is the barycentric radial velocity correction
    (positive toward the target)."""
    mjd = float(hdr["MJD-OBS"])
    dit = float(hdr.get("HIERARCH ESO DET SEQ1 DIT", hdr.get("EXPTIME", 0.0)))
    t = Time(mjd + 0.5 * dit / 86400.0, format="mjd", scale="utc",
             location=_PARANAL)
    ra = float(hdr.get("RA", hdr.get("HIERARCH ESO TEL TARG ALPHA", 0.0)))
    dec = float(hdr.get("DEC", hdr.get("HIERARCH ESO TEL TARG DELTA", 0.0)))
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    berv = target.radial_velocity_correction(obstime=t).to(u.km / u.s).value
    bjd = (t.tdb + t.light_travel_time(target, kind="barycentric")).jd
    return bjd, berv


def airmass_of(hdr):
    a0 = hdr.get("HIERARCH ESO TEL AIRM START")
    a1 = hdr.get("HIERARCH ESO TEL AIRM END")
    if a0 is not None and a1 is not None:
        return 0.5 * (float(a0) + float(a1))
    return float(hdr.get("AIRMASS", 1.0))


def ensure_molecfit(reduction_dir, run_missing=True, molecules=None):
    """Make sure the molecfit products exist for both nodding positions.

    If ``molecfit/molecfit_nod<AB>.npz`` is absent and ``run_missing`` is set,
    run the molecfit wrapper (this needs the ESO esorex/molecfit tools on the
    PATH; see exoplore.instruments.crires_molecfit).  The npz caches the result
    so this slow step runs only once.  ``molecules`` is an optional (names,
    flags) override of the per band telluric species."""
    reduced = os.path.join(reduction_dir, "reduced")
    for nod in ("A", "B"):
        f = os.path.join(reduction_dir, "molecfit", f"molecfit_nod{nod}.npz")
        if os.path.exists(f) or not run_missing:
            continue
        if not glob.glob(os.path.join(
                reduced, f"pair_*/cr2res_obs_nodding_extracted{nod}.fits")):
            continue  # this nodding position was not observed
        print(f"molecfit products missing for nod {nod}; running the wrapper")
        crires_molecfit.run_night(reduced, nod, molecules=molecules,
                                  out_dir=os.path.join(reduction_dir, "molecfit"))


def load_molecfit(reduction_dir):
    """Load per segment refined wave (nm) and transmittance per nodding
    position, keyed by segment name.  The refined wave is already gated and
    intra order corrected by the wrapper, so it is used as is.  Missing => empty
    (fall back to the reduction wavelength solution, no telluric model)."""
    out = {"A": {}, "B": {}}
    for nod in ("A", "B"):
        f = os.path.join(reduction_dir, "molecfit", f"molecfit_nod{nod}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        for name, wr, tr in zip(d["names"], d["wave_refined"],
                                d["transmittance"]):
            out[nod][str(name)] = (np.asarray(wr, float), np.asarray(tr, float))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reduction_dir")
    ap.add_argument("output_dir")
    ap.add_argument("-n", "--night-index", type=int, default=0)
    ap.add_argument("--species", default=None,
                    help="telluric molecules for molecfit if it must be run, "
                         "e.g. 'H2O,CO2,CH4' (see crires_molecfit.parse_species)")
    args = ap.parse_args()

    manifest = os.path.join(args.reduction_dir, "reduced",
                            "timeseries_manifest.txt")
    rows = []
    with open(manifest) as fh:
        for line in fh:
            mjd, path = line.split()
            rows.append((float(mjd), path))
    rows.sort()

    # Frame quality cut: drop adaptive optics glitch frames whose spatial PSF
    # FWHM or S/N is a strong outlier for this night (the papers observed clean
    # nights; such frames have a very different, degraded resolution).
    _paths = [p for _, p in rows]
    _keep, _fw, _sn = crires_molecfit.quality_keep(_paths)
    if not _keep.all():
        _drop = ", ".join(
            os.path.basename(os.path.dirname(_paths[k]))
            + ("/A" if "extractedA" in _paths[k] else "/B")
            for k in np.where(~_keep)[0])
        print(f"  quality cut: excluding {int((~_keep).sum())} frames "
              f"(PSF FWHM / S-N outliers): {_drop}")
        rows = [r for r, k in zip(rows, _keep) if k]
    n_spectra = len(rows)

    species = crires_molecfit.parse_species(args.species) if args.species else None
    ensure_molecfit(args.reduction_dir, molecules=species)
    molecfit = load_molecfit(args.reduction_dir)

    # Establish the segment order and pixel count from the first spectrum.
    ref_segs = read_segments(rows[0][1])
    seg_names = [s[0] for s in ref_segs]
    n_orders = len(seg_names)
    n_pixels = ref_segs[0][1].size

    all_flux = np.full((n_spectra, n_orders, n_pixels), np.nan)
    all_sig = np.full((n_spectra, n_orders, n_pixels), np.nan)
    wave_pf = np.full((n_spectra, n_orders, n_pixels), np.nan)
    bjd = np.zeros(n_spectra)
    berv = np.zeros(n_spectra)
    airmass = np.zeros(n_spectra)
    nods = np.empty(n_spectra, dtype="<U1")

    for i, (mjd, path) in enumerate(rows):
        nod = "A" if "extractedA" in os.path.basename(path) else "B"
        nods[i] = nod
        hdr = fits.getheader(path)
        bjd[i], berv[i] = barycentric(hdr)
        airmass[i] = airmass_of(hdr)
        for h, (name, wl_nm, flux, err) in enumerate(read_segments(path)):
            all_flux[i, h] = flux
            all_sig[i, h] = err
            # Refined wavelength solution for this exposure's nodding position.
            # The wrapper already gated it and applied the intra order / Fabry
            # Perot fallback, so it is used as is; if no molecfit product exists
            # the native reduction grid is kept.
            wr = molecfit.get(nod, {}).get(name)
            use = np.asarray(wr[0], float) if wr is not None else wl_nm
            wave_pf[i, h] = use / 1000.0  # um

    # Theoretical telluric transmittance per segment for the normalisation
    # masks (median of the two nodding positions where both fit).  Fine for
    # masking; note the two positions carry different per beam wavelength
    # solutions, so this blend must not be used as a line position reference.
    trans = np.ones((n_orders, n_pixels))
    for h, name in enumerate(seg_names):
        ts = [molecfit[_nod][name][1] for _nod in ("A", "B")
              if name in molecfit.get(_nod, {})]
        if ts:
            trans[h] = np.nanmedian(np.vstack(ts), axis=0)

    # Cosmic ray / bad pixel rejection along the time series: robust (Tukey
    # biweight) polynomial fit per channel, replacing 5 sigma outliers with the
    # fit value (Lesjak 2025).  Reuses the engine's Robust_Outlier_Removal, so
    # the fit itself is not pulled by the cosmic ray being removed.
    from exoplore.pipelines.masking import Robust_Outlier_Removal
    _ncr = 0
    for h in range(n_orders):
        _before = np.copy(all_flux[:, h])
        all_flux[:, h], all_sig[:, h] = Robust_Outlier_Removal(
            all_flux[:, h], all_sig[:, h], threshold=5)
        _ncr += int(np.sum(all_flux[:, h] != _before))
    print(f"  cosmic ray clean: replaced {_ncr} time series outliers (5 sigma)")

    # Per exposure telluric frame wavelength alignment (Nortmann 2024 A.2):
    # cross correlate every exposure against the reference (earliest good
    # B position exposure) over the telluric lines, measure the A/B offset and
    # drift, and correct only if they are significant.
    wave_nm, _dr = crires_molecfit.align_telluric_frame(
        all_flux, wave_pf * 1000.0, nods, bjd, trans)
    wave_pf = wave_nm / 1000.0
    print(f"  wavelength alignment: A/B offset {_dr['ab_offset']:+.3f} km/s, "
          f"drift {_dr['drift']:+.3f} km/s (ref exp {_dr['reference']}) -> "
          f"{'CORRECTED' if _dr['applied'] else 'negligible, kept as is'}")

    # Reference grid: the median per exposure solution (µm).
    wave_ref = np.nanmedian(wave_pf, axis=0)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    n = args.night_index
    fits.writeto(f"{out}/julian_date_{n}.fits", bjd, overwrite=True)
    fits.writeto(f"{out}/airmass_{n}.fits", airmass, overwrite=True)
    fits.writeto(f"{out}/observations_berv_{n}.fits", berv, overwrite=True)
    fits.writeto(f"{out}/sig_{n}.fits", all_sig, overwrite=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        fits.writeto(f"{out}/snr_{n}.fits", all_flux / all_sig, overwrite=True)
    fits.writeto(f"{out}/wave_{n}.fits", wave_ref, overwrite=True)
    fits.writeto(f"{out}/wave.fits", wave_ref, overwrite=True)
    fits.writeto(f"{out}/wave_perframe_{n}.fits", wave_pf, overwrite=True)
    fits.writeto(f"{out}/telluric_model_{n}.fits", trans, overwrite=True)
    for h in range(n_orders):
        fits.writeto(f"{out}/observations_night_{n}_order_{h}.fits",
                     all_flux[:, h, :], overwrite=True)

    # Effective spectral resolution per segment from the cr2res slit function
    # (Nortmann A.1 super-resolution check).  Saved so the analysis convolves
    # the H2S template to the night's measured resolution rather than the
    # nominal value: our star can underfill the 0.2" slit and be super-resolved.
    try:
        reso = crires_molecfit.measure_resolution(
            os.path.join(args.reduction_dir, "reduced"), slit_arcsec=0.2)
        R_arr = np.array([reso["R_per_order"].get(nm, reso["R"])
                          for nm in seg_names], float)
        fits.writeto(f"{out}/resolution_{n}.fits", R_arr, overwrite=True)
        print(f"  resolution: median R {reso['R']:.0f} "
              f"({'SUPER-RESOLUTION' if reso['super_resolution'] else 'nominal'})"
              f", per-order {R_arr.min():.0f}-{R_arr.max():.0f} "
              f"(PSF FWHM {reso['fwhm_pix']:.2f} px)")
    except Exception as _e:
        print(f"  resolution: measurement skipped ({_e})")

    n_fit = int(np.sum(np.any(trans < 1.0, axis=1)))
    print(f"Wrote {n_spectra} exposures x {n_orders} segments x {n_pixels} px "
          f"to {out}")
    print(f"  wavelength {np.nanmin(wave_ref):.4f}-{np.nanmax(wave_ref):.4f} µm "
          f"| telluric model on {n_fit}/{n_orders} segments")
    print(f"  BJD {bjd.min():.5f}-{bjd.max():.5f} | "
          f"BERV {berv.min():+.3f}..{berv.max():+.3f} km/s | "
          f"airmass {airmass.min():.2f}-{airmass.max():.2f}")


if __name__ == "__main__":
    main()
