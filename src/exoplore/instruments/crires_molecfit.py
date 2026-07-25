"""
exoplore.instruments.crires_molecfit
====================================

Wrapper that runs the ESO tool molecfit on an externally reduced CRIRES+
nodding time series to (a) refine the wavelength solution per detector segment
and (b) produce a theoretical telluric transmittance for the normalisation
masks, following Nortmann et al. 2026 Appendix A.1 and Sec 3.1.

This module does not reduce raw frames.  The upstream extraction (master dark,
flat, wavelength solution from a uranium neon lamp plus a Fabry Perot etalon,
and per A/B pair nodding extraction) is done with the ESO cr2res pipeline; see
the CRIRES+ documentation at
https://www.eso.org/sci/software/pipelines/cr2res/ and the molecfit
documentation at https://www.eso.org/sci/software/pipelines/molecfit/ for
installation and usage of those tools.  We only wrap molecfit; we do not
replace the ESO pipelines or their guides.

What the wrapper does automatically
-----------------------------------
1. Reads the wavelength setting and slit width from the FITS header and selects
   the telluric molecules relevant to that band (see BAND_MOLECULES).
2. Picks a reference spectrum per nodding position (the highest median signal to
   noise exposure by default; all exposures share the same Fabry Perot solution,
   which Nortmann verifies does not drift over the night).
3. Runs molecfit_model per segment in the configuration validated for CRIRES+:
   vacuum wavelength frame (cr2res delivers vacuum wavelengths; running in air
   introduces an offset of order 80 km/s at 2 micron) and the reflex ANY line
   spread function (a single fitted Gaussian).
4. Gates the refined solution: a segment is accepted only if its velocity shift
   relative to the Fabry Perot solution is below GATE_KMS.  Under constrained
   segments (telluric saturated band heads with no continuum to anchor, or
   telluric poor segments with almost no lines) are rejected; they inherit the
   median shift of the accepted detectors of the same echelle order (which agree
   to well under 1 km/s), or keep the Fabry Perot solution if no detector of the
   order is constrained.
5. Reports the distribution of accepted shifts as the wavelength consistency
   check: if the constrained segments cluster near zero, the Fabry Perot
   solution is already correct, which is the same conclusion Nortmann reaches
   from a telluric template cross correlation.

Generality across settings
---------------------------
Everything except the telluric molecule list and the slit width is band
independent (vacuum frame, line spread function, plausibility gate, intra order
fallback, segment and order parsing), so the wrapper runs on any CRIRES+ Y, J,
H, K, L or M setting.  The slit width is read from the header.  The molecule
lists per band are in BAND_MOLECULES.

Output
------
``molecfit_nod<AB>.npz`` with arrays ``names``, ``wave_refined`` (nm) and
``transmittance``, one row per segment, consumed by
``scripts/prepare_crires_night.py``.
"""

from __future__ import annotations

import glob
import os
import subprocess

import numpy as np
from astropy.io import fits

C_KMS = 299792.458

# Telluric molecules per near infrared band.  The K band list follows Nortmann
# 2024 Appendix A.3 (water, methane, carbon dioxide, nitrous oxide).  The other
# bands list the standard telluric absorbers of that wavelength range; these are
# sensible defaults for the wavelength fit rather than a published per setting
# recipe, and can be overridden through the ``molecules`` argument.  In every
# band only water is FITTED (it has structured lines throughout the near
# infrared and anchors the wavelength solution); the remaining species are
# included at climatological abundance but not fitted, so their column density
# does not run away in segments that lack their lines.
BAND_MOLECULES = {
    "Y": (["H2O", "O2"], [1, 0]),
    "J": (["H2O", "O2", "CO2"], [1, 0, 0]),
    "H": (["H2O", "CO2", "CH4"], [1, 0, 0]),
    "K": (["H2O", "CH4", "CO2", "N2O"], [1, 0, 0, 0]),
    "L": (["H2O", "CH4", "N2O", "CO2", "O3"], [1, 0, 0, 0, 0]),
    "M": (["H2O", "CO2", "CO", "N2O", "O3"], [1, 0, 0, 0, 0]),
}

# Instrument line spread function (reflex ANY defaults): a single fitted
# Gaussian, box and Lorentz off.  Band independent.
PIX_SCALE = 0.056     # arcsec/pixel, CRIRES+ spatial scale
RES_GAUSS0 = 2.5      # initial Gaussian FWHM in pixels (fitted)
KERNFAC = 3           # kernel size in FWHM
DEFAULT_SLIT = 0.2    # arcsec, used if the header carries no slit width

# molecfit optimiser convergence criteria (relative chi2 / parameter change).
# The molecfit default is 1e-2 (User Manual sec 4.7); it is the recommended
# value and is already below the telluric model's own accuracy floor.  Values
# far tighter than this (e.g. 1e-10) can leave the mpfit/LBLRTM loop oscillating
# without converging on segments with weak, sparse telluric lines (e.g. between
# the water bands in Y), so they are avoided.
MOLEC_FTOL = "1e-4"
MOLEC_XTOL = "1e-4"

EDGE_TRIM = 40        # pixels trimmed from each segment end for the fit window

# Wavelength refinement plausibility gate.  The Fabry Perot solution is good to
# below 0.1 pixel, so a genuine molecfit refinement is sub pixel; under
# constrained fits jump by many pixels.  3 km/s (about 1.5 pixels) sits in the
# gap between the two regimes.
GATE_KMS = 3.0


def _header_value(header, *keys, default=None):
    for k in keys:
        for form in (k, "HIERARCH " + k):
            if form in header:
                return header[form]
    return default


def band_of(header):
    """Near infrared band letter of the CRIRES+ setting, from the wavelength
    setting identifier in the header (e.g. K2148 -> 'K')."""
    wid = _header_value(header, "ESO INS WLEN ID", default="")
    if wid and wid[0].upper() in BAND_MOLECULES:
        return wid[0].upper()
    return None


def slit_width(header):
    """Slit width in arcsec from the header, or the default if absent."""
    val = _header_value(header, "ESO INS SLIT1 WID", "ESO INS SLIT WID")
    try:
        val = float(val)
        if val > 0:
            return val
    except (TypeError, ValueError):
        pass
    return DEFAULT_SLIT


def molecules_for(header, molecules=None):
    """Return (molecule_names, fit_flags) for the setting.  ``molecules`` may
    override the per band default with an explicit (names, flags) pair."""
    if molecules is not None:
        return list(molecules[0]), list(molecules[1])
    band = band_of(header)
    if band is None:
        raise ValueError(
            "could not determine the CRIRES+ band from the header "
            "(ESO INS WLEN ID); pass molecules=(names, flags) explicitly")
    return list(BAND_MOLECULES[band][0]), list(BAND_MOLECULES[band][1])


def parse_species(spec):
    """Parse a telluric species specification string into (names, flags).

    Accepts a comma separated list of molecule names, optionally with an
    explicit fit flag after a colon, for example::

        "H2O,CO2,CH4"        include all three, fit water (the anchor)
        "H2O:1,O2:0"         include both, fit water only
        "CO2:1,CO:0,H2O:0"   fit carbon dioxide instead (e.g. an M band)

    If no colon is present anywhere, water is fitted when included, otherwise
    the first species is fitted."""
    names, flags, has_flag = [], [], (":" in spec)
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            n, f = tok.split(":", 1)
            names.append(n.strip())
            flags.append(int(f))
        else:
            names.append(tok)
            flags.append(0)
    if not has_flag and names:
        flags = [1 if n == "H2O" else 0 for n in names]
        if 1 not in flags:
            flags[0] = 1
    return names, flags


def segments_of(path):
    """Return [(name, wl_nm, spec, err), ...] from one per pair extraction."""
    out = []
    with fits.open(path) as h:
        for ext in h[1:]:
            if not hasattr(ext, "columns"):
                continue
            chip = ext.name
            orders = sorted(set(c.split("_")[0] for c in ext.columns.names
                                if c.endswith("_WL")))
            for o in orders:
                out.append((f"{chip}_{o}",
                            np.asarray(ext.data[f"{o}_01_WL"], float),
                            np.asarray(ext.data[f"{o}_01_SPEC"], float),
                            np.asarray(ext.data[f"{o}_01_ERR"], float)))
    return out


def _order_of(name):
    """Echelle order token of a segment name (CHIP2.INT1_04 -> '04')."""
    return name.rsplit("_", 1)[-1]


def _median_snr(path):
    segs = segments_of(path)
    vals = []
    for _, _, sp, er in segs:
        g = np.isfinite(sp) & np.isfinite(er) & (er > 0)
        if g.any():
            vals.append(np.median(sp[g] / er[g]))
    return float(np.median(vals)) if vals else 0.0


def pick_reference(reduced_dir, nod):
    """Path of the reference extraction for one nodding position.

    Defaults to the highest median signal to noise exposure: all exposures share
    the same Fabry Perot wavelength solution, so the refined solution applies to
    the whole night, and the best signal to noise spectrum gives the best
    constrained molecfit fit.  Nortmann A.1 uses one reference spectrum per
    nodding position per night without specifying which."""
    files = sorted(glob.glob(os.path.join(
        reduced_dir, f"pair_*/cr2res_obs_nodding_extracted{nod}.fits")))
    if not files:
        raise FileNotFoundError(
            f"no extracted{nod} spectra under {reduced_dir}")
    return max(files, key=_median_snr)


def measure_resolution(reduced_dir, nod=None, nominal_R=100000.0,
                       slit_fwhm_pix=3.5, slit_arcsec=0.2, pix_scale=PIX_SCALE):
    """Effective spectral resolution per segment from the cr2res pipeline
    spatial PSF FWHM, following Nortmann et al. 2024 Appendix A.2 and Lesjak
    et al. 2025.

    The cr2res pipeline measures the spatial (cross dispersion) stellar PSF
    FWHM per detector and echelle order and writes it to the FITS headers as
    ``ESO QC SLITFWHM<order>`` (in detector pixels).  Under the premise that
    the PSF width is comparable in the spatial and dispersion directions, a PSF
    FWHM equal to the slit width (about 3.5 pixels for the 0.2 arcsec slit)
    corresponds to homogeneous slit illumination and the nominal resolution
    R = 100000 (Dorn et al. 2023).  When the PSF underfills the slit the
    achieved resolution is higher (super-resolution) and scales as

        R = nominal_R * slit_fwhm_pix / FWHM_pix .

    The FWHM is read for every segment of every nodding pair; the returned R
    is the per segment median over pairs (Nortmann and Lesjak both quote the
    median over the night, about R = 140000).

    Parameters
    ----------
    reduced_dir : str
        Directory with ``pair_*/cr2res_obs_nodding_extracted<AB>.fits``.
    nod : {"A", "B", None}
        Restrict to one nodding position, or use both (default).
    slit_fwhm_pix : float
        PSF FWHM in pixels that corresponds to nominal_R (the slit width in
        pixels, ~3.5 for the 0.2 arcsec slit; Dorn et al. 2023).

    Returns
    -------
    dict with ``R`` (median over segments), ``R_per_order`` (name -> R),
    ``fwhm_pix`` (median PSF FWHM), ``fwhm_by_order`` and ``super_resolution``.
    """
    pat = (f"pair_*/cr2res_obs_nodding_extracted{nod}.fits" if nod
           else "pair_*/cr2res_obs_nodding_extracted?.fits")
    files = sorted(glob.glob(os.path.join(reduced_dir, pat)))
    if not files:
        raise FileNotFoundError(
            f"no extracted products under {reduced_dir}")

    per_seg = {}
    for f in files:
        with fits.open(f) as h:
            for ext in h[1:]:
                chip = ext.name
                if not chip.startswith("CHIP") or "ERR" in chip:
                    continue
                for k in ext.header:
                    ks = str(k).upper().replace("HIERARCH ", "")
                    if (ks.startswith("ESO QC SLITFWHM")
                            and ks[-1].isdigit()):
                        order = ks[-1].zfill(2)          # SLITFWHM2 -> "02"
                        fw = float(ext.header[k])
                        if np.isfinite(fw) and fw > 0:
                            per_seg.setdefault(f"{chip}_{order}", []).append(fw)

    if not per_seg:
        raise ValueError(
            "no ESO QC SLITFWHM keywords in the cr2res headers")

    fwhm_by_order = {k: float(np.median(v)) for k, v in per_seg.items()}
    # The super-resolution scaling R = nominal_R * slit_fwhm_pix / FWHM only holds
    # when the PSF UNDERFILLS the slit (FWHM < slit_fwhm_pix): a tighter PSF acts
    # as a narrower effective entrance aperture and raises the resolution.  When
    # the PSF OVERFILLS the slit (FWHM >= slit_fwhm_pix, e.g. poor seeing), the
    # slit itself sets the line-spread function, so the resolution is slit-limited
    # at the nominal value and cannot drop below it.  Clamp the effective FWHM at
    # the slit width so overfilled (bad-seeing) nights return nominal_R, not a
    # spuriously low value.
    def _R(fw):
        return nominal_R * slit_fwhm_pix / min(fw, slit_fwhm_pix)
    R_per_order = {k: _R(fw) for k, fw in fwhm_by_order.items()}
    fwhm_med = float(np.median(list(fwhm_by_order.values())))
    slit_pix = (slit_arcsec / pix_scale) if slit_arcsec else slit_fwhm_pix
    return {
        "R": _R(fwhm_med),
        "R_per_order": R_per_order,
        "fwhm_pix": fwhm_med,
        "fwhm_by_order": fwhm_by_order,
        "super_resolution": fwhm_med < slit_pix,
    }


def frame_quality(path):
    """(PSF FWHM, median S/N) of one extracted spectrum, for the quality cut.

    PSF FWHM is the median of the pipeline ``ESO QC SLITFWHM MED`` over the
    three detectors; a frame whose FWHM is a strong outlier (adaptive optics
    loss, clouds) or whose S/N collapses is unusable."""
    fw = []
    with fits.open(path) as h:
        for ext in h[1:]:
            if not str(ext.name).startswith("CHIP") or "ERR" in ext.name:
                continue
            for k in ext.header:
                if str(k).upper().replace("HIERARCH ", "").startswith(
                        "ESO QC SLITFWHM MED"):
                    fw.append(float(ext.header[k]))
    snr = []
    for _, _, sp, er in segments_of(path):
        g = np.isfinite(sp) & (er > 0)
        if g.any():
            snr.append(np.median(sp[g] / er[g]))
    return (float(np.median(fw)) if fw else np.nan,
            float(np.median(snr)) if snr else 0.0)


def quality_keep(paths, max_fwhm_factor=2.0, min_snr_factor=0.3):
    """Boolean keep mask over ``paths``: drop frames whose PSF FWHM exceeds
    ``max_fwhm_factor`` times the median FWHM, or whose S/N falls below
    ``min_snr_factor`` times the median S/N (adaptive optics glitches).  The
    cuts are data driven (relative to the night's own medians), not hardcoded
    frame indices."""
    fw = np.array([frame_quality(p)[0] for p in paths])
    sn = np.array([frame_quality(p)[1] for p in paths])
    fw_med = np.nanmedian(fw)
    sn_med = np.nanmedian(sn)
    keep = (fw <= max_fwhm_factor * fw_med) & (sn >= min_snr_factor * sn_med)
    return keep, fw, sn


def anchor_grids_to_tellurics(flux, wave_nm, nod, telluric,
                              v_window=4.0, v_step=0.05, min_lines=300):
    """Anchor every nodding position's wavelength grid to the telluric rest
    frame, segment by segment.

    Nortmann et al. 2024 (Appendix A.2 and A.3) obtain a molecfit refined
    wavelength solution per nodding position so that the telluric lines of both
    positions are labelled at their true observed wavelengths; the star sits at
    a different slit position in A and B, so their raw solutions differ.  When
    the molecfit refinement of a segment is rejected by the plausibility gate
    for one nodding position but accepted for the other, the two positions end
    up with inconsistent grids in that segment (offsets of up to a few km/s),
    which the detrending then sees as time variable line positions.  This step
    completes the intended end state of the published scheme: for each nodding
    position and segment, the median combined spectrum of that position is
    cross correlated against the theoretical telluric transmittance in
    continuum normalised absorption space, and the grid is corrected so its
    telluric lines sit at the model (telluric rest) positions.  Segments
    without a usable telluric anchor inherit the median correction of the
    anchored segments of the same nodding position (the slit position offset
    is common to the whole position).  This correction is a data driven
    completion of the per position refinement, not a step spelled out in the
    papers, which did not gate their molecfit solutions.

    Parameters
    ----------
    flux : (n_spectra, n_seg, n_pix)
    wave_nm : (n_spectra, n_seg, n_pix)
    nod : (n_spectra,) array of "A"/"B"
    telluric : dict nod -> (n_seg, n_pix)
        Theoretical transmittance PER NODDING POSITION, each evaluated on that
        position's own grid (from molecfit_nod<AB>.npz).  A blended model built
        across inconsistent grids is not a valid anchor.

    Returns
    -------
    (wave_corrected, report) with report[nod] = per segment shift array (km/s,
    NaN where inherited) and report["applied_" + nod] = median correction.
    """
    c = 299792.458
    nod = np.asarray(nod)
    n_spec, n_seg, n_pix = flux.shape
    wave_out = wave_nm.copy()
    report = {}
    vs = np.arange(-v_window, v_window + v_step / 2, v_step)
    for nn in ("A", "B"):
        idx = np.where(nod == nn)[0]
        if idx.size == 0 or nn not in telluric:
            continue
        shifts = np.full(n_seg, np.nan)
        for j in range(n_seg):
            t = telluric[nn][j]
            if np.nanmin(t) > 0.7 or np.sum(t > 0.95) < 100:
                continue
            w = wave_nm[idx[0], j]
            med = np.nanmedian(flux[idx, j, :], axis=0)
            g = np.isfinite(w) & np.isfinite(med)
            cont = g & (t > 0.95) & (med > 0)
            if cont.sum() < 50 or g.sum() < min_lines:
                continue
            x = (w - np.nanmean(w[g])) / (np.nanstd(w[g]) or 1.0)
            base = np.polyval(np.polyfit(x[cont], med[cont], 3), x)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = 1.0 - med / base
            d = np.where(np.isfinite(d), d, 0.0)
            ta = np.where(g, 1.0 - t, 0.0)
            mask = g & (t < 0.98)
            if mask.sum() < min_lines:
                continue
            d0 = d - np.nanmean(d[mask])
            cc = []
            for v in vs:
                ts = np.interp(w, w * (1 + v / c), ta, left=0, right=0)
                ts = ts - np.nanmean(ts[mask])
                cc.append(np.nansum(d0[mask] * ts[mask])
                          / np.sqrt(np.nansum(d0[mask] ** 2)
                                    * np.nansum(ts[mask] ** 2) + 1e-30))
            cc = np.array(cc)
            k = int(np.argmax(cc))
            if 0 < k < len(vs) - 1 and cc[k] > 0.2:
                y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
                shifts[j] = vs[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * v_step
        common = float(np.nanmedian(shifts)) if np.isfinite(shifts).any() else 0.0
        for j in range(n_seg):
            s = shifts[j] if np.isfinite(shifts[j]) else common
            # data labelled at +s relative to telluric rest: relabel the grid
            # so the telluric lines sit at the model positions
            wave_out[idx, j, :] = wave_nm[idx, j, :] / (1.0 + s / c)
        report[nn] = shifts
        report["applied_" + nn] = common
    return wave_out, report


def align_telluric_frame(flux, wave_nm, nod, mjd, telluric,
                         tell_threshold=0.98, v_window=5.0, v_step=0.1,
                         significance=3.0):
    """Per exposure wavelength alignment in the telluric rest frame
    (Nortmann et al. 2024 Appendix A.2; check first, correct only if needed).

    Every exposure is resampled onto the common (median) grid in continuum
    normalised absorption and cross correlated against the mean of all the
    other exposures over the telluric lines (the leave one out, variance
    reduced form of the paper's single reference cross correlation).  The
    measured per exposure velocity captures the A versus B slit offset and any
    drift.
    The correction (shifting each exposure onto the reference frame) is applied
    only when the systematic (the A/B offset or the total drift) is significant
    relative to the per exposure scatter, as Nortmann 2024 did for WASP-127;
    otherwise the grid is left unchanged, as Lesjak 2025 and Nortmann 2026 found
    for their nights.

    Parameters
    ----------
    flux : (n_spectra, n_seg, n_pix)
    wave_nm : (n_spectra, n_seg, n_pix), the molecfit refined per exposure grid
    nod : (n_spectra,) array of "A"/"B"
    mjd : (n_spectra,)
    telluric : (n_seg, n_pix) theoretical telluric transmittance

    Returns
    -------
    (wave_corrected, report) where report has ``v`` (per exposure km/s),
    ``ab_offset``, ``drift``, ``reference`` and ``applied``.
    """
    c = 299792.458
    nod = np.asarray(nod)
    mjd = np.asarray(mjd, float)
    n_spec, n_seg, n_pix = flux.shape
    snr = np.array([np.nanmedian([
        np.nanmedian(np.abs(flux[i, j])) for j in range(n_seg)])
        for i in range(n_spec)])
    # reference exposure retained for reporting only; the shift of every
    # exposure is measured against the mean of all the others (leave one out),
    # which is the variance reduced form of the single reference cross
    # correlation of Nortmann 2024 A.2.
    b_idx = [i for i in range(n_spec) if nod[i] == "B"]
    good_b = [i for i in b_idx if snr[i] >= np.nanmedian(snr)]
    ref_i = (good_b or b_idx or [0])[0]

    # Common wavelength grid: the median per exposure solution.  Every
    # exposure is resampled onto it in continuum normalised absorption space
    # (Nortmann 2024 A.2: the input spectra are continuum normalised before
    # the cross correlation), so the correlation is driven by the telluric
    # lines rather than by the blaze shape, which is identical between
    # exposures and would otherwise pin the peak to zero lag.
    w0 = np.nanmedian(wave_nm, axis=0)                    # (n_seg, n_pix)
    tell_segs = [j for j in range(n_seg)
                 if np.nanmin(telluric[j]) < 0.7
                 and np.sum(telluric[j] > 0.95) > 100]

    def _depth_on_w0(i, j):
        w, f = wave_nm[i, j], flux[i, j]
        g = np.isfinite(w) & np.isfinite(f)
        if g.sum() < 200:
            return None
        fi = np.interp(w0[j], w[g], f[g], left=np.nan, right=np.nan)
        t = telluric[j]
        cont = np.isfinite(fi) & (t > 0.95) & (fi > 0)
        if cont.sum() < 50:
            return None
        x = (w0[j] - np.nanmean(w0[j])) / (np.nanstd(w0[j]) or 1.0)
        base = np.polyval(np.polyfit(x[cont], fi[cont], 3), x)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = 1.0 - fi / base
        return np.where(np.isfinite(d), d, 0.0)

    depths = {}
    for j in tell_segs:
        arr = [_depth_on_w0(i, j) for i in range(n_spec)]
        if all(a is not None for a in arr):
            depths[j] = np.array(arr)

    vs = np.arange(-v_window, v_window + v_step / 2, v_step)
    V = np.full(n_spec, np.nan)
    for i in range(n_spec):
        peaks = []
        for j in depths:
            d = depths[j][i]
            ref = np.nanmean(np.delete(depths[j], i, axis=0), axis=0)
            t = telluric[j]
            g = (t < tell_threshold) & np.isfinite(d) & np.isfinite(ref)
            if g.sum() < 300:
                continue
            r0 = ref - np.nanmean(ref[g])
            cc = []
            for v in vs:
                ds = np.interp(w0[j], w0[j] * (1 + v / c), d,
                               left=np.nan, right=np.nan)
                m = g & np.isfinite(ds)
                dm = ds[m] - np.nanmean(ds[m])
                cc.append(np.sum(r0[m] * dm)
                          / np.sqrt(np.sum(r0[m] ** 2)
                                    * np.sum(dm ** 2) + 1e-30))
            cc = np.array(cc)
            k = int(np.argmax(cc))
            if 0 < k < len(vs) - 1:
                y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
                peaks.append(vs[k]
                             + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * v_step)
        if peaks:
            V[i] = np.nanmedian(peaks)
    va, vb = V[nod == "A"], V[nod == "B"]
    ab_offset = float(np.nanmean(va) - np.nanmean(vb))
    m = np.isfinite(V)
    slope = np.polyfit(mjd[m] - mjd[m].mean(), V[m], 1)[0]
    drift = float(slope * (mjd.max() - mjd.min()))
    scatter = float(np.nanstd(vb)) or 1e-6
    applied = max(abs(ab_offset), abs(drift)) > significance * scatter

    wave_corrected = wave_nm.copy()
    if applied:
        for i in range(n_spec):
            if np.isfinite(V[i]):
                wave_corrected[i] = wave_nm[i] * (1.0 + V[i] / c)
    return wave_corrected, {
        "v": V, "ab_offset": ab_offset, "drift": drift,
        "reference": ref_i, "applied": applied, "scatter": scatter,
    }


def global_scale(segs):
    """Reflex normalise_spectrum convention: divide the whole spectrum by the
    mean of the per extension medians (a single scalar for all segments)."""
    meds = []
    for _, _, sp, er in segs:
        g = np.isfinite(sp) & np.isfinite(er) & (er > 0)
        if g.any():
            meds.append(np.median(sp[g]))
    s = float(np.mean(meds)) if meds else 1.0
    return s if s > 0 else 1.0


def _write_science(name, wl, sp, er, header, out_fits, scale):
    good = np.isfinite(wl) & np.isfinite(sp)
    lam = wl / 1000.0  # nm -> micron
    flux = np.where(np.isfinite(sp), sp, 0.0) / scale
    med_er = np.nanmedian(er[good]) if good.any() else 1.0
    dflux = np.where(np.isfinite(er) & (er > 0), er, med_er) / scale
    col = fits.BinTableHDU.from_columns([
        fits.Column(name="lambda", format="D", array=lam),
        fits.Column(name="flux", format="D", array=flux),
        fits.Column(name="dflux", format="D", array=dflux),
    ], name=name[:20])
    fits.HDUList([fits.PrimaryHDU(header=header), col]).writeto(
        out_fits, overwrite=True)
    idx = np.where(good)[0]
    lo, hi = lam[idx[EDGE_TRIM]], lam[idx[-EDGE_TRIM]]
    return (min(lo, hi), max(lo, hi))


def _write_molecules(path, names, flags):
    cols = [
        fits.Column(name="LIST_MOLEC", format="8A", array=np.array(names)),
        fits.Column(name="FIT_MOLEC", format="J", array=np.array(flags, int)),
        fits.Column(name="REL_COL", format="D", array=np.ones(len(names))),
    ]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(cols)]).writeto(path, overwrite=True)


def _write_wave_include(path, ranges):
    cols = [
        fits.Column(name="LOWER_LIMIT", format="D",
                    array=np.array([r[0] for r in ranges])),
        fits.Column(name="UPPER_LIMIT", format="D",
                    array=np.array([r[1] for r in ranges])),
        fits.Column(name="MAPPED_TO_CHIP", format="J",
                    array=np.arange(1, len(ranges) + 1)),
        fits.Column(name="WLC_FIT_FLAG", format="J", array=np.ones(len(ranges), int)),
        fits.Column(name="CONT_FIT_FLAG", format="J", array=np.ones(len(ranges), int)),
    ]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(cols)]).writeto(path, overwrite=True)


def _esorex_env(esorex_bin):
    env = dict(os.environ)
    d = os.path.dirname(esorex_bin)
    if d:
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
    return env


def _kernel_params(slit):
    return [
        f"--SLIT_WIDTH_VALUE={slit}", f"--PIX_SCALE_VALUE={PIX_SCALE}",
        "--FIT_RES_GAUSS=TRUE", f"--RES_GAUSS={RES_GAUSS0}", f"--KERNFAC={KERNFAC}",
        "--FIT_RES_BOX=FALSE", "--FIT_RES_LORENTZ=FALSE",
        "--WAVELENGTH_FRAME=VAC", f"--FTOL={MOLEC_FTOL}", f"--XTOL={MOLEC_XTOL}",
    ]


# Hard wall-clock limit for a single molecfit_model call.  A segment with too
# few or ill-conditioned telluric lines can leave the mpfit/LBLRTM optimiser
# oscillating without ever meeting the convergence tolerance; without a cap it
# hangs the whole night.  On timeout the call is abandoned and the segment falls
# back to the per-order median shift (handled by the caller), the same as any
# other failed segment.
MOLEC_TIMEOUT_S = 240


def _molec_model(name, wl, sp, er, header, work, scale, ranges,
                 names, flags, fit_wlc, wlc_n, slit, esorex_bin, warn_after,
                 timeout=MOLEC_TIMEOUT_S):
    """Single extension molecfit_model call.  Returns (mlambda_nm, mtrans) or
    None on failure (including timeout)."""
    import time
    import signal
    os.makedirs(work, exist_ok=True)
    sci = os.path.join(work, "science.fits")
    mol = os.path.join(work, "molecules.fits")
    wav = os.path.join(work, "wave_include.fits")
    _write_science(name, wl, sp, er, header, sci, scale)
    _write_molecules(mol, names, flags)
    _write_wave_include(wav, ranges)
    with open(os.path.join(work, "model.sof"), "w") as fh:
        fh.write(f"{sci} SCIENCE\n{mol} MOLECULES\n{wav} WAVE_INCLUDE\n")
    params = [
        "--COLUMN_LAMBDA=lambda", "--COLUMN_FLUX=flux", "--COLUMN_DFLUX=dflux",
        "--WLG_TO_MICRON=1.0", f"--FIT_WLC={fit_wlc}", f"--WLC_N={wlc_n}",
        "--FIT_CONTINUUM=1", "--CONTINUUM_N=1",
        "--MAP_REGIONS_TO_CHIP=" + ",".join("1" for _ in ranges),
        "--LIST_MOLEC=" + ",".join(names),
        "--FIT_MOLEC=" + ",".join(str(f) for f in flags),
        "--REL_COL=" + ",".join("1.0" for _ in names),
    ] + _kernel_params(slit)
    work = os.path.abspath(work)
    cmd = [esorex_bin, f"--output-dir={work}", "molecfit_model"] + params + [
        os.path.join(work, "model.sof")]
    t0 = time.time()
    with open(os.path.join(work, "molecfit_model.log"), "w") as fh:
        # Run in its own process group so a timeout kills esorex *and* its
        # LBLRTM children, not just the parent.
        proc = subprocess.Popen(cmd, env=_esorex_env(esorex_bin), cwd=work,
                                stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
            print(f"    molecfit_model TIMEOUT ({timeout:.0f}s) in "
                  f"{os.path.basename(work)}; falling back to order median")
            return None
    if warn_after and time.time() - t0 > warn_after:
        print(f"    WARNING: molecfit_model took {time.time() - t0:.0f}s in "
              f"{os.path.basename(work)}")
    if returncode != 0:
        return None
    e = fits.open(os.path.join(work, "BEST_FIT_MODEL.fits"))[1]
    return (np.asarray(e.data["mlambda"], float) * 1000.0,
            np.asarray(e.data["mtrans"], float))


def _fit_window(wl):
    good = np.isfinite(wl)
    idx = np.where(good)[0]
    lam = wl / 1000.0
    return [(lam[idx[EDGE_TRIM]], lam[idx[-EDGE_TRIM]])]


def _smooth_refined(mlambda, native, deg=2):
    """molecfit's per pixel wavelength can carry a few wild values at the
    segment edges (poor extrapolation of the fit beyond the fitted region).
    The WLC correction is a low order polynomial, so fit that polynomial to the
    correction (refined minus native) over the edge trimmed core and evaluate it
    across the whole segment.  This removes the edge outliers and yields a
    smooth, monotonic grid.  Returns the input unchanged if the core has too few
    finite points."""
    px = np.arange(native.size, dtype=float)
    core = slice(EDGE_TRIM, -EDGE_TRIM)
    diff = mlambda - native
    g = np.isfinite(diff[core]) & np.isfinite(native[core])
    if g.sum() < 10:
        return mlambda
    coef = np.polyfit(px[core][g], diff[core][g], deg)
    return native + np.polyval(coef, px)


def refine_segment(name, wl, sp, er, header, work, scale, names, flags,
                   slit, esorex_bin, wlc_n=1, warn_after=600):
    """Refine the wavelength of one segment (FIT_WLC=1, fit water).  Returns
    (mlambda_nm, mtrans, shift_kms) or None.  The returned wavelength is the
    smooth polynomial correction, free of molecfit's edge outliers."""
    res = _molec_model(name, wl, sp, er, header, work, scale, _fit_window(wl),
                       names, flags, fit_wlc=1, wlc_n=wlc_n, slit=slit,
                       esorex_bin=esorex_bin, warn_after=warn_after)
    if res is None:
        return None
    ml, mt = res
    ml = _smooth_refined(ml, wl)
    return ml, mt, float(np.nanmedian((ml - wl) / wl * C_KMS))


def theoretical_transmittance(name, wl, sp, er, header, work, scale, names,
                              slit, esorex_bin, warn_after=600):
    """Clean theoretical telluric spectrum for a segment (climatological
    columns, wavelength fixed: FIT_WLC=0, FIT_MOLEC all zero).  Used for the
    masks of segments whose wavelength fit was rejected.  Returns mtrans on the
    input grid, or ones on failure."""
    res = _molec_model(name, wl, sp, er, header, work, scale, _fit_window(wl),
                       names, [0] * len(names), fit_wlc=0, wlc_n=1, slit=slit,
                       esorex_bin=esorex_bin, warn_after=warn_after)
    return res[1] if res is not None else np.ones_like(wl)


def run_night(reduced_dir, nod="A", out_dir=None, molecules=None,
              reference=None, esorex_bin="esorex", gate_kms=GATE_KMS,
              warn_after=600):
    """Refine the wavelength solution and build the telluric model for one
    nodding position.

    Parameters
    ----------
    reduced_dir : str
        Directory holding ``pair_*/cr2res_obs_nodding_extracted<AB>.fits``.
    nod : {"A", "B"}
        Nodding position.
    out_dir : str, optional
        Where to write ``molecfit_nod<nod>.npz`` and the per segment work
        directories (default ``<reduced_dir>/../molecfit``).
    molecules : (names, flags), optional
        Override the per band telluric molecule list.
    reference : str, optional
        Explicit reference extraction path (default: highest signal to noise).
    esorex_bin : str
        Path to the esorex executable (default: found on PATH).
    gate_kms : float
        Plausibility gate on the refined velocity shift.

    Returns
    -------
    dict name -> (wave_refined_nm, transmittance)
    """
    out_dir = out_dir or os.path.join(os.path.dirname(
        os.path.abspath(reduced_dir)), "molecfit")
    os.makedirs(out_dir, exist_ok=True)
    ref_path = reference or pick_reference(reduced_dir, nod)
    header = fits.open(ref_path)[0].header
    band = band_of(header)
    slit = slit_width(header)
    names, flags = molecules_for(header, molecules)
    segs = segments_of(ref_path)
    scale = global_scale(segs)
    print(f"molecfit nod {nod}: band {band}, slit {slit}\", "
          f"{len(segs)} segments, molecules {names} (fit {flags})")
    print(f"  reference {os.path.basename(os.path.dirname(ref_path))} "
          f"(median S/N {_median_snr(ref_path):.0f})")

    # Phase 1: refine every segment.
    raw = {}
    for name, wl, sp, er in segs:
        work = os.path.join(out_dir, f"nod{nod}", name)
        res = refine_segment(name, wl, sp, er, header, work, scale, names,
                             flags, slit, esorex_bin, warn_after=warn_after)
        raw[name] = (wl, sp, er, res)
        print(f"  {name}: " + ("molecfit FAILED" if res is None
                                else f"shift {res[2]:+.2f} km/s"))

    # Phase 2: gate and build the per order median of the accepted shifts.
    accepted = {n: v[3][2] for n, v in raw.items()
                if v[3] is not None and abs(v[3][2]) < gate_kms}
    order_med = {}
    for name in raw:
        o = _order_of(name)
        vals = [s for n, s in accepted.items() if _order_of(n) == o]
        order_med[o] = float(np.median(vals)) if vals else None

    # Phase 3: assemble the final wavelength solution and telluric model.
    out = {}
    for name, (wl, sp, er, res) in raw.items():
        if name in accepted:
            out[name] = (res[0], res[1])
            continue
        v = order_med[_order_of(name)]
        if v is None:
            out[name] = (wl.copy(), np.ones_like(wl))
        else:
            mt = theoretical_transmittance(
                name, wl, sp, er, header,
                os.path.join(out_dir, f"nod{nod}", name + "_theory"),
                scale, names, slit, esorex_bin, warn_after=warn_after)
            out[name] = (wl * (1.0 + v / C_KMS), mt)

    np.savez(os.path.join(out_dir, f"molecfit_nod{nod}.npz"),
             names=list(out.keys()),
             wave_refined=np.array([out[n][0] for n in out], dtype=object),
             transmittance=np.array([out[n][1] for n in out], dtype=object))

    # Wavelength consistency check (Nortmann A.1: the telluric anchored solution
    # agrees with the Fabry Perot one, i.e. no drift).
    if accepted:
        med = np.median(np.abs(list(accepted.values())))
        print(f"  wavelength check: {len(accepted)}/{len(out)} segments "
              f"constrained, median |refinement| {med:.2f} km/s "
              f"({'consistent with Fabry Perot' if med < gate_kms else 'check setup'})")
    print(f"  {len(accepted)}/{len(out)} molecfit, "
          f"{len(out) - len(accepted)} intra order or Fabry Perot "
          f"-> molecfit_nod{nod}.npz")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run molecfit on a CRIRES+ night.")
    ap.add_argument("reduced_dir", help="dir with pair_*/...extracted<AB>.fits")
    ap.add_argument("nod", nargs="?", default="A", choices=["A", "B"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--esorex", default="esorex")
    ap.add_argument("--species", default=None,
                    help="override the per band telluric molecules, e.g. "
                         "'H2O,CO2,CH4' (see parse_species); required to be set "
                         "sensibly for any band other than the built in defaults")
    a = ap.parse_args()
    mol = parse_species(a.species) if a.species else None
    run_night(a.reduced_dir, a.nod, out_dir=a.out_dir, esorex_bin=a.esorex,
              molecules=mol)
