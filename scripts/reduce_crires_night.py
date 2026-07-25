#!/usr/bin/env python3
"""
reduce_crires_night.py
======================

Stage 0 raw reduction driver for a CRIRES+ nodding time series, wrapping the
ESO cr2res pipeline (esorex, command line) following the reduction methodology
of Nortmann et al. 2026 Appendix A.1.  It turns a directory of raw frames plus
their associated calibrations into the extracted 1D spectral time series (one
spectrum per nodding exposure) that ``prepare_crires_night.py`` ingests.

This is only a convenience wrapper around the ESO tools; it does not replace
them.  You must install the cr2res pipeline yourself and have ``esorex`` on the
PATH.  For installation and for the authoritative description of every recipe
see the ESO CRIRES+ pipeline pages:
https://www.eso.org/sci/software/pipelines/cr2res/ .

Cascade (each step reads the products of the previous one):
  1. cr2res_cal_dark    : master darks per DIT and bad pixel maps
  2. cr2res_cal_flat    : master flat, blaze and trace wave
  3. cr2res_cal_wave    : wavelength solution from a uranium neon lamp and a
                          Fabry Perot etalon
  4. cr2res_obs_nodding : per A/B pair extraction, keeping each nodding
                          position as a separate spectrum in the time series

The calibration exposure times (dark, flat, wave, science DIT) are read from
the frames themselves, so the driver is not tied to one observing programme.

Run::

    python scripts/reduce_crires_night.py <raw_dir> <step>

where step is one of {dark, flat, wave, nod, all} and <raw_dir> holds the raw
FITS frames.  Products are written under <raw_dir>/../ in calib/ and reduced/.
"""
import argparse
import glob
import os
import subprocess
import sys
from collections import Counter

from astropy.io import fits


def _kw(h, key, default=""):
    for k in (key, "HIERARCH " + key):
        if k in h:
            return h[k]
    return default


class Reducer:
    def __init__(self, raw_dir, esorex="esorex"):
        self.RAW = os.path.abspath(raw_dir)
        root = os.path.dirname(self.RAW)
        self.SOF = os.path.join(root, "sof")
        self.CAL = os.path.join(root, "calib")
        self.RED = os.path.join(root, "reduced")
        self.LOG = os.path.join(root, "logs")
        for d in (self.SOF, self.CAL, self.RED, self.LOG):
            os.makedirs(d, exist_ok=True)
        self.esorex = esorex
        env = dict(os.environ)
        d = os.path.dirname(os.path.abspath(esorex)) if os.path.sep in esorex else ""
        if d:
            env["PATH"] = d + os.pathsep + env.get("PATH", "")
        self.ENV = env
        self.frames, self.masters = self._classify()

    def _classify(self):
        """dpr type -> list of (path, DIT); plus static master products."""
        frames, masters = {}, {}
        for f in sorted(glob.glob(os.path.join(self.RAW, "*.fits"))):
            h = fits.getheader(f)
            pro = _kw(h, "ESO PRO CATG")
            if pro:
                masters[pro] = f
                continue
            typ = _kw(h, "ESO DPR TYPE")
            dit = float(_kw(h, "ESO DET SEQ1 DIT", 0) or 0)
            frames.setdefault(typ, []).append((f, dit))
        return frames, masters

    # --- helpers -----------------------------------------------------------
    def write_sof(self, name, entries):
        p = os.path.join(self.SOF, name)
        with open(p, "w") as fh:
            for path, tag in entries:
                fh.write(f"{path} {tag}\n")
        return p

    def run(self, recipe, sof, extra=None, outdir=None):
        outdir = outdir or self.CAL
        cmd = [self.esorex, f"--output-dir={outdir}"] + (extra or []) + [recipe, sof]
        logf = os.path.join(self.LOG, f"{recipe}.log")
        print(f"  running {recipe}  (log: {logf})")
        with open(logf, "w") as fh:
            r = subprocess.run(cmd, env=self.ENV, cwd=outdir, stdout=fh,
                               stderr=subprocess.STDOUT)
        print(f"    exit {r.returncode} ({'OK' if r.returncode == 0 else 'FAIL, see log'})")
        return r.returncode == 0

    def find(self, outdir, pattern):
        return sorted(glob.glob(os.path.join(outdir, pattern)))

    def dark_for_dit(self, dit):
        ms = [m for m in self.find(self.CAL, "cr2res_cal_dark_*master.fits")
              if f"_{dit:g}" in m]
        bp = [b for b in self.find(self.CAL, "cr2res_cal_dark_*bpm.fits")
              if f"_{dit:g}" in b]
        return (ms[0] if ms else None), (bp[0] if bp else None)

    def _dit_of(self, typ):
        """Most common DIT of a raw frame type (e.g. FLAT, OBJECT)."""
        dits = [d for _, d in self.frames.get(typ, [])]
        return Counter(dits).most_common(1)[0][0] if dits else None

    # --- cascade steps -----------------------------------------------------
    def step_dark(self):
        darks = [(f, "DARK") for f, _ in self.frames.get("DARK", [])]
        ok = self.run("cr2res_cal_dark", self.write_sof("dark.sof", darks))
        ms = self.find(self.CAL, "cr2res_cal_dark_*master.fits")
        print(f"    master darks: {len(ms)}")
        return ok

    def step_flat(self):
        flats = [(f, "FLAT") for f, _ in self.frames.get("FLAT", [])]
        flat_dit = self._dit_of("FLAT")
        md, bpm = self.dark_for_dit(flat_dit) if flat_dit else (None, None)
        tw = self.masters.get("UTIL_WAVE_TW")
        entries = list(flats)
        if md:
            entries.append((md, "CAL_DARK_MASTER"))
        if bpm:
            entries.append((bpm, "CAL_DARK_BPM"))
        if tw:
            entries.append((tw, "CAL_FLAT_TW"))
        ok = self.run("cr2res_cal_flat", self.write_sof("flat.sof", entries))
        print(f"    master flat: {len(self.find(self.CAL, 'cr2res_cal_flat_*master_flat.fits'))}")
        return ok

    def step_wave(self):
        une = [(f, "WAVE_UNE") for f, _ in self.frames.get("WAVE,UNE", [])]
        fpet = [(f, "WAVE_FPET") for f, _ in self.frames.get("WAVE,FPET", [])]
        ftw = (self.find(self.CAL, "cr2res_cal_flat_*tw_merged.fits")
               or self.find(self.CAL, "cr2res_cal_flat_*tw.fits"))
        mf = self.find(self.CAL, "cr2res_cal_flat_*master_flat.fits")
        lines = self.masters.get("EMISSION_LINES")
        wave_dit = self._dit_of("WAVE,UNE") or self._dit_of("WAVE,FPET")
        md, bpm = self.dark_for_dit(wave_dit) if wave_dit else (None, None)
        entries = une + fpet
        if ftw:
            entries.append((ftw[0], "CAL_FLAT_TW"))
        if mf:
            entries.append((mf[0], "CAL_FLAT_MASTER"))
        if lines:
            entries.append((lines, "EMISSION_LINES"))
        if md:
            entries.append((md, "CAL_DARK_MASTER"))
        if bpm:
            entries.append((bpm, "CAL_DARK_BPM"))
        ok = self.run("cr2res_cal_wave", self.write_sof("wave.sof", entries))
        print(f"    wave trace: {[os.path.basename(w) for w in self.find(self.CAL, 'cr2res_cal_wave_tw.fits')]}")
        return ok

    def _nod_calib(self, dit=None):
        """Calibration entries for one nodding pair; the master dark is picked
        for the pair's own DIT (nights can mix exposure times)."""
        wtw = (self.find(self.CAL, "cr2res_cal_wave_tw_fpet.fits")
               or self.find(self.CAL, "cr2res_cal_wave_tw*.fits")
               or self.find(self.CAL, "cr2res_cal_flat_*tw_merged.fits"))
        mf = self.find(self.CAL, "cr2res_cal_flat_*master_flat.fits")
        blaze = self.find(self.CAL, "cr2res_cal_flat_*blaze.fits")
        md, bpm = self.dark_for_dit(dit if dit else self._dit_of("OBJECT"))
        pf = self.masters.get("PHOTO_FLUX")
        e = []
        if wtw:
            e.append((wtw[0], "CAL_FLAT_TW"))
        if mf:
            e.append((mf[0], "CAL_FLAT_MASTER"))
        if blaze:
            e.append((blaze[0], "CAL_FLAT_EXTRACT_1D"))
        if md:
            e.append((md, "CAL_DARK_MASTER"))
        if bpm:
            e.append((bpm, "CAL_DARK_BPM"))
        if pf:
            e.append((pf, "PHOTO_FLUX"))
        return e

    @staticmethod
    def _pair_frames(sci):
        """Pair each science frame with the temporally nearest opposite nod
        frame OF THE SAME DIT, giving A/B pairs for per pair extraction (time
        series).  The same DIT requirement keeps pairs from straddling an
        exposure time change during the night."""
        items = sorted(sci, key=lambda r: r["mjd"])
        used = [False] * len(items)
        pairs = []
        for i, a in enumerate(items):
            if used[i]:
                continue
            best, bestdt = None, 1e9
            for j, b in enumerate(items):
                if (used[j] or j == i or b["nod"] == a["nod"]
                        or abs(b.get("dit", 0) - a.get("dit", 0)) >= 1.0):
                    continue
                dt = abs(b["mjd"] - a["mjd"])
                if dt < bestdt:
                    best, bestdt = j, dt
            if best is not None:
                used[i] = used[best] = True
                pairs.append((a, items[best]))
        return pairs

    def step_nod(self):
        # Keep every OBJECT frame regardless of DIT: programmes may change the
        # exposure time during the night (for example shorter frames out of
        # transit and longer ones in transit).  The A minus B subtraction
        # removes the dark to first order, and pairs are consecutive so both
        # members share one DIT.
        sci = []
        for f, d in self.frames.get("OBJECT", []):
            h = fits.getheader(f)
            sci.append(dict(path=f, nod=_kw(h, "ESO SEQ NODPOS"),
                            mjd=float(_kw(h, "MJD-OBS", 0)), dit=d))
        pairs = self._pair_frames(sci)
        print(f"    {len(sci)} science frames -> {len(pairs)} A/B pairs")
        manifest = []
        for n, (a, b) in enumerate(pairs):
            outdir = os.path.join(self.RED, f"pair_{n:02d}")
            os.makedirs(outdir, exist_ok=True)
            calib = self._nod_calib(dit=a.get("dit"))
            entries = [(a["path"], "OBS_NODDING_OTHER"),
                       (b["path"], "OBS_NODDING_OTHER")] + calib
            sof = self.write_sof(f"nod_{n:02d}.sof", entries)
            okp = self.run("cr2res_obs_nodding", sof, outdir=outdir)
            ea = self.find(outdir, "cr2res_obs_nodding_extractedA.fits")
            eb = self.find(outdir, "cr2res_obs_nodding_extractedB.fits")
            if okp and ea and eb:
                aa, bb = (a, b) if a["nod"] == "A" else (b, a)
                manifest.append((aa["mjd"], ea[0]))
                manifest.append((bb["mjd"], eb[0]))
            print(f"      pair {n:02d}: {'OK' if okp else 'FAIL'}")
        manifest.sort()
        with open(os.path.join(self.RED, "timeseries_manifest.txt"), "w") as fh:
            for mjd, path in manifest:
                fh.write(f"{mjd:.8f} {path}\n")
        print(f"    time series: {len(manifest)} spectra "
              f"-> reduced/timeseries_manifest.txt")
        return len(manifest) > 0


def main():
    ap = argparse.ArgumentParser(description="Reduce a CRIRES+ nodding night with cr2res.")
    ap.add_argument("raw_dir", help="directory with raw FITS frames")
    ap.add_argument("step", nargs="?", default="all",
                    choices=["dark", "flat", "wave", "nod", "all"])
    ap.add_argument("--esorex", default="esorex")
    a = ap.parse_args()
    r = Reducer(a.raw_dir, esorex=a.esorex)
    print("classified raw frames:")
    for t, L in r.frames.items():
        print(f"  {t:16s} {len(L):2d}  DITs {sorted(set(f'{d:g}' for _, d in L))}")
    print()
    if a.step in ("dark", "all"):
        print("STEP 1: master dark");        r.step_dark()
    if a.step in ("flat", "all"):
        print("STEP 2: master flat + trace"); r.step_flat()
    if a.step in ("wave", "all"):
        print("STEP 3: wavelength solution"); r.step_wave()
    if a.step in ("nod", "all"):
        print("STEP 4: nodding extraction");  r.step_nod()


if __name__ == "__main__":
    main()
