#!/usr/bin/env python3
"""
Prepare CARMENES reference-night input files for EXoPLORE.

Reads per-exposure CARMENES FITS files produced by the CARACAL pipeline,
applies quality cuts and NaN correction, and writes the EXoPLORE input
files into the target inputs directory.

Usage
-----
    python scripts/prepare_carmenes_night.py \\
        --night_dir  /path/to/raw/night/ \\
        --output_dir inputs/CARMENES_NIR/MyPlanet/reference_night/ \\
        --night_index 0 \\
        --first      car-20170907T20h00m00s-sci-gtoc-nir_A.fits \\
        --last       car-20170907T23h59m59s-sci-gtoc-nir_A.fits

Output files written (all in --output_dir)
------------------------------------------
    julian_date_{N}.fits            BJD timestamps, shape (n_spectra,)
    airmass_{N}.fits                airmass, shape (n_spectra,)
    sig_{N}.fits                    uncertainty array, shape (n_spectra, n_orders, n_pixels)
    snr_{N}.fits                    S/N array  = spec/sig, shape (n_spectra, n_orders, n_pixels)
    observations_berv_{N}.fits      BERV in km/s, shape (n_spectra,)
    observations_night_{N}_order_{K}.fits   spectra for order K, shape (n_spectra, n_pixels)

where N is --night_index and K is the zero-based order number.

Notes
-----
- CARMENES FITS structure (CARACAL pipeline):
    hdu[0].header : observing metadata (AIRMASS, BJD, BERV, ...)
    hdu[1].data   : spectra,      shape (n_pixels, n_orders)
    hdu[2].data   : continuum,    shape (n_pixels, n_orders)
    hdu[3].data   : uncertainties shape (n_pixels, n_orders)
    hdu[4].data   : wavelengths,  shape (n_pixels, n_orders), units Angstrom

- The CARACAL BJD header keyword stores BJD - 2400000; this script adds
  2400000 back to obtain the full BJD.

- Exposures with mean S/N < --snr_threshold are discarded before saving.

- NaN values in spectra and uncertainties are replaced with the median of
  the valid pixels in the same exposure and order. Uncertainty values below
  1e-7 are replaced with 800 (flagged as unreliable).
"""

import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CARMENES instrument constants
# ---------------------------------------------------------------------------
_CHANNEL_PARAMS = {
    "NIR": {"n_orders": 28, "n_pixels": 4080, "keyword": "*sci-*nir_A.fits"},
    "VIS": {"n_orders": 44, "n_pixels": 4096, "keyword": "*sci-*vis_A.fits"},
}

_BJD_OFFSET = 2_400_000.0   # CARACAL stores BJD - 2400000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _select_files(night_dir: str, keyword: str,
                  first: str | None, last: str | None) -> list[str]:
    """Return sorted list of science FITS files in [first, last], FP removed."""
    all_files = sorted(glob.glob(os.path.join(night_dir, keyword)))
    if not all_files:
        sys.exit(f"No files matching '{keyword}' found in {night_dir}")

    # Trim to [first, last] if provided
    if first is not None:
        first_path = os.path.join(night_dir, first) if not os.path.isabs(first) else first
        if first_path not in all_files:
            sys.exit(f"--first file not found in directory: {first_path}")
        all_files = all_files[all_files.index(first_path):]

    if last is not None:
        last_path = os.path.join(night_dir, last) if not os.path.isabs(last) else last
        if last_path not in all_files:
            sys.exit(f"--last file not found in directory: {last_path}")
        all_files = all_files[:all_files.index(last_path) + 1]

    # Remove Fabry-Pérot calibration frames
    science = []
    for f in all_files:
        with fits.open(f) as hdu:
            mode = hdu[0].header.get("HIERARCH CAHA INS ICS FIB-MODE", "")
        if mode != "FP,FP":
            science.append(f)

    print(f"  Selected {len(science)} science exposures "
          f"({len(all_files) - len(science)} FP frames removed)")
    return science


def _correct_nans(spec: np.ndarray, sig: np.ndarray,
                  sig_floor: float = 1e-7, sig_fill: float = 800.0):
    """
    Replace NaN values in spec and sig with the per-exposure median.
    Also floor suspiciously small sigma values (< sig_floor) with sig_fill.

    Parameters
    ----------
    spec : (n_spectra, n_pixels)   spectrum for one order
    sig  : (n_spectra, n_pixels)   uncertainty for one order
    """
    for i in range(spec.shape[0]):
        finite = np.isfinite(spec[i])
        if not np.all(finite):
            med_spec = np.median(spec[i, finite]) if finite.any() else 0.0
            med_sig  = np.median(sig[i, finite])  if finite.any() else sig_fill
            spec[i, ~finite] = med_spec
            sig[i, ~finite]  = med_sig
        sig[i, sig[i] < sig_floor] = sig_fill
    return spec, sig


def _read_night(files: list[str], n_orders: int, n_pixels: int) -> dict:
    """Read all exposures and return a dict of arrays."""
    n = len(files)
    wave  = np.zeros((n, n_orders, n_pixels))
    spec  = np.zeros((n, n_orders, n_pixels))
    sig   = np.zeros((n, n_orders, n_pixels))
    bjd   = np.zeros(n)
    berv  = np.zeros(n)
    airm  = np.zeros(n)

    for i, f in enumerate(tqdm(files, desc="  Reading FITS", unit="exp")):
        with fits.open(f) as hdu:
            # Arrays in FITS are (n_pixels, n_orders); transpose to (n_orders, n_pixels)
            wave[i] = np.transpose(hdu[4].data, (1, 0)) * 1e-4   # Å → µm
            spec[i] = np.transpose(hdu[1].data, (1, 0))
            sig[i]  = np.transpose(hdu[3].data, (1, 0))
            h = hdu[0].header
            bjd[i]  = h["HIERARCH CARACAL BJD"] + _BJD_OFFSET
            berv[i] = h["HIERARCH CARACAL BERV"]
            airm[i] = h["AIRMASS"]

    return {"wave": wave, "spec": spec, "sig": sig,
            "bjd": bjd, "berv": berv, "airmass": airm}


def _snr_cut(data: dict, threshold: float) -> dict:
    """Drop exposures with mean S/N below threshold."""
    snr_mean = np.nanmean(data["spec"] / data["sig"], axis=(1, 2))
    keep = snr_mean >= threshold
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} exposures with mean S/N < {threshold}")
    return {k: v[keep] if v.ndim >= 1 and v.shape[0] == keep.shape[0] else v
            for k, v in data.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CARMENES reference-night input files for EXoPLORE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--night_dir",   required=True,
                        help="Directory containing raw CARMENES FITS files for one night.")
    parser.add_argument("--output_dir",  required=True,
                        help="Output directory (e.g. inputs/CARMENES_NIR/MyPlanet/reference_night/).")
    parser.add_argument("--night_index", type=int, default=0,
                        help="Night index N used in output filenames (default: 0).")
    parser.add_argument("--channel",     choices=["NIR", "VIS"], default="NIR",
                        help="CARMENES channel (default: NIR).")
    parser.add_argument("--keyword",     default=None,
                        help="Glob pattern for FITS files. "
                             "Defaults to '*sci-*nir_A.fits' for NIR, '*sci-*vis_A.fits' for VIS.")
    parser.add_argument("--first",       default=None,
                        help="Filename of first exposure to include (basename only).")
    parser.add_argument("--last",        default=None,
                        help="Filename of last exposure to include (basename only).")
    parser.add_argument("--snr_threshold", type=float, default=20.0,
                        help="Minimum mean S/N per exposure (default: 20).")
    args = parser.parse_args()

    params  = _CHANNEL_PARAMS[args.channel]
    keyword = args.keyword or params["keyword"]
    n_ord   = params["n_orders"]
    n_pix   = params["n_pixels"]
    N       = args.night_index

    print(f"\nPreparing CARMENES {args.channel} night {N}")
    print(f"  Source : {args.night_dir}")
    print(f"  Output : {args.output_dir}")

    # 1. Select files
    files = _select_files(args.night_dir, keyword, args.first, args.last)

    # 2. Read
    print("  Reading exposures...")
    data = _read_night(files, n_ord, n_pix)

    # 3. S/N cut
    data = _snr_cut(data, args.snr_threshold)
    n_spectra = data["bjd"].shape[0]
    print(f"  Remaining exposures: {n_spectra}")

    # 4. NaN correction per order
    print("  Correcting NaN values and flagging bad pixels...")
    for h in range(n_ord):
        data["spec"][:, h, :], data["sig"][:, h, :] = _correct_nans(
            data["spec"][:, h, :], data["sig"][:, h, :])

    # 5. Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    fits.writeto(os.path.join(args.output_dir, f"julian_date_{N}.fits"),
                 data["bjd"], overwrite=True)
    fits.writeto(os.path.join(args.output_dir, f"airmass_{N}.fits"),
                 data["airmass"], overwrite=True)
    fits.writeto(os.path.join(args.output_dir, f"sig_{N}.fits"),
                 data["sig"], overwrite=True)
    fits.writeto(os.path.join(args.output_dir, f"snr_{N}.fits"),
                 data["spec"] / data["sig"], overwrite=True)
    fits.writeto(os.path.join(args.output_dir, f"observations_berv_{N}.fits"),
                 data["berv"], overwrite=True)

    for h in range(n_ord):
        fits.writeto(
            os.path.join(args.output_dir, f"observations_night_{N}_order_{h}.fits"),
            data["spec"][:, h, :], overwrite=True)

    print(f"\n  Done. {n_spectra} exposures, {n_ord} orders written to:")
    print(f"  {args.output_dir}\n")


if __name__ == "__main__":
    main()
