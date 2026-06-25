#!/usr/bin/env python3
"""
make_wave_carmenes_nir.py
─────────────────────────
Convert the raw CARACAL wave_NIR.fits (shape 46 × 4080 × 28)
to the clean EXoPLORE format wave_CARMENES_NIR.fits (shape 28 × 4080).

The raw file contains the wavelength solution from 46 exposures of a real
CARMENES night.  All 46 solutions are essentially identical (thermal/mechanical
drift is < 1e-5 µm); we take the mean and drop the time dimension.

Usage (from repo root, with venv active):
    python scripts/make_wave_carmenes_nir.py \
        --input  /path/to/inputs/wave_NIR.fits \
        --output src/exoplore/instruments/data/wave_CARMENES_NIR.fits
"""

import argparse
import numpy as np
from astropy.io import fits
from pathlib import Path


def convert(input_path: str, output_path: str) -> None:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    print(f"Reading  : {input_path}")
    with fits.open(input_path) as hdul:
        raw = hdul[0].data          # shape (46, 4080, 28), CARACAL convention

    print(f"  Raw shape : {raw.shape}   dtype={raw.dtype}")
    assert raw.ndim == 3, f"Expected 3-D array, got {raw.ndim}-D"

    n_spectra, n_pixels, n_orders = raw.shape   # 46, 4080, 28
    print(f"  n_spectra={n_spectra}  n_orders={n_orders}  n_pixels={n_pixels}")

    # CARACAL returns (n_spectra, n_pixels, n_orders).
    # Transpose to (n_spectra, n_orders, n_pixels) then take mean over spectra.
    wvl = np.transpose(raw, (0, 2, 1))          # (46, 28, 4080)
    wvl_mean = np.nanmean(wvl, axis=0)          # (28, 4080)

    max_drift = np.nanmax(np.nanstd(wvl, axis=0))
    print(f"  Max wavelength drift across {n_spectra} spectra: {max_drift:.2e} µm")
    print(f"  Output shape : {wvl_mean.shape}")
    print(f"  Order  0 range : {wvl_mean[0, 0]:.6f} to {wvl_mean[0, -1]:.6f} µm")
    print(f"  Order 27 range : {wvl_mean[-1, 0]:.6f} to {wvl_mean[-1, -1]:.6f} µm")
    assert np.isfinite(wvl_mean).all(), "NaN/Inf found in mean wavelength solution!"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    hdr['INSTRUME'] = ('CARMENES_NIR', 'CARMENES near-infrared channel')
    hdr['BUNIT']    = ('micron',       'wavelength unit')
    hdr['ORIGIN']   = ('EXoPLORE',     'converted from CARACAL wave_NIR.fits')
    hdr.add_comment(f'Mean of {n_spectra} exposures; max drift {max_drift:.2e} um')
    hdr.add_comment('Shape: (n_orders=28, n_pixels=4080)')

    hdu = fits.PrimaryHDU(data=wvl_mean.astype(np.float64), header=hdr)
    hdu.writeto(str(output_path), overwrite=True)
    print(f"Written  : {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',  required=True,
                        help='Path to raw wave_NIR.fits (CARACAL format)')
    parser.add_argument('--output', required=True,
                        help='Output path for wave_CARMENES_NIR.fits')
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
