"""Prepare a Turbospectrum angular export for rm_clv.model_file.

Run with --help for the required paths and wavelength convention. Stellar
synthesis is performed beforehand with Turbospectrum. EXoPLORE subsequently
applies stellar rotation, the transit geometry and instrumental resolution.


The output is the NPZ selected by rm_clv.model_file in the simulation JSON.
Use matching line and continuum exports from one stellar atmosphere and
explicitly declare whether the synthesis wavelengths are in air or vacuum.
"""

import argparse
from pathlib import Path
import sys

# Use this checkout's source, matching scripts/run_exoplore.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoplore.io.turbospectrum import save_turbospectrum_bundle


def main():
    # Describe the required stellar exports and wavelength conventions at the command line.
    parser = argparse.ArgumentParser(description=__doc__)
    # Read the line-intensity export and choose a fresh output bundle path.
    parser.add_argument("intensities", help="Plane-parallel Turbospectrum intensity ASCII export")
    parser.add_argument("output", help="New NPZ file for rm_clv.model_file")
    # An optional line-free export gives the continuum on the same angular grid.
    parser.add_argument("--continuum", help="Matching intensity export with no line lists")
    # Additional exports supply further viewing angles when the synthesis limits each run.
    parser.add_argument("--additional-intensities", action="append", default=[],
                        help="Further output angles from the same synthesis (repeatable)")
    parser.add_argument("--additional-continuum", action="append", default=[],
                        help="Continuum for each additional export, in the same order")
    # Declare the input and desired output wavelength conventions explicitly.
    parser.add_argument("--input-medium", choices=("air", "vacuum"), required=True)
    parser.add_argument("--output-medium", choices=("air", "vacuum"), default="vacuum")
    # Record the stellar parameters, line data and synthesis version in the bundle.
    parser.add_argument("--source", required=True,
                        help="Synthesis version, stellar parameters and abundance assumptions")
    # Record any instrumental broadening already applied to the synthesis.
    # Spatial coupling uses an intrinsic synthesis with resolving power zero.
    parser.add_argument("--resolving-power", type=float, default=0.,
                        help="Already applied instrumental R, or 0 for an unconvolved synthesis")
    # Read the requested paths and conversion choices.
    args = parser.parse_args()
    # Pair each additional angular intensity export with its matching continuum file.
    if args.additional_continuum and len(args.additional_continuum) != len(args.additional_intensities):
        parser.error("Supply one additional continuum for each additional intensity export")
    # Use the reader's continuum-recovery option for additional exports lacking a separate file.
    continua = args.additional_continuum or [None]*len(args.additional_intensities)
    # Read, validate, convert and merge the exports through the shared stellar-input writer.
    local = save_turbospectrum_bundle(args.intensities, args.output,
        source=args.source, input_medium=args.input_medium, output_medium=args.output_medium,
        continuum_path=args.continuum, resolving_power=args.resolving_power,
        additional_exports=zip(args.additional_intensities, continua))
    # Report the number of viewing angles and wavelengths available to the transit model.
    print(f"Saved {len(local.mu)} angles and {len(local.wavelength_um)} wavelengths to {args.output}")


# Run the conversion when this file is invoked as a command-line script.
if __name__ == "__main__":
    main()
