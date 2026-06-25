"""
exoplore.core.cli
=================

Command-line entry point for the clean EXoPLORE runner.

Usage::

    exoplore configs/hd189733b_andes_transit_clean.json
    python scripts/run_exoplore.py configs/hd189733b_andes_transit_clean.json
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from exoplore.config import SimulationConfig
from exoplore.core.simulator import ExoploreSimulator

# ── Suppress known-safe cosmetic warnings ────────────────────────
# Two warnings appear in every normal run and are safe to silence:
#
#   RankWarning from np.polyfit (BL19 telluric correction): fires for
#   spectral orders with a flat telluric spectrum, the fit result is
#   still valid.
#
#   RuntimeWarning from the rotation-broadening kernel:
#   np.sqrt is called on marginally-negative floats at the domain boundary;
#   the nan values are never used in the final convolution output.
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in sqrt",
    category=RuntimeWarning,
)
# ─────────────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="exoplore",
        description="EXoPLORE: high-resolution exoplanet atmosphere simulator",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the simulation JSON config file.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Actually run the simulation (requires full dependencies).",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    print(f"\nLoading config: {args.config}")
    cfg = SimulationConfig.from_json(args.config)
    sim = ExoploreSimulator(cfg)
    print(sim.summary())

    if args.run:
        sim.run()
    else:
        print("  (Pass --run to execute the simulation.)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
