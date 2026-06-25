#!/usr/bin/env python3
"""
run_exoplore.py, Clean EXoPLORE runner.

Usage
-----
    python scripts/run_exoplore.py configs/hd189733b_andes_transit_clean.json
    python scripts/run_exoplore.py configs/hd189733b_andes_transit_clean.json --run

The first form validates the config and prints a simulation summary.
Pass ``--run`` to execute the full simulation (requires all dependencies).
"""
import sys
import warnings
from pathlib import Path

# --- Targeted warning suppressions ---
# Suppress only the two known-safe cosmetic warnings that appear during a
# normal simulation run, so that any genuinely unexpected warnings remain visible.

# RankWarning from np.polyfit in the BL19 telluric correction:
# fires for spectral orders where the telluric spectrum is nearly flat (no telluric
# absorption), making the degree-2 polynomial fit ill-conditioned.  The fit result
# is still used and is correct for flat orders.
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

# RuntimeWarning from the rotation-broadening kernel:
# np.sqrt is evaluated at argument values that are slightly negative at the edges
# of the integration domain (floating-point rounding).  The kernel clips these
# gracefully; the sqrt(negative) → nan is never used in the final convolution.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in sqrt",
    category=RuntimeWarning,
)
# ──────────────────────────────────────────────────────────────────────────────

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exoplore.core.cli import main

if __name__ == "__main__":
    sys.exit(main())
