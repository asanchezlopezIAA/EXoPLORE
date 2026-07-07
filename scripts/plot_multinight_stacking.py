#!/usr/bin/env python
"""Plot per-night and combined 1D CCFs for a simple multi-night co-add.

Reads the Kp-Vsys S/N maps written by a run with ``n_nights > 1`` and
``different_nights: false`` (the per-night maps ``..._night{b}.npz`` and the
combined map ``....npz``), takes the 1D CCF slice at each map's own Kp of
maximum significance, and overlays them. Given two run directories it draws a
two-panel comparison (e.g. CARMENES NIR next to ANDES).

Usage
-----
    python scripts/plot_multinight_stacking.py \
        --run  "<carmenes_matrices_dir>" --label "CARMENES NIR (3.5 m)" \
        --run  "<andes_matrices_dir>"    --label "ANDES YJHK (39 m)" \
        --out  docs/figures/tutorial5a_stacking.png

Each ``--run`` is the ``matrices`` directory of one simulation. Panels are drawn
in the order the ``--run`` flags are given.
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def _peak_slice(npz_path):
    """Return (v_rest, ccf_slice_at_best_kp, peak_sn) for one map file."""
    d = np.load(npz_path)
    m = d["ccf_tot_sn"]
    v_rest = d["v_rest"]
    vi, ki = np.unravel_index(np.nanargmax(m), m.shape)
    return v_rest, m[:, ki], float(m[vi, ki])


def _collect(matrices_dir):
    """Gather the per-night slices and the combined slice from a run dir."""
    files = glob.glob(os.path.join(matrices_dir, "ccf_tot_sn_map_*.npz"))
    nights, combined = [], None
    for f in sorted(files):
        if re.search(r"_night\d+\.npz$", f):
            nights.append(_peak_slice(f))
        else:
            combined = _peak_slice(f)
    return nights, combined


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="matrices/ directory of one run (repeatable)")
    ap.add_argument("--label", action="append", required=True,
                    help="panel title for the matching --run (repeatable)")
    ap.add_argument("--out", required=True, help="output figure path")
    ap.add_argument("--xlim", type=float, default=100.0,
                    help="v_rest half-range in km/s (default 100)")
    args = ap.parse_args()

    if len(args.run) != len(args.label):
        ap.error("give one --label per --run")

    n = len(args.run)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.4), squeeze=False)
    axes = axes[0]

    night_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]

    for ax, run, label in zip(axes, args.run, args.label):
        nights, combined = _collect(run)
        for i, (v, s, pk) in enumerate(nights):
            ax.plot(v, s, color=night_colors[i % len(night_colors)], lw=1.1,
                    alpha=0.85, label=f"night {i} ({pk:.1f})")
        if combined is not None:
            v, s, pk = combined
            ax.plot(v, s, color="k", lw=2.0, label=f"combined ({pk:.1f})")
        ax.axvline(0.0, color="0.6", ls="--", lw=0.8)
        ax.set_xlim(-args.xlim, args.xlim)
        ax.set_xlabel(r"$v_{\rm rest}$ (km s$^{-1}$)")
        ax.set_ylabel("CCF S/N")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
