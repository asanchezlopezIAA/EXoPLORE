"""Overlay corner plot of a noise statistical study (significance, Kp, Vrest).

EXoPLORE writes the per-realisation statistics of a noise statistical study
(``statistics.enabled: true`` with ``n_nights`` realisations and
``all_significance_metrics: true``) as ``stats_*.npz`` arrays of shape
(n_realisations, 3) with columns [significance, Kp, Vrest], both in the run's
single ``matrices`` directory: the S/N metric as ``stats_<name>.npz`` and the
Welch t-test as ``stats_welch_<name>.npz``.

This driver overlays those distributions in a single corner plot, distinguishing
the significance metric by colour and the cross-correlation velocity step by
line style, so that the velocity-sampling behaviour of the two metrics can be
compared directly.  The single-case version of this corner (one metric, one
run) is produced by ``exoplore.analysis.stats.plot_stats``.

Usage
-----
    python scripts/plot_significance_study.py \
        --oversampled-root /path/to/sig_study_oversampled \
        --critical-root    /path/to/sig_study_critical \
        --kp-truth 149.4 --vrest-truth 0.0 \
        --output docs/figures/significance_study_corner.png
"""

from __future__ import annotations

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(root: str, metric: str) -> np.ndarray:
    # Both metrics live in the single consolidated ``matrices`` directory (there
    # are no separate ``matrices_SNR`` / ``matrices_Welch`` folders).  The S/N
    # stats are ``stats_<name>.npz`` and the Welch stats ``stats_welch_<name>.npz``.
    if metric == "Welch":
        matches = glob.glob(f"{root}/**/matrices/stats_welch_*.npz", recursive=True)
    else:
        matches = [m for m in glob.glob(f"{root}/**/matrices/stats_*.npz",
                                        recursive=True)
                   if "stats_welch_" not in m]
    if not matches:
        raise FileNotFoundError(
            f"no {metric} stats under {root}/**/matrices/")
    return np.load(matches[0])["a"]   # (n, 3): significance, Kp, Vrest


def make_figure(over_root: str, crit_root: str, output: str,
                kp_truth: float = 149.4, vrest_truth: float = 0.0) -> None:
    cases = {
        "S/N, oversampled":   (_load(over_root, "SNR"),   "#2166ac", "-"),
        "S/N, critical":      (_load(crit_root, "SNR"),   "#2166ac", "--"),
        "Welch, oversampled": (_load(over_root, "Welch"), "#d6604d", "-"),
        "Welch, critical":    (_load(crit_root, "Welch"), "#d6604d", "--"),
    }

    labels = ["significance", r"$K_{\rm P}$ (km s$^{-1}$)", r"$V_{\rm rest}$ (km s$^{-1}$)"]
    truths = [None, kp_truth, vrest_truth]

    # Axis ranges: significance from the data, Kp/Vrest from the search grid
    all_sig = np.concatenate([a[:, 0] for a, _, _ in cases.values()])
    ranges = [
        (max(0, all_sig.min() - 1), all_sig.max() + 1),
        (-350, 350),
        (-100, 100),
    ]
    n_bins = [24, 24, 20]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.08, wspace=0.08)

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.set_facecolor("white")
            if col > row:
                ax.set_visible(False)
                continue

            if row == col:
                bins = np.linspace(*ranges[col], n_bins[col] + 1)
                for arr, colour, ls in cases.values():
                    h, e = np.histogram(arr[:, col], bins=bins, density=True)
                    cx = 0.5 * (e[:-1] + e[1:])
                    ax.step(cx, h, where="mid", color=colour, ls=ls, lw=1.6)
                if truths[col] is not None:
                    ax.axvline(truths[col], color="black", ls=":", lw=1.2)
                ax.set_xlim(ranges[col]); ax.set_yticks([])
            else:
                for arr, colour, ls in cases.values():
                    marker = "o" if ls == "-" else "x"
                    ax.scatter(arr[:, col], arr[:, row], s=10, alpha=0.45,
                               color=colour, marker=marker, linewidths=0.8)
                if truths[col] is not None:
                    ax.axvline(truths[col], color="black", ls=":", lw=1.0)
                if truths[row] is not None:
                    ax.axhline(truths[row], color="black", ls=":", lw=1.0)
                ax.set_xlim(ranges[col]); ax.set_ylim(ranges[row])

            if row == 2:
                ax.set_xlabel(labels[col], fontsize=11)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=11)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    # Legend in the unused upper-right area
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, ls=ls, lw=2,
                      label=name) for name, (_, c, ls) in cases.items()]
    handles.append(Line2D([0], [0], color="black", ls=":", lw=1.2, label="injected truth"))
    axes[0, 2].set_visible(True); axes[0, 2].axis("off")
    axes[0, 2].legend(handles=handles, loc="center", fontsize=9, frameon=False)

    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")
    for name, (arr, _, _) in cases.items():
        print(f"  {name:<20}: significance median = {np.median(arr[:,0]):.2f}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oversampled-root", required=True)
    p.add_argument("--critical-root", required=True)
    p.add_argument("--output", default="significance_study_corner.png")
    p.add_argument("--kp-truth", type=float, default=149.4)
    p.add_argument("--vrest-truth", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    make_figure(a.oversampled_root, a.critical_root, a.output, a.kp_truth, a.vrest_truth)
