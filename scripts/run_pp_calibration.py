"""Build a probability-probability (p-p) calibration plot from real retrievals.

This driver runs the EXoPLORE retrieval on many independent noise realisations
of the same injected atmosphere, then asks whether the resulting credible
intervals are statistically calibrated.  Unlike the toy in
``scripts/illustrate_pp_calibration.py``, this uses the full forward model,
preparation pipeline, and Bayesian retrieval, so the p-p plot reflects the
actual behaviour of a given pipeline and log-likelihood.

How it works
------------
For each realisation i in 0 .. N-1 the driver:

  1. Copies the base config, sets ``noise.noise_seed = base_seed + i`` (a new
     noise draw with the same injected truth), and sets
     ``paths.output_root`` to a per-realisation directory so nothing is
     overwritten.
  2. Runs ``scripts/run_exoplore.py <config> --run``.
  3. Locates the retrieval posterior and weights written for that realisation.

After the realisations are available, the driver computes, for each parameter,
the posterior percentile of the injected truth, and from those percentiles the
empirical coverage as a function of the nominal credible level (the p-p plot).

The driver is **resumable**: a realisation whose retrieval output already
exists is skipped.  Run it repeatedly, or with ``--plot-only``, to update the
figure as more realisations finish (useful when the full set is running in the
background).

Usage
-----
Run 30 realisations and build the plot::

    python scripts/run_pp_calibration.py \
        --config configs/hd189733b_andes_retrieval_blain24_noisy.json \
        --n 30 --base-seed 1000 --live-points 100 \
        --output-root /path/to/pp_calibration \
        --truths -3.0 149.4 1170.0 0.0 \
        --param-names "log10(X_H2O)" "Kp" "T_eq" "v_wind" \
        --figure docs/figures/pp_calibration_real.png

Re-plot from whatever has finished so far, without running anything::

    python scripts/run_pp_calibration.py ... --plot-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _realisation_dir(output_root: str, i: int) -> str:
    return os.path.join(output_root, f"pp_real_{i:03d}")


def _find_posterior(realisation_root: str):
    """Return (samples, weights) for the retrieval in a realisation, or None."""
    dat = glob.glob(os.path.join(realisation_root, "**",
                                 "retrieval_night_0_dat_*.npz"), recursive=True)
    if not dat:
        return None
    dat_path = dat[0]
    wts_path = dat_path.replace("retrieval_night_0_dat_",
                                "retrieval_night_0_weights_")
    if not os.path.exists(wts_path):
        return None
    samples = np.load(dat_path)["a"]
    weights = np.load(wts_path)["a"]
    weights = weights / weights.sum()
    return samples, weights


def _run_one(base_config: str, i: int, base_seed: int, output_root: str,
             live_points: int | None) -> None:
    """Write a per-realisation config and run the EXoPLORE retrieval."""
    with open(base_config) as f:
        cfg = json.load(f)

    cfg["noise"]["noise_seed"] = base_seed + i
    cfg["paths"]["output_root"] = _realisation_dir(output_root, i)
    if live_points is not None:
        cfg["retrieval"]["live_points"] = live_points
    cfg["retrieval"]["enabled"] = True

    os.makedirs(_realisation_dir(output_root, i), exist_ok=True)
    tmp_config = os.path.join(_realisation_dir(output_root, i), "config.json")
    with open(tmp_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[pp] realisation {i:03d}: seed={base_seed + i}, running ...", flush=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(repo_root, "src"))
    subprocess.run(
        [sys.executable, "-u", "scripts/run_exoplore.py", tmp_config, "--run"],
        cwd=repo_root, env=env, check=True,
    )


def _weighted_percentile_of(samples_col: np.ndarray, weights: np.ndarray,
                            truth: float) -> float:
    """Posterior CDF evaluated at ``truth`` (weighted fraction below truth)."""
    below = samples_col < truth
    return float(weights[below].sum())


def build_plot(output_root: str, n: int, truths, param_names, figure: str) -> None:
    """Collect finished realisations and write the coverage p-p plot."""
    # percentiles[param] = list of truth-percentiles across realisations
    n_par = len(truths)
    percentiles = [[] for _ in range(n_par)]

    n_done = 0
    for i in range(n):
        res = _find_posterior(_realisation_dir(output_root, i))
        if res is None:
            continue
        samples, weights = res
        n_done += 1
        for p in range(min(n_par, samples.shape[1])):
            q = _weighted_percentile_of(samples[:, p], weights, truths[p])
            percentiles[p].append(q)

    if n_done == 0:
        print("[pp] no finished realisations yet; nothing to plot.")
        return

    print(f"[pp] building coverage plot from {n_done} finished realisations.")

    levels = np.linspace(0.0, 1.0, 200)
    colours = ["#2166ac", "#d6604d", "#4dac26", "#8856a7", "#e08214", "#5e3c99"]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    # confidence band from binomial standard error
    for k, alpha in [(1.0, 0.20), (2.0, 0.10)]:
        se = k * np.sqrt(np.clip(levels * (1.0 - levels), 0, None) / n_done)
        ax.fill_between(levels, levels - se, levels + se, color="0.5", alpha=alpha, lw=0)
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1.2, label="ideal (calibrated)")

    for p in range(n_par):
        if not percentiles[p]:
            continue
        dist = np.abs(np.array(percentiles[p]) - 0.5)
        coverage = np.array([np.mean(dist <= c / 2.0) for c in levels])
        name = param_names[p] if p < len(param_names) else f"param {p}"
        ax.plot(levels, coverage, color=colours[p % len(colours)], lw=2.0, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("nominal credible level", fontsize=12)
    ax.set_ylabel("empirical coverage", fontsize=12)
    ax.set_title(f"Retrieval coverage (p-p), {n_done} realisations", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(figure, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[pp] saved: {figure}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Base noisy retrieval config.")
    p.add_argument("--n", type=int, default=30, help="Number of realisations.")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--output-root", required=True,
                   help="Directory holding the per-realisation output dirs.")
    p.add_argument("--live-points", type=int, default=None,
                   help="Override retrieval live points (lower = faster).")
    p.add_argument("--truths", nargs="+", type=float, required=True)
    p.add_argument("--param-names", nargs="+", default=[])
    p.add_argument("--figure", default="pp_calibration_real.png")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip running; build the plot from finished realisations.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if not args.plot_only:
        for i in range(args.n):
            if _find_posterior(_realisation_dir(args.output_root, i)) is not None:
                print(f"[pp] realisation {i:03d}: already done, skipping.")
                continue
            _run_one(args.config, i, args.base_seed, args.output_root,
                     args.live_points)
            # Update the figure after each realisation so progress is visible
            build_plot(args.output_root, args.n, args.truths,
                       args.param_names, args.figure)

    build_plot(args.output_root, args.n, args.truths,
               args.param_names, args.figure)
