"""Illustrate the velocity-sampling bias of the Welch t-test significance.

Two significance metrics are in common use for cross-correlation detections:
the cross-correlation S/N (peak value divided by the off-peak standard
deviation) and Welch's t-test between the in-trail and out-of-trail
distributions of CCF values.  The Welch t-test assumes the values it compares
are independent.  A cross-correlation function is a smooth, matched-filter
response, so its values are correlated on the scale of the instrumental
resolution element.  When the CCF is sampled more finely than the resolution
element (oversampling, common when a small velocity step is used), the in-trail
window contains more points than there are independent ones, and the t-test
treats correlated points as independent.  This inflates the Welch significance,
while the S/N is unaffected.

This standalone script demonstrates the effect on a controlled toy CCF, drawing
many noise realisations of the same injected peak and comparing the two metrics
at two velocity steps: oversampled (step finer than the resolution element) and
critically sampled (step equal to the resolution element).  To first order the
Welch/S-N offset is approximately √(Resolution / Sampling), the square root of
the number of correlated points per resolution element.

Running this script
-------------------
    python scripts/illustrate_significance_sampling.py --output significance_sampling.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind


# Toy CCF parameters (km/s)
RESEL = 3.0            # resolution element (correlation length of the CCF)
V_MAX = 250.0          # CCF velocity half-range
FINE_STEP = 0.3        # underlying fine grid the CCF is generated on
IN_TRAIL_HALF = 3.0    # in-trail window half-width around the peak
SAFETY = 25.0          # exclusion window around the trail for the out-of-trail set
PEAK_AMP = 6.0         # injected peak amplitude in units of the noise sigma


def _make_ccf(rng, v_fine):
    """A toy CCF: a Gaussian peak at v=0 plus noise correlated over RESEL."""
    white = rng.normal(0.0, 1.0, len(v_fine))
    sigma_corr_pix = (RESEL / 2.355) / FINE_STEP
    noise = gaussian_filter1d(white, sigma_corr_pix, mode="nearest")
    noise /= noise.std()
    signal = PEAK_AMP * np.exp(-0.5 * (v_fine / (RESEL / 2.355)) ** 2)
    return signal + noise


def _metrics(v, ccf):
    """Return (S/N, Welch t-value) for a sampled CCF."""
    out_mask = np.abs(v) > SAFETY
    in_mask = np.abs(v) <= IN_TRAIL_HALF
    noise_std = ccf[out_mask].std()
    peak = ccf[np.argmin(np.abs(v))]
    snr = peak / noise_std
    t_value, _ = ttest_ind(ccf[in_mask], ccf[out_mask], equal_var=False)
    return snr, t_value


def _grid(step):
    # symmetric grid that always includes v = 0, so the peak is sampled
    # identically at every step and the S/N is free of a grid artefact
    pos = np.arange(0.0, V_MAX + step, step)
    return np.concatenate([-pos[::-1][:-1], pos])


def make_figure(output: str, n_real: int = 600, seed: int = 12345) -> None:
    rng = np.random.default_rng(seed)
    v_fine = np.arange(-V_MAX, V_MAX + FINE_STEP, FINE_STEP)

    # Scan the CCF velocity step from finely oversampled up to the resolution
    # element (critical sampling). Steps larger than the resolution element
    # undersample the line and are not used in practice.
    steps = np.array([0.5, 0.75, 1.0, 1.3, 1.8, 2.4, 3.0])

    snr_med = np.zeros_like(steps)
    welch_med = np.zeros_like(steps)
    snr_lo, snr_hi = np.zeros_like(steps), np.zeros_like(steps)
    welch_lo, welch_hi = np.zeros_like(steps), np.zeros_like(steps)

    grids = {s: _grid(s) for s in steps}
    snr_all = {s: [] for s in steps}
    welch_all = {s: [] for s in steps}

    for _ in range(n_real):
        ccf_fine = _make_ccf(rng, v_fine)
        for s in steps:
            ccf = np.interp(grids[s], v_fine, ccf_fine)
            snr, welch = _metrics(grids[s], ccf)
            snr_all[s].append(snr)
            welch_all[s].append(welch)

    for k, s in enumerate(steps):
        snr_med[k] = np.median(snr_all[s])
        welch_med[k] = np.median(welch_all[s])
        snr_lo[k], snr_hi[k] = np.percentile(snr_all[s], [16, 84])
        welch_lo[k], welch_hi[k] = np.percentile(welch_all[s], [16, 84])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor("white")

    ax.fill_between(steps, snr_lo, snr_hi, color="#2166ac", alpha=0.15, lw=0)
    ax.plot(steps, snr_med, "-o", color="#2166ac", lw=2, label="cross-correlation S/N")

    ax.fill_between(steps, welch_lo, welch_hi, color="#d6604d", alpha=0.15, lw=0)
    ax.plot(steps, welch_med, "-o", color="#d6604d", lw=2, label="Welch t-value")

    ax.axvline(RESEL, color="0.5", ls="--", lw=1.0)
    ax.text(RESEL * 0.97, ax.get_ylim()[1] * 0.6, "resolution\nelement",
            fontsize=9, color="0.4", va="top", ha="right")

    ax.set_xlabel("CCF velocity step Δv (km/s)", fontsize=12)
    ax.set_ylabel("median significance", fontsize=12)
    ax.set_title("S/N is robust to the velocity step; the Welch t-test is not",
                 fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.tick_params(labelsize=10)
    ax.set_xlim(steps.min() - 0.2, steps.max() + 0.2)

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")
    print(f"  S/N median across steps:   {snr_med.min():.2f} to {snr_med.max():.2f} (flat)")
    print(f"  Welch median: {welch_med[0]:.2f} at Δv={steps[0]} km/s "
          f"vs {welch_med[-1]:.2f} at Δv={steps[-1]} km/s")
    print(f"  Welch ratio (finest/critical): {welch_med[0]/welch_med[-1]:.2f}  "
          f"expected √(Resel/Δv)={np.sqrt(RESEL/steps[0]):.2f}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="significance_sampling.png")
    p.add_argument("--n-real", type=int, default=600)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    make_figure(args.output, args.n_real, args.seed)
