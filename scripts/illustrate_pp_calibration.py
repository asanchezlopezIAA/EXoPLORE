"""Illustrate probability-probability (p-p) plots for retrieval calibration.

A p-p plot tests whether the credible intervals of a retrieval mean what they
claim.  The construction repeats an injection-recovery experiment many times:
for each simulated dataset with a known truth, one records the posterior
percentile at which the true value lies (the posterior CDF evaluated at the
truth).  From those percentiles one builds a coverage curve: for each nominal
credible level c, the fraction of simulations in which the truth falls inside
the central c credible interval.  If the inference is well calibrated, the
truth lies inside the central c interval a fraction c of the time, so the curve
follows the diagonal.  Deviations diagnose miscalibration:

  * below the diagonal  -> posteriors too narrow (over-confident, under-covering)
  * above the diagonal  -> posteriors too wide  (conservative, over-covering)

This standalone teaching script demonstrates both cases on a fast linear-Gaussian
toy problem, where the posterior is known analytically, so it runs in a second
and isolates the statistical idea from the cost of a real retrieval.  The
EXoPLORE documentation (Tutorial 8) uses the same diagnostic on real retrievals
via ``scripts/run_pp_calibration.py``.

Running this script
-------------------
    python scripts/illustrate_pp_calibration.py --output pp_calibration.png

Left panel: a well-specified inference (the noise used in the posterior matches
the noise in the data) yields a p-p plot on the diagonal.  Right panel: an
over-confident inference (the posterior underestimates the noise by a factor)
sags below the diagonal, the signature of credible intervals that are too tight.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _percentiles(n_sims: int, sigma_model_factor: float, rng) -> np.ndarray:
    """Return the posterior percentile of the truth for ``n_sims`` experiments.

    Linear-Gaussian model: theta ~ N(0, tau^2) is the truth, the datum is
    d = theta + noise with noise ~ N(0, sigma^2).  The conjugate posterior for
    theta given d is Gaussian.  ``sigma_model_factor`` multiplies the noise
    assumed by the posterior: 1.0 is well specified, < 1.0 is over-confident
    (the posterior believes the data are more precise than they are).
    """
    tau = 1.0          # prior standard deviation on theta
    sigma = 1.0        # true noise standard deviation of the data

    percentiles = np.empty(n_sims)
    sigma_model = sigma * sigma_model_factor

    for i in range(n_sims):
        theta_true = rng.normal(0.0, tau)
        datum = theta_true + rng.normal(0.0, sigma)

        # Conjugate Gaussian posterior using the (possibly wrong) sigma_model
        post_var = 1.0 / (1.0 / tau**2 + 1.0 / sigma_model**2)
        post_mean = post_var * (datum / sigma_model**2)
        post_std = np.sqrt(post_var)

        # Percentile of the truth within the posterior (posterior CDF at truth)
        percentiles[i] = stats.norm.cdf(theta_true, loc=post_mean, scale=post_std)

    return percentiles


def _coverage_curve(percentiles: np.ndarray):
    """Empirical coverage as a function of the nominal central credible level.

    The truth lies inside the central credible interval of level c when its
    posterior percentile q satisfies |q - 0.5| <= c / 2.  For a calibrated
    inference the percentiles are uniform, so the coverage equals c.
    """
    levels = np.linspace(0.0, 1.0, 200)
    dist = np.abs(percentiles - 0.5)
    coverage = np.array([np.mean(dist <= c / 2.0) for c in levels])
    return levels, coverage


def make_figure(output: str, n_sims: int = 300, seed: int = 12345) -> None:
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    cases = [
        (axes[0], 1.0,  "Well-specified inference", "#2166ac"),
        (axes[1], 0.65, "Over-confident inference (noise underestimated)", "#d6604d"),
    ]

    for ax, factor, title, colour in cases:
        perc = _percentiles(n_sims, factor, rng)
        x, coverage = _coverage_curve(perc)

        # 1-sigma and 2-sigma confidence bands (binomial standard error on the
        # coverage, which is a fraction of n_sims at each level)
        for k, alpha in [(1.0, 0.20), (2.0, 0.10)]:
            se = k * np.sqrt(np.clip(x * (1.0 - x), 0, None) / n_sims)
            ax.fill_between(x, x - se, x + se, color="0.5", alpha=alpha, lw=0)

        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1.2, label="ideal (calibrated)")
        ax.plot(x, coverage, color=colour, lw=2.2, label="empirical")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("nominal credible level", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10, frameon=False, loc="upper left")
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("empirical coverage", fontsize=12)

    fig.suptitle(f"Probability-probability (p-p) plots, {n_sims} simulations",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")
    print("  Left (well-specified): empirical curve follows the diagonal.")
    print("  Right (over-confident): empirical curve sags below the diagonal.")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="pp_calibration.png")
    p.add_argument("--n-sims", type=int, default=300)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    make_figure(args.output, args.n_sims, args.seed)
