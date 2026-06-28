"""Illustrate why the choice of log-likelihood matters in HRS retrievals.

This standalone teaching script reproduces, on a controlled toy spectrum,
the mechanism behind the difference between the Brogi & Line (2019) and the
Blain, Sanchez-Lopez & Molliere (2024) log-likelihood formulations used in
EXoPLORE.  It is referenced by the Concepts primer (docs/concepts.md) and by
Tutorial 7 of the documentation.

The key result
--------------
Both formulations describe Gaussian noise, but they treat the noise scale
differently:

  * BL19 estimates a SINGLE global noise level per spectrum from the
    residuals themselves (its log-likelihood is -N/2 ln of the mean squared
    residual).  There is no per-pixel sigma anywhere in the formula.

  * Blain24 uses the KNOWN per-pixel uncertainty sigma(n), so each pixel is
    weighted by 1/sigma(n)^2.

With UNIFORM (homoscedastic) noise the two are almost identical.  When the
noise varies pixel-to-pixel (heteroscedastic, as is frequently the case in
real spectra, where telluric absorption and the blaze raise sigma in some
channels) the per-pixel weighting of Blain24 may yield a tighter constraint:
it down-weights the noisy pixels and retains the signal in the clean ones,
whereas BL19's single self-estimated noise level is raised by the noisy
minority, which dilutes the signal in the clean majority.  This reflects the
noise structure of a given dataset rather than a general ordering of the two
formulations; BL19 was developed and validated on photon-noise-dominated
simulated CRIRES data, where it recovers statistically correct credibility
intervals and does not require reliable per-pixel uncertainties.

Running this script
-------------------
    python scripts/illustrate_likelihood_weighting.py --output likelihood_weighting.png

The figure shows, for a toy absorption spectrum, the log-likelihood as a
function of the line-depth parameter alpha (truth = 1), under uniform and
heteroscedastic noise of identical total variance.  The 1-sigma posterior
widths are printed and annotated.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Toy spectrum: a handful of Gaussian absorption lines over N pixels
# ---------------------------------------------------------------------------

N_PIXELS = 400
_LINE_CENTRES = [80, 150, 175, 250, 320, 360]


def template(amp: float = 1.0) -> np.ndarray:
    """A toy absorption spectrum scaled by ``amp`` (line depth parameter)."""
    x = np.arange(N_PIXELS)
    g = np.zeros(N_PIXELS)
    for c in _LINE_CENTRES:
        g -= np.exp(-0.5 * ((x - c) / 2.5) ** 2)
    return amp * g


# ---------------------------------------------------------------------------
# The two log-likelihoods, exactly as in src/exoplore/likelihood.py
# ---------------------------------------------------------------------------

def lnL_bl19(data: np.ndarray, model: np.ndarray) -> float:
    """Brogi & Line (2019): -N/2 ln(s_f^2 - 2R + s_g^2).

    Equivalent to -N/2 ln of the mean squared residual.  The noise scale is
    estimated from the data itself; no per-pixel sigma enters.
    """
    f = data - data.mean()
    g = model - model.mean()
    n = len(f)
    sf2 = np.mean(f ** 2)
    sg2 = np.mean(g ** 2)
    R = np.mean(f * g)
    return -0.5 * n * np.log(sf2 - 2.0 * R + sg2)


def lnL_blain24(data: np.ndarray, model: np.ndarray, sigma: np.ndarray) -> float:
    """Blain et al. (2024): -1/2 sum ((d - m) / sigma)^2 with known sigma(n)."""
    return -0.5 * np.sum(((data - model) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Build the demonstration
# ---------------------------------------------------------------------------

def _one_sigma_width(alphas: np.ndarray, curve: np.ndarray) -> float:
    """Width of the region within Delta lnL > -0.5 of the peak."""
    inside = alphas[curve > curve.max() - 0.5]
    return float(inside.max() - inside.min()) if len(inside) else float("nan")


def make_figure(output: str, seed: int = 12345) -> None:
    rng = np.random.default_rng(seed)

    g_true = template(1.0)
    alphas = np.linspace(-1.0, 3.0, 600)

    # --- Uniform noise (homoscedastic) ---
    sig_uniform = np.full(N_PIXELS, 0.5)
    f_uniform = g_true + rng.normal(0.0, 1.0, N_PIXELS) * sig_uniform

    # --- Heteroscedastic noise: same total variance, concentrated in 15% of pixels ---
    sig_het = np.full(N_PIXELS, 0.15)
    contaminated = rng.choice(N_PIXELS, size=60, replace=False)
    sig_het[contaminated] = 1.6
    sig_het *= np.sqrt((sig_uniform ** 2).sum() / (sig_het ** 2).sum())
    f_het = g_true + rng.normal(0.0, 1.0, N_PIXELS) * sig_het

    def curves(data, sigma):
        bl19 = np.array([lnL_bl19(data, template(a)) for a in alphas])
        blasp = np.array([lnL_blain24(data, template(a), sigma) for a in alphas])
        return bl19 - bl19.max(), blasp - blasp.max()

    bl19_u, blasp_u = curves(f_uniform, sig_uniform)
    bl19_h, blasp_h = curves(f_het, sig_het)

    c_bl19, c_blasp = "#2166ac", "#d6604d"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (bl19, blasp), title in [
        (axes[0], (bl19_u, blasp_u), "Uniform noise (homoscedastic)"),
        (axes[1], (bl19_h, blasp_h), "Heteroscedastic noise (same total variance)"),
    ]:
        ax.plot(alphas, bl19, color=c_bl19, lw=2.2, label="Brogi & Line (2019)")
        ax.plot(alphas, blasp, color=c_blasp, lw=2.2, label="Blain et al. (2024)")
        ax.axvline(1.0, color="black", ls="--", lw=1.3, label="Truth")
        ax.set_xlim(alphas.min(), alphas.max())
        ax.set_ylim(-8, 0.4)
        ax.set_xlabel(r"line-depth scaling $\alpha$", fontsize=12)
        ax.set_title(title, fontsize=12)
        w_bl19 = _one_sigma_width(alphas, bl19)
        w_blasp = _one_sigma_width(alphas, blasp)
        ax.text(0.04, 0.06,
                f"1$\\sigma$ width\nBL19  = {w_bl19:.2f}\nBlain24 = {w_blasp:.2f}\n"
                f"ratio = {w_bl19 / w_blasp:.1f}$\\times$",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(r"$\Delta \ln \mathcal{L}$", fontsize=12)
    axes[1].legend(fontsize=10, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")
    print(f"  Uniform:        BL19/Blain24 1-sigma width ratio = "
          f"{_one_sigma_width(alphas, bl19_u) / _one_sigma_width(alphas, blasp_u):.1f}x")
    print(f"  Heteroscedastic: BL19/Blain24 1-sigma width ratio = "
          f"{_one_sigma_width(alphas, bl19_h) / _one_sigma_width(alphas, blasp_h):.1f}x")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="likelihood_weighting.png",
                   help="Output figure path.")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    make_figure(args.output, args.seed)
