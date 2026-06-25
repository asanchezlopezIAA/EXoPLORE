"""Illustrate the Doppler separation that underlies high-resolution spectroscopy.

A schematic of a time-series spectral matrix during a transit. The telluric and
stellar lines are essentially stationary in wavelength from one exposure to the
next, so they form vertical features. The planet's lines Doppler-shift as its
radial velocity changes during the transit, so they trace an inclined trail.
This separation in the time-wavelength plane is what allows the planet signal to
be isolated from the much stronger contaminants, and it is the basis of the
cross-correlation technique.

This script generates a teaching schematic (no real data) for the EXoPLORE
Concepts primer.

    python scripts/illustrate_doppler_trick.py --output doppler_trick.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(output: str) -> None:
    n_exp = 60                  # exposures (time / orbital phase)
    n_pix = 600                 # wavelength channels
    wav = np.linspace(0.0, 1.0, n_pix)
    rng = np.random.default_rng(0)

    img = np.ones((n_exp, n_pix))

    def gaussian_line(centre, depth, width):
        return depth * np.exp(-0.5 * ((wav - centre) / width) ** 2)

    # Stationary telluric and stellar lines: fixed wavelength across exposures
    tell_centres = rng.uniform(0.05, 0.95, 14)
    tell_depths = rng.uniform(0.25, 0.7, 14)
    for c, d in zip(tell_centres, tell_depths):
        img -= gaussian_line(c, d, 0.004)[None, :]

    # Planet lines: shift in wavelength with exposure (radial-velocity change)
    phase = np.linspace(0, 1, n_exp)
    shift = 0.10 * (phase - 0.5)          # net drift across the sequence
    planet_centres = np.array([0.30, 0.45, 0.60, 0.72])
    for t in range(n_exp):
        for c in planet_centres:
            img[t] -= gaussian_line(c + shift[t], 0.10, 0.004)

    # Mild photon noise
    img += rng.normal(0, 0.015, img.shape)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.imshow(img, aspect="auto", cmap="gray", origin="lower",
              extent=(0, 1, 0, 1), vmin=0.1, vmax=1.0,
              interpolation="nearest")

    ax.set_xlabel("wavelength", fontsize=12)
    ax.set_ylabel("time  /  orbital phase", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    # Annotate a stationary line (vertical) and the planet trail (inclined)
    ax.annotate("telluric and stellar lines\n(stationary in wavelength)",
                xy=(tell_centres[3], 0.5), xytext=(0.02, 1.12),
                textcoords="axes fraction", fontsize=11, ha="left",
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.5),
                color="#2166ac")

    # Arrow following the planet trail
    x0, x1 = planet_centres[1] + shift[5], planet_centres[1] + shift[-6]
    ax.annotate("", xy=(x1, 0.92), xytext=(x0, 0.08),
                arrowprops=dict(arrowstyle="->", color="#d6604d", lw=2.0))
    ax.text(0.74, 1.12, "planet signal\n(Doppler-shifts during transit)",
            transform=ax.transAxes, fontsize=11, ha="right", color="#d6604d")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="doppler_trick.png")
    return p.parse_args()


if __name__ == "__main__":
    make_figure(_parse().output)
