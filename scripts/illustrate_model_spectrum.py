"""Generate model transmission spectra for the EXoPLORE documentation.

Builds a petitRADTRANS forward model with EXoPLORE's own routines and produces
two teaching figures:

  * model_spectrum.png      A single transmission spectrum (the planet signal
                            that EXoPLORE injects and cross-correlates against),
                            for the Concepts primer.
  * chemistry_co_ratio.png  The same atmosphere at two carbon-to-oxygen ratios,
                            showing how a carbon-rich composition suppresses H2O
                            and enhances carbon-bearing species, for Tutorial 3
                            (Example A).

The opacity tables are loaded once (slow, a few minutes); the two spectra are
then computed from the same Radtrans object.

    python scripts/illustrate_model_spectrum.py \
        --model-output docs/figures/model_spectrum.png \
        --co-output    docs/figures/chemistry_co_ratio.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from petitRADTRANS.radtrans import Radtrans          # noqa: E402
from exoplore.atmosphere.prt import call_pRT          # noqa: E402

# HD 189733 b parameters in CGS (from the planet parameter file)
_INP = {
    "Gravity": None,
    "M_pl": 2.1828495e30,      # g
    "R_pl": 8.1357896e9,       # cm
    "R_star": 5.5987750e10,    # cm
    "T_star": None,            # None -> skip stellar spectrum (we only need the planet)
    "event": "transit",
    "T_int": 200.0,            # internal temperature for the Guillot profile
    "res": 100000.0,           # instrument resolving power
}

_SPECIES = ["H2", "He", "H2O", "CH4", "NH3", "CO", "H2S", "HCN"]
_WAV_MIN, _WAV_MAX = 1.00, 1.70   # NIR window (µm)


def _spectrum(prt, pressures, c_to_o):
    """Return (wavelength µm, transit depth %) for a given C/O ratio."""
    wave, spec, *_ = call_pRT(
        _INP, pressures, prt,
        _SPECIES, None, 2.33, 0.01,
        False, None, False, None, None,
        0.01, 0.4, 1170.0,
        0.53, c_to_o,
        use_easyCHEM=True, P_cloud=None, easychem_CtoO_ret=True,
    )
    return wave, spec * 100.0      # transit depth in per cent


def make_figures(model_output: str, co_output: str) -> None:
    pressures = np.logspace(-6, 2, 100)

    print("Building Radtrans (loading opacities, slow the first time) ...")
    prt = Radtrans(
        pressures=pressures,
        line_species=_SPECIES[2:],
        rayleigh_species=["H2", "He"],
        gas_continuum_contributors=["H2--H2", "H2--He"],
        wavelength_boundaries=[_WAV_MIN - 0.01, _WAV_MAX + 0.01],
        line_opacity_mode="lbl",
    )

    print("Computing spectrum at C/O = 0.55 ...")
    wav, depth_solar = _spectrum(prt, pressures, 0.55)
    print("Computing spectrum at C/O = 0.90 ...")
    _, depth_rich = _spectrum(prt, pressures, 0.90)

    # --- Figure 1: single model spectrum (Concepts primer) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.plot(wav, depth_solar, color="#2166ac", lw=0.6)
    ax.set_xlabel("wavelength (µm)", fontsize=12)
    ax.set_ylabel("transit depth (%)", fontsize=12)
    ax.set_title("Model transmission spectrum (HD 189733 b, C/O = 0.55)", fontsize=12)
    ax.set_xlim(_WAV_MIN, _WAV_MAX)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(model_output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {model_output}")

    # --- Figure 2: C/O comparison (Tutorial 3, Example A) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.plot(wav, depth_solar, color="#2166ac", lw=0.6, label="C/O = 0.55 (H₂O-rich)")
    ax.plot(wav, depth_rich,  color="#d6604d", lw=0.6, label="C/O = 0.90 (carbon-rich)")
    ax.set_xlabel("wavelength (µm)", fontsize=12)
    ax.set_ylabel("transit depth (%)", fontsize=12)
    ax.set_title("Effect of the carbon-to-oxygen ratio on the spectrum", fontsize=12)
    ax.set_xlim(_WAV_MIN, _WAV_MAX)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(co_output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {co_output}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-output", default="model_spectrum.png")
    p.add_argument("--co-output", default="chemistry_co_ratio.png")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    make_figures(a.model_output, a.co_output)
