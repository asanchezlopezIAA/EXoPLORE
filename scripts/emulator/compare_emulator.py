"""Compare emulator predictions against true petitRADTRANS spectra.

Produces a PDF with one row per test case, each row showing:
  Top panel    : true pRT spectrum (black) and emulated spectrum (red)
  Bottom panel : residuals (true - emulated), normalised by the RMS of the true spectrum

Usage
-----
    python scripts/emulator/compare_emulator.py \\
        --emulator_dir emulators/carmenes_nir/order_23
"""

import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# ── Test parameter sets (log10_X_H2O, T_eq) ───────────────────────────────────
TEST_CASES = [
    (-3.0, 800.0,  "low T"),
    (-3.0, 1200.0, "mid T"),
    (-3.0, 1800.0, "high T"),
    (-6.0, 1200.0, "low X"),
    (-1.0, 1200.0, "high X"),
]

# ── Planet constants for call_pRT ─────────────────────────────────────────────
import petitRADTRANS.physical_constants as cst

GRAVITY   = 2201.20
R_PL_CM   = 8.1358e+09
R_STAR_CM = 5.5988e+10
R_INSTRUMENT = 80400.0

INP_DAT = {
    "Gravity":           GRAVITY,
    "M_pl":              None,
    "R_pl":              R_PL_CM,
    "event":             "transit",
    "T_star":            None,
    "R_star":            R_STAR_CM,
    "v_rotsini":         350000.0,
    "R_instrument":      R_INSTRUMENT,
    "pixels_per_res_el": 3.3,
    "res":               R_INSTRUMENT,
}

WAVE_BOUNDS = (1.51258, 1.54068)


def _init_prt():
    from petitRADTRANS.radtrans import Radtrans
    pressures = np.logspace(-6, 2, 100)
    atm = Radtrans(
        pressures=pressures,
        wavelength_boundaries=[WAVE_BOUNDS[0] - 0.01, WAVE_BOUNDS[1] + 0.01],
        line_species=["H2O"],
        gas_continuum_contributors=["H2--H2", "H2--He"],
        rayleigh_species=["H2", "He"],
        line_opacity_mode="lbl",
    )
    return atm, pressures


def _call_prt(atm, pressures, log10_x_h2o, T_eq):
    from exoplore.atmosphere.prt import call_pRT
    result = call_pRT(
        inp_dat=INP_DAT,
        pressures=pressures,
        prt_object=atm,
        species=["H2", "He", "H2O"],
        vmr=[0.0, 0.0, 10 ** log10_x_h2o],
        MMW=2.33,
        p0=0.01,
        isothermal=True,
        iso_T_value=T_eq,
        two_point_T=False,
        p_points=None,
        t_points=None,
        kappa=0.01,
        gamma=0.4,
        T_equil=T_eq,
        metallicity=0.0,
        C_to_O=0.55,
        use_easyCHEM=False,
    )
    return np.asarray(result[0]), np.asarray(result[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emulator_dir", default="emulators/carmenes_nir/order_23")
    ap.add_argument("--output",       default="emulators/carmenes_nir/order_23/comparison.pdf")
    args = ap.parse_args()

    from exoplore.retrieval.emulator import PrtEmulator
    print("Loading emulator...")
    em = PrtEmulator.load(args.emulator_dir)

    print("Initialising pRT (this takes ~60 s)...")
    atm, pressures = _init_prt()

    n_cases = len(TEST_CASES)
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 3 * n_cases),
                             gridspec_kw={"height_ratios": [1] * n_cases,
                                          "hspace": 0.05, "wspace": 0.3})
    if n_cases == 1:
        axes = [axes]

    for row, (log10_x, T_eq, label) in enumerate(TEST_CASES):
        print(f"  Computing: log10(X)={log10_x}, T={T_eq} K  [{label}]")
        wave_true, spec_true = _call_prt(atm, pressures, log10_x, T_eq)
        wave_emul, spec_emul = em.predict(T_eq=T_eq, log10_x_h2o=log10_x)

        # Align to same wavelength grid (emulator and pRT should be identical)
        wave = wave_true
        resid = spec_true - spec_emul
        rms   = np.std(spec_true)
        ccf   = float(np.dot(spec_true - spec_true.mean(),
                             spec_emul - spec_emul.mean()) /
                      (np.linalg.norm(spec_true - spec_true.mean()) *
                       np.linalg.norm(spec_emul - spec_emul.mean()) + 1e-30))

        # Spectrum panel
        ax = axes[row][0] if n_cases > 1 else axes[0]
        ax.plot(wave, spec_true, color="k",       lw=0.6, label="pRT (true)")
        ax.plot(wave, spec_emul, color="firebrick", lw=0.6, alpha=0.8, label="Emulator")
        ax.set_ylabel("Flux (a.u.)", fontsize=10)
        ax.set_title(f"{label}  |  log₁₀(X)={log10_x}, T={T_eq:.0f} K  |  CCF={ccf:.5f}",
                     fontsize=9)
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")
        if row < n_cases - 1:
            ax.set_xticklabels([])

        # Residuals panel
        ax2 = axes[row][1] if n_cases > 1 else axes[1]
        ax2.plot(wave, resid / rms * 100, color="steelblue", lw=0.5)
        ax2.axhline(0, color="k", lw=0.5, ls="--")
        ax2.set_ylabel("Residual (% of RMS)", fontsize=10)
        ax2.set_title(f"RMSE = {np.sqrt(np.mean(resid**2)):.2e}", fontsize=9)
        if row < n_cases - 1:
            ax2.set_xticklabels([])

    for ax in [axes[-1][0], axes[-1][1]] if n_cases > 1 else axes:
        ax.set_xlabel("Wavelength (µm)", fontsize=10)

    fig.suptitle("Emulator vs petitRADTRANS, CARMENES NIR order 23", fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
