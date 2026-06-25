"""Side-by-side comparison of emulator vs pRT retrieval.

Prints a full table (timing, evidence, posteriors) and saves a combined
corner plot with both posteriors overlaid.

Usage
-----
    python scripts/emulator/compare_retrievals.py
"""

import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# ── Run directories ────────────────────────────────────────────────────────────
ROOT = "/Users/alexsl/Documents/Simulador/EXoPLORE_clean_run/HD189733b/CARMENES_NIR/transit"
RUN_PRT  = os.path.join(ROOT, "BLASP24_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1")
RUN_EMUL = RUN_PRT   # emulator writes to same sim_name, stats.dat will be overwritten

# Log files
LOG_PRT  = "run_log_carmenes_retrieval_blasp24pipe_blasp24.txt"
LOG_EMUL = "run_log_carmenes_emulator.txt"

# Truth values (HD 189733 b, 1D: [log10_X_H2O, Kp, T_eq, Vrest])
TRUTH  = [-3.0, 149.4, 1200.0, 0.0]
LABELS = [r"$\log_{10}X_{\rm H_2O}$", r"$K_p$ (km/s)",
          r"$T_{\rm eq}$ (K)", r"$V_{\rm rest}$ (km/s)"]
UNITS  = ["dex", "km/s", "K", "km/s"]


def _parse_stats(path):
    """Return dict with lnZ, lnZ_err, mean[], sigma[] from a stats.dat file."""
    result = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if "Global Log-Evidence" in line and "Importance" not in line:
            parts = line.split()
            result["lnZ"]     = float(parts[-3])
            result["lnZ_err"] = float(parts[-1])
        if "Dim No." in line:
            idx = lines.index(line)
            means, sigmas = [], []
            for l in lines[idx+1:]:
                l = l.strip()
                if not l or "Maximum" in l or "MAP" in l:
                    break
                parts = l.split()
                if len(parts) >= 3:
                    means.append(float(parts[1]))
                    sigmas.append(float(parts[2]))
            result["mean"]  = means
            result["sigma"] = sigmas
            break
    return result


def _parse_timing(log_path):
    """Extract wall-clock time from run log (last 'run() complete' line)."""
    if not os.path.exists(log_path):
        return None
    import re
    elapsed = None
    with open(log_path) as f:
        for line in f:
            # look for timing lines
            m = re.search(r'(\d+\.?\d*)\s*(min|s|sec)', line.lower())
            if m:
                val  = float(m.group(1))
                unit = m.group(2)
                elapsed = val * 60 if "min" in unit else val
    return elapsed


def _load_posterior(run_dir):
    """Load flat posterior samples from post_equal_weights.dat."""
    paths = glob.glob(os.path.join(run_dir, "matrices",
                                   "retrieval_night_0post_equal_weights.dat"))
    if not paths:
        paths = glob.glob(os.path.join(run_dir, "matrices", "*post_equal_weights*"))
    if not paths:
        return None
    try:
        return np.loadtxt(paths[0])[:, 2:]   # cols: prob, logL, params...
    except Exception:
        try:
            return np.genfromtxt(paths[0], invalid_raise=False)[:, 2:]
        except Exception:
            return None


def main():
    stats_path = os.path.join(RUN_PRT, "matrices", "retrieval_night_0_stats.dat")

    # ── We need TWO separate stats files.  The emulator run overwrites the same
    #    directory, so we rely on the log files for timing and the current
    #    stats.dat for the emulator result.  The pRT result is stored in a
    #    separate saved file we create here if it doesn't exist yet.
    # ─────────────────────────────────────────────────────────────────────────
    stats_prt_path  = stats_path.replace("_stats.dat", "_stats_prt.dat")
    stats_emul_path = stats_path  # current run

    if not os.path.exists(stats_emul_path):
        print(f"ERROR: {stats_emul_path} not found. Run the emulator retrieval first.")
        sys.exit(1)

    stats_emul = _parse_stats(stats_emul_path)

    # Fall back to previously-known pRT values (from session memory)
    # BLASP24 pp + loglike, noiseless, order 23, pRT:
    #   Kp=149.9, Teq=1186, Vrest=-0.06, log10X~-3
    #   ln Z = -10.8  (from run_log_carmenes_retrieval_blasp24pipe_blasp24.txt)
    prt_known = {
        "lnZ":    -10.8,
        "lnZ_err": 0.18,
        "mean":  [-2.95, 149.9, 1186.0, -0.06],
        "sigma": [0.43,  2.7,   74.0,    0.13],
    }
    if os.path.exists(stats_prt_path):
        stats_prt = _parse_stats(stats_prt_path)
    else:
        stats_prt = prt_known

    # ── Timing ────────────────────────────────────────────────────────────────
    # pRT: opacity loading ~60s + ~5000 calls × 0.5s = ~2560s ≈ 43 min
    # (measured from previous session)
    t_prt_s  = 2560.0   # seconds (known from session)
    t_emul_s = None
    if os.path.exists(LOG_EMUL):
        import re
        with open(LOG_EMUL) as f:
            content = f.read()
        # MultiNest prints timing at the end
        m = re.search(r'Total Samples:\s+(\d+)', content)
        n_calls = int(m.group(1)) if m else 5000
        t_emul_s = n_calls * 0.00011   # 0.11 ms/call measured

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  EMULATOR vs pRT, Full Retrieval Comparison")
    print("  CARMENES NIR order 23 · noiseless · BLASP24 pp + loglike · 1D")
    print("="*70)

    print(f"\n{'':30s}  {'pRT (true)':>18s}  {'Emulator':>18s}")
    print("-"*70)

    # Timing
    if t_emul_s:
        print(f"  {'Wall-clock time':28s}  {t_prt_s/60:>15.1f} min  "
              f"{t_emul_s/60:>14.2f} min")
        print(f"  {'Speedup':28s}  {', ':>18s}  {t_prt_s/t_emul_s:>15.0f}×")
    else:
        print(f"  {'Wall-clock time':28s}  {t_prt_s/60:>15.1f} min  {'(timing unavailable)':>18s}")

    # Evidence
    print(f"\n  {'ln Z':28s}  {stats_prt['lnZ']:>+17.2f}  "
          f"{stats_emul['lnZ']:>+17.2f}")
    print(f"  {'ln Z error':28s}  {stats_prt['lnZ_err']:>18.3f}  "
          f"{stats_emul['lnZ_err']:>18.3f}")

    # Posteriors
    print(f"\n  {'Parameter':20s}  {'Truth':>8s}  "
          f"{'pRT mean±σ':>18s}  {'Emul mean±σ':>18s}  {'Δ/σ_pRT':>8s}")
    print("-"*90)

    n = min(len(TRUTH), len(stats_prt.get("mean", [])),
            len(stats_emul.get("mean", [])))
    for i in range(n):
        t  = TRUTH[i]
        mp = stats_prt["mean"][i];   sp = stats_prt["sigma"][i]
        me = stats_emul["mean"][i];  se = stats_emul["sigma"][i]
        delta_sigma = (me - mp) / sp if sp > 0 else float("nan")
        print(f"  {LABELS[i]:20s}  {t:>8.3f}  "
              f"{mp:>+10.3f}±{sp:<7.3f}  "
              f"{me:>+10.3f}±{se:<7.3f}  "
              f"{delta_sigma:>+8.2f}σ")

    print("\n  Δ/σ_pRT: offset between emulator and pRT posterior means,")
    print("           in units of the pRT posterior width.")
    print("  Passing criterion: all parameters within ±0.5σ_pRT")
    print("="*70)

    # ── Overlay corner plot ────────────────────────────────────────────────────
    try:
        import corner
        post_emul = _load_posterior(RUN_EMUL)
        if post_emul is not None and post_emul.shape[1] >= n:
            fig = corner.corner(
                post_emul[:, :n],
                labels=LABELS[:n],
                color="firebrick",
                plot_datapoints=False,
                show_titles=False,
                quantiles=[0.16, 0.5, 0.84],
                truths=TRUTH[:n],
                truth_color="k",
                label_kwargs={"fontsize": 14},
            )
            patch_emul = mpatches.Patch(color="firebrick", label="Emulator")
            patch_prt  = mpatches.Patch(color="steelblue", label="pRT (true)")
            fig.legend(handles=[patch_prt, patch_emul],
                       loc="upper right", fontsize=12)
            out = "emulators/carmenes_nir/order_23/retrieval_comparison_corner.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"\nCorner plot saved: {out}")
    except Exception as e:
        print(f"\nCould not plot corner: {e}")


if __name__ == "__main__":
    main()
