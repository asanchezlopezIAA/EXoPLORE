#!/usr/bin/env python3
"""
verify_clean_run.py, Scientific verification of EXoPLORE clean-run outputs.

Run with the venv active:
    python "/Users/alexsl/Documents/Claude/Projects/EXoPLORE Repository/scripts/verify_clean_run.py"
"""

import sys, os, glob
import numpy as np

# ── base directories ───────────────────────────────────────────────────────────
MATRICES_PARENT = os.path.expanduser(
    "~/Documents/Simulador/EXoPLORE_clean_run/"
    "HD189733b/ANDES/transit/matrices/"
)

# Find the run subdirectory
candidates = sorted(glob.glob(os.path.join(MATRICES_PARENT, "matrices_BL19_*")))
if not candidates:
    sys.exit(f"\n[ERROR] No matrices_BL19_* subdirectory found under:\n  {MATRICES_PARENT}")
MAT_DIR = candidates[-1]

# Derive suffix from subdir name:  matrices_<SUFFIX>  → <SUFFIX>
SFX = os.path.basename(MAT_DIR)[len("matrices_"):]

print(f"\n{'='*72}")
print(f"  EXoPLORE clean-run scientific verification")
print(f"{'='*72}")
print(f"  Run subdir : {os.path.basename(MAT_DIR)}")
print(f"  Suffix     : {SFX}")

# ── helpers ────────────────────────────────────────────────────────────────────
def load_run(stem):
    """Load <MAT_DIR>/<stem>_<SFX>.npz, key 'a'."""
    path = os.path.join(MAT_DIR, f"{stem}_{SFX}.npz")
    d = np.load(path)
    return d[list(d.keys())[0]]

def try_load_run(stem):
    try:
        return load_run(stem)
    except FileNotFoundError:
        return None

def load_parent_fits(fname):
    """Load a FITS array, tries several candidate directories."""
    from astropy.io import fits
    # transit/matrices/  (original guess)
    # transit/<SFX>/matrices/  (actual location from output structure)
    TRANSIT_DIR = os.path.dirname(MATRICES_PARENT.rstrip("/"))
    candidates = [
        os.path.join(MATRICES_PARENT, fname),
        os.path.join(TRANSIT_DIR, SFX, "matrices", fname),
        os.path.join(MAT_DIR, fname),
    ]
    for path in candidates:
        if os.path.exists(path):
            with fits.open(path) as hdul:
                return hdul[0].data
    return None

def section(title):
    print(f"\n  {'─'*64}")
    print(f"  {title}")
    print(f"  {'─'*64}")

ok_count = fail_count = 0
KP_LIT = 149.4   # km/s  HD189733b literature

def check(label, cond, detail=""):
    global ok_count, fail_count
    sym = "✓" if cond else "✗"
    msg = f"  {sym}  {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    if cond:
        ok_count += 1
    else:
        fail_count += 1

def skip(reason):
    print(f"     [SKIP] {reason}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Velocity / Kp grids
# ══════════════════════════════════════════════════════════════════════════════
section("1. Velocity grids")

v_ccf    = try_load_run("v_ccf")
kp_range = try_load_run("kp_range")

for name, arr in [("v_ccf", v_ccf), ("kp_range", kp_range)]:
    if arr is None:
        print(f"     {name}_{SFX}.npz : NOT FOUND in run subdir")
    else:
        print(f"     {name}: shape={arr.shape}  min={arr.min():.1f}  max={arr.max():.1f}  km/s")

if v_ccf is not None:
    check("v_ccf covers ±300 km/s",
          v_ccf.min() <= -300 and v_ccf.max() >= 300,
          f"{v_ccf.min():.0f} to {v_ccf.max():.0f} km/s")
if kp_range is not None:
    check(f"kp_range brackets Kp_lit={KP_LIT} km/s",
          kp_range.min() < KP_LIT < kp_range.max(),
          f"{kp_range.min():.0f} to {kp_range.max():.0f} km/s")
    ikp = np.argmin(np.abs(kp_range - KP_LIT))
    print(f"     Closest kp_range to {KP_LIT}: {kp_range[ikp]:.2f} km/s  (idx {ikp})")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-order CCF matrices
# ══════════════════════════════════════════════════════════════════════════════
section("2. Per-order CCF matrices")

ccf_files = sorted(glob.glob(os.path.join(MAT_DIR, f"ccf_store_order_*_{SFX}.npz")))
print(f"     Found {len(ccf_files)} ccf_store_order files")
check("76 orders present", len(ccf_files) == 76, f"found {len(ccf_files)}")

ccf_cube = []
if ccf_files:
    for idx in [0, len(ccf_files)//2, -1]:
        f   = ccf_files[idx]
        d   = np.load(f)
        arr = d[list(d.keys())[0]]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        lbl = f"order_{os.path.basename(f).split('_')[3]}"
        print(f"     {lbl}: shape={arr.shape}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  finite={np.isfinite(arr).all()}")
        check(f"{lbl} 2D", arr.ndim == 2, f"ndim={arr.ndim}")
        check(f"{lbl} all finite", np.isfinite(arr).all())
        check(f"{lbl} CCF in [-1,1]",
              arr.min() >= -1.01 and arr.max() <= 1.01,
              f"[{arr.min():.3f}, {arr.max():.3f}]")

    for f in ccf_files:
        d   = np.load(f)
        arr = d[list(d.keys())[0]]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if np.isfinite(arr).all():
            ccf_cube.append(arr)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Masking arrays
# ══════════════════════════════════════════════════════════════════════════════
section("3. Masking arrays")

for stem in ["mask", "mask_snr", "mask_inter",
             "useful_spectral_points", "useful_spectral_points_snr"]:
    arr = try_load_run(stem)
    if arr is None:
        print(f"     {stem}_{SFX}.npz : NOT FOUND")
    else:
        print(f"     {stem}: shape={arr.shape}  dtype={arr.dtype}")
        check(f"{stem} non-empty", arr.size > 0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Phase / JD arrays
# ══════════════════════════════════════════════════════════════════════════════
section("4. Phase / JD arrays")

phase = None
for fname, label in [("phase.fits", "phase"), ("syn_jd.fits", "syn_jd")]:
    # Try parent matrices/ dir first, then run subdir
    arr = load_parent_fits(fname)
    if arr is None:
        # check run subdir too
        run_path = os.path.join(MAT_DIR, fname)
        if os.path.exists(run_path):
            from astropy.io import fits
            with fits.open(run_path) as hdul:
                arr = hdul[0].data
    if arr is None:
        print(f"     {fname} : NOT FOUND (checked both parent and run subdir)")
    else:
        print(f"     {label}: shape={arr.shape}  min={arr.min():.5f}  max={arr.max():.5f}")
        check(f"{label} 1D with ~390 spectra",
              arr.ndim == 1 and 100 < arr.size < 800,
              f"size={arr.size}")
        if label == "phase":
            phase = arr

# also try phase.npz (Different_nights path)
if phase is None:
    arr = try_load_run("phase")
    if arr is None:
        path_npz = os.path.join(MATRICES_PARENT, "phase.npz")
        if os.path.exists(path_npz):
            d = np.load(path_npz)
            arr = d[list(d.keys())[0]]
    if arr is not None:
        print(f"     phase (npz): shape={arr.shape}")
        phase = arr.ravel() if arr.ndim > 1 else arr

# ══════════════════════════════════════════════════════════════════════════════
# 5. Kp-Vsys detection map
# ══════════════════════════════════════════════════════════════════════════════
section("5. Kp-Vsys detection map  (built from ccf_store)")

if v_ccf is None or kp_range is None:
    skip("v_ccf or kp_range not loaded")
elif not ccf_cube:
    skip("no finite CCF order arrays found")
elif phase is None:
    skip("phase array not loaded, cannot compute Kp shifts")
else:
    ccf_sum = np.sum(ccf_cube, axis=0)        # (n_lags, n_spectra)
    n_lags, n_spec = ccf_sum.shape
    print(f"     Combined CCF : ({n_lags}, {n_spec})  [{len(ccf_cube)}/{len(ccf_files)} orders]")

    # Trim phase to n_spec if needed
    ph = phase[:n_spec] if len(phase) >= n_spec else phase

    # Build Kp-Vsys map by shifting each spectrum by Kp*sin(2π*phase) and co-adding
    sn_map = np.zeros((len(kp_range), n_lags))
    for i, kp in enumerate(kp_range):
        col = np.zeros(n_lags)
        for j in range(len(ph)):
            rv = kp * np.sin(2 * np.pi * ph[j])
            # Shift CCF left by rv: result[i] = ccf(v[i] + rv)
            # This brings the planet peak (at Kp*sin + Vsys) to Vsys
            shifted = np.interp(v_ccf + rv, v_ccf, ccf_sum[:, j],
                                left=np.nan, right=np.nan)
            ok = np.isfinite(shifted)
            col[ok] += shifted[ok]
        sn_map[i, :] = col

    # Normalise to S/N using the out-of-trail baseline (|Vsys| > 50 km/s)
    out = np.abs(v_ccf) > 50
    if out.sum() > 20:
        sigma = np.std(sn_map[:, out])
        sn_norm = sn_map / sigma if sigma > 0 else sn_map
    else:
        sn_norm = sn_map

    peak_val = float(np.nanmax(sn_norm))
    pidx     = np.unravel_index(np.nanargmax(sn_norm), sn_norm.shape)
    peak_kp  = float(kp_range[pidx[0]])
    peak_vs  = float(v_ccf[pidx[1]])

    print(f"\n     ┌──────────────────────────────────────┐")
    print(f"     │  Peak S/N  = {peak_val:8.2f}                │")
    print(f"     │  Peak Kp   = {peak_kp:8.1f} km/s  (expect ~{KP_LIT:.0f})  │")
    print(f"     │  Peak Vsys = {peak_vs:8.1f} km/s  (expect ~0)       │")
    print(f"     └──────────────────────────────────────┘")

    check("Peak S/N > 3  (detectable signal)",
          peak_val > 3, f"SNR={peak_val:.2f}")
    check("|ΔKp| < 30 km/s from literature",
          abs(peak_kp - KP_LIT) < 30,
          f"|{peak_kp:.1f} - {KP_LIT}| = {abs(peak_kp-KP_LIT):.1f} km/s")
    check("|Vsys| < 30 km/s  (near zero)",
          abs(peak_vs) < 30, f"Vsys={peak_vs:.1f} km/s")

    # 1D slice at peak Kp
    row   = sn_norm[pidx[0], :]
    iv0   = int(np.argmin(np.abs(v_ccf)))
    print(f"\n     1D CCF slice at Kp={peak_kp:.0f} km/s:")
    print(f"       SNR at Vsys = 0    : {row[iv0]:.2f}")
    print(f"       SNR at peak        : {row[pidx[1]]:.2f}  (Vsys={peak_vs:.1f} km/s)")

    # Sanity: out-of-peak noise level
    noise_row = np.concatenate([row[v_ccf < -50], row[v_ccf > 50]])
    if len(noise_row) > 5:
        print(f"       Out-of-peak RMS    : {np.std(noise_row):.2f}  (should be ~1)")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
section("Summary")
total = ok_count + fail_count
print(f"     Passed : {ok_count}/{total}")
print(f"     Failed : {fail_count}/{total}")
if fail_count == 0:
    print("\n  ✓  All checks passed, clean-run looks scientifically correct.\n")
else:
    print(f"\n  ✗  {fail_count} check(s) failed, see output above.\n")
