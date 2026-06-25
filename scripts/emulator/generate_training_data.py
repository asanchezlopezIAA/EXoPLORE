"""Generate petitRADTRANS training spectra and fit PCA for the pRT emulator.

Two-pass strategy: generates spectra twice using the same Latin-hypercube
parameter set (deterministic, fixed seed).  First pass fits IncrementalPCA
without saving any raw spectra to disk.  Second pass transforms each batch
to PCA coefficients and saves only those.  Peak disk usage is ~40 MB for
100k samples instead of ~8 GB for raw spectra.

Usage
-----
    python scripts/emulator/generate_training_data.py \\
        --n_samples 100000 \\
        --n_workers 7 \\
        --n_pca     100 \\
        --output_dir emulators/carmenes_nir/order_23

Output
------
    params_train.npy          (N, 2)     log10(X_H2O), T_eq
    pca_coefficients.npy      (N, k)     PCA-projected training targets
    pca_mean.npy              (n_wave,)  spectral mean
    pca_components.npy        (k, n_wave)PCA basis vectors
    param_bounds.npy          (2, 2)     [[min, max], ...] per parameter
    wave_prt.npy              (n_wave,)  wavelength grid in µm
    metadata.json
"""

import argparse, json, multiprocessing, os, sys, time
import numpy as np
from scipy.stats.qmc import LatinHypercube
from sklearn.decomposition import IncrementalPCA

# ── Prior bounds, narrowed to retrieval-relevant range ──────────────────────
# Wider range ([-8, 0]) covers the full prior but wastes capacity on
# log10_X < -5 where the spectrum is nearly flat and the retrieval
# never samples.  Narrow priors mean denser coverage → better accuracy
# in the region MultiNest actually explores.
LOG10_X_H2O_MIN = -5.0
LOG10_X_H2O_MAX = -1.0
T_EQ_MIN        = 600.0
T_EQ_MAX        = 1800.0

# ── HD 189733 b constants ─────────────────────────────────────────────────────
GRAVITY      = 2201.20
R_PL_CM      = 8.1358e+09
R_STAR_CM    = 5.5988e+10
R_INSTRUMENT = 80400.0
WAVE_BOUNDS  = (1.51258, 1.54068)   # CARMENES NIR order 23 (µm)

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

_atm = None
_pressures = None
_wave_prt  = None


def _worker_init(wave_min, wave_max):
    global _atm, _pressures, _wave_prt
    import petitRADTRANS.physical_constants as cst
    from petitRADTRANS.radtrans import Radtrans
    _pressures = np.logspace(-6, 2, 100)
    _atm = Radtrans(
        pressures=_pressures,
        wavelength_boundaries=[wave_min - 0.01, wave_max + 0.01],
        line_species=["H2O"],
        gas_continuum_contributors=["H2--H2", "H2--He"],
        rayleigh_species=["H2", "He"],
        line_opacity_mode="lbl",
    )
    _wave_prt = cst.c / _atm.frequencies / 1e-4


def _compute_spectrum(args):
    log10_x_h2o, t_eq = args
    from exoplore.atmosphere.prt import call_pRT
    try:
        result = call_pRT(
            inp_dat=_inp_dat if "_inp_dat" in globals() else INP_DAT,
            pressures=_pressures,
            prt_object=_atm,
            species=["H2", "He", "H2O"],
            vmr=[0.0, 0.0, 10 ** log10_x_h2o],
            MMW=2.33,
            p0=0.01,
            isothermal=True,
            iso_T_value=t_eq,
            two_point_T=False,
            p_points=None,
            t_points=None,
            kappa=0.01,
            gamma=0.4,
            T_equil=t_eq,
            metallicity=0.0,
            C_to_O=0.55,
            use_easyCHEM=False,
        )
        return np.asarray(result[1], dtype=np.float32)
    except Exception as e:
        return None


def _generate_batch(params_batch, n_workers, wave_min, wave_max):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_worker_init,
                  initargs=(wave_min, wave_max)) as pool:
        results = list(pool.imap(
            _compute_spectrum,
            [(p[0], p[1]) for p in params_batch],
            chunksize=10,
        ))
    good_mask = [r is not None for r in results]
    good_specs = np.array([r for r in results if r is not None], dtype=np.float32)
    good_params = params_batch[good_mask]
    return good_params, good_specs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_samples",   type=int,   default=100_000)
    ap.add_argument("--n_workers",   type=int,
                    default=max(1, multiprocessing.cpu_count() - 1))
    ap.add_argument("--n_pca",       type=int,   default=100)
    ap.add_argument("--batch_size",  type=int,   default=10_000)
    ap.add_argument("--output_dir",  default="emulators/carmenes_nir/order_23")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Sample parameter space once (fixed seed → deterministic) ─────────────
    print(f"Sampling {args.n_samples} points (narrowed prior)...")
    print(f"  log10(X_H2O): [{LOG10_X_H2O_MIN}, {LOG10_X_H2O_MAX}]")
    print(f"  T_eq:         [{T_EQ_MIN}, {T_EQ_MAX}] K")
    sampler = LatinHypercube(d=2, seed=42)
    unit = sampler.random(n=args.n_samples)
    log10_x = LOG10_X_H2O_MIN + (LOG10_X_H2O_MAX - LOG10_X_H2O_MIN) * unit[:, 0]
    t_eq    = T_EQ_MIN        + (T_EQ_MAX        - T_EQ_MIN)        * unit[:, 1]
    params  = np.column_stack([log10_x, t_eq]).astype(np.float32)

    n_batches = int(np.ceil(args.n_samples / args.batch_size))

    # ── Initialise one worker to get wave_pRT shape ───────────────────────────
    _worker_init(WAVE_BOUNDS[0], WAVE_BOUNDS[1])
    n_wave = len(_wave_prt)
    wave   = _wave_prt.copy()
    print(f"\npRT grid: {n_wave} points, {wave.min():.4f} to {wave.max():.4f} µm")
    est_time = args.n_samples / 8.1 / 60
    print(f"Estimated generation time (2 passes): {2*est_time:.0f} min")
    print(f"Estimated storage (PCA coefficients only): "
          f"{args.n_samples * args.n_pca * 4 / 1e6:.0f} MB\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 1, fit IncrementalPCA (no spectra saved to disk)
    # ─────────────────────────────────────────────────────────────────────────
    print("Pass 1/2, fitting IncrementalPCA...")
    ipca = IncrementalPCA(n_components=args.n_pca)
    t0 = time.time()
    good_params_all = []

    for b in range(n_batches):
        i0 = b * args.batch_size
        i1 = min(i0 + args.batch_size, args.n_samples)
        gp, gs = _generate_batch(params[i0:i1], args.n_workers,
                                 WAVE_BOUNDS[0], WAVE_BOUNDS[1])
        good_params_all.append(gp)
        if len(gs) >= args.n_pca:
            ipca.partial_fit(gs.astype(np.float64))
        elapsed = time.time() - t0
        rate    = (i1 - i0) / elapsed * (b + 1)
        print(f"  batch {b+1:3d}/{n_batches}  {i1:6d}/{args.n_samples}  "
              f"{rate:.1f} spec/s  "
              f"var={ipca.explained_variance_ratio_.sum()*100:.3f}%",
              flush=True)

    params_good = np.vstack(good_params_all)
    n_good = len(params_good)
    elapsed_p1 = time.time() - t0
    print(f"\nPass 1 done in {elapsed_p1/60:.1f} min  "
          f"({n_good} good spectra)")
    print(f"Cumulative variance explained: "
          f"{ipca.explained_variance_ratio_.sum()*100:.4f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 2, transform batches to PCA coefficients (no raw spectra saved)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPass 2/2, computing PCA coefficients...")
    t0 = time.time()
    coeffs_all = []
    params_good2 = []   # reorder to match only successful spectra
    ptr = 0

    for b in range(n_batches):
        i0 = b * args.batch_size
        i1 = min(i0 + args.batch_size, args.n_samples)
        n_batch = i1 - i0
        gp, gs = _generate_batch(params[i0:i1], args.n_workers,
                                 WAVE_BOUNDS[0], WAVE_BOUNDS[1])
        if len(gs) > 0:
            coeffs = ipca.transform(gs.astype(np.float64)).astype(np.float32)
            coeffs_all.append(coeffs)
            params_good2.append(gp)
        elapsed = time.time() - t0
        print(f"  batch {b+1:3d}/{n_batches}  {i1:6d}/{args.n_samples}  "
              f"{len(gs)/elapsed*(b+1):.1f} spec/s", flush=True)

    coefficients = np.vstack(coeffs_all)    # (N, k)
    params_final = np.vstack(params_good2)  # (N, 2)
    elapsed_p2   = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, "params_train.npy"),
            params_final)
    np.save(os.path.join(args.output_dir, "pca_coefficients.npy"),
            coefficients)
    np.save(os.path.join(args.output_dir, "pca_mean.npy"),
            ipca.mean_.astype(np.float32))
    np.save(os.path.join(args.output_dir, "pca_components.npy"),
            ipca.components_.astype(np.float32))
    np.save(os.path.join(args.output_dir, "param_bounds.npy"),
            np.array([[LOG10_X_H2O_MIN, T_EQ_MIN],
                      [LOG10_X_H2O_MAX, T_EQ_MAX]], dtype=np.float32))
    np.save(os.path.join(args.output_dir, "wave_prt.npy"),
            wave.astype(np.float32))

    # Remove old raw spectra file if present (no longer needed)
    old = os.path.join(args.output_dir, "spectra_train.npz")
    if os.path.exists(old):
        os.remove(old)
        print(f"Removed old raw spectra file: {old}")

    total_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
        for f in ["params_train.npy", "pca_coefficients.npy",
                  "pca_mean.npy", "pca_components.npy",
                  "param_bounds.npy", "wave_prt.npy"]
    )

    meta = {
        "n_samples":            int(len(params_final)),
        "n_pca":                args.n_pca,
        "n_wave":               int(n_wave),
        "wave_min_um":          float(wave.min()),
        "wave_max_um":          float(wave.max()),
        "param_names":          ["log10_X_H2O", "T_eq_K"],
        "param_bounds":         [[LOG10_X_H2O_MIN, LOG10_X_H2O_MAX],
                                 [T_EQ_MIN, T_EQ_MAX]],
        "variance_explained":   float(ipca.explained_variance_ratio_.sum()),
        "elapsed_pass1_s":      round(elapsed_p1, 1),
        "elapsed_pass2_s":      round(elapsed_p2, 1),
        "total_storage_mb":     round(total_mb, 1),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.output_dir}/  ({total_mb:.0f} MB total)")
    print(f"  params_train.npy, pca_coefficients.npy")
    print(f"  pca_mean.npy, pca_components.npy, param_bounds.npy, wave_prt.npy")
    print(f"\nDone. {len(params_final)} spectra, "
          f"{meta['variance_explained']*100:.4f}% variance explained.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    main()
