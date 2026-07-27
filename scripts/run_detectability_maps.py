#!/usr/bin/env python3
"""
run_detectability_maps.py
=========================

Sweep a simulation over a 2-D grid of two atmospheric variables and record the
recovered cross-correlation significance at each grid point, producing a
detectability map, one per molecule.

The sweep is driven by the ``detectability`` block of a normal EXoPLORE config
(see ``exoplore.config.models.DetectabilityConfig``).  Two variable pairs are
supported:

  * metallicity vs C/O    (x=metallicity_wrt_solar, y=carbon_to_oxygen_ratio)
  * metallicity vs clouds (x=metallicity_wrt_solar, y=cloud_pressure_bar)

For each molecule and each (x, y) grid point this script writes a temporary
config that (a) sets a single-species template for that molecule, (b) sets the
two swept atmosphere variables, and (c) writes minimal output to a per-point
directory, then runs the engine as a **separate subprocess** (memory-isolated,
cluster/job-array friendly, exactly as the legacy code ran recursively from the
terminal).  It then reads the run's Kp-Vsys significance map and records the
significance at the injected planet position.

Output (under ``<output_root>/detectability/<molecule>/``):
  * ``detectability_x<...>_y<...>.txt`` -- one line ``x  y  significance`` per
    grid point (the format consumed by
    ``exoplore.plotting.plot_detectability_map``).
  * ``runs/`` -- the minimal per-point run directories.

Usage::

    python scripts/run_detectability_maps.py configs/my_detectability.json [--plot]

``--plot`` also renders the maps at the end with
``exoplore.plotting.plot_detectability_map``.
"""
import argparse
import copy
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from exoplore.config.models import SimulationConfig  # noqa: E402

# The two supported axis pairs (x is always metallicity, given as log10 Z/Zsun).
_SUPPORTED_PAIRS = {
    ("metallicity_wrt_solar", "carbon_to_oxygen_ratio"),
    ("metallicity_wrt_solar", "cloud_pressure_bar"),
}
# Default grids used when the config leaves x_values / y_values empty.
# metallicity_wrt_solar is log10(Z/Zsun).
_DEFAULT_GRIDS = {
    # metallicity vs C/O: log10 Z in [-1.5, +1.5] (~0.03-31x solar), C/O in [0.3, 1.1]
    ("metallicity_wrt_solar", "carbon_to_oxygen_ratio"): (
        [-1.5, -1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
        [0.30, 0.35, 0.40, 0.41, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10],
    ),
    # metallicity vs clouds: log10 Z in [0, 3] (1-1000x solar), cloud-top pressure
    # in [1e-4, 1] bar (both log-spaced), i.e. the GJ 1214 b sub-Neptune range.
    ("metallicity_wrt_solar", "cloud_pressure_bar"): (
        [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0],
        [1e-4, 2.78e-4, 7.74e-4, 2.15e-3, 6e-3, 1.67e-2, 4.64e-2, 0.129, 0.359, 1.0],
    ),
}
_AXIS_REGION_FIELDS = {
    "metallicity_wrt_solar", "carbon_to_oxygen_ratio", "cloud_pressure_bar",
}


def _tag(value: float) -> str:
    """Filename-safe tag for a float (avoid '.' and '-')."""
    return f"{value:+.3g}".replace(".", "p").replace("-", "m").replace("+", "")


def _planet_kp(base: dict, repo_root: Path) -> float:
    """Kp (km/s) of the injected planet, from the planet-parameter file."""
    pf = base.get("planet", {}).get("parameter_file", "")
    pdir = base.get("paths", {}).get("planet_parameter_dir", "planet_params")
    for cand in (Path(pf), repo_root / pf, repo_root / pdir / Path(pf).name):
        if cand.is_file():
            p = json.loads(cand.read_text())
            if p.get("kp_kms") is not None:
                return float(p["kp_kms"])
    raise SystemExit(
        "Could not read kp_kms from the planet parameter file; needed to locate "
        "the injected planet cell.")


def _significance_from_map(map_npz: str, kp_planet: float, det) -> float:
    """Significance at the injected position from a Kp-Vsys S/N map."""
    d = np.load(map_npz, allow_pickle=True)
    sn = np.asarray(d["ccf_tot_sn"], float)
    v = np.asarray(d["v_rest"], float)
    kp = np.asarray(d["kp_range"], float)
    if det.significance == "injected_point":
        iv = int(np.argmin(np.abs(v - 0.0)))
        ik = int(np.argmin(np.abs(kp - kp_planet)))
        return float(sn[iv, ik])
    # cell_box: max in a small box around (kp_planet, v_rest=0)
    m = ((np.abs(v)[:, None] <= det.box_half_vrest_kms)
         & (np.abs(kp - kp_planet)[None, :] <= det.box_half_kp_kms))
    if not m.any():
        iv = int(np.argmin(np.abs(v))); ik = int(np.argmin(np.abs(kp - kp_planet)))
        return float(sn[iv, ik])
    return float(np.nanmax(np.where(m, sn, np.nan)))


def _run_grid_point(molecule, xv, yv, base, det, kp_planet, runner,
                    runs_dir, mol_dir):
    """Run one grid point in its own subprocess and record its significance.

    Independent of every other point (own config, own output directory, own
    result file), so points can run concurrently.  Returns a one-line status.
    """
    tag = f"x{_tag(xv)}_y{_tag(yv)}"
    point = copy.deepcopy(base)
    atm = point.setdefault("atmosphere", {})
    # Injected atmosphere: push ONLY the swept metallicity / C-O onto every
    # region that builds the injected model, and change nothing else, exactly
    # as the legacy sweep did (it set Metallicity_wrt_solar and C_to_O on both
    # limbs at each grid point and left species, chemistry mode, mass fractions
    # and the per-region temperature untouched).  Each limb keeps its own
    # temperature, so equilibrium chemistry gives a different abundance per limb
    # and the combined transit spectrum stays asymmetric.  When the base config
    # simulates limb asymmetries the injected model is built from the four
    # terminator regions (not planet_model), so all of them must follow the
    # grid; otherwise the injected atmosphere stays fixed and the map is flat.
    inject_regions = ["planet_model"]
    for limb in ("morning_day", "morning_night", "evening_day", "evening_night"):
        if limb in atm:
            inject_regions.append(limb)
    for region in inject_regions:
        r = atm.setdefault(region, {})
        r[det.x_variable] = xv
        r[det.y_variable] = yv
    # Cross-correlation template: a single-species template for this molecule at
    # the same swept composition (legacy species_cc = ['H2','He',<molecule>]
    # with use_easyCHEM_cc = True).
    tpl = atm.setdefault("ccf_template", {})
    tpl["species"] = ["H2", "He", molecule]
    tpl["use_easychem"] = True
    tpl["mass_fractions"] = []
    tpl[det.x_variable] = xv
    tpl[det.y_variable] = yv
    run_out = os.path.join(runs_dir, tag) + os.sep
    point.setdefault("paths", {})["output_root"] = run_out
    # keep only what we need to read the peak (small on disk): switch off every
    # per-order matrix product; the Kp-Vsys S/N map is written regardless.
    out = point.setdefault("output", {})
    for _flag in ("save_mat_res", "save_mat_back", "save_ccf_store",
                  "save_propag_noise", "save_U_sysrem", "save_mat_cc",
                  "save_mat_noise", "save_std_noise"):
        out[_flag] = False
    point.setdefault("cross_correlation", {})["ccf_snr"] = True
    point.setdefault("retrieval", {})["enabled"] = False
    point.setdefault("detectability", {})["enabled"] = False
    tmp_cfg = os.path.join(runs_dir, f"_cfg_{tag}.json")
    Path(tmp_cfg).write_text(json.dumps(point, indent=1))

    r = subprocess.run([sys.executable, runner, tmp_cfg, "--run"],
                       capture_output=True, text=True)
    maps = glob.glob(os.path.join(run_out, "**", "matrices",
                                  "ccf_tot_sn_map_*.npz"), recursive=True)
    if r.returncode != 0 or not maps:
        return f"  {tag}: FAILED (rc={r.returncode}); {r.stderr.strip()[-200:]}"
    sig = _significance_from_map(maps[0], kp_planet, det)
    with open(os.path.join(mol_dir, f"detectability_{tag}.txt"), "w") as f:
        f.write(f"{xv:.6g} {yv:.6g} {sig:.4f}\n")
    return (f"  {tag}: {det.x_variable}={xv:g} {det.y_variable}={yv:g} "
            f"-> S/N {sig:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--plot", action="store_true",
                    help="render the maps at the end")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of grid-point simulations to run concurrently "
                         "(each is an independent subprocess; raise on a machine "
                         "with spare cores and RAM)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base = json.loads(Path(args.config).read_text())
    cfg = SimulationConfig.from_dict(base)
    det = cfg.detectability
    if not det.enabled:
        raise SystemExit("detectability.enabled is false in this config.")
    pair = (det.x_variable, det.y_variable)
    if pair not in _SUPPORTED_PAIRS:
        raise SystemExit(
            f"unsupported axis pair {pair}; supported: {sorted(_SUPPORTED_PAIRS)}")
    # Fall back to the default grid for this pair when the config omits values.
    x_values = list(det.x_values) or _DEFAULT_GRIDS[pair][0]
    y_values = list(det.y_values) or _DEFAULT_GRIDS[pair][1]
    print(f"grid: {len(x_values)} x {len(y_values)} = "
          f"{len(x_values) * len(y_values)} points "
          f"({'config' if det.x_values else 'default'} x, "
          f"{'config' if det.y_values else 'default'} y)")

    kp_planet = _planet_kp(base, repo_root)
    out_root = base.get("paths", {}).get("output_root", "outputs/")
    runner = str(repo_root / "scripts" / "run_exoplore.py")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    for molecule in det.molecules:
        mol_dir = os.path.join(out_root, "detectability", molecule)
        runs_dir = os.path.join(mol_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        # resume: only the grid points without a result file are (re)run.
        pending = [(xv, yv) for xv in x_values for yv in y_values
                   if not os.path.isfile(os.path.join(
                       mol_dir, f"detectability_x{_tag(xv)}_y{_tag(yv)}.txt"))]
        ndone = len(x_values) * len(y_values) - len(pending)
        print(f"\n=== detectability map: {molecule} "
              f"({det.x_variable} x {det.y_variable}), "
              f"{len(x_values)}x{len(y_values)} grid; {len(pending)} to run, "
              f"{ndone} already done, {args.workers} worker(s) ===", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(_run_grid_point, molecule, xv, yv, base, det,
                                 kp_planet, runner, runs_dir, mol_dir)
                       for xv, yv in pending]
            for fut in as_completed(futures):
                print(fut.result(), flush=True)

    if args.plot:
        from exoplore.plotting import plot_detectability_map
        for molecule in det.molecules:
            mol_dir = os.path.join(out_root, "detectability", molecule)
            plot_detectability_map(
                mol_dir, x_variable=det.x_variable, y_variable=det.y_variable,
                molecule=molecule,
                fname=os.path.join(mol_dir, f"detectability_{molecule}.png"))
    print("\ndetectability sweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
