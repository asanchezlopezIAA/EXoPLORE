#!/usr/bin/env python3
"""Plot saved CRIRES+ helium preparation products without repeating any fit.

The exposure view shows only the fitted telluric windows. Its continuum
scaling is for display; scientific correction uses pure calctrans transmission.
Every output is created exclusively. Source corrections remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def read_reports(directory: Path) -> list[dict]:
    """Read chronological exposure reports from current or validated caches."""
    content = json.loads((directory / "night_report.json").read_text())
    rows = content if isinstance(content, list) else content["exposures"]
    if not rows:
        raise ValueError("No exposure reports")
    return sorted(rows, key=lambda row: row["exposure"])


def save_figure(figure: plt.Figure, output: Path) -> None:
    """Create a PNG exclusively; never replace an earlier diagnostic."""
    with output.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_exposure(directory: Path, number: int, output: Path) -> None:
    """Show the fit, pure-transmission correction, and weighted residuals."""
    selected = [row for row in read_reports(directory) if row["exposure"] == number]
    if len(selected) != 1:
        raise ValueError("Exposure number must identify one saved report")
    report = selected[0]
    work = directory / f"exposure_{number:02d}_{report['nod']}"
    model = fits.getdata(work / "BEST_FIT_MODEL.fits", 1)
    with np.load(work / "correction.npz", allow_pickle=False) as cache:
        transmission = cache["transmittance"].copy()
        native_wave = cache["native_wave_nm"].copy()
    wavelength = model["lambda"] * 1000
    if len(wavelength) != len(native_wave) or not np.allclose(wavelength, native_wave, rtol=0, atol=1e-7):
        raise ValueError("Model and correction must have the same native pixels")
    windows = report["fit_windows_nm"]
    figure, axes = plt.subplots(3, len(windows), figsize=(4.4 * len(windows), 9.5), squeeze=False, sharex="col", gridspec_kw={"hspace": 0})
    for column, (lower, upper) in enumerate(windows):
        valid = ((wavelength >= lower) & (wavelength <= upper)
                 & (model["mrange"] > 0) & (model["weight"] > 0)
                 & (model["mscal"] > 0) & (transmission > 0))
        if not np.any(valid):
            raise ValueError("Saved fit window has no usable pixels")
        x = wavelength[valid] / 1000
        observed, fitted = model["flux"][valid], model["mflux"][valid]
        continuum = model["mscal"][valid]
        axes[0, column].plot(x, observed, color="black", label="Observed")
        axes[0, column].plot(x, fitted, color="tab:orange", label="Telluric fit")
        axes[0, column].set_title(f"{lower / 1000:.5f}–{upper / 1000:.5f} µm", fontsize=16)
        axes[1, column].plot(x, observed / continuum, color="0.6", label="Before correction")
        axes[1, column].plot(x, observed / continuum / transmission[valid], color="tab:blue", label="After correction")
        axes[1, column].axhline(1, color="0.5", ls=":")
        residual = (observed - fitted) * model["weight"][valid]
        axes[2, column].plot(x, residual, color="black")
        axes[2, column].axhline(0, color="0.5", ls=":")
        axes[2, column].set_xlabel("Wavelength (µm)")
        for axis in axes[:, column]:
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.xaxis.set_major_locator(MaxNLocator(3))
            axis.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
            axis.tick_params(labelsize=13, length=5.6, width=1.1)
            axis.xaxis.label.set_size(17)
            axis.yaxis.label.set_size(17)
        for axis in axes[:2, column]:
            axis.tick_params(axis="x", labelbottom=False)
    axes[0, 0].set_ylabel("Extracted flux")
    axes[1, 0].set_ylabel("Normalised flux")
    axes[2, 0].set_ylabel("(Data − model) / error")
    axes[0, 0].legend(fontsize=12)
    axes[1, 0].legend(fontsize=12)
    figure.suptitle(f"Exposure {number}, nod {report['nod']}: wavelength adjustment {report['he_shift_kms']:+.3f} km/s", fontsize=17)
    figure.subplots_adjust(hspace=0, wspace=0.30, top=0.88, bottom=0.10, left=0.08, right=0.99)
    save_figure(figure, output)


def plot_night(directory: Path, output: Path) -> None:
    """Compare wavelength refinement, fitted Gaussian width, and fit quality."""
    reports = read_reports(directory)
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fields = [("he_shift_kms", "Wavelength adjustment (km/s)"),
              ("gaussfwhm", "Gaussian FWHM (pixels)"),
              ("reduced_chi2", "Reduced chi-squared")]
    for nod, colour, marker in (("A", "tab:blue", "o"), ("B", "tab:orange", "s")):
        rows = [row for row in reports if row["nod"] == nod]
        for axis, (field, label) in zip(axes, fields):
            values = [row[field] if field == "he_shift_kms" else row["parameters"][field] for row in rows]
            axis.plot([row["exposure"] for row in rows], values, marker=marker, color=colour, label=f"Nod {nod}", ms=4)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
    axes[0].axhline(0, color="0.5", ls=":")
    axes[0].legend(fontsize=14)
    axes[-1].set_xlabel("Chronological exposure number")
    for axis in axes:
        axis.tick_params(labelsize=14, length=5.6, width=1.1)
        axis.xaxis.label.set_size(17)
        axis.yaxis.label.set_size(17)
    figure.suptitle("Telluric correction across the observing sequence", fontsize=17)
    figure.tight_layout()
    save_figure(figure, output)


def main() -> None:
    """Plot an exposure, night summary, or transmission result from disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    for name in ("exposure", "night"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
        command.add_argument("output", type=Path)
        if name == "exposure":
            command.add_argument("--exposure", type=int, required=True)
    command = commands.add_parser("transmission")
    command.add_argument("config", type=Path)
    command.add_argument("products", type=Path)
    command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename; existing figures are preserved")
    if args.kind == "exposure":
        plot_exposure(args.directory, args.exposure, args.output)
    elif args.kind == "night":
        plot_night(args.directory, args.output)
    else:
        from exoplore.config import SimulationConfig
        from exoplore.pipelines.czesla2024 import plot_transmission
        config = SimulationConfig.from_json(args.config).pipeline.czesla2024
        if config is None:
            parser.error("Use a czesla2024 configuration")
        with np.load(args.products, allow_pickle=False) as data:
            result = {key: data[key].copy() for key in data.files}
        plot_transmission(result, result["phase"], result["planet_rv_kms"], result["berv_kms"], config, args.output)
    print(f"Saved {args.output}; no fits or noise simulations were repeated.")


if __name__ == "__main__":
    main()
