"""
exoplore.plotting.detectability
===============================

Detectability maps: the recovered cross-correlation significance across a 2-D
grid of two atmospheric variables (produced by
``scripts/run_detectability_maps.py``), with optional detection-threshold
contours.

``plot_detectability_map`` reads the per-grid-point ``detectability_*.txt``
files written by the sweep (one line ``x  y  significance`` each) and renders a
filled-contour map for one molecule.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Sequence, Tuple

import numpy as np

# Human-readable axis labels and whether the axis is log-scaled.
_AXIS_LABEL = {
    "metallicity_wrt_solar": (r"[Fe/H] (log$_{10}$ Z/Z$_\odot$)", False),
    "carbon_to_oxygen_ratio": ("C/O", False),
    "cloud_pressure_bar": (r"cloud-top pressure (bar)", True),
}


def _read_points(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read all ``detectability_*.txt`` in a directory into (x, y, sn)."""
    files = sorted(glob.glob(os.path.join(data_dir, "detectability_*.txt")))
    xs, ys, sn = [], [], []
    for fpath in files:
        try:
            with open(fpath) as fh:
                a, b, c = map(float, fh.readline().split())
            xs.append(a); ys.append(b); sn.append(c)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {os.path.basename(fpath)}: {exc}")
    return np.asarray(xs), np.asarray(ys), np.asarray(sn)


def plot_detectability_map(
    data_dir: str,
    x_variable: str = "metallicity_wrt_solar",
    y_variable: str = "carbon_to_oxygen_ratio",
    molecule: str = "",
    contour_levels: Sequence[float] = (3.0, 5.0),
    truth: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    grid_size: int = 200,
    fname: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 5.5),
    save_plot: bool = True,
    show_plot: bool = False,
):
    """Filled-contour detectability map for one molecule.

    Parameters
    ----------
    data_dir : str
        Directory holding the ``detectability_*.txt`` files for this molecule.
    x_variable, y_variable : str
        The swept atmosphere variables (for axis labels / log scaling).
    molecule : str
        Species name, used in the title.
    contour_levels : sequence of float
        Significance levels to overlay as line contours (e.g. S/N = 3, 5).
    truth : (x, y), optional
        A reference point to mark with a star (e.g. the true composition).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    x, y, sn = _read_points(data_dir)
    if x.size == 0:
        raise RuntimeError(f"no detectability_*.txt found in {data_dir}")

    xlabel, xlog = _AXIS_LABEL.get(x_variable, (x_variable, False))
    ylabel, ylog = _AXIS_LABEL.get(y_variable, (y_variable, False))
    xp = np.log10(x) if xlog else x
    yp = np.log10(y) if ylog else y

    # regular grid for the filled contours
    gx = np.linspace(xp.min(), xp.max(), grid_size)
    gy = np.linspace(yp.min(), yp.max(), grid_size)
    GX, GY = np.meshgrid(gx, gy)
    if np.unique(xp).size > 1 and np.unique(yp).size > 1:
        GZ = griddata((xp, yp), sn, (GX, GY), method="cubic")
        GZ = np.where(np.isnan(GZ),
                      griddata((xp, yp), sn, (GX, GY), method="nearest"), GZ)
    else:  # degenerate (1-D) grid: nearest only
        GZ = griddata((xp, yp), sn, (GX, GY), method="nearest")

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.contourf(GX, GY, GZ, levels=24, cmap=cmap)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("cross-correlation S/N")

    levels = sorted(L for L in contour_levels
                    if np.nanmin(GZ) < L < np.nanmax(GZ))
    if levels:
        cs = ax.contour(GX, GY, GZ, levels=levels, colors="white",
                        linewidths=2.0)
        ax.clabel(cs, fmt=lambda v: f"S/N={v:.0f}", fontsize=10)

    ax.scatter(xp, yp, s=8, c="k", alpha=0.35)  # sampled grid points
    if truth is not None:
        tx = np.log10(truth[0]) if xlog else truth[0]
        ty = np.log10(truth[1]) if ylog else truth[1]
        ax.plot(tx, ty, marker="*", ms=18, color="gold",
                markeredgecolor="k", label="truth")
        ax.legend(loc="best", fontsize=10, frameon=False)

    ax.set_xlabel((r"log$_{10}$ " + xlabel) if xlog else xlabel)
    ax.set_ylabel((r"log$_{10}$ " + ylabel) if ylog else ylabel)
    imax = int(np.nanargmax(sn))
    ax.set_title(
        f"{molecule} detectability  (peak S/N {sn[imax]:.1f} at "
        f"{x_variable.split('_')[0]}={x[imax]:g}, {y[imax]:g})", fontsize=11)

    if save_plot:
        out = fname or os.path.join(data_dir, f"detectability_{molecule}.png")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    if show_plot:
        plt.show()
    return fig, ax
