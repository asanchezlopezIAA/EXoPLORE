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

# Axis labels (as in the reference implementation) and whether the plotted
# values are log10 of the stored ones.  Metallicity is already log10 Z/Zsun.
_AXIS_LABEL = {
    "metallicity_wrt_solar": ("[Fe/H]", False),
    "carbon_to_oxygen_ratio": ("C/O", False),
    "cloud_pressure_bar": (r"log$_{10}$ P$_{\rm cloud}$ (bar)", True),
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


def _auto_last3_levels(snr_max: float) -> np.ndarray:
    """The detection contour levels: the last three multiples of 5 at or below
    the map maximum (so contours appear even when the whole map is high S/N).
    Matches the ``auto_last3`` default of the reference implementation."""
    highest = int(np.floor(snr_max / 5.0) * 5)
    if highest < 5:
        return np.array([], dtype=float)
    return np.arange(5, highest + 1, 5)[-3:].astype(float)


def plot_detectability_map(
    data_dir: str,
    x_variable: str = "metallicity_wrt_solar",
    y_variable: str = "carbon_to_oxygen_ratio",
    molecule: str = "",
    contour_levels="auto_last3",
    truth: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    grid_size: int = 280,
    contour_linewidth: float = 3.0,
    contour_fontsize: float = 27.0,
    fname: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 7.0),
    save_plot: bool = True,
    show_plot: bool = False,
):
    """Filled-contour detectability map for one molecule.

    A faithful port of the reference detectability plot: linear + nearest
    ``griddata`` interpolation with light Gaussian smoothing, a 0-to-maximum
    colour scale, dashed black detection contours, an optional gold truth star,
    and a horizontal colorbar.

    Parameters
    ----------
    data_dir : str
        Directory holding the ``detectability_*.txt`` files for this molecule.
    x_variable, y_variable : str
        The swept atmosphere variables (for axis labels and log scaling; the
        cloud-pressure axis is plotted in log10).
    molecule : str
        Species name, used in the title.
    contour_levels : "auto_last3", sequence of float, or None
        Detection contours to overlay.  ``"auto_last3"`` (default) draws the
        last three multiples of 5 at or below the map maximum, so contours
        appear even when the whole map is well above 5 sigma; a sequence draws
        those explicit S/N levels (kept if 0 < level < max); ``None`` draws none.
    truth : (x, y), optional
        Composition to mark with a gold star.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    x, y, sn = _read_points(data_dir)
    if x.size == 0:
        raise RuntimeError(f"no detectability_*.txt found in {data_dir}")

    xlabel, xlog = _AXIS_LABEL.get(x_variable, (x_variable, False))
    ylabel, ylog = _AXIS_LABEL.get(y_variable, (y_variable, False))
    xp = np.log10(x) if xlog else x
    yp = np.log10(y) if ylog else y

    # Interpolation grid: linear, nearest-fill for the convex-hull edges, then a
    # light Gaussian smooth (as in the reference implementation).
    gx = np.linspace(xp.min(), xp.max(), grid_size)
    gy = np.linspace(yp.min(), yp.max(), grid_size)
    GX, GY = np.meshgrid(gx, gy)
    GZ_lin = griddata((xp, yp), sn, (GX, GY), method="linear")
    GZ_near = griddata((xp, yp), sn, (GX, GY), method="nearest")
    GZ = np.where(np.isnan(GZ_lin), GZ_near, GZ_lin)
    GZ = gaussian_filter(GZ, sigma=0.4)
    GZ = np.nan_to_num(GZ, nan=float(np.nanmin(sn)))

    # Colour scale 0 to max, 30 filled levels.
    vmin, vmax = 0.0, max(1.0, float(np.nanmax(sn)))
    levels = np.linspace(vmin, vmax, 30)

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(GX, GY, GZ, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
                     extend="both", alpha=0.95, antialiased=False)

    # Detection contours.
    if contour_levels is None:
        clev = np.array([], dtype=float)
    elif isinstance(contour_levels, str) and contour_levels == "auto_last3":
        clev = _auto_last3_levels(float(np.nanmax(GZ)))
    else:
        clev = np.asarray(contour_levels, dtype=float)
        clev = np.unique(clev[(clev > 0.0) & (clev < float(np.nanmax(GZ)))])
    if clev.size:
        cs = ax.contour(GX, GY, GZ, levels=clev, colors="k", linestyles="--",
                        linewidths=contour_linewidth)
        fmt = {lev: f"S/N={lev:.0f}" for lev in clev}
        texts = ax.clabel(cs, levels=clev, fmt=fmt, inline=False,
                          inline_spacing=8, fontsize=contour_fontsize,
                          colors="k", rightside_up=True)
        for txt in texts:
            txt.set_bbox(dict(facecolor="none", edgecolor="none", pad=0.0))

    if truth is not None:
        tx = np.log10(truth[0]) if xlog else truth[0]
        ty = np.log10(truth[1]) if ylog else truth[1]
        ax.scatter(tx, ty, marker="*", s=1500, facecolor="none",
                   edgecolor="k", linewidth=1.0, zorder=40)
        ax.scatter(tx, ty, marker="*", s=1500, facecolor="gold",
                   edgecolor="k", linewidth=1.4, zorder=41)

    # Horizontal colorbar with adaptive integer ticks (step 1 / 2 / 5).
    vmax_int = int(np.floor(vmax))
    step = 1 if vmax_int <= 10 else (2 if vmax_int <= 20 else 5)
    ticks = np.arange(0, vmax_int + 1, step)
    # Add the exact maximum as a tick only when it is at least a full step past
    # the last multiple, so the two labels do not overlap (e.g. "45 48").
    if ticks.size == 0 or (vmax_int - ticks[-1]) >= step:
        ticks = np.append(ticks, vmax_int)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.14,
                      fraction=0.06, ticks=ticks)
    cb.set_label("S/N", fontsize=18)
    cb.ax.tick_params(labelsize=17)

    ax.set_title(molecule, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(xp.min(), xp.max())
    ax.set_ylim(yp.min(), yp.max())
    ax.tick_params(axis="both", which="both", labelsize=16, width=1.2, length=6)

    if save_plot:
        out = fname or os.path.join(data_dir, f"detectability_{molecule}.png")
        fig.savefig(out, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    return fig, ax
