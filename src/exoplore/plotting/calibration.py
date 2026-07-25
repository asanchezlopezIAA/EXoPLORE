"""
exoplore.plotting.calibration
==============================

Visualisation of the CRIRES+ calibration stages, for documentation and
diagnostic inspection of a reduced night:

  ``plot_telluric_correction``
      One echelle segment before and after the telluric correction: the
      extracted spectrum, the fitted molecfit transmittance, and the
      corrected spectrum (observed / transmittance).

  ``plot_resolution_per_order``
      The effective spectral resolving power measured per order, showing the
      super-resolution regime (PSF underfilling the slit) versus the nominal,
      slit-limited value.

These illustrate the reduction and telluric-correction methodology of
Nortmann et al. (2025, 2026); the underlying recipes are the ESO ``cr2res``
pipeline and ``molecfit`` (see the CRIRES+ reduction reference page for the
recipe details and links).  The functions take plain NumPy arrays so they can
be called on any reduced night, and optionally save a figure.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _continuum_normalise(flux: np.ndarray, deg: int = 3) -> np.ndarray:
    """Divide a spectrum by a low-order polynomial fit to its upper envelope,
    so different segments are comparable on a continuum of ~1."""
    x = np.arange(flux.size, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    if good.sum() < deg + 2:
        return flux
    # iterative upper-envelope fit (reject points well below the fit)
    keep = good.copy()
    for _ in range(3):
        coef = np.polyfit(x[keep], flux[keep], deg)
        model = np.polyval(coef, x)
        resid = flux - model
        keep = good & (resid > -1.5 * np.nanstd(resid[good]))
        if keep.sum() < deg + 2:
            break
    model = np.polyval(np.polyfit(x[keep], flux[keep], deg), x)
    model[model == 0] = np.nan
    return flux / model


def plot_telluric_correction(
    wavelength_nm: np.ndarray,
    flux: np.ndarray,
    transmittance: np.ndarray,
    segment_label: str = "",
    normalise: bool = True,
    edge_trim: int = 0,
    deep_core_mask: Optional[float] = 0.1,
    fname: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 5.5),
    save_plot: bool = True,
    show_plot: bool = False,
):
    """Illustrate the molecfit telluric correction for one echelle segment.

    Parameters
    ----------
    wavelength_nm : array
        Wavelength solution of the segment, in nm.
    flux : array
        Extracted (observed) flux of the segment, one exposure.
    transmittance : array
        Fitted telluric transmittance (1 = no absorption) on the same grid,
        e.g. from ``molecfit_nod<AB>.npz``.
    segment_label : str
        Label for the segment (for the title), e.g. ``"CHIP2.INT1_04"``.
    normalise : bool
        Continuum-normalise the observed flux before plotting so the panels
        read on a common ~1 continuum.
    edge_trim : int
        Drop this many pixels from each segment end (the extraction edges carry
        low-throughput artefacts).
    deep_core_mask : float or None
        Blank the corrected spectrum where the transmittance falls below this
        value: the cores of saturated telluric lines divide two near-zero
        numbers and are excluded from the analysis (the deep-line mask of
        Nortmann et al. 2025).  Set ``None`` to disable.
    fname : str, optional
        Output path.  If ``None`` and ``save_plot``, a name is derived from
        the segment label.

    Returns
    -------
    fig, (ax_top, ax_bottom)
    """
    import matplotlib.pyplot as plt

    w = np.asarray(wavelength_nm, float)
    f = np.asarray(flux, float)
    t = np.asarray(transmittance, float)
    if edge_trim > 0:
        sl = slice(edge_trim, -edge_trim)
        w, f, t = w[sl], f[sl], t[sl]
    if normalise:
        f = _continuum_normalise(f)
    t_safe = np.where(np.isfinite(t) & (t > 1e-3), t, np.nan)
    corrected = f / t_safe
    if deep_core_mask is not None:
        corrected = np.where(t >= deep_core_mask, corrected, np.nan)

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
    )

    ax0.plot(w, f, color="0.15", lw=0.7, label="observed (telluric + stellar)")
    ax0.plot(w, t, color="tab:blue", lw=0.9, alpha=0.9,
             label="molecfit transmittance")
    ax0.set_ylabel("normalised flux")
    ax0.legend(loc="lower left", fontsize=9, frameon=False)
    ttl = "Telluric correction (molecfit)"
    if segment_label:
        ttl += f" ({segment_label})"
    ax0.set_title(ttl, fontsize=11)

    ax1.plot(w, corrected, color="tab:red", lw=0.7,
             label="corrected (observed / transmittance)")
    ax1.axhline(1.0, color="0.6", lw=0.6, ls="--")
    ax1.set_ylabel("corrected")
    ax1.set_xlabel(r"wavelength (nm)")
    ax1.legend(loc="lower left", fontsize=9, frameon=False)

    if save_plot:
        out = fname or f"telluric_correction_{segment_label or 'segment'}.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
    if show_plot:
        plt.show()
    return fig, (ax0, ax1)


def plot_resolution_per_order(
    order_wavelength_um: Sequence[float],
    resolving_power: Sequence[float],
    nominal_R: float = 100000.0,
    slit_arcsec: float = 0.2,
    fname: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
    save_plot: bool = True,
    show_plot: bool = False,
):
    """Plot the measured effective resolving power per order.

    The super-resolution regime (R above the nominal, slit-limited value)
    occurs when the seeing-limited PSF underfills the slit; overfilling
    (poor seeing) is slit-limited and clamped to ``nominal_R``.

    Parameters
    ----------
    order_wavelength_um : sequence
        Central wavelength of each order/segment, in microns.
    resolving_power : sequence
        Measured resolving power R for each order/segment.
    nominal_R : float
        Nominal, slit-limited resolving power (~100000 for the 0.2" slit).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    w = np.asarray(order_wavelength_um, float)
    R = np.asarray(resolving_power, float)
    order = np.argsort(w)
    w, R = w[order], R[order]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(nominal_R, color="0.5", lw=0.8, ls="--",
               label=f"nominal (slit-limited), R = {nominal_R:.0f}")
    ax.fill_between(w, nominal_R, np.maximum(R, nominal_R),
                    where=R > nominal_R, color="tab:green", alpha=0.15,
                    label="super-resolution (PSF underfills slit)")
    ax.plot(w, R, "o-", color="tab:blue", ms=4, lw=1.0,
            label="measured per order")

    ax.set_xlabel(r"order central wavelength ($\mu$m)")
    ax.set_ylabel("effective resolving power R")
    med = float(np.nanmedian(R))
    ax.set_title(
        f"CRIRES+ effective resolution (median R = {med:.0f}, "
        f"{slit_arcsec}\" slit)", fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=False)

    if save_plot:
        out = fname or "resolution_per_order.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
    if show_plot:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Convenience loaders (from a reduced night on disk)
# ---------------------------------------------------------------------------

def telluric_correction_from_night(
    molecfit_npz: str,
    reduced_extracted_fits: str,
    segment_index: int = 0,
    **kwargs,
):
    """Convenience wrapper: read one segment's transmittance from a
    ``molecfit_nod<AB>.npz`` and the matching extracted spectrum from a
    ``cr2res_obs_nodding_extracted<AB>.fits`` product, then call
    :func:`plot_telluric_correction`.

    ``segment_index`` indexes the ``names`` array in the molecfit file.
    """
    from astropy.io import fits

    d = np.load(molecfit_npz, allow_pickle=True)
    names = list(d["names"])
    name = names[segment_index]
    wave = np.asarray(d["wave_refined"][segment_index], float)
    trans = np.asarray(d["transmittance"][segment_index], float)

    # match the extracted-product columns: "<order>_01_SPEC" / "_WL"
    order = name.split("_")[-1]  # e.g. CHIP2.INT1_04 -> "04"
    hdul = fits.open(reduced_extracted_fits)
    flux = None
    for hdu in hdul:
        if hdu.data is None or not hasattr(hdu.data, "names"):
            continue
        for col in hdu.data.names:
            if col.endswith("_SPEC") and col.split("_")[0].endswith(order):
                flux = np.asarray(hdu.data[col], float)
                break
        if flux is not None:
            break
    if flux is None:  # fall back to first SPEC column
        for hdu in hdul:
            if hdu.data is not None and hasattr(hdu.data, "names"):
                sc = [c for c in hdu.data.names if c.endswith("_SPEC")]
                if sc:
                    flux = np.asarray(hdu.data[sc[0]], float)
                    break
    return plot_telluric_correction(wave, flux, trans,
                                    segment_label=name, **kwargs)
