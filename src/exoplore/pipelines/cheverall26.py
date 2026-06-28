"""
exoplore.pipelines.cheverall26
==============================

Cheverall et al. (2026) data-preparation pipeline for IGRINS
high-resolution transmission spectroscopy.

Designed for real-data analysis (Mode C) of IGRINS observations,
following the methodology of Cheverall et al. (2026, MNRAS) for
the detection of atmospheric species in temperate exoplanets with
small radial-velocity changes during transit.

Pipeline steps (applied in this order per spectral order):
    1.  Rescale, divide each spectrum by its median flux.
    2.  Outlier removal, sigma-clip per wavelength column across time.
    3.  Pseudo-continuum normalisation, 2nd-order polynomial fitted to
        the maxima in 80 wavelength bins (``pipeline_pseudocontinuum_norm``
        from ``exoplore.pipelines.bl19``), tracing the blaze/throughput
        envelope without distortion from absorption lines.
    4.  SYSREM detrending with Max_Diff/ΔCCF optimisation, inject H2S
        model at offset velocity (+19 km/s, ``kp_vrest_injection=[70, 19]``),
        run SYSREM for N=1..max, select N where marginal ΔCCF gain is
        maximised (Holmberg & Madhusudhan 2022; Cheverall et al. 2023; 2026).

Notes
-----
Edge trimming (100 px each side, Brogi et al. 2023) and secondary
wavelength calibration (2nd-order stretch-and-shift to align each
spectrum to the last exposure, Cheverall et al. 2026 Section 2.3)
are applied upstream in ``scripts/prepare_igrins_night.py`` before
the data reaches this pipeline.

References
----------
Cheverall, C. J. et al. (2026), MNRAS, L 98-59 d atmospheric
    characterisation with IGRINS on Gemini South.
Brogi, M. et al. (2023), AJ, 165, 91, edge trimming convention.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Step 1, Rescaling
# ---------------------------------------------------------------------------

def chev26_rescale(
    data: np.ndarray,
    noise: np.ndarray,
    good_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale each spectrum by dividing by its median flux.

    Removes the per-exposure flux level so that all spectra share a common
    continuum level of ~1, making subsequent polynomial fitting and
    cross-correlation well-conditioned.

    Parameters
    ----------
    data : ndarray, shape (n_spectra, n_pixels)
        Input flux time series for one order.
    noise : ndarray, shape (n_spectra, n_pixels)
        Associated uncertainties.
    good_pixels : ndarray of int
        Indices of unmasked pixels used to compute the median.

    Returns
    -------
    data_out : ndarray, shape (n_spectra, n_pixels)
        Rescaled flux; each row divided by its median over good pixels.
    noise_out : ndarray, shape (n_spectra, n_pixels)
        Uncertainties rescaled by the same factor.
    """
    data_out  = data.copy().astype(float)
    noise_out = noise.copy().astype(float)

    for i in range(data.shape[0]):
        row_good = data[i, good_pixels]
        med = np.nanmedian(row_good)
        if med != 0 and np.isfinite(med):
            data_out[i]  = data[i]  / med
            noise_out[i] = noise[i] / med

    return data_out, noise_out


# ---------------------------------------------------------------------------
# Step 2, Outlier removal
# ---------------------------------------------------------------------------

def chev26_outlier_removal(
    data: np.ndarray,
    noise: np.ndarray,
    good_pixels: np.ndarray,
    sigma_threshold: float = 5.0,
    n_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma-clip outliers per wavelength column across the time axis.

    For each wavelength channel, computes the median and MAD of the
    time-series values among good pixels, then flags values more than
    ``sigma_threshold`` × MAD from the median as outliers.  Outlier
    pixels are replaced with the column median and their noise is set
    to a large value (1e6) so they do not contribute to subsequent fits
    or the cross-correlation.

    This implements the "cleaned of bad pixels and outliers" step of
    Cheverall et al. (2026) Section 2.3.

    Parameters
    ----------
    data : ndarray, shape (n_spectra, n_pixels)
        Rescaled flux time series (output of ``chev26_rescale``).
    noise : ndarray, shape (n_spectra, n_pixels)
        Associated uncertainties.
    good_pixels : ndarray of int
        Indices of unmasked pixels.
    sigma_threshold : float
        Clipping threshold in units of MAD-estimated sigma.  Default 5.
    n_iterations : int
        Number of sigma-clipping passes.  Default 2.

    Returns
    -------
    data_out : ndarray, shape (n_spectra, n_pixels)
        Flux with outliers replaced by the column median.
    noise_out : ndarray, shape (n_spectra, n_pixels)
        Uncertainties with outlier entries set to 1e6.
    outlier_mask : ndarray of bool, shape (n_spectra, n_pixels)
        True where a pixel was flagged as an outlier.
    """
    data_out     = data.copy().astype(float)
    noise_out    = noise.copy().astype(float)
    outlier_mask = np.zeros(data.shape, dtype=bool)

    for px in good_pixels:
        col = data_out[:, px].copy()

        for _ in range(n_iterations):
            med = np.nanmedian(col)
            mad = np.nanmedian(np.abs(col - med))
            # Convert MAD to sigma-equivalent (factor 1.4826 for Gaussian)
            sigma = 1.4826 * mad
            if sigma == 0:
                break
            flagged = np.abs(col - med) > sigma_threshold * sigma
            col[flagged] = med   # replace for next iteration

        # Final pass: flag against the clipped median
        med_final = np.nanmedian(col)
        mad_final = np.nanmedian(np.abs(col - med_final))
        sigma_final = 1.4826 * mad_final
        if sigma_final > 0:
            flags = np.abs(data_out[:, px] - med_final) > sigma_threshold * sigma_final
            outlier_mask[:, px] = flags
            data_out[flags, px]  = med_final
            noise_out[flags, px] = 1e6

    return data_out, noise_out, outlier_mask


# ---------------------------------------------------------------------------
# Step 3, Pseudo-continuum normalisation
# ---------------------------------------------------------------------------

def chev26_normalise(
    wave: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    good_pixels: np.ndarray,
    window: int = 31,
    percentile: float = 95.0,
    degree: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-continuum normalisation following Cheverall & Madhusudhan (2024).

    This is the exact recipe Cheverall et al. (2026) inherit ("normalized
    using a second-order polynomial fit to the continuum … following the
    methods of Cheverall & Madhusudhan 2024").  Per spectrum (each already
    divided by its median in :func:`chev26_rescale`):

      1.  A **sliding 31-pixel filter selects the 95th-percentile value**
          within its window, tracing the pseudo-continuum while rejecting
          absorption-line cores *and* the top ~5% (hot pixels / emission
          spikes), unlike a max-based envelope, which a single hot pixel
          would bias upward.
      2.  A **quadratic (2nd-order) polynomial** is fit to these
          95th-percentile values across the order.
      3.  The spectrum is **divided by the fit**.

    Note this is NOT the ASL19 maxima method (80-bin maxima, see
    :func:`chev26_normalise_maxima`), it is the 95th-percentile sliding
    envelope specific to Cheverall & Madhusudhan (2024).

    Parameters
    ----------
    wave : ndarray, shape (n_pixels,)
        Wavelength grid for the order in µm.
    data : ndarray, shape (n_spectra, n_pixels)
        Median-rescaled, outlier-cleaned flux (output of Steps 1 to 2).
    noise : ndarray, shape (n_spectra, n_pixels)
        Associated uncertainties.
    good_pixels : ndarray of int
        Indices of unmasked pixels.
    window : int
        Sliding-filter width in pixels (C&M 2024: 31).
    percentile : float
        Percentile selected within each window (C&M 2024: 95).
    degree : int
        Polynomial degree of the continuum fit (C&M 2024 / Cheverall 2026: 2).

    Returns
    -------
    data_norm : ndarray, shape (n_spectra, n_pixels)
        Normalised flux array (full n_pixels width; masked pixels = 1).
    noise_norm : ndarray, shape (n_spectra, n_pixels)
        Propagated uncertainties (masked pixels = 1).
    """
    from scipy.ndimage import percentile_filter

    data_norm  = np.ones_like(data)
    noise_norm = np.ones_like(noise)

    d = data[:, good_pixels]
    e = noise[:, good_pixels]
    x = wave[good_pixels].astype(float)
    # Condition the abscissa for a stable polynomial fit.
    xs = x.std()
    xc = (x - x.mean()) / (xs if xs > 0 else 1.0)

    res = np.ones_like(d)
    err = np.ones_like(e)
    for n in range(d.shape[0]):
        spec = d[n]
        # (1) sliding 95th-percentile pseudo-continuum sample at every pixel
        cont_pts = percentile_filter(
            spec, percentile=percentile, size=window, mode="nearest"
        )
        # (2) quadratic fit to those values; (3) divide
        fit = np.polyval(np.polyfit(xc, cont_pts, degree), xc)
        fit = np.where(np.abs(fit) < 1e-30, 1e-30, fit)
        res[n] = spec / fit
        denom = np.where(np.abs(spec) < 1e-30, 1e-30, spec)
        err[n] = np.abs(res[n]) * np.abs(e[n] / denom)

    data_norm[:, good_pixels]  = res
    noise_norm[:, good_pixels] = err

    return data_norm, noise_norm


def chev26_normalise_maxima(
    wave: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    good_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ASL19-style pseudo-continuum normalisation (80-bin maxima, degree 2).

    Divides each spectrum by a 2nd-order polynomial fitted to the maximum
    flux in 80 equal-width wavelength bins.  This is the ASL19 maxima
    approach (delegates to ``pipeline_pseudocontinuum_norm``), retained as a
    selectable alternative (``continuum_method='maxima'``).  It is NOT the
    Cheverall & Madhusudhan (2024) recipe, see :func:`chev26_normalise`.
    """
    from exoplore.pipelines.bl19 import pipeline_pseudocontinuum_norm

    data_norm  = np.ones_like(data)
    noise_norm = np.ones_like(noise)

    result, error = pipeline_pseudocontinuum_norm(
        wave, data, noise, good_pixels
    )

    data_norm[:, good_pixels]  = result
    noise_norm[:, good_pixels] = error

    return data_norm, noise_norm


# ---------------------------------------------------------------------------
# Step 3b, Literal continuum normalisation (per-exposure polynomial fit)
# ---------------------------------------------------------------------------

def chev26_normalise_polyfit(
    wave: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    good_pixels: np.ndarray,
    degree: int = 2,
    n_iter: int = 5,
    low_sigma: float = 1.5,
    high_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-exposure 2nd-order polynomial fit to the continuum.

    Implements the literal description in Cheverall et al. (2026) Section 2.3,
    "normalized using a second-order polynomial fit to the continuum of each
    order and exposure": for each spectrum (order, exposure) a polynomial of
    ``degree`` is fitted to flux vs wavelength, iteratively rejecting points
    that fall below the fit by more than ``low_sigma`` × residual-RMS (i.e.
    absorption features) and above by ``high_sigma`` (emission spikes), so the
    fit converges onto the continuum.  Each spectrum is divided by its final
    continuum polynomial.

    This differs from :func:`chev26_normalise` (which traces the continuum via
    the maxima of 80 bins); both yield a continuum near unity, but this one is
    the literal per-exposure polynomial continuum fit.

    Parameters mirror :func:`chev26_normalise`.

    Returns
    -------
    data_norm, noise_norm : ndarray, shape (n_spectra, n_pixels)
        Continuum-normalised flux and propagated uncertainties (masked
        pixels = 1).
    """
    n_spectra, n_pixels = data.shape
    data_norm  = np.ones_like(data, dtype=float)
    noise_norm = np.ones_like(noise, dtype=float)

    gp = np.asarray(good_pixels)
    w = wave[gp].astype(float)
    # Normalise the abscissa for numerical conditioning of polyfit.
    span = (w.max() - w.min()) or 1.0
    x = (w - w.mean()) / span

    for i in range(n_spectra):
        y = data[i, gp].astype(float)
        keep = np.isfinite(y) & (y > 0)
        if keep.sum() <= degree + 1:
            continue
        coeffs = None
        for _ in range(n_iter):
            if keep.sum() <= degree + 1:
                break
            coeffs = np.polyfit(x[keep], y[keep], degree)
            fit = np.polyval(coeffs, x)
            resid = y - fit
            rms = np.nanstd(resid[keep])
            if rms == 0:
                break
            keep = (resid > -low_sigma * rms) & (resid < high_sigma * rms) \
                & np.isfinite(y) & (y > 0)
        if coeffs is None:
            continue
        cont = np.polyval(coeffs, x)
        ok = cont > 0
        data_norm[i, gp[ok]]  = y[ok] / cont[ok]
        noise_norm[i, gp[ok]] = noise[i, gp[ok]] / cont[ok]

    return data_norm, noise_norm


# ---------------------------------------------------------------------------
# Statistical-study extras for the Cheverall26 route (local; gated in the
# simulator to pipeline.name == "Cheverall26").  Reuse the per-night CCF cube
# already produced by the run to add the timing (in-transit duration) map
# alongside the Kp-Vsys map, and store the planet-position and maximum values
# of BOTH maps so statistical_study-style corner plots can be made.
# ---------------------------------------------------------------------------

def chev26_duration_map(ccf_exp, v_ccf, phase, berv, kp, vsys,
                        exclude=15.0, vwin=50.0, vmax_noise=None):
    """Cheverall+26 Fig. 4-right duration (N_in) map for ONE night, fixed Kp.

    Single tested implementation used for BOTH the saved map/figure and
    the scalar metrics, so the two can never diverge.

    At fixed Kp, shift each exposure's order-summed matched-filter CCF to the
    planet rest frame (``vp = Kp sin(2π φ) + Vsys - BERV``), order the exposures
    by proximity to mid-transit, and co-add the N_in most-central ones with
    equal weight.  S/N uses the 'point' normalisation
    (per-velocity std of co-add values >``exclude`` km/s away), identical to
    the main Kp-Vsys map.

    Parameters
    ----------
    ccf_exp : ndarray (n_lags, n_spectra)
        Per-exposure matched-filter CCF for this night (orders already summed).
    v_ccf : ndarray   Earth-frame CCF velocity grid (km/s).
    phase, berv : ndarray (n_spectra,)   Orbital phase and BERV per exposure.
    kp, vsys : float   Fixed Kp and systemic velocity (km/s).

    Returns
    -------
    snmap : ndarray (n_spectra, n_vrest)   S/N vs (N_in, V_rest).
    vsys_axis : ndarray (n_vrest,)   Systemic-velocity axis (= v_rest + vsys).
    iv0 : int   Column index of the expected V_sys (v_rest = 0).
    """
    n_lag, n_exp = ccf_exp.shape
    vp = kp * np.sin(2.0 * np.pi * phase) + vsys - berv
    dv = np.median(np.diff(v_ccf))
    # Display/measurement half-window (kept for the map, plot and metrics;
    # ~ the Kp-Vsys ±40 scale).
    _disp_half = vwin + exclude + 5
    # Noise half-range: match the main Kp-Vsys map (±vmax_noise) so the 'point'
    # noise denominator is estimated over the SAME velocity span; clipped so
    # vp+v_rest never leaves the CCF grid (no extrapolation).
    if vmax_noise is None:
        _half = _disp_half
    else:
        _feas = (min(float(np.nanmax(v_ccf) - np.nanmax(vp)),
                     float(np.nanmin(vp) - np.nanmin(v_ccf))) - 2.0 * dv)
        _half = max(_disp_half, min(float(vmax_noise) + exclude + 5, _feas))
    v_rest = np.arange(-_half, _half + dv, dv)
    iv0 = int(np.argmin(np.abs(v_rest)))
    shifted = np.array([
        np.interp(vp[i] + v_rest, v_ccf, ccf_exp[:, i], left=np.nan, right=np.nan)
        for i in range(n_exp)])
    order = np.argsort(np.abs(phase))
    # Noise normalisation = 'point' convention, IDENTICAL to
    # the main Kp-Vsys map (get_max_CCF_peak): for each velocity the noise is the
    # mean/std of the co-add values MORE than `exclude` km/s from THAT velocity
    # (the trail under test), via a sliding window, NOT the std around the
    # co-add's own peak.  Keeps the timing map and the Kp-Vsys map on the same
    # S/N scale.
    n = v_rest.size
    _W = int(round(exclude / dv)) if dv > 0 else 0
    _idx = np.arange(n)
    _lo = np.maximum(0, _idx - _W)
    _hi = np.minimum(n, _idx + _W + 1)
    snmap = np.full((n_exp, n), np.nan)
    for k in range(1, n_exp + 1):
        co = np.nansum(shifted[order[:k]], axis=0)
        if not np.any(np.isfinite(co)):
            continue
        _c = np.nan_to_num(co)
        _cs = np.concatenate(([0.0], np.cumsum(_c)))
        _css = np.concatenate(([0.0], np.cumsum(_c ** 2)))
        _wn = _hi - _lo
        _nn = n - _wn
        _ns = _cs[-1] - (_cs[_hi] - _cs[_lo])
        _nss = _css[-1] - (_css[_hi] - _css[_lo])
        _nmean = _ns / _nn
        _nstd = np.sqrt(np.maximum(_nss / _nn - _nmean ** 2, 0.0))
        snmap[k - 1] = np.where(_nstd > 0, (co - _nmean) / _nstd, 0.0)
    # Restrict the OUTPUT to the display/measurement window (the noise above was
    # estimated over the full ±vmax_noise span); keeps the 'timing max' metric
    # on the same ±vwin scale as the Kp-Vsys ±40 map.
    _keep = np.abs(v_rest) <= _disp_half
    snmap = snmap[:, _keep]
    v_rest = v_rest[_keep]
    iv0 = int(np.argmin(np.abs(v_rest)))
    return snmap, v_rest + vsys, iv0


def chev26_timing_metrics_one_night(ccf_exp, v_ccf, phase, berv, kp, vsys,
                                    n_in_cross=14, exclude=15.0, vwin=50.0,
                                    vmax_noise=None):
    """Crosshair (N_in, V_rest=0) and map-max S/N from :func:`chev26_duration_map`."""
    snmap, _vsys_axis, iv0 = chev26_duration_map(
        ccf_exp, v_ccf, phase, berv, kp, vsys, exclude=exclude, vwin=vwin,
        vmax_noise=vmax_noise)
    kx = int(np.clip(n_in_cross - 1, 0, snmap.shape[0] - 1))
    pp = float(snmap[kx, iv0]) if np.isfinite(snmap[kx, iv0]) else np.nan
    mx = float(np.nanmax(snmap)) if np.any(np.isfinite(snmap)) else np.nan
    return pp, mx


def chev26_timing_stats(ccf_all_orders, v_ccf, phase, berv, kp, vsys,
                        n_nights, n_in_cross=14, exclude=15.0, vmax_noise=None):
    """Loop nights: timing planet-pos + max for each.  Returns two arrays."""
    pos = np.full(n_nights, np.nan)
    mx = np.full(n_nights, np.nan)
    for b in range(n_nights):
        ccf_exp = np.nansum(ccf_all_orders[:, b, :, :], axis=0)  # (n_lags, n_spectra)
        ph = phase[b] if (isinstance(phase, list) or np.ndim(phase) > 1) else phase
        be = berv[b] if (isinstance(berv, list) or np.ndim(berv) > 1) else berv
        pos[b], mx[b] = chev26_timing_metrics_one_night(
            ccf_exp, v_ccf, ph, be, kp, vsys, n_in_cross, exclude,
            vmax_noise=vmax_noise)
    return pos, mx


def chev26_plot_duration_map(snmap, vsys_axis, vsys_expected, out_path,
                             kp=None, vwin=50.0, title=None, n_in_expected=None):
    """Save the Cheverall+26 Fig. 4-right duration-test figure.

    Left: the S/N(N_in, V_sys) map with crosshairs at the expected V_sys and
    the expected number of in-transit spectra (the transit duration,
    ``n_in_expected``).  Right: two 1-D slices vs N_in, at the expected V_sys
    (red) and at the V_sys of the global map maximum (orange), plus a
    horizontal line marking the expected transit duration, to expose whether
    the signal builds coherently with N_in or is a localised bump.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_in = snmap.shape[0]
    disp = (vsys_axis >= vsys_expected - vwin) & (vsys_axis <= vsys_expected + vwin)
    iv = int(np.argmin(np.abs(vsys_axis - vsys_expected)))
    col = snmap[:, iv]
    n_axis = np.arange(1, n_in + 1)
    kbest = int(np.nanargmax(col)) + 1 if np.any(np.isfinite(col)) else 0
    sbest = float(np.nanmax(col)) if np.any(np.isfinite(col)) else np.nan
    # global map maximum and the V_sys at which it occurs
    if np.any(np.isfinite(snmap)):
        kmx, ivmx = np.unravel_index(np.nanargmax(snmap), snmap.shape)
        col_mx = snmap[:, ivmx]
        vsys_mx = float(vsys_axis[ivmx])
        smax = float(snmap[kmx, ivmx])
    else:
        col_mx, vsys_mx, smax, kmx = None, np.nan, np.nan, 0

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.5, 6), sharey=True,
        gridspec_kw={"width_ratios": [3.0, 1.5]})
    extent = [vsys_axis[disp].min(), vsys_axis[disp].max(), 1, n_in]
    im = axL.imshow(snmap[:, disp], origin="lower", aspect="auto",
                    extent=extent, cmap="viridis")
    axL.axvline(vsys_expected, color="r", ls="--", lw=1.2, alpha=0.85)
    if n_in_expected is not None:
        axL.axhline(n_in_expected, color="r", ls="--", lw=1.2, alpha=0.85)
    axL.set_xlabel(r"$V_{\rm sys}$ (km s$^{-1}$)", fontsize=13)
    axL.set_ylabel(r"$N_{\rm in}$ (in-transit spectra co-added)", fontsize=13)
    fig.colorbar(im, ax=axL, label="S/N")
    # Right: slices vs N_in
    axR.plot(col, n_axis, color="#c1121f", lw=1.5, zorder=3,
             label=f"expected $V_{{\\rm sys}}$ ({vsys_expected:.1f})")
    if col_mx is not None and abs(vsys_mx - vsys_expected) > 1e-6:
        axR.plot(col_mx, n_axis, color="#e8870c", lw=1.3, ls="-", zorder=2,
                 label=f"max $V_{{\\rm sys}}$ ({vsys_mx:.1f})")
    axR.axvline(0.0, color="0.6", lw=0.8)
    if n_in_expected is not None:
        axR.axhline(n_in_expected, color="r", ls="--", lw=1.2, alpha=0.85,
                    label=f"$T_{{14}}$: $N_{{\\rm in}}$={n_in_expected}")
    axR.set_xlabel(r"S/N vs $N_{\rm in}$", fontsize=12)
    axR.set_title(f"@expected: {sbest:.2f}@N={kbest}\n"
                  f"@max: {smax:.2f}@N={kmx+1}", fontsize=9)
    axR.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    axR.grid(alpha=0.25)
    if title is None:
        title = ("H$_2$S duration test"
                 + (f" (Kp={kp:.0f})" if kp is not None else ""))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def chev26_fourmetric_corner(metrics, out_path, truths=None, title=None):
    """Corner plot of the 4 significance metrics for the Cheverall26 route.

    metrics : dict {label: 1-D array per night}.  Diagonals = histograms,
    lower triangle = 2-D scatter.  Reuses matplotlib/seaborn (as plot_stats).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        _sns = True
    except Exception:
        _sns = False
    labels = list(metrics.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(2.7 * n, 2.7 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            xi = np.asarray(metrics[labels[j]], float)
            yi = np.asarray(metrics[labels[i]], float)
            if i == j:
                d = xi[np.isfinite(xi)]
                if _sns:
                    sns.histplot(d, ax=ax, color="0.3", bins=20)
                else:
                    ax.hist(d, bins=20, color="0.3")
                if truths is not None and labels[i] in truths:
                    ax.axvline(truths[labels[i]], color="r", ls=":", lw=1.2)
            elif i > j:
                ok = np.isfinite(xi) & np.isfinite(yi)
                ax.scatter(xi[ok], yi[ok], s=8, alpha=0.4, color="0.2")
                if truths is not None and labels[j] in truths and labels[i] in truths:
                    ax.axvline(truths[labels[j]], color="r", ls=":", lw=0.9)
                    ax.axhline(truths[labels[i]], color="r", ls=":", lw=0.9)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Bayes factor -> frequentist detection significance.
# This conversion is GENERIC (Sellke, Bayarri & Berger 2001; Benneke & Seager
# 2013; Welbanks & Madhusudhan 2021, Eq. 17), not Cheverall-specific, so it now
# lives in exoplore.analysis.stats.bayes_factor_to_sigma.
# ---------------------------------------------------------------------------
