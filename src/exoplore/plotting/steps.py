"""Pipeline steps diagnostic plot.

Four-panel figure showing the successive data-simulation and preparation
steps for a single spectral order, following Fig. 5 of Sánchez-López et al.
(2022, A&A) and the ANDES paper:

  Panel A, 1-D flux comparison: noiseless model spectrum (black) and
             noisy realisation (red) at mid-transit.
  Panel B, 2-D noiseless spectral matrix (phase × wavelength).
  Panel C, 2-D noisy spectral matrix (with throughput variations).
  Panel D, 2-D residual matrix after pipeline preparation (masked pixels
             set to NaN so they appear white).
"""

import numpy as np


def plot_pipeline_steps(
        sim_name,
        plots_dir,
        wave_ins,
        phase,
        with_signal,
        without_signal,
        useful_spectral_points,
        mat_noiseless,
        mat_noisy,
        mat_residual,
        spec_idx=None,
        order_label="",
        xlim_1d=None,
        sysrem_stages=None,
        sysrem_iters=None,
        use_real_data=False,
        save_plot=True,
        show_plot=False,
):
    """Four-panel pipeline-steps diagnostic figure.

    Parameters
    ----------
    sim_name : str
        Simulation name used for the output filename.
    plots_dir : str
        Directory where the PDF is saved.
    wave_ins : ndarray  shape (n_pixels,)
        Wavelength array in µm.
    phase : ndarray  shape (n_spectra,)
        Orbital phase of each spectrum.
    with_signal : ndarray
        Integer indices of in-transit spectra.
    without_signal : ndarray
        Integer indices of out-of-transit spectra.
    useful_spectral_points : ndarray
        Boolean or integer indices of unmasked (good) pixels.
    mat_noiseless : ndarray  shape (n_spectra, n_pixels)
        Noiseless spectral matrix (e.g. spec_mat or spec_mat_shift).
    mat_noisy : ndarray  shape (n_spectra, n_pixels)
        Noisy matrix with throughput and telluric variations.
    mat_residual : ndarray  shape (n_spectra, n_pixels)
        Residual matrix after preparing_pipeline (SYSREM / BL19 / etc.).
    spec_idx : int or None
        Which spectrum to display in Panel A.  Defaults to the spectrum
        nearest to mid-transit.
    order_label : str
        Short string appended to the title (e.g. "order 23").
    xlim_1d : tuple (lo, hi) in µm or None
        Optional wavelength window for Panel A only.  If the requested
        range falls outside the order's coverage, the full range is used.
        The 2D panels always show the complete good-pixel wavelength range.
    save_plot : bool
    show_plot : bool
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_spectra, n_pixels = mat_noiseless.shape

    # ── good-pixel mask ──────────────────────────────────────────────────────
    if useful_spectral_points.dtype == bool:
        good = np.where(useful_spectral_points)[0]
    else:
        good = useful_spectral_points.astype(int)

    wave_good = wave_ins[good]

    # ── spectrum index for Panel A ────────────────────────────────────────────
    if spec_idx is None:
        if len(with_signal) > 0:
            # spectrum closest to phase = 0 (mid-transit)
            spec_idx = with_signal[np.argmin(np.abs(phase[with_signal]))]
        else:
            spec_idx = n_spectra // 2

    # ── SYSREM pipelines: stacked iteration waterfall (replaces the 4-panel) ───
    if sysrem_stages:
        _plot_sysrem_waterfall(
            sim_name, plots_dir, wave_ins, phase, with_signal, good,
            mat_noiseless, mat_noisy, sysrem_stages, sysrem_iters,
            spec_idx, order_label, xlim_1d, use_real_data,
            save_plot, show_plot,
        )
        return

    # ── build 2-D plotting matrices ───────────────────────────────────────────
    # Mask bad pixels with NaN so they appear white in pcolormesh
    def _masked(mat):
        out = np.full_like(mat, np.nan, dtype=float)
        out[:, good] = mat[:, good]
        return out

    mat_b = _masked(mat_noiseless)   # Panel B
    mat_c = _masked(mat_noisy)       # Panel C
    mat_d = _masked(mat_residual)    # Panel D

    # Phase grid for 2-D panels, all spectra
    ph_all = phase

    # ── figure ────────────────────────────────────────────────────────────────
    plt.close('all')
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
    })

    fig, axes = plt.subplots(4, 1, figsize=(9, 14),
                             gridspec_kw={"height_ratios": [1, 1.5, 1.5, 1.5]})
    plt.subplots_adjust(hspace=0.05)

    cmap = cm.viridis

    # ── Panel A: 1-D spectrum ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(wave_good,
            mat_noiseless[spec_idx, good],
            'k', linewidth=1.5, label='Noiseless')
    ax.plot(wave_good,
            mat_noisy[spec_idx, good],
            color='firebrick', linewidth=0.8, alpha=0.7, label='Noisy')
    # x-axis for Panel A, zoom if xlim_1d overlaps order coverage
    _xlo = wave_good.min()
    _xhi = wave_good.max()
    if xlim_1d is not None:
        _lo, _hi = xlim_1d
        if _lo < _xhi and _hi > _xlo:          # ranges overlap
            _xlo = max(_lo, _xlo)
            _xhi = min(_hi, _xhi)
    ax.set_xlim(_xlo, _xhi)
    ax.set_ylabel(
        'Measured flux (a.u.)' if use_real_data else 'In-silico flux (a.u.)',
        fontsize=13)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.6)
    ax.tick_params(direction='in', which='both')
    ax.set_xticklabels([])

    # ── Panels B / C / D: 2-D matrices ───────────────────────────────────────
    panel_data   = [mat_b,   mat_c,   mat_d]
    panel_labels = ['(B) Noiseless matrix',
                    '(C) Noisy + throughput',
                    '(D) Pipeline residuals (masked)']

    for k, (mat, label) in enumerate(zip(panel_data, panel_labels)):
        ax = axes[k + 1]
        vmin = np.nanpercentile(mat, 1)
        vmax = np.nanpercentile(mat, 99)
        im = ax.pcolormesh(
            wave_good,
            ph_all,
            mat[:, good],
            cmap=cmap,
            shading='auto',
            vmin=vmin,
            vmax=vmax,
        )
        # Mark ingress / egress
        if len(with_signal) > 0:
            ph_in  = phase[with_signal[0]]
            ph_out = phase[with_signal[-1]]
            ax.axhline(ph_in,  color='w', linestyle='--', linewidth=1.2)
            ax.axhline(ph_out, color='w', linestyle='--', linewidth=1.2)

        ax.set_xlim(wave_good.min(), wave_good.max())
        ax.set_ylabel('Phase', fontsize=13)
        ax.tick_params(direction='in', which='both')

        cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.ax.tick_params(labelsize=10)

        ax.text(0.01, 0.97, label,
                transform=ax.transAxes,
                fontsize=11, color='white',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='k', alpha=0.4))

        if k < 2:
            ax.set_xticklabels([])

    axes[-1].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=13)
    axes[-1].ticklabel_format(useOffset=False)

    if save_plot:
        fname = f"{plots_dir}pipeline_steps_{sim_name}.pdf"
        fig.savefig(fname, bbox_inches='tight', dpi=150)

    if show_plot:
        plt.show()


def reconstruct_sysrem_stages(
        wave, mat_noisy, noise, good_pixels, iterations,
        use_normalised_errors=True):
    """Reconstruct the residual matrix after each requested SYSREM iteration.

    Replays one order of the SYSREM preparation (pseudo-continuum
    normalisation followed by SYSREM) so the intermediate cleaning stages
    can be shown in the pipeline-steps diagnostic.  Uses :func:`apply_sysrem`,
    which returns the residual after *k* iterations, so no change to the
    production pipeline is needed.

    Parameters
    ----------
    wave : ndarray, shape (n_pixels,)
    mat_noisy : ndarray, shape (n_spectra, n_pixels)
        Noisy data matrix before preparation.
    noise : ndarray, shape (n_spectra, n_pixels)
        Per-pixel uncertainties.
    good_pixels : ndarray
        Integer indices of the unmasked columns.
    iterations : list of int
        1-based SYSREM iteration indices to reconstruct (e.g. [1, 5]).
    use_normalised_errors : bool
        ASL19 weights SYSREM by the normalised uncertainties; Gibson22 by
        the original noise.  Set False for Gibson22.

    Returns
    -------
    dict  ``{iteration: matrix}`` with masked columns set to NaN,
          each matrix of shape ``(n_spectra, n_pixels)``.
    """
    from exoplore.pipelines.bl19 import pipeline_pseudocontinuum_norm
    from exoplore.pipelines.sysrem import apply_sysrem

    gp = np.asarray(good_pixels, dtype=int)
    n_spec, n_pix = mat_noisy.shape
    norm, norm_err = pipeline_pseudocontinuum_norm(wave, mat_noisy, noise, gp)

    full = np.ones((n_spec, n_pix), dtype=float)
    full[:, gp] = norm
    full_err = np.ones((n_spec, n_pix), dtype=float)
    full_err[:, gp] = norm_err if use_normalised_errors else noise[:, gp]

    stages = {}
    for k in iterations:
        d, _ = apply_sysrem(full, full_err, int(k), gp)
        out = np.full((n_spec, n_pix), np.nan, dtype=float)
        out[:, gp] = d
        stages[int(k)] = out
    return stages


def _plot_sysrem_waterfall(
        sim_name, plots_dir, wave_ins, phase, with_signal, good,
        mat_noiseless, mat_noisy, sysrem_stages, sysrem_iters,
        spec_idx, order_label, xlim_1d, use_real_data,
        save_plot, show_plot):
    """Stacked grayscale SYSREM waterfall.

    One column shared across panels: a 1-D spectrum on top, then the raw
    data matrix, then one panel per requested SYSREM iteration, so the
    common-mode systematics are seen peeling away from one panel to the
    next.  Used only for SYSREM pipelines (ASL19, Gibson22).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    wave_good = wave_ins[good]
    if sysrem_iters is None:
        iters = sorted(sysrem_stages.keys())
    else:
        iters = [i for i in sysrem_iters if i in sysrem_stages]

    # The raw matrix is unmasked; the stages already carry NaN at the masked
    # columns, shown as a uniform colour.  Integer keys are SYSREM iterations;
    # a string key (e.g. polynomial "Corrected (residual)") is used verbatim.
    panels = [(mat_noisy, 'Raw matrix (throughput + tellurics)')]
    for k in iters:
        _lbl = f'After SYSREM iteration {k}' if isinstance(k, int) else str(k)
        panels.append((sysrem_stages[k], _lbl))

    n_panels = 1 + len(panels)   # 1-D spectrum + the 2-D panels

    # Wavelength window shared by all panels: the xlim_1d zoom if set, else
    # the full order.
    if xlim_1d is not None:
        _mxlo = max(xlim_1d[0], wave_ins.min())
        _mxhi = min(xlim_1d[1], wave_ins.max())
        if not _mxlo < _mxhi:
            _mxlo, _mxhi = wave_ins.min(), wave_ins.max()
    else:
        _mxlo, _mxhi = wave_ins.min(), wave_ins.max()
    _colvis = (wave_ins >= _mxlo) & (wave_ins <= _mxhi)

    plt.close('all')
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    })
    heights = [1.0] + [1.25] * (n_panels - 1)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 1.8 * n_panels),
        gridspec_kw={"height_ratios": heights, "hspace": 0.0})
    plt.subplots_adjust(hspace=0.0)

    # ── Panel 0: 1-D spectrum (same wavelength window as the matrices) ────────
    ax = axes[0]
    _g1 = (wave_ins >= _mxlo) & (wave_ins <= _mxhi)
    if use_real_data:
        ax.plot(wave_ins[_g1], mat_noisy[spec_idx, _g1], 'k', lw=1.0,
                label='Measured')
        _ylab = 'Measured flux (a.u.)'
    else:
        _nl = mat_noiseless[spec_idx, _g1]
        # Display only: exaggerate the noise by 40% so the noisy curve is
        # visibly distinct from the noiseless one in this diagnostic. The data
        # and all matrices are unchanged; this affects the top panel only.
        _ny = _nl + 1.4 * (mat_noisy[spec_idx, _g1] - _nl)
        ax.plot(wave_ins[_g1], _nl, 'k', lw=1.2, label='Noiseless')
        ax.plot(wave_ins[_g1], _ny, color='firebrick',
                lw=0.8, alpha=0.8, label='Noisy')
        _ylab = 'In-silico flux (a.u.)'
    ax.set_xlim(_mxlo, _mxhi)
    ax.set_ylabel(_ylab, fontsize=12)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.6)
    ax.tick_params(direction='in', which='both')
    ax.set_xticklabels([])

    # ── 2-D panels: raw matrix then one per iteration ────────────────────────
    import copy as _copy
    n_spectra = mat_noisy.shape[0]
    _cmap = _copy.copy(cm.binary_r)
    _cmap.set_bad('0.55')   # uniform grey for masked columns
    for j, (mat, label) in enumerate(panels):
        ax = axes[j + 1]
        _vis = mat[:, _colvis]
        vmin, vmax = np.nanpercentile(_vis, 2), np.nanpercentile(_vis, 98)
        ax.pcolormesh(wave_ins, np.arange(n_spectra) + 1, mat,
                      cmap=_cmap, shading='auto', vmin=vmin, vmax=vmax)
        if len(with_signal) > 0:
            ax.axhline(with_signal[0] + 1, color='red', ls='--', lw=1.0,
                       alpha=0.8)
            ax.axhline(with_signal[-1] + 1, color='red', ls='--', lw=1.0,
                       alpha=0.8)
        ax.set_xlim(_mxlo, _mxhi)
        ax.set_ylim(1, n_spectra)
        ax.set_ylabel('Spectrum', fontsize=11)
        ax.tick_params(direction='in', which='both')
        ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=10,
                color='red', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))
        if j < len(panels) - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=12)
    axes[-1].ticklabel_format(useOffset=False, axis='x')

    if save_plot:
        fig.savefig(f"{plots_dir}pipeline_steps_{sim_name}.pdf",
                    bbox_inches='tight', dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)

    plt.close(fig)
