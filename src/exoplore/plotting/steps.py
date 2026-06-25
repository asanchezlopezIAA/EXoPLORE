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
    ax.set_ylabel('Flux (a.u.)', fontsize=13)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.6)
    ax.tick_params(direction='in', which='both')
    ax.set_xticklabels([])
    title = f"Pipeline steps, {sim_name}"
    if order_label:
        title += f"  [{order_label}]"
    ax.set_title(title, fontsize=13, pad=6)

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

    plt.close(fig)
