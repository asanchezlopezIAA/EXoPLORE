"""
exoplore.plotting.kpvsys
========================

Kp-Vsys (planet semi-amplitude vs. systemic velocity) detection maps.

These maps are the standard visualisation for high-resolution
cross-correlation detections.  The co-added CCF signal is evaluated
on a grid of (Kp, Vsys) values; the planet atmosphere is detected at
the expected (Kp_true, Vsys_true) coordinate.

This module provides:

- :func:`plot_kp_vsys_map`, colourmap with optional SNR contours.
- :func:`plot_1d_ccf`, collapsed 1-D CCF at the peak Kp.

All functions return the Matplotlib ``Figure`` object so callers can
save or further customise the figure.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def plot_kp_vsys_map(
    kp_grid: np.ndarray,
    vsys_grid: np.ndarray,
    ccf_map: np.ndarray,
    snr_map: Optional[np.ndarray] = None,
    kp_true: Optional[float] = None,
    vsys_true: Optional[float] = None,
    title: str = "Kp-Vsys detection map",
    xlabel: str = r"$V_{\rm sys}$ (km s$^{-1}$)",
    ylabel: str = r"$K_p$ (km s$^{-1}$)",
    cmap: str = "RdBu_r",
    snr_contour_levels: Sequence[float] = (3.0, 5.0),
    figsize: Tuple[float, float] = (7.0, 5.5),
):
    """Plot a Kp-Vsys CCF map.

    Parameters
    ----------
    kp_grid:
        1-D array of Kp values in km/s.
    vsys_grid:
        1-D array of systemic velocity offsets in km/s.
    ccf_map:
        2-D CCF map, shape ``(len(kp_grid), len(vsys_grid))``.
        Can be a raw sum-CCF or a pre-computed SNR map.
    snr_map:
        Optional 2-D SNR map, same shape as ``ccf_map``.  If
        provided, SNR contours are overplotted.
    kp_true, vsys_true:
        Expected planet position.  If provided, a crosshair is drawn.
    title:
        Figure title.
    xlabel, ylabel:
        Axis labels.
    cmap:
        Matplotlib colormap name.
    snr_contour_levels:
        SNR values at which to draw contour lines.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=figsize)

    vextreme = np.nanmax(np.abs(ccf_map))
    norm = TwoSlopeNorm(vmin=-vextreme, vcenter=0.0, vmax=vextreme)

    im = ax.pcolormesh(
        vsys_grid, kp_grid, ccf_map,
        cmap=cmap, norm=norm, shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CCF" if snr_map is None else "CCF (normalised)")

    if snr_map is not None:
        ax.contour(
            vsys_grid, kp_grid, snr_map,
            levels=list(snr_contour_levels),
            colors="k", linewidths=0.8, linestyles="--",
        )

    if kp_true is not None:
        ax.axhline(kp_true, color="lime", lw=1.0, ls=":")
    if vsys_true is not None:
        ax.axvline(vsys_true, color="lime", lw=1.0, ls=":")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1-D CCF slice at peak Kp
# ---------------------------------------------------------------------------


def plot_1d_ccf(
    vsys_grid: np.ndarray,
    ccf_1d: np.ndarray,
    vsys_true: Optional[float] = None,
    snr_1d: Optional[np.ndarray] = None,
    title: str = "CCF at peak Kp",
    xlabel: str = r"$V_{\rm sys}$ (km s$^{-1}$)",
    ylabel: str = "CCF",
    figsize: Tuple[float, float] = (6.0, 3.5),
):
    """Plot a 1-D CCF slice.

    Parameters
    ----------
    vsys_grid:
        Velocity grid in km/s.
    ccf_1d:
        1-D CCF array, same length as ``vsys_grid``.
    vsys_true:
        Expected systemic velocity.  A vertical dashed line is drawn.
    snr_1d:
        Optional 1-D SNR array.  If provided, the y-axis is relabelled
        as SNR and ``ccf_1d`` is ignored in favour of ``snr_1d``.
    title, xlabel, ylabel, figsize:
        Plot aesthetics.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=figsize)

    plot_y = snr_1d if snr_1d is not None else ccf_1d
    ax.plot(vsys_grid, plot_y, color="steelblue", lw=1.5)
    ax.axhline(0, color="grey", lw=0.5, ls="--")

    if vsys_true is not None:
        ax.axvline(vsys_true, color="tomato", lw=1.2, ls="--", label=r"$V_{\rm sys,true}$")
        ax.legend(frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("SNR" if snr_1d is not None else ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def plot_1D_CCF(
        inp_dat, v_rest, ccf_tot_sn, max_kp, max_sn, n_kp,
        max_v_wind, xlims, show_plot=False, save_plot=True,
        CCF_Noise=False, sysrem_opt=False
        ):
    """
    Plots the 1D CCF obtained at the Kp of maximum significance of
    the grid (kp_range in the main code) explored.

    Args:
        v_rest: Array of v_rest values (0 km/s is exoplanet rest frame).
        ccf_tot_sn: 2D array of CCF values (v_rest.shape, kp_range.shape).
        max_kp: Maximum Kp value.
        max_sn: Maximum S/N value.
        inp_dat: Dictionary containing input parameters.
        n_kp: Number of Kp values.
        max_v_wind: Maximum v_wind value.

    Returns:
        None

    """

    """
    # Plot for proposals
    import matplotlib.pyplot as plt

    # Plot with enhancements
    plt.figure(figsize=(10, 5))  # Adjust figure size
    plt.plot(v_rest, ccf_tot_sn_stat[:,int(stats[37,1])+320,37], linewidth = 3, color='royalblue', label='CCF Signal')

    # Add labels and title
    plt.xlabel("Rest-frame Velocity (km/s)", fontsize=14)
    plt.ylabel("Cross-Correlation Function (S/N)", fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='best')

    # Improve tick styles
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("/Users/alexsl/Documents/Simulador/IGRINS2/HD219134b/4transits_CCF_H2O.pdf")

    # Show the plot
    plt.show()
    """
    import matplotlib.pyplot as plt

    plt.close('all')

    if not sysrem_opt:
        fig = plt.figure(figsize=(9,5))
        plt.plot(v_rest, ccf_tot_sn[:, int(max_kp)], color='black')

        # Set the y-axis limits
        plt.ylim(-2.5, round(max_sn, 2) + 0.6)

        # Set the x-axis limits and ticks
        if xlims == None:
            plt.xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
            labels = np.arange(-inp_dat['MAX_CCF_V_STD'] + 5,
                               inp_dat['MAX_CCF_V_STD'] - 5 +
                               inp_dat['PLOT_CCF_XSTEP'], inp_dat['PLOT_CCF_XSTEP'])
            plt.xticks(labels)
            # Add text to the plot
            plt.text(
                -inp_dat['MAX_CCF_V_STD'] + 10, 3,
                f"{round(max_sn, 2)}, {int(max_kp-n_kp/2)}, {np.round(max_v_wind, 2)}",
                color='black', fontsize=10
                )
        else:
            plt.xlim(xlims[0], xlims[1])
            # Add text to the plot
            plt.text(
                xlims[0]+10, 3,
                f"{round(max_sn, 2)}, {int(max_kp-n_kp/2)}, {np.round(max_v_wind, 2)}",
                color='black', fontsize=10
                )


    else:

        # Calculate number of rows (SYSREM its) and columns (subplots)
        n_rows = inp_dat["sysrem_its"]

        # Create a figure with a grid of subplots
        fig, axes = plt.subplots(n_rows, 1, figsize=(18, 5 * n_rows))

        for l in range(n_rows):
            ax = axes[l] if n_rows > 1 else axes[l]  # Select the correct subplot
            ax.plot(v_rest, ccf_tot_sn[:, int(max_kp[0, l]), 0, l], color='black', label = "Nominal")
            ax.plot(v_rest, ccf_tot_sn[:, int(max_kp[1, l]), 1, l], color='firebrick', label  = "Injected")

            if l == n_rows-1:
                ax.set_xlabel("v$_{\mathrm{rest}}$ ($\mathrm{km}$ $\mathrm{s}^{-1}$)",
                              fontsize=17)
            ax.set_ylabel('Cross correlation (S/N)', fontsize=17)

            # Set the y-axis limits
            ax.set_ylim(-5, round(np.amax([max_sn[0, l],max_sn[1, l]]), 2) + 0.6)

            # Set the x-axis limits and ticks
            if xlims == None:
                ax.set_xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
                labels = np.arange(-inp_dat['MAX_CCF_V_STD'] + 5,
                                   inp_dat['MAX_CCF_V_STD'] - 5 +
                                   inp_dat['PLOT_CCF_XSTEP'], inp_dat['PLOT_CCF_XSTEP'])
                ax.set_xticks(labels)
                # Add text to the plot
                for n in range(2):
                    color = 'k' if n == 0 else 'firebrick'
                    ax.text(
                        -inp_dat['MAX_CCF_V_STD'] + 10, np.median( ccf_tot_sn[:, int(max_kp[0, l]), 0, l])+n*1.5,
                        f"{round(max_sn[n, l], 2)}, {int(max_kp[n, l]-n_kp/2)}, {np.round(max_v_wind[n, l], 2)}",
                        color=color, fontsize=17
                        )
            else:
                ax.set_xlim(xlims[0], xlims[1])
                # Add text to the plot
                for n in range(2):
                    color = 'k' if n == 0 else 'firebrick'
                    ax.text(
                        xlims[0]+10, np.median( ccf_tot_sn[:, int(max_kp[0, l]), 0, l])+n*1.5,
                        f"{round(max_sn[n, l], 2)}, {int(max_kp[n, l]-n_kp/2)}, {np.round(max_v_wind[n, l], 2)}",
                        color=color, fontsize=17
                        )

            ax.legend(prop={'size': 17}, loc = "best")


    # Set the tick parameters
    plt.tick_params(axis='both', width=1.5, direction='in')

    # Axis labels
    plt.xlabel(r"$v_{\rm rest}$ (km s$^{-1}$)", fontsize=12)
    plt.ylabel("S/N", fontsize=12)

    # Show a reference in v_rest = 0 km/s
    plt.axvline(x=0., linestyle='--', linewidth=1, color='black')


    if save_plot and not CCF_Noise:
        filename = f"{inp_dat['plots_dir']}1D_CCF_{inp_dat['Simulation_name']}.pdf"
        fig.savefig(filename)
    elif save_plot and CCF_Noise:
        filename = f"{inp_dat['plots_dir']}1D_CCF_noise_{inp_dat['Simulation_name']}.pdf"
        fig.savefig(filename)
    if show_plot: plt.show()
    plt.close()
    return


def plot_multi_night_1D_CCF(
        inp_dat,
        v_rest,
        night_slices,
        combined_slice,
        xlims=None,
        show_plot=False,
        save_plot=True,
):
    """Overlay 1D CCF slices for individual nights and the combined stack.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dict (needs ``plots_dir``, ``Simulation_name``).
    v_rest : ndarray
        Velocity axis in km/s, shape (n_v,).
    night_slices : list of (ndarray, float)
        Per-night ``(snr_1d, peak_snr)`` tuples in night order.
    combined_slice : (ndarray, float)
        ``(snr_1d, peak_snr)`` for the combined (all-night) stack.
    xlims : list [vmin, vmax], optional
        x-axis limits; defaults to [-100, 100].
    show_plot, save_plot : bool
    """
    import matplotlib.pyplot as plt

    _xlims = xlims if xlims is not None else [-100, 100]
    _colors = ['steelblue', 'firebrick', 'darkorange', 'mediumpurple',
               'seagreen', 'goldenrod']

    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 5))

    for nn, (sl, peak) in enumerate(night_slices):
        col = _colors[nn % len(_colors)]
        ax.plot(v_rest, sl, color=col, lw=1.5,
                label=f"Night {nn}  ({peak:.1f}σ)")

    comb_sl, comb_peak = combined_slice
    ax.plot(v_rest, comb_sl, color='black', lw=2.2,
            label=f"Combined  ({comb_peak:.1f}σ)")

    ax.axvline(0, color='grey', lw=1, ls='--')
    ax.set_xlabel(r"$v_{\rm rest}$ (km s$^{-1}$)", fontsize=12)
    ax.set_ylabel("S/N", fontsize=12)
    ax.set_xlim(_xlims[0], _xlims[1])
    ax.legend(fontsize=10, loc='upper left')
    ax.tick_params(axis='both', direction='in', width=1.5)
    plt.tight_layout()

    if save_plot:
        filename = (f"{inp_dat['plots_dir']}"
                    f"1D_CCF_{inp_dat['Simulation_name']}_nights_combined.pdf")
        fig.savefig(filename)
        fig.savefig(filename.replace('.pdf', '.png'), dpi=150)
    if show_plot:
        plt.show()
    plt.close()


def plot_combined_KpVrest_1DCCF(
        inp_dat,
        v_rest,
        kp_range,
        ccf_tot_sn,
        max_kp_idx,
        title="",
        filename="",
        panel_label=None,
        show_plot=False,
        save_plot=False,
        ):

    """
    Plot the 1D CCF at the Kp of maximum significance on top,
    and the 2D Kp-V_rest significance map below, with a shared
    x-axis and a dedicated colorbar column.

    Args:
        inp_dat: dict of input parameters, must contain
            'MAX_CCF_V_STD', 'kp_max', 'PLOT_CCF_XSTEP',
            'K_p', 'plots_dir', 'Simulation_name'.
        v_rest: 1D array of rest-frame velocities.
        kp_range: 1D array of Kp values.
        ccf_tot_sn: 2D array of S/N values shaped (len(v_rest), len(kp_range)).
        max_kp_idx: integer index of the peak-significance Kp in kp_range.
        title: optional title for the top panel.
        show_plot: whether to call plt.show().
        save_plot: whether to save to PDF (and PNG).
    """
    import matplotlib.pyplot as plt

    # Build the figure with a 2×2 GridSpec: left column for plots, right for colorbar
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.4, 1.2],
        width_ratios=[0.95, 0.05],
        wspace=0.02,  # narrow gap
        hspace=0.0
    )

    # Axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])

    # Panel label outside panels (figure coords)
    if panel_label:
        fig.text(
            0.01, 0.9,
            panel_label,
            transform=fig.transFigure,
            fontsize=40,
            fontweight='bold',
            va='top',
            ha='left'
        )

    linecolor = "tomato"

    # Top panel: 1D CCF slice
    line_data = ccf_tot_sn[:, max_kp_idx]
    ax1.plot(v_rest, line_data, lw=5, color="dodgerblue")
    ax1.axvline(0, color=linecolor, ls="--", lw=4, alpha=0.7)
    ax1.set_ylabel("S/N", fontsize=22)
    ax1.tick_params(labelbottom=False, labelsize=20)
    if title:
        ax1.set_title(title, fontsize=22)
    # Add Kp max value as integer text in upper-right
    kp_val = int(np.round(kp_range[max_kp_idx]))
    ax1.text(
        0.98, 0.90,
        f"K$_P$={kp_val}"r" kms$^{-1}$",
        transform=ax1.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        ha='right'
    )
    # Bottom panel: Kp-V_rest map
    levels = np.arange(
        np.floor(ccf_tot_sn.min()),
        np.ceil(ccf_tot_sn.max()) - 0.5, 1
    )
    cf = ax2.contourf(
        v_rest, kp_range, ccf_tot_sn.T,
        levels=levels, cmap=plt.cm.viridis
    )
    # reference lines
    for sign in (-1, 1):
        ax2.plot(
            sign * np.array([20, 500]),
            [inp_dat["K_p"]] * 2,
            color=linecolor, ls="--", lw=4, alpha=0.7
        )
    # vertical dashed line at V_rest=0 with gap in the center
    y_min, y_max = ax2.get_ylim()
    center = inp_dat["K_p"]
    gap = 40  # size of gap in km/s
    ax2.plot([0, 0], [y_min, center-gap], color=linecolor, lw=4, ls="--", alpha=0.7)
    ax2.plot([0, 0], [center+gap, y_max], color=linecolor, lw=4, ls="--", alpha=0.7)

    ax2.set_xlabel("V$_{rest}$ (km s$^{-1}$)", fontsize=22)
    ax2.set_ylabel("K$_{P}$ (km s$^{-1}$)", fontsize=22)
    ax2.set_xlim(
        -inp_dat["MAX_CCF_V_STD"] + 5,
         inp_dat["MAX_CCF_V_STD"] - 5
    )
    ax2.set_ylim(-inp_dat["kp_max"], inp_dat["kp_max"])
    ax2.tick_params(labelsize=20)

    # Colorbar
    step = max(1, len(levels) // 5)
    cb = fig.colorbar(cf, cax=cax, ticks=levels[::step])
    cb.set_label("S/N", fontsize=22)
    cax.tick_params(labelsize=20)

    # Finalize
    plt.tight_layout()
    if save_plot:
        out_pdf = filename
        fig.savefig(out_pdf)
        out_png = out_pdf.replace(".pdf", ".png")
        fig.savefig(out_png, transparent=True)
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_Kp_Vrest(
        inp_dat, kp_range, ccf_tot_sn, v_rest, title="",
        show_plot=False, save_plot=False, xrange=None, yrange=None,
        CCF_Noise=False, sysrem_opt=False, cc_values=False
        ):
    """Plot the Kp-Vrest S/N detection map.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``'MAX_CCF_V_STD'``,
        ``'PLOT_CCF_XSTEP'``, ``'kp_max'``, ``'K_p'``,
        ``'sysrem_its'``, ``'plots_dir'``, ``'Simulation_name'``.
    kp_range : ndarray
        Array of trial Kp values (km/s).
    ccf_tot_sn : ndarray
        S/N map.  Shape (n_v, n_kp) for the standard case, or
        (n_v, n_kp, 2, sysrem_its) when ``sysrem_opt=True``.
    v_rest : ndarray
        Rest-frame velocity grid (km/s).
    title : str
        Figure title.
    show_plot : bool
        If ``True``, call ``plt.show()``.
    save_plot : bool
        If ``True``, save the figure as a PDF.
    xrange : list or None
        ``[xmin, xmax]`` x-axis limits.
    yrange : list or None
        ``[ymin, ymax]`` y-axis limits.
    CCF_Noise : bool
        If ``True``, appends ``"_noise"`` to the saved filename.
    sysrem_opt : bool
        If ``True``, produces a multi-panel figure for all SYSREM iterations.
    cc_values : bool
        If ``True``, uses ``"CCVal"`` in the filename (instead of ``"SNR"``).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    plt.close('all')

    # A 2D map always uses the single-panel path, even when sysrem_opt is True
    # (e.g. the Welch t-test map is collapsed to (n_v, n_kp) and has no
    # per-iteration axis, unlike the optimisation-aware CCF S/N map).
    if not sysrem_opt or np.ndim(ccf_tot_sn) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_variable = np.transpose(ccf_tot_sn, (1, 0))
        levels = np.arange(np.floor(ccf_tot_sn.min()),
                           np.ceil(ccf_tot_sn.max()) - 0.7, 1)
        # x-axis ticks: keep the configured PLOT_CCF_XSTEP when it yields a
        # readable number of labels; otherwise widen to the smallest "nice"
        # step giving <= ~12 ticks over the displayed velocity range, so a
        # small step over a wide v_rest range does not cram the axis.
        _vspan = 2.0 * inp_dat['MAX_CCF_V_STD']
        _xstep = inp_dat['PLOT_CCF_XSTEP']
        if _vspan / max(_xstep, 1e-9) > 12:
            _xstep = next((s for s in (10, 20, 25, 50, 100, 200, 500)
                           if _vspan / s <= 12), 500)
        # x-axis (v_rest) ticks anchored on 0 (so 0 is always labelled even when
        # the step does not divide the velocity range evenly); ticks beyond the
        # range are clipped by the axis limits.
        _vmax_tick = int(np.ceil(inp_dat['MAX_CCF_V_STD'] / _xstep) * _xstep)
        xlabels = np.arange(-_vmax_tick, _vmax_tick + _xstep, _xstep)
        # y-axis (Kp) ticks: a clean 50 km/s grid anchored on 0 (so 0 is always
        # labelled), spanning at least -200 to 200 km/s inclusive; ticks beyond
        # the Kp range are clipped by the axis limits.
        _kstep = 50
        _kmax = max(200, int(np.ceil(inp_dat['kp_max'] / _kstep) * _kstep))
        ylabels = np.arange(-_kmax, _kmax + _kstep, _kstep)

        kp_plot = ax.contourf(v_rest, kp_range, plot_variable, levels,
                              cmap=cm.viridis)

        ax.tick_params(axis='both', width=1.5, direction='out', labelsize=16)
        ax.set_xticks(xlabels)
        ax.set_yticks(ylabels)
        smin, smax = np.amin(plot_variable), np.amax(plot_variable)
        norm = matplotlib.colors.Normalize(vmin=smin, vmax=smax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=kp_plot.cmap)
        sm.set_array([])
        _cbar_step = 5
        _cbar_ticks = np.arange(
            int(np.ceil(smin / _cbar_step)) * _cbar_step,
            int(np.floor(smax / _cbar_step)) * _cbar_step + _cbar_step,
            _cbar_step,
        )
        cbar = plt.colorbar(sm, ax=ax, ticks=_cbar_ticks)
        cbar.set_label('S/N', fontsize=16)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel(r"v$_{\mathrm{rest}}$ ($\mathrm{km}$ $\mathrm{s}^{-1}$)",
                      fontsize=17)
        ax.set_ylabel(r"K$_\mathrm{P}$ (km s$^{-1}$)", fontsize=17)
        ax.plot([-500, -20], [inp_dat['K_p'], inp_dat['K_p']], color='r',
                linestyle='--', linewidth=2., alpha=0.6)
        ax.plot([20, 500], [inp_dat['K_p'], inp_dat['K_p']], color='r',
                linestyle='--', linewidth=2., alpha=0.6)
        ax.plot([0., 0.], [-500, inp_dat['K_p'] - 20], linestyle='--',
                linewidth=2., color='r', alpha=0.6)
        ax.plot([0., 0.], [inp_dat['K_p'] + 20, 500], linestyle='--',
                linewidth=2., color='r', alpha=0.6)
        if (xrange is None and yrange is not None) or (xrange is not None and yrange is None):
            raise ValueError(
                "Please provide either both xrange and yrange or None of them"
                )
        if xrange is None and yrange is None:
            ax.set_xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
            ax.set_ylim(-inp_dat['kp_max'], inp_dat['kp_max'])
        else:
            ax.set_xlim(xrange[0], xrange[1])
            ax.set_ylim(yrange[0], yrange[1])

    else:
        n_rows = inp_dat["sysrem_its"]
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j] if n_rows > 1 else axes[j]
                plot_variable = np.transpose(ccf_tot_sn[:, :, j, i], (1, 0))
                levels = np.arange(np.floor(ccf_tot_sn.min()),
                                   np.ceil(ccf_tot_sn.max()) - 0.7, 1)
                # x-axis ticks anchored on 0 (see 2D path), so 0 is labelled
                # even when the step does not divide the velocity range evenly.
                _xstep_m = inp_dat['PLOT_CCF_XSTEP']
                _vmax_tick = int(np.ceil(inp_dat['MAX_CCF_V_STD'] / _xstep_m)
                                 * _xstep_m)
                xlabels = np.arange(-_vmax_tick, _vmax_tick + _xstep_m, _xstep_m)
                # y-axis (Kp) ticks on a clean 50 km/s grid anchored on 0,
                # spanning at least -200 to 200 km/s inclusive (see 2D path).
                _kstep = 50
                _kmax = max(200,
                            int(np.ceil(inp_dat['kp_max'] / _kstep) * _kstep))
                ylabels = np.arange(-_kmax, _kmax + _kstep, _kstep)
                kp_plot = ax.contourf(v_rest, kp_range, plot_variable, levels,
                                      cmap=cm.viridis)
                ax.tick_params(axis='both', width=1.5, direction='out', labelsize=16)
                ax.set_xticks(xlabels)
                ax.set_yticks(ylabels)
                smin, smax = np.amin(plot_variable), np.amax(plot_variable)
                norm = matplotlib.colors.Normalize(vmin=smin, vmax=smax)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=kp_plot.cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, ticks=kp_plot.levels)
                if i == n_rows - 1:
                    ax.set_xlabel(
                        r"v$_{\mathrm{rest}}$ ($\mathrm{km}$ $\mathrm{s}^{-1}$)",
                        fontsize=17)
                if j == n_cols - 1:
                    ax.set_ylabel(r"K$_\mathrm{P}$ (km s$^{-1}$)", fontsize=17)
                ax.plot([-500, -20], [inp_dat['K_p'], inp_dat['K_p']],
                        color='firebrick', linestyle='--', linewidth=2., alpha=0.6)
                ax.plot([20, 500], [inp_dat['K_p'], inp_dat['K_p']],
                        color='firebrick', linestyle='--', linewidth=2., alpha=0.6)
                ax.plot([0., 0.], [-500, inp_dat['K_p'] - 20], linestyle='--',
                        linewidth=2., color='firebrick', alpha=0.6)
                ax.plot([0., 0.], [inp_dat['K_p'] + 20, 500], linestyle='--',
                        linewidth=2., color='firebrick', alpha=0.6)
                if (xrange is None and yrange is not None) or (xrange is not None and yrange is None):
                    raise ValueError(
                        "Please provide either both xrange and yrange or None of them"
                        )
                if xrange is None and yrange is None:
                    ax.set_xlim(-inp_dat['MAX_CCF_V_STD'] + 5,
                                inp_dat['MAX_CCF_V_STD'] - 5)
                    ax.set_ylim(-inp_dat['kp_max'], inp_dat['kp_max'])
                else:
                    ax.set_xlim(xrange[0], xrange[1])
                    ax.set_ylim(yrange[0], yrange[1])

    if cc_values:
        flag = "CCVal"
    else:
        flag = "SNR"
    if save_plot and not CCF_Noise:
        filename = (f"{inp_dat['plots_dir']}"
                    f"sn_map_{inp_dat['Simulation_name']}_{flag}.pdf")
        fig.savefig(filename)
        filename = (f"{inp_dat['plots_dir']}"
                    f"sn_map_{inp_dat['Simulation_name']}_{flag}.png")
        fig.savefig(filename, transparent=True)
    elif save_plot and CCF_Noise:
        filename = (f"{inp_dat['plots_dir']}"
                    f"sn_map_noise_{inp_dat['Simulation_name']}_{flag}.pdf")
        fig.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    return
