"""
exoplore.plotting.matrices
===========================

Spectral matrix and CCF time-series visualisation.

These plots are used for diagnostic inspection of the data at
different pipeline stages.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def plot_spectral_matrix(
    wavelengths: np.ndarray,
    phases: np.ndarray,
    matrix: np.ndarray,
    in_transit_mask: Optional[np.ndarray] = None,
    title: str = "Spectral time series",
    xlabel: str = r"Wavelength ($\mu$m)",
    ylabel: str = "Orbital phase",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (8.0, 5.5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot a 2-D spectral time-series matrix (phase vs. wavelength).

    Parameters
    ----------
    wavelengths:
        Wavelength array in μm, shape ``(n_pixels,)``.
    phases:
        Orbital phase array, shape ``(n_spectra,)``.
    matrix:
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    in_transit_mask:
        Boolean or integer index array of in-transit frames.  If
        provided, horizontal dashed lines are drawn at the transit
        ingress/egress phases.
    title, xlabel, ylabel, cmap, figsize, vmin, vmax:
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

    vext = vmin if vmin is not None else -np.nanpercentile(np.abs(matrix), 99)
    vext_max = vmax if vmax is not None else np.nanpercentile(np.abs(matrix), 99)

    ax.pcolormesh(
        wavelengths, phases, matrix,
        cmap=cmap, vmin=vext, vmax=vext_max, shading="auto",
    )

    if in_transit_mask is not None and len(in_transit_mask) > 0:
        ph_in = phases[in_transit_mask]
        ax.axhline(ph_in.min(), color="lime", lw=0.8, ls="--", label="Ingress / egress")
        ax.axhline(ph_in.max(), color="lime", lw=0.8, ls="--")
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_ccf_timeseries(
    velocity_grid: np.ndarray,
    phases: np.ndarray,
    ccf_matrix: np.ndarray,
    kp_shift: Optional[float] = None,
    title: str = "CCF time series",
    xlabel: str = r"Velocity (km s$^{-1}$)",
    ylabel: str = "Orbital phase",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (7.0, 5.5),
):
    """Plot a 2-D CCF time series (phase vs. velocity lag).

    Parameters
    ----------
    velocity_grid:
        CCF velocity lag grid in km/s, shape ``(n_lags,)``.
    phases:
        Orbital phase array, shape ``(n_spectra,)``.
    ccf_matrix:
        CCF matrix, shape ``(n_spectra, n_lags)``.
    kp_shift:
        If provided, shift the velocity axis by ``-Kp * sin(2π phase)``
        so that the planet trail aligns at ``V=0``.  This produces the
        "aligned" co-added CCF visualisation.
    title, xlabel, ylabel, cmap, figsize:
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

    vext = np.nanpercentile(np.abs(ccf_matrix), 99)
    ax.pcolormesh(
        velocity_grid, phases, ccf_matrix,
        cmap=cmap, vmin=-vext, vmax=vext, shading="auto",
    )

    if kp_shift is not None:
        # Overlay expected planet trail
        trail_v = kp_shift * np.sin(2.0 * np.pi * phases)
        ax.plot(trail_v, phases, "lime", lw=0.8, ls="--", label=f"Kp={kp_shift:.0f} km/s trail")
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def CCF_matrix_ERF(
        inp_dat, v_ccf, phase, ccf_complete, with_signal,
        without_signal, v_planet, ingress_idx=None, egress_idx=None,
        show_plot=False, save_plot=True, xlims=None, ylims=None,
        CCF_Noise=False
        ):
    """Plot the CCF matrix in the Earth's rest-frame (ERF).

    Shows the cross-correlation function as a function of velocity lag
    (x-axis, Earth rest frame) and orbital phase (y-axis).  The expected
    planet velocity trail is overplotted as a dashed white line.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``'Opt_PCA_its_ord_by_ord'``,
        ``'event'``, ``'sysrem_its'``, ``'plots_dir'``, ``'Simulation_name'``.
    v_ccf : ndarray
        Earth-frame velocity grid (km/s).
    phase : ndarray
        Orbital phase of each exposure.
    ccf_complete : ndarray
        CCF matrix.  Shape (n_lags, n_spectra) for the standard case, or
        (n_lags, n_spectra, 2, sysrem_its) when order-by-order SYSREM is used.
    with_signal : ndarray
        Indices of in-transit exposures.
    without_signal : ndarray
        Indices of out-of-transit exposures.
    v_planet : ndarray
        Planet velocity in the Earth rest frame for each exposure.
    ingress_idx : ndarray or None
        Indices of ingress contact exposures (for T1/T2 annotation).
    egress_idx : ndarray or None
        Indices of egress contact exposures (for T3/T4 annotation).
    show_plot : bool
        If ``True``, call ``plt.show()``.
    save_plot : bool
        If ``True``, save the figure to disk.
    xlims : list or None
        ``[xmin, xmax]`` velocity axis limits.
    ylims : list or None
        ``[ymin, ymax]`` phase axis limits.
    CCF_Noise : bool
        If ``True``, appends ``"_noise"`` to the saved filename.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Compute T1, T4 if contact indices are available
    if ingress_idx is not None and egress_idx is not None:
        T1 = phase[ingress_idx[0]]
        T2 = phase[ingress_idx[-1]]
        T3 = phase[egress_idx[0]]
        T4 = phase[egress_idx[-1]]

    if not inp_dat["Opt_PCA_its_ord_by_ord"] or ccf_complete.ndim == 2:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(9, 7))

        if xlims is None:
            ax.set_xlim([v_ccf.min(), v_ccf.max()])
        else:
            ax.set_xlim([xlims[0], xlims[1]])

        if ylims is None:
            ax.set_ylim([phase[with_signal[0]], phase[with_signal[-1]]])
        else:
            ax.set_ylim([ylims[0], ylims[1]])

        pcm = ax.pcolormesh(v_ccf, phase, np.transpose(ccf_complete),
                            cmap=cm.viridis, shading='auto')
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', width=1.5, direction='in', labelsize=15)
        ax.set_xlabel("Earth's rest-frame velocity (km s$^{-1}$)", fontsize=17, color='k')
        ax.set_ylabel('Phase', fontsize=17, color='k')

        if inp_dat['event'] == 'transit':
            ax.axhline(y=phase[with_signal[0]], xmin=-500, xmax=500,
                       color='w', linestyle='--')
            ax.axhline(y=phase[with_signal[-1]], xmin=-500, xmax=500,
                       color='w', linestyle='--')
            _i0 = without_signal[min(3, len(without_signal) - 1)]
            _i1 = with_signal[min(5, len(with_signal) - 1)]
            _i2 = with_signal[max(-5, -len(with_signal))]
            _i3 = without_signal[-1]
            ax.plot(v_planet[_i0:_i1], phase[_i0:_i1],
                    'w', linestyle='--', linewidth=2.5, alpha=0.9)
            ax.plot(v_planet[_i2:_i3], phase[_i2:_i3],
                    'w', linestyle='--', linewidth=2.5, alpha=0.9)

            if ingress_idx is not None and egress_idx is not None:
                ax.axhline(y=T1, color='cyan', linestyle='--')
                ax.axhline(y=T2, color='cyan', linestyle='--')
                ax.axhline(y=T3, color='cyan', linestyle='--')
                ax.axhline(y=T4, color='cyan', linestyle='--')
                x_text = -xlims[1] + 0.02 * (xlims[1] - xlims[0])
                ax.text(x_text, T1 + 0.0005, "T$_1$",
                        color='cyan', fontsize=20, ha='left', va='bottom', fontweight='bold')
                ax.text(x_text, T2 + 0.006, "T$_2$",
                        color='cyan', fontsize=20, ha='left', va='top', fontweight='bold')
                ax.text(x_text, T3 - 0.0009, "T$_3$",
                        color='cyan', fontsize=20, ha='left', va='top', fontweight='bold')
                ax.text(x_text, T4 - 0.0005, "T$_4$",
                        color='cyan', fontsize=20, ha='left', va='top', fontweight='bold')
        else:
            ax.axhline(y=phase[without_signal[0]], xmin=-500, xmax=500,
                       color='w', linestyle='--')
            ax.axhline(y=phase[without_signal[-1]], xmin=-500, xmax=500,
                       color='w', linestyle='--')
            ax.plot(v_planet[with_signal[0]:without_signal[0]],
                    phase[with_signal[0]:without_signal[0]],
                    'w', linestyle='--', linewidth=2., alpha=0.7)
            ax.plot(v_planet[without_signal[-1]:with_signal[-1]],
                    phase[without_signal[-1]:with_signal[-1]],
                    'w', linestyle='--', linewidth=2., alpha=0.7)

        cb = plt.colorbar(pcm, ax=ax)
        cb.ax.tick_params(labelsize=17)
        cb.set_label('CCF value', fontsize=17)

        if save_plot and not CCF_Noise:
            fig.savefig(
                f"{inp_dat['plots_dir']}"
                f"CC_ERF_{inp_dat['Simulation_name']}.pdf",
                bbox_inches='tight'
            )
        elif save_plot and CCF_Noise:
            fig.savefig(
                f"{inp_dat['plots_dir']}"
                f"CC_ERF_noise_{inp_dat['Simulation_name']}.pdf",
                bbox_inches='tight'
            )

        if show_plot:
            plt.show()
        plt.close()

    else:
        plt.close('all')
        n_rows = inp_dat["sysrem_its"]
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j] if n_rows > 1 else axes[j]

                if xlims is None:
                    ax.set_xlim([v_ccf.min(), v_ccf.max()])
                else:
                    ax.set_xlim([xlims[0], xlims[1]])

                if ylims is None:
                    ax.set_ylim([phase[with_signal[0]], phase[with_signal[-1]]])
                else:
                    ax.set_ylim([ylims[0], ylims[1]])

                pcm = ax.pcolormesh(v_ccf, phase,
                                    np.transpose(ccf_complete[:, :, j, i]),
                                    cmap=cm.viridis, shading='auto')
                ax.ticklabel_format(useOffset=False)
                ax.tick_params(axis='both', width=1.5, direction='in', labelsize=15)

                if i == n_rows - 1:
                    ax.set_xlabel("Earth's rest-frame velocity (km s$^{-1}$)", fontsize=17)
                if j == 0:
                    ax.set_ylabel("Phase", fontsize=17)

                if inp_dat['event'] == 'transit':
                    ax.axhline(y=phase[with_signal[0]], color='w', linestyle='--')
                    ax.axhline(y=phase[with_signal[-1]], color='w', linestyle='--')
                    ax.plot(v_planet[without_signal[3]:with_signal[5]],
                            phase[without_signal[3]:with_signal[5]],
                            'w', linestyle='--', linewidth=2.5, alpha=0.9)
                    ax.plot(v_planet[with_signal[-5]:without_signal[-1]],
                            phase[with_signal[-5]:without_signal[-1]],
                            'w', linestyle='--', linewidth=2.5, alpha=0.9)

                cb = plt.colorbar(pcm, ax=ax)
                cb.ax.tick_params(labelsize=17)
                if j == n_cols - 1:
                    cb.set_label("CCF value", fontsize=17)

        if save_plot and not CCF_Noise:
            fig.savefig(
                f"{inp_dat['plots_dir']}"
                f"CC_ERF_{inp_dat['Simulation_name']}.pdf",
                bbox_inches='tight'
            )
        elif save_plot and CCF_Noise:
            fig.savefig(
                f"{inp_dat['plots_dir']}"
                f"CC_ERF_noise_{inp_dat['Simulation_name']}.pdf",
                bbox_inches='tight'
            )

        if show_plot:
            plt.show()
        plt.close()
        plt.tight_layout()

    return


def plot_ccf_matrices_per_night(
        inp_dat, ccf_store, output_dir, v_ccf, phase, with_signal
        ):
    """
    CALL WITH:
        output_directory = "/Users/alexsl/Documents/Simulador/CARMENES_NIR/GJ436b/transit/plots/"

        exosims.plot_ccf_matrices_per_night(inp_dat,ccf_store, output_directory, v_ccf, phase, with_signal)

    Plots the CCF matrices for each night in a gridspec layout.

    Parameters:
    - ccf_store: numpy array of shape (n_orders, n_nights, ccf_lags, n_spectra).
    - output_dir: str, directory to save the plots.

    Returns:
    - None
    """
    import os
    from matplotlib.gridspec import GridSpec
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    n_orders, n_nights, ccf_lags, n_spectra = ccf_store.shape

    #ipdb.set_trace()
    for night_idx in range(n_nights):
        if inp_dat["Different_nights"]:
            phase_run = phase[night_idx]
            with_signal_run = with_signal[night_idx]
        else:
            phase_run = phase
            with_signal_run = with_signal

        # Create figure with gridspec
        fig = plt.figure(figsize=(16, 12))  # Adjust size as needed
        gs = GridSpec(5, 5, figure=fig)  # 5x5 grid (23 orders fit here)

        for order_idx in range(n_orders):
            ccf_matrix = ccf_store[order_idx, night_idx, :, with_signal_run]
            ax = fig.add_subplot(gs[order_idx])

            # Plot the CCF matrix
            im = ax.imshow(
                ccf_matrix, aspect='auto', cmap='viridis',
                origin='lower',
                extent=[v_ccf.min(), v_ccf.max(),
                        phase_run[with_signal_run].min(),
                        phase_run[with_signal_run].max()]
            )

            # Set title and axis labels
            ax.set_title(f"Order {order_idx}", fontsize=17)
            ax.tick_params(axis='both', which='both', labelsize=10)

            # Set the y-limits to phase_run values
            ax.set_ylim(phase_run[with_signal_run].min(),
                        phase_run[with_signal_run].max())

        # Adjust layout and add colorbar
        fig.tight_layout()
        # Adjusting colorbar to be horizontal
        #cbar_ax = fig.add_axes([1, 0.1, 0.15, 0.03])  # Smaller horizontal colorbar
        #cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        # Optionally, adjust colorbar scale and other properties
        #cbar.set_label('CCF value', fontsize=17)  # Add a label to the colorbar
        #cbar.ax.tick_params(labelsize=17)  # Adjust the size of the colorbar ticks

        # Save plot
        import os as _os
        output_path = _os.path.join(output_dir, f"ccf_gridspec_night_{night_idx + 1}.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
        plt.close(fig)

        print(f"Saved plot for night {night_idx + 1}: {output_path}")


def plot_mat_with_collapse(x, y, z, inp_dat, h, name,
                           with_signal=None,
                           xrange=None, yrange=None,
                           ccf_diff=False,
                           save_plot=False, only_std=False,
                           with_collapse=False):
    """
    This function generates a 2x2 grid of subplots for visualizing a
    matrix `z` as a contour plot in the upper-left subplot (bigger)
    and co-added profiles along the x-axis and y-axis in the remaining
    subplots. It also optionally plots a signal range if `ccf_diff`
    is True.

    Parameters:
        x (ndarray): 1D array representing the x-axis values.
        y (ndarray): 1D array representing the y-axis values.
        z (ndarray): 2D array representing the data to visualize.
        inp_dat (dict): Input data containing metadata.
        save_plot (bool): Whether to save the plot as an image.
        h (int): Index or identifier related to the plot.
        name (str): Name for the saved plot.
        ccf_diff (bool): Whether to plot a signal range
        (Cross-Correlation Function difference).
        with_signal (list): List of indices specifying the signal
        range (if ccf_diff is True).

    Returns:
        None
    """
    if with_collapse:
        if not only_std:
            # Create a 2x2 grid of subplots
            fig, axs = plt.subplots(
                2, 2, figsize=(14,10),
                gridspec_kw={'width_ratios': [8, 1.5], 'height_ratios': [5, 1.5]})

            # Create the contour plot in the upper-left subplot (bigger)
            plot = axs[0, 0].contourf(x, y, z, cmap='viridis')
            axs[0, 0].set_xticklabels([])
            axs[0, 0].tick_params(axis='both', width=1.4, direction='in',
                                  which='major', labelsize=17)
            axs[0, 0].set_xticklabels([])
            axs[0, 0].set_ylabel('Orbital phase', fontsize = 17)
            #axs[0, 0].set_title(f"Order {inp_dat['order_selection'][h]}", fontsize = 17)
            axs[0, 0].set_xlim([x.min(), x.max()])
            axs[0, 0].set_ylim([y.min(), y.max()])

            if ccf_diff:
                axs[0, 0].set_ylim(y[with_signal[0]],
                                   y[with_signal[-1]])

            # Create the co-added plot along the x-axis in the upper-right subplot
            co_added_y = np.mean(z, axis=1)
            axs[0, 1].plot(co_added_y * 1e3, y, color='k')
            axs[0, 1].tick_params(axis='both', width=1.4, direction='in',
                                  which='major', labelsize=17)
            axs[0, 1].set_yticklabels([])
            axs[0, 1].set_xlabel('x$10^{3}$')
            axs[0, 1].set_ylim([y.min(), y.max()])
            #axs[0, 1].set_xticklabels([])
            # Draw a line at zero
            axs[0, 1].plot(co_added_y*0., y, color='k', linestyle = '--')

            # Create the co-added plot along the x-axis in the lower-left subplot
            co_added_x = np.mean(z, axis=0)
            axs[1, 0].plot(x, co_added_x, color='k')
            axs[1, 0].tick_params(axis='both', width=1.4, direction='in',
                                  which='major', labelsize=17)
            axs[1, 0].set_xlim([x.min(), x.max()])
            #axs[1, 0].set_yticklabels([])
            if ccf_diff:
                axs[1, 0].set_xlabel('Radial velocity (km s$^{-1}$)',
                                     fontsize = 17)
            else:
                axs[1, 0].set_xlabel(r'Wavelength ($\mu m$)', fontsize = 17)
            # Leave the lower-right subplot empty
            axs[1, 1].axis('off')

            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0., hspace=0.)
    else:
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(
            2,1, figsize=(14,10),
            gridspec_kw={'height_ratios': [1, 2]})

        # Create the contour plot in the upper-left subplot (bigger)
        # Define the desired color scale range
        color_scale_min = np.min(z[with_signal, :])
        color_scale_max = np.max(z[with_signal, :])
        n_levels = 10

        axs[1].contourf(
            x, y, z, cmap='viridis',
            vmin=color_scale_min, vmax=color_scale_max,
            levels=np.linspace(color_scale_min, color_scale_max, n_levels)
            )

        axs[1].tick_params(axis='both', width=1.4, direction='in',
                              which='major', labelsize=17)
        axs[1].set_ylabel('Orbital phase', fontsize = 17)
        axs[1].set_xlabel(r'Wavelength ($\mu m$)', fontsize = 17)
        if xrange is None and yrange is None:
            axs[1].set_xlim([x.min(), x.max()])
            axs[0].set_xlim([x.min(), x.max()])
            axs[1].set_ylim([y.min(), y.max()])
        elif xrange is None and yrange is not None:
            axs[1].set_ylim([yrange[0], yrange[1]])
            axs[1].set_xlim([x.min(), x.max()])
            axs[0].set_xlim([x.min(), x.max()])

        axs[0].plot(x, z[with_signal[2], :], color='k')
        axs[0].tick_params(axis='both', width=1.4, direction='in',
                             which='major', labelsize=17)

        axs[0].set_xticklabels([])

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0., hspace=0.)

    # Save the plot if requested
    if save_plot:
        os.makedirs(f"{inp_dat['plots_dir']}statistical/gauss_noise_minmax", exist_ok=True)
        fig.savefig(
            f"{inp_dat['plots_dir']}statistical/"
            f"gauss_noise_minmax/{name}_{inp_dat['order_selection'][h]}.png",
            bbox_inches='tight'
            )


    # Show the plot
    plt.show()

    if with_collapse:
        if only_std:
            return np.std(co_added_x), np.std(co_added_y)
        else:
            return np.std(np.mean(z, axis=0)), np.std(np.mean(z, axis=1))
    else: return


def plot_steps(
    inp_dat, wave_ins, n_panels, mat_list, spec_idx, phase,
    with_signal, useful_spectral_points
    ):
    """
    exosims.plot_steps(
        inp_dat, wave_ins, 4, [spec_mat, mat_noise[0], mat_res[0]],
        3, phase, with_signal, useful_spectral_points
    )
    """
    if len(mat_list) != n_panels - 1:
        raise Exception("Provide as many matrices as n_panels-1")

    # Explicitly close any existing figures to prevent duplicates
    plt.close('all')

    # Create a new figure
    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(9, 5))

    # Set plotting parameters
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.tick_params(axis='both', width=1.4, direction='in', which='major')

    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim([wave_ins.min(), wave_ins.max()])

        if i == 0:
            # First panel: plot raw data
            ax.plot(wave_ins[useful_spectral_points], mat_list[i][spec_idx, useful_spectral_points], 'k', linewidth=2)
            ax.set_ylabel('Raw', fontsize=17)
            ax.legend(prop={'size': 15})
        else:
            mat_plot = mat_list[i - 1][with_signal][:, useful_spectral_points]
            vmin = mat_plot.min()
            vmax = mat_plot.max()
            # Remaining panels: plot matrices as heatmaps
            im = ax.pcolormesh(
                wave_ins[useful_spectral_points],
                phase[with_signal],
                mat_plot,
                cmap=cm.viridis,
                shading='auto',
                vmin=vmin,  # Use global color limits
                vmax=vmax
            )
            ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])

        # Remove x-axis tick labels except for the last subplot
        if i != len(axes) - 1:
            ax.set_xticklabels([])

    # Set common parameters for all subplots
    axes[-1].ticklabel_format(useOffset=False)
    axes[-1].set_xlabel(r'$\lambda$ [$\mu m$]', fontsize=17)
    plt.ylabel('Phase', fontsize=17)
    plt.subplots_adjust(hspace=0)  # Adjust the vertical spacing

    # Show the figure and explicitly close it
    plt.show(block=False)
    plt.close(fig)  # Close the figure after showing it


def plot_matrix_difference(wave, matrix1, matrix2, with_signal):
    """Plot two spectral matrices side-by-side with a percentage-difference panel.

    Creates a three-panel figure: ``matrix1`` (top), ``matrix2`` (middle), and
    the signed percentage difference ``(matrix2 - matrix1) / matrix1 * 100``
    (bottom), restricted to the in-transit spectra identified by ``with_signal``.
    The matrix variable names are inferred from the caller's local scope via the
    ``inspect`` module and used as subplot titles.  A symmetric colour scale is
    applied to the difference panel.  If both matrices are identical a warning
    is printed.

    Parameters
    ----------
    wave : np.ndarray, shape (n_pixels,)
        Wavelength array in any consistent unit (used for the x-axis extent).
    matrix1 : np.ndarray, shape (n_spectra, n_pixels)
        Reference spectral matrix (e.g., noiseless residuals).
    matrix2 : np.ndarray, shape (n_spectra, n_pixels)
        Comparison spectral matrix (e.g., noisy residuals or model-injected).
    with_signal : array-like of int or bool
        Indices (or Boolean mask) selecting the in-transit rows to compare.
    """
    import inspect
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import AutoMinorLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    frame = inspect.currentframe().f_back
    matrix1_name = next((n for n, v in frame.f_locals.items() if v is matrix1), "Matrix 1")
    matrix2_name = next((n for n, v in frame.f_locals.items() if v is matrix2), "Matrix 2")

    # Mask rows
    mat1 = matrix1[with_signal, :]
    mat2 = matrix2[with_signal, :]

    # Percentage diff
    diff = (mat2 - mat1) / mat1 * 100
    if np.allclose(diff, 0):
        diff_vmin, diff_vmax = -1, 1
    else:
        m = np.max(np.abs(diff))
        diff_vmin, diff_vmax = -m, m

    # Prepare subplots
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0.05}
    )

    # Plot specs: ax, data, cmap, vmin, vmax, title, ylabel, xlabel
    specs = [
        (axes[0], mat1,   'viridis',  mat1.min(),  mat1.max(),  matrix1_name,         'Spectra Index', None),
        (axes[1], mat2,   'viridis',  mat2.min(),  mat2.max(),  matrix2_name,         'Spectra Index', None),
        (axes[2], diff,   'coolwarm', diff_vmin,   diff_vmax,   'Percentage Difference (%)', 'Spectra Index', 'Wavelength')
    ]

    for ax, data, cmap, vmin, vmax, title, ylabel, xlabel in specs:
        im = ax.imshow(
            data, aspect='auto', origin='lower',
            extent=[wave.min(), wave.max(), 0, data.shape[0]],
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14)

        # separate colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        # ticks
        ax.tick_params(labelsize=12, direction='in', top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.show()

    if np.array_equal(mat1, mat2):
        print("BOTH MATRICES ARE THE SAME")
