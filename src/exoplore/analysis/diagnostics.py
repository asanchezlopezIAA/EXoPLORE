"""
exoplore.analysis.diagnostics
==============================

Diagnostic and model-comparison plots for high-resolution spectroscopy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Tuple, Set

def plot_absolute_differences(inp_dat, matrix, name, stat,
                              night_ref=0, night_max = 0, night_min=0,
                              per_order=False, save_plot=False):
    """
    Plot the absolute differences between noise matrices for 
    multiple nights and orders.

    Parameters:
        inp_dat (dict): Input data.
        matrix (ndarray): 4D array of matrices 
        for nights and orders.
        name (str): Name for the plot.
        stat (ndarray): Statistical data for each night.
        night_ref (int): Reference night to exclude from the plots.
        night_min (int): Minimum night value for per-order plots.
        per_order (bool): Whether to create per-order plots.
        save_plot (bool): Whether to save the plot as an image.

    Returns:
        None
    """
    if save_plot:
        os.makedirs(f"{inp_dat['plots_dir']}statistical/gauss_noise_minmax", exist_ok=True)
    # Calculate the absolute differences between noise matrices
    abs_diff = np.zeros((inp_dat['n_nights'], inp_dat['n_orders']))
    amp = np.zeros((inp_dat['n_nights'], inp_dat['n_orders']))
    for n in range(inp_dat['n_nights']):
        for h in range(inp_dat['n_orders']):
            abs_diff[n, h] = np.sum(
                np.abs(
                    matrix[night_ref, h, :, :] - matrix[n, h, :, :]
                    )
                ) 
            amp[n, h] = np.ptp(
                    matrix[night_ref, h, :, :] - matrix[n, h, :, :]
                    ) 

    # Create a mask to exclude the reference night from the plots
    mask = np.ones(inp_dat['n_nights'], dtype=bool)
    mask[night_ref] = False

    if not per_order:
        nights_x = np.arange(0, inp_dat['n_nights'], 1)[mask]
        for data_y, ylabel, fname_prefix in [
            (np.sum(abs_diff, axis=1)[mask], 'Absolute difference', ''),
            (np.sum(amp, axis=1)[mask],      'Amplitude',           'amp_'),
        ]:
            plt.close()
            plt.plot(nights_x, data_y, 'ko')
            plt.xlabel('Night', fontsize=17)
            plt.ylabel(ylabel, fontsize=17)
            plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
            plt.tick_params(axis='both', width=1.5, direction='in', labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)
            if save_plot:
                plt.savefig(
                    f"{inp_dat['plots_dir']}statistical/"
                    f"gauss_noise_minmax/{fname_prefix}{name}.png",
                    bbox_inches='tight')
            plt.close()

        # Calculate Pearson correlation coefficient and p-value
        pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                    np.sum(abs_diff, axis=1)[mask])
        # Plot absolute differents vs. the S/N of the night
        plt.plot(stat[:, 0][mask], np.sum(abs_diff, axis=1)[mask], 'ko',
                 label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Absolute difference', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.legend(prop={'size': 10}, loc='best')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data

        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"gauss_noise_minmax/{name}_SNR.png",
                bbox_inches='tight')

        plt.show()
        plt.close()
        
        # Calculate Pearson correlation coefficient and p-value
        pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                    np.sum(amp, axis=1)[mask])
        # Plot absolute differents vs. the S/N of the night
        plt.plot(stat[:, 0][mask], np.sum(amp, axis=1)[mask], 'ko',
                 label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Amplitude', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.legend(prop={'size': 10}, loc='best')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data

        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"gauss_noise_minmax/amp_{name}_SNR.png",
                bbox_inches='tight')

        plt.show()
        plt.close()
    else:
        for h in range(inp_dat['n_orders']):
            plt.close()
            # Plot absolute differences vs. night (excluding reference night)
            plt.plot(np.arange(0, inp_dat['n_nights'], 1)[mask], 
                     abs_diff[:, h][mask], 'ko')
            plt.title(f"Order {inp_dat['order_selection'][h]}", 
                      fontsize=17)
            plt.xlabel('Night', fontsize=17)
            plt.ylabel('Absolute difference', fontsize=17)
            plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
            plt.axvline(x=night_min, color='r', linestyle='--', 
                        linewidth=0.5, label = 'Night_min')
            plt.axvline(x=night_max, color='b', linestyle='--', 
                        linewidth=0.5, label = 'Night_max')
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            plt.legend()
            if save_plot:
                plt.savefig(
                    f"{inp_dat['plots_dir']}statistical/"
                    f"gauss_noise_minmax/{name}_{inp_dat['order_selection'][h]}.png",
                    bbox_inches='tight')

            plt.show()
            plt.close()
            
            plt.close()
            # Plot absolute differences vs. night (excluding reference night)
            plt.plot(np.arange(0, inp_dat['n_nights'], 1)[mask], 
                     amp[:, h][mask], 'ko')
            plt.title(f"Order {inp_dat['order_selection'][h]}", 
                      fontsize=17)
            plt.xlabel('Night', fontsize=17)
            plt.ylabel('Amplitude', fontsize=17)
            plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
            plt.axvline(x=night_min, color='r', linestyle='--', 
                        linewidth=0.5, label = 'Night_min')
            plt.axvline(x=night_max, color='b', linestyle='--', 
                        linewidth=0.5, label = 'Night_max')
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            plt.legend()
            if save_plot:
                plt.savefig(
                    f"{inp_dat['plots_dir']}statistical/"
                    f"gauss_noise_minmax/amp_{name}_{inp_dat['order_selection'][h]}.png",
                    bbox_inches='tight')

            plt.show()
            plt.close()

            # Calculate Pearson correlation coefficient and p-value
            pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                        abs_diff[:, h][mask])
            # Plot absolute differents vs. the S/N of the night
            plt.plot(stat[:, 0][mask], abs_diff[:, h][mask], 'ko',
                     label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Absolute difference', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.legend(prop={'size': 10}, loc='best')
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
            if save_plot:
                plt.savefig(
                    f"{inp_dat['plots_dir']}statistical/"
                    f"gauss_noise_minmax/{name}_SNR_{inp_dat['order_selection'][h]}.png",
                    bbox_inches='tight')

        plt.show()
        plt.close()
        
        pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                    amp[:, h][mask])
        # Plot absolute differents vs. the S/N of the night
        plt.plot(stat[:, 0][mask], amp[:, h][mask], 'ko',
                 label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Amplitude', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.legend(prop={'size': 10}, loc='best')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data

        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"gauss_noise_minmax/amp_{name}_SNR_{inp_dat['order_selection'][h]}.png",
                bbox_inches='tight')

    plt.show()
    plt.close()

    return


def diff_res_model(inp_dat, matrix_res, model, stats,
                   save_plot=False, per_order=True):
    """Plot the absolute difference between the residual matrix and a forward model.

    Compares the SysRem-processed spectral residual matrix for each night
    against a reference forward-model template, visualising the deviation as a
    function of spectral order and/or night S/N.  This is used to assess
    whether the systematic removal has over- or under-subtracted the planetary
    signal relative to the injected model.

    Night index 0 is always treated as the reference (noiseless) night and
    excluded from the comparison; the remaining ``n_nights - 1`` nights are
    compared against ``model``.

    Parameters
    ----------
    inp_dat : dict
        Parameter dictionary containing ``'n_nights'``, ``'n_orders'``,
        ``'instrument'``, ``'Exoplanet_name'``, ``'event'``, and
        ``'Simulation_name'``.
    matrix_res : np.ndarray, shape (n_nights, n_orders, n_spectra, n_pixels)
        4D residual matrix from the pipeline.  Night index 0 is excluded;
        the remaining nights are compared to the model.
    model : np.ndarray, shape (n_orders, n_spectra, n_pixels)
        Reference forward-model template for comparison.
    stats : np.ndarray, shape (n_nights, ...)
        Per-night statistics array; column 0 is the CCF S/N used for colour
        coding the scatter plots.
    save_plot : bool
        If True, save figures to disk.
    per_order : bool
        If True, scatter absolute differences per spectral order (colour-coded
        by S/N).  If False, sum over all orders and plot vs. S/N.
    """
    matrix_res = matrix_res[1:,:,:,:]

    # Calculate the absolute differences
    diff_resmmodel = np.zeros_like(matrix_res)
    for n in range(inp_dat['n_nights']-1):
        diff_resmmodel[n, :, :, :] = np.abs(
            matrix_res[n, :,:,:] - model[:, :, :]
            )
            
    plt.close()
    # Create a colormap (Viridis in this case)
    norm = plt.Normalize(stats[1:,0].min(), stats[1:,0].max())
    cmap = cm.viridis

    # Create a figure and axis
    if save_plot:
        os.makedirs(f"{inp_dat['plots_dir']}statistical/gauss_noise_minmax", exist_ok=True)
    plt.figure(figsize=(8, 6))

    if per_order:
        abs_diff = np.zeros((inp_dat['n_nights']-1, inp_dat['n_orders']))
        amp = np.zeros((inp_dat['n_nights']-1, inp_dat['n_orders']))
        for n in range(inp_dat['n_nights']-1):
            for h in range(inp_dat['n_orders']):
                abs_diff[n, h] = np.sum(
                    np.abs(diff_resmmodel[n, h, :, :])
                    )
                amp[n, h] = np.ptp(
                    diff_resmmodel[n, h, :, :]
                    )
            
            plt.scatter(
                inp_dat['order_selection'], abs_diff[n, :], c = cmap(norm(stats[n, 0])),
                norm = norm, cmap = cmap, marker = 'o', linewidth = 1, 
                #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
                )
         
        # Add a colorbar to indicate the colorscale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # An empty array is sufficient
        cbar = plt.colorbar(sm, label='S/N MSS')


        plt.xlabel('Spectral order', fontsize=17)
        plt.ylabel('Abs. diff. wrt TC-data', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        #plt.legend()
        #plt.xlim([800,1600])
        
        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"ABS_Diff_TCdata_model_perorder.png",
                bbox_inches='tight')
            
        plt.show()
        plt.close()
          
        for n in range(inp_dat['n_nights']-1):
            plt.scatter(
                inp_dat['order_selection'], amp[n, :], c = cmap(norm(stats[n, 0])),
                norm = norm, cmap = cmap, marker = 'o', linewidth = 1, 
                #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
                )
         
        # Add a colorbar to indicate the colorscale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # An empty array is sufficient
        cbar = plt.colorbar(sm, label='S/N MSS')

        plt.xlabel('Spectral order', fontsize=17)
        plt.ylabel('Amplitude', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        #plt.legend()
        #plt.xlim([800,1600])
        
        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"Amplitude_TCdata_model_perorder.png",
                bbox_inches='tight')
        
        plt.show()
        plt.close()
            
    else:
        abs_diff = np.zeros((inp_dat['n_nights']-1))
        amp = np.zeros((inp_dat['n_nights']-1))
        for n in range(inp_dat['n_nights']-1):
            abs_diff[n] = np.sum(np.abs(
                diff_resmmodel[n, :, :, :]
                ))
            amp[n] = np.ptp(
                diff_resmmodel[n, :, :, :]
                )
        
            
        plt.scatter(
            stats[1:,0], abs_diff, marker = 'o', 
            linewidth = 1, 
            #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
            )
        
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Abs. diff. wrt TC-data', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        #plt.legend()
        #plt.xlim([800,1600])
        
        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"ABS_Diff_TCdata_model.png",
                bbox_inches='tight')
        
        plt.show()
        plt.close()
        
        plt.scatter(
            stats[1:,0], amp, marker = 'o', 
            linewidth = 1, 
            #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
            )
        
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Amplitude wrt TC-data', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        #plt.legend()
        #plt.xlim([800,1600])
        
        if save_plot:
            plt.savefig(
                f"{inp_dat['plots_dir']}statistical/"
                f"ABS_Diff_TCdata_model.png",
                bbox_inches='tight')
        
        plt.show()
        plt.close()
            
    return


def plot_difference(wave, spec1, spec2):
    """Plot two spectra on a shared wavelength axis with a percentage-difference panel.

    Creates a two-panel figure: the upper panel overlays ``spec1`` and ``spec2``
    as a function of wavelength, and the lower panel shows the signed percentage
    difference ``(spec2 - spec1) / spec1 * 100``.  Useful for visually comparing
    two model spectra or a model vs. a telluric-corrected observation.

    Parameters
    ----------
    wave : array-like, shape (n_pixels,)
        Wavelength array in any consistent unit (nm, µm, or Å).
    spec1 : array-like, shape (n_pixels,)
        Reference spectrum (plotted in blue; denominator of the % difference).
    spec2 : array-like, shape (n_pixels,)
        Comparison spectrum (plotted in red).
    """
    # Example functions
    x = wave
    f1 = spec1
    f2 = spec2

    # Percentage difference
    diff_percent = (f2 - f1) / f1 * 100

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Top panel: Functions
    ax1.plot(x, f1, label='$f_1$', color='blue', linewidth=2)
    ax1.plot(x, f2, label='$f_2$', color='red', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Function Value', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(visible=True, which='both', linestyle='--', alpha=0.6)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title('Comparison of $f_1$ and $f_2$', fontsize=16)

    # Bottom panel: Percentage difference
    ax2.plot(x, diff_percent, label='Difference (%)', color='black', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('$x$', fontsize=14)
    ax2.set_ylabel('Diff (%)', fontsize=14)
    ax2.grid(visible=True, which='both', linestyle='--', alpha=0.6)
    ax2.tick_params(axis='both', labelsize=12)

    # Save and display
    plt.tight_layout()
    plt.show()


def autocorrelation_examples(n_lines = 500, filepath = ""
        ):
    """Generate illustrative cross-correlation figures for a simulated ro-vib line forest.

    Simulates a synthetic molecular template with ``n_lines`` Gaussian absorption
    features on a 1000-pixel wavelength grid (1.0 to 1.8 µm), then cross-correlates
    it with noisy versions of itself at three noise levels (zero, low, high).
    Produces a four-panel figure: the template spectrum and three CCF traces,
    useful for demonstrating how S/N and line density affect the CCF peak shape.

    Parameters
    ----------
    n_lines : int
        Number of simulated molecular lines to inject into the template.
        Higher values increase the autocorrelation peak sharpness and S/N.
    filepath : str
        Directory prefix for the output PDF.  The figure is saved to
        ``{filepath}autocorrelation_game.pdf``.
    """
    # 1) Simulate a dense ro-vib line forest template
    np.random.seed(42)
    n_pixels = 1000
    # Create a realistic wavelength grid (e.g., 1.0 to 1.8 microns)
    wavelength = np.linspace(1.0, 1.8, n_pixels)
    
    # baseline continuum fluctuations
    continuum = 0.01 * np.random.randn(n_pixels)
    
    # random line parameters
    n_lines = n_lines
    line_centers = np.random.uniform(wavelength.min(), wavelength.max(), n_lines)
    line_strengths = np.random.lognormal(mean=0, sigma=1, size=n_lines)
    line_widths = np.random.uniform(0.0002, 0.002, n_lines)  # in microns
    
    # build template
    template = continuum.copy()
    for center, strength, width in zip(line_centers, line_strengths, line_widths):
        template += strength * np.exp(-0.5 * ((wavelength - center) / width) ** 2)
    template -= np.mean(template)
    
    # 2) Create noisy versions of the template
    noise_levels = [0.0, 5, 20]
    templates_noisy = [template + nl * np.random.normal(size=n_pixels) for nl in noise_levels]
    
    # 3) Compute cross-correlations
    lags = np.arange(-n_pixels+1, n_pixels)
    ccfs = [np.correlate(template, t_noisy, mode='full') for t_noisy in templates_noisy]
    ccfs_norm = [ccf / np.max(np.abs(ccf)) for ccf in ccfs]
    
    # 4) Plot four subplots: first in wavelength, next three in lag
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), gridspec_kw={'hspace': 0.2})
    
    # 4.1) Template vs wavelength
    axes[0].plot(wavelength, template, color='teal')
    axes[0].set_ylabel("Flux (cont.-subtr.)")
    axes[0].set_title("0) Simulated Ro-Vib Line Forest Template")
    axes[0].grid(alpha=0.5)
    
    # 4.2) Autocorrelation vs lag
    axes[1].plot(lags, ccfs_norm[0], color='orange')
    axes[1].axhline(0, linestyle='--', color='gray')
    axes[1].set_xlim(-200, 200)
    axes[1].set_ylabel("CCF (norm.)")
    axes[1].set_title("1) Template Autocorrelation (no noise)")
    axes[1].grid(alpha=0.5)
    
    # 4.3) Low noise CCF
    axes[2].plot(lags, ccfs_norm[1], color='orange')
    axes[2].axhline(0, linestyle='--', color='gray')
    axes[2].set_xlim(-200, 200)
    axes[2].set_ylabel("CCF (norm.)")
    axes[2].set_title("2) CCF with Low Noise")
    axes[2].grid(alpha=0.5)
    
    # 4.4) High noise CCF
    axes[3].plot(lags, ccfs_norm[2], color='orange')
    axes[3].axhline(0, linestyle='--', color='gray')
    axes[3].set_xlim(-200, 200)
    axes[3].set_ylabel("CCF (norm.)")
    axes[3].set_title("3) CCF with Higher Noise")
    axes[3].set_xlabel("Lag (pixels)")
    axes[3].grid(alpha=0.5)
    
    # Style ticks
    for ax in axes:
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{filepath}autocorrelation_game.pdf")

    plt.show()


def plot_params_vs_order(base_dir: str,
    night_index: int = 0,
    param_labels: Optional[List[str]] = None,
    truths: Optional[List[float]] = None,
    output_png: Optional[str] = None,
    save_csv: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Scan betatest_order_* subfolders under base_dir, read retrieval_night_{night_index}_stats.json
    from each, extract parameter medians and 1-sigma, and plot each parameter vs order.

    Returns (orders, medians_arr, sigmas_arr, used_param_labels)
      - orders: shape (n_orders,)
      - medians_arr: shape (n_orders, n_params)
      - sigmas_arr: shape (n_orders, n_params)
      - used_param_labels: list length n_params
      
    CALL LIKE:
    -----------
    base_dir = "/Users/alexsl/Documents/Simulador/CARMENES_NIR/GJ436b/transit/matrices/betatest_GJ436b_night_3/"
    param_labels = ["log_{10}(Z) $Z_\\odot$", "$log_{10}(p)$ (bar)","$\\beta$"]
    truths = [0, -1, 1]

    orders, meds, sigs, labels = exosims.plot_params_vs_order(
        base_dir=base_dir,
        night_index=0,
        param_labels=param_labels,
        truths=None,
        output_png=os.path.join(base_dir, "params_vs_order.png"),
        save_csv=True
    )
    """
    import os
    import glob
    import json
    import re
    import numpy as np
    import matplotlib.pyplot as plt

    # find order folders
    order_dirs = sorted(glob.glob(os.path.join(base_dir, "order_*")))
    if not order_dirs:
        raise FileNotFoundError(f"No folders matching betatest_order_* found in {base_dir}")

    def order_from_name(path):
        """Extract the spectral order number from a folder name.

        Tries two regex patterns in sequence: first looks for an explicit
        ``order_<N>`` or ``order-<N>`` substring, then falls back to the first
        integer found anywhere in the basename.  Returns ``None`` if no digit
        sequence can be found, in which case the folder is skipped.

        Parameters
        ----------
        path : str
            Full path to an ``order_*`` subdirectory.

        Returns
        -------
        int or None
            Parsed spectral order number, or ``None`` if parsing fails.
        """
        name = os.path.basename(path)
        m = re.search(r'order[_\-]?(\d+)', name)
        if m:
            return int(m.group(1))
        m2 = re.search(r'(\d+)', name)
        return int(m2.group(1)) if m2 else None

    orders = []
    medians_list = []
    sigmas_list = []
    param_names_from_first = None

    for d in order_dirs:
        order_num = order_from_name(d)
        if order_num is None:
            print(f"Skipping {d}: couldn't parse order number")
            continue

        # search recursively for the stats json
        pattern = os.path.join(d, "**", f"retrieval_night_0_stats.json")
        found = glob.glob(pattern, recursive=True)
        if not found:
            print(f"[order {order_num}] Stats JSON not found in {d}")
            continue
        stats_fn = found[0]

        try:
            with open(stats_fn, 'r') as fh:
                stats = json.load(fh)
        except Exception as e:
            print(f"[order {order_num}] Failed to read {stats_fn}: {e}")
            continue

        marginals = stats.get('marginals') or stats.get('marginal') or None
        if not marginals or not isinstance(marginals, list):
            print(f"[order {order_num}] No 'marginals' list found in {stats_fn}")
            continue

        # record param names from the first valid file
        if param_names_from_first is None:
            names_tmp = []
            for m in marginals:
                nm = m.get('name') or m.get('parameter') or m.get('param') or m.get('label')
                names_tmp.append(nm if nm is not None else None)
            param_names_from_first = names_tmp

        n_params = len(marginals)
        meds = np.full(n_params, np.nan)
        sigs = np.full(n_params, np.nan)

        for i, m in enumerate(marginals):
            # median extraction heuristics
            med = None
            if 'median' in m:
                med = m['median']
            elif 'mean' in m:
                med = m['mean']
            elif 'quantiles' in m and isinstance(m['quantiles'], dict):
                med = m['quantiles'].get('50%') or m['quantiles'].get('0.5')
            try:
                if med is not None:
                    meds[i] = float(med)
            except Exception:
                meds[i] = np.nan

            # sigma extraction heuristics
            if '1sigma' in m and isinstance(m['1sigma'], (list, tuple)) and len(m['1sigma']) == 2:
                lo, hi = m['1sigma']
                try:
                    sigs[i] = (float(hi) - float(lo)) / 2.0
                except Exception:
                    sigs[i] = np.nan
            elif 'sigma' in m:
                try:
                    sigs[i] = float(m['sigma'])
                except Exception:
                    sigs[i] = np.nan
            else:
                # try quantiles dict with 16%/84%
                if 'quantiles' in m and isinstance(m['quantiles'], dict):
                    lo = m['quantiles'].get('16%') or m['quantiles'].get('0.16')
                    hi = m['quantiles'].get('84%') or m['quantiles'].get('0.84')
                    if lo is not None and hi is not None:
                        try:
                            sigs[i] = (float(hi) - float(lo)) / 2.0
                        except Exception:
                            sigs[i] = np.nan
                # else leave NaN

        orders.append(order_num)
        medians_list.append(meds)
        sigmas_list.append(sigs)

    if not orders:
        raise RuntimeError("No valid stats files were loaded. Check base_dir and night_index.")

    # sort by order number
    idx = np.argsort(orders)
    orders = np.array(orders)[idx]
    medians_arr = np.vstack(medians_list)[idx, :]
    sigmas_arr = np.vstack(sigmas_list)[idx, :]

    n_orders, n_params = medians_arr.shape

    # determine labels to use
    if param_labels is not None and len(param_labels) == n_params:
        used_labels = param_labels
    else:
        # try param names from JSON if available
        if param_names_from_first and len(param_names_from_first) == n_params and all(x is not None for x in param_names_from_first):
            used_labels = param_names_from_first
        else:
            # fallback: generic names possibly extended with provided list
            if param_labels is None:
                used_labels = [f"p{i}" for i in range(n_params)]
            else:
                # if user passed a list of labels but length differs, align/truncate/pad
                if len(param_labels) < n_params:
                    used_labels = param_labels + [f"p{i}" for i in range(len(param_labels), n_params)]
                else:
                    used_labels = param_labels[:n_params]

    # prepare output filenames
    if output_png is None:
        output_png = os.path.join(base_dir, "params_vs_order.png")
    output_csv = os.path.join(base_dir, "params_vs_order.csv")

    # ---------------------------
    # PLOTTING LAYOUT: 5 rows minimum, shared x-axis, no titles, hspace=0
    # label fontsizes = 20, ticklabels size = 16, tick width increased
    # ---------------------------
    nrows_plot = max(5, n_params)
    ncols_plot = 1
    plt.close('all')
    fig, axes2d = plt.subplots(nrows_plot, ncols_plot,
                              figsize=(10, 2.8 * nrows_plot),
                              sharex=True, squeeze=False)
    axes = [axes2d[r, 0] for r in range(nrows_plot)]
    x = orders

    for i in range(n_params):
        ax = axes[i]
        y = medians_arr[:, i]
        yerr = sigmas_arr[:, i]
        has_err = ~np.isnan(yerr)

        if np.any(has_err):
            ax.errorbar(x[has_err], y[has_err], yerr=yerr[has_err], fmt='o-', capsize=3, label='median ± sigma')
            if np.any(~has_err):
                ax.plot(x[~has_err], y[~has_err], 's', alpha=0.6, label='median (no sigma)')
        else:
            ax.plot(x, y, 'o-', label='median')

        # truth line if provided
        if truths is not None and len(truths) >= n_params and not np.isnan(truths[i]):
            ax.axhline(truths[i], color='firebrick', linestyle='--', label='truth')

        # No titles per your request
        ax.set_ylabel(used_labels[i], fontsize=20)           # <-- label fontsize 20
        ax.grid(alpha=0.25)
        ax.set_xticks(x)
        #ax.legend(fontsize=8, loc='best')

        # increase tick width and ticklabel size for both x and y
        ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)

    # hide any extra (unused) axes if n_params < nrows_plot
    for j in range(n_params, nrows_plot):
        axes[j].set_visible(False)

    # shared x-label on the bottom-most visible axis
    bottom_ax = None
    for ax in reversed(axes):
        if ax.get_visible():
            bottom_ax = ax
            break
    if bottom_ax is not None:
        bottom_ax.set_xlabel("Order number", fontsize=20)   # <-- label fontsize 20
        bottom_ax.tick_params(axis='x', rotation=45, labelsize=16, width=2, length=6)

    # set vertical spacing to zero
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    fig.savefig(output_png, dpi=200)
    print("Saved plot to:", output_png)

    if save_csv:
        # build a friendly CSV: columns: order, param1_med, param1_sigma, param2_med, param2_sigma, ...
        header = ["order"]
        for lab in used_labels:
            header += [f"{lab}_med", f"{lab}_sigma"]
        rows = []
        for oi in range(n_orders):
            row = [int(orders[oi]) if np.isfinite(orders[oi]) else orders[oi]]
            # build med/sig pairs
            for pi in range(n_params):
                row.append(medians_arr[oi, pi])
                row.append(sigmas_arr[oi, pi])
            rows.append(row)
        import csv
        with open(output_csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for r in rows:
                writer.writerow(r)
        print("Saved CSV to:", output_csv)

    plt.show()
    return orders, medians_arr, sigmas_arr, used_labels





# ------------------ Core function to combine retrievals: read, combine, plot with style2 ------------------
# ------------------ Core function to combine retrievals: read, combine, plot with style2 ------------------
# ------------------ Core function to combine retrievals: read, combine, plot with style2 ------------------
# ------------------ Core function to combine retrievals: read, combine, plot with style2 ------------------


def plot_detectability_maps_FeH_CO(data_dirs,
    cmaps,
    labels=None,
    sn5_contour=True,
    sn10_contour=False,
    figsize=(6, 7),
    simulate=False,
    truths=None,
    output_file="erase.pdf",
    contour_levels_by_map=None,
    contour_default="auto_last3",
    contour_linewidth=3,
    contour_fontsize=27,
    contour_inline=False,
    contour_label_fmt="S/N={:.0f}"
    ):
    """
    ============================================================================
    Plot multiple Fe/H vs C/O detectability maps.

    Each directory must contain files named:
        detectability_*.txt

    Each file must contain one line:
        FeH  C/O  SNR

    Example line:
        -1.5000 0.8000 15.9063

    PARAMETERS
    ----------
    data_dirs : list of str
        List of paths to directories containing detectability files.

    cmaps : list of str
        List of matplotlib colormaps. Must have same length as data_dirs.

    labels : list of str, optional
        Titles for each subplot, e.g. ["H2O", "CO2", "CH4"].

    sn5_contour : bool
        Kept for backwards compatibility. If contour_levels_by_map is None
        and contour_default is not used, this can request S/N=5.

    sn10_contour : bool
        Kept for backwards compatibility. If contour_levels_by_map is None
        and contour_default is not used, this can request S/N=10.

    figsize : tuple
        Size of each subplot as (width, height).

    simulate : bool
        If False, all expected Fe/H--C/O grid files must be present.
        If True, the function reads all available real files and fills missing
        grid points with simulated S/N values, allowing incomplete maps to be
        visualized while simulations are still running.

    truths : list, tuple, or None
        If not None, should be [Fe/H, C/O]. Plotted as a gold star.

    output_file : str
        Path to save output figure.

    contour_levels_by_map : None, list, tuple, or dict
        Flexible contour control.

        If None:
            use contour_default.

        If list/tuple:
            same contour levels are used for all panels.
            Example:
                contour_levels_by_map=[5, 10, 15]

        If dict:
            keys can be subplot indices or labels.
            Example:
                contour_levels_by_map={
                    0: [5, 10, 15],
                    1: [5],
                    "Fe (UBV)": [35, 40],
                    "Ca (UBV)": [20]
                }

            Use [] or None for a panel to draw no contours.

    contour_default : str or list
        Default contour behaviour when no explicit levels are given.

        "auto_last3":
            draw the last three available multiples of 5.

        "fixed":
            draw S/N=5 and/or S/N=10 according to sn5_contour/sn10_contour.

        list/tuple:
            use this list as default levels for every panel.

        None:
            draw no default contours.

    contour_linewidth : float
        Contour linewidth.

    contour_fontsize : float
        Contour label fontsize.

    contour_inline : bool
        Passed to ax.clabel(..., inline=contour_inline).
        Set False to avoid the white rectangular gaps under labels.

    contour_label_fmt : str
        Format string for contour labels.
        Example:
            "S/N={:.0f}"
            "{:.0f}"
            r"$S/N={:.0f}$"

    RETURNS
    -------
    fig, axes, peak_summary
    
    Example of use:
    fig, axes, peak_summary = exosims.plot_detectability_maps_FeH_CO(
        data_dirs=[
            "/Users/alexsl/Documents/Simulador/ANDES/WASP76b/transit/matrices/h2o/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/",
            "/Users/alexsl/Documents/Simulador/ANDES/WASP76b/transit/matrices/co/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/",
            "/Users/alexsl/Documents/Simulador/ANDES/WASP76b/transit/matrices/fe/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/",
            "/Users/alexsl/Documents/Simulador/ANDES/WASP76b/transit/matrices/ti/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/",
            "/Users/alexsl/Documents/Simulador/ANDES/WASP76b/transit/matrices/v/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/",
            "/Users/alexsl/Documents/Simulador/ANDES/WASP76b/transit/matrices/ca/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/",
        ],
        labels=[
            "H$_2$O (YHJK)",
            "CO (K)",
            "Fe (UBV)",
            "Ti (UBV)",
            "V (UBV)",
            "Ca (UBV)"
        ],
        cmaps=[
            "gist_earth_r",
            "YlGnBu_r",
            "bone",
            "gray",
            "summer",
            "copper"
        ],
        simulate=True,
        truths=[-1.34, 0.8],
        contour_levels_by_map={
            "H$_2$O (YHJK)": [5, 10, 15],
            "CO (K)": [10],
            "Fe (UBV)": [35, 40],
            "Ti (UBV)": [],
            "V (UBV)": [25],
            "Ca (UBV)": [20]
        },
        contour_inline=False,
        output_file="/Users/alexsl/Documents/Simulador/ANDES/HD189733b/transit/plots/detectability_maps_CCF_w76b.pdf"
    )
    ============================================================================
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import glob
    import os

    # -------------------------------------------------------------------------
    # Expected Fe/H and C/O grids
    # -------------------------------------------------------------------------
    feh_vals = np.array(
        [-1.50, -1.00, -0.70, -0.50, -0.30,
          0.00,  0.30,  0.50,  0.70,  1.00, 1.50]
    )

    co_vals = np.array(
        [0.30, 0.35, 0.40, 0.50,
         0.60, 0.70, 0.80, 0.90, 1.00, 1.10]
    )

    expected_grid = [(float(f), float(c)) for f in feh_vals for c in co_vals]

    # -------------------------------------------------------------------------
    # Basic checks
    # -------------------------------------------------------------------------
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    if isinstance(cmaps, str):
        cmaps = [cmaps]

    n_maps = len(data_dirs)

    if len(cmaps) != n_maps:
        raise ValueError("cmaps must have the same length as data_dirs")

    if labels is None:
        labels = [f"Map {i + 1}" for i in range(n_maps)]

    if len(labels) != n_maps:
        raise ValueError("labels must have the same length as data_dirs")

    # -------------------------------------------------------------------------
    # Helper for contour levels
    # -------------------------------------------------------------------------
    def _get_contour_levels(i, label, snr_max):
        """
        Decide which contour levels to draw for one map.
        """

        # Explicit user control has highest priority.
        if contour_levels_by_map is not None:

            if isinstance(contour_levels_by_map, dict):

                if i in contour_levels_by_map:
                    levels_user = contour_levels_by_map[i]
                elif label in contour_levels_by_map:
                    levels_user = contour_levels_by_map[label]
                else:
                    levels_user = contour_default

            else:
                levels_user = contour_levels_by_map

        else:
            levels_user = contour_default

        # No contours.
        if levels_user is None:
            return np.array([], dtype=float)

        # Automatic last-three multiples of 5.
        if isinstance(levels_user, str):

            if levels_user == "auto_last3":
                highest_level = int(np.floor(snr_max / 5.0) * 5)

                if highest_level < 5:
                    return np.array([], dtype=float)

                levels_auto = np.arange(5, highest_level + 1, 5)
                return levels_auto[-3:].astype(float)

            elif levels_user == "fixed":
                levels_fixed = []

                if sn5_contour:
                    levels_fixed.append(5.0)

                if sn10_contour:
                    levels_fixed.append(10.0)

                return np.asarray(levels_fixed, dtype=float)

            else:
                raise ValueError(
                    "contour_default must be 'auto_last3', 'fixed', "
                    "a list/tuple of levels, or None."
                )

        # Explicit list of levels.
        levels_user = np.asarray(levels_user, dtype=float)

        if levels_user.size == 0:
            return np.array([], dtype=float)

        # Only keep levels that are meaningful for this map.
        levels_user = levels_user[np.isfinite(levels_user)]
        levels_user = levels_user[(levels_user > 0.0) & (levels_user < snr_max)]

        return np.unique(levels_user)

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    if n_maps <= 3:
        ncols = n_maps
    elif n_maps <= 6:
        ncols = 3
    else:
        ncols = 4

    nrows = int(np.ceil(n_maps / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    peak_summary = {}

    # -------------------------------------------------------------------------
    # Loop over maps
    # -------------------------------------------------------------------------
    for i, (data_dir, cmap, label) in enumerate(zip(data_dirs, cmaps, labels)):

        ax = axes[i]

        files = sorted(glob.glob(os.path.join(data_dir, "detectability_*.txt")))

        real_data = {}

        for file in files:
            with open(file) as f:
                try:
                    feh_val, co_val, snr_val = map(float, f.readline().split())
                    key = (round(feh_val, 2), round(co_val, 2))
                    real_data[key] = snr_val
                except Exception as e:
                    print(f"Skipping {file}: {e}")

        missing = []

        for fval, cval in expected_grid:
            key = (round(fval, 2), round(cval, 2))

            if key not in real_data:
                missing.append(key)

        if missing and not simulate:
            raise FileNotFoundError(
                f"[{label}] Missing {len(missing)} / {len(expected_grid)} "
                f"detectability files in {data_dir}. "
                f"Set simulate=True to fill missing grid points temporarily."
            )

        feh_list = []
        co_list = []
        snr_list = []

        rng = np.random.default_rng(seed=42 + i)

        for fval, cval in expected_grid:

            key = (round(fval, 2), round(cval, 2))

            feh_list.append(fval)
            co_list.append(cval)

            if key in real_data:
                snr_list.append(real_data[key])

            else:
                center_feh = [-0.2, 0.2, 0.6][i % 3]
                center_co = [0.5, 0.7, 0.9][i % 3]
                amp = [9.0, 7.0, 13.0][i % 3]

                snr_fake = (
                    amp
                    * np.exp(-((fval - center_feh) ** 2) / 0.65)
                    * np.exp(-((cval - center_co) ** 2) / 0.08)
                )

                snr_fake += 1.5 * (fval + 1.5) / 2.5
                snr_fake += rng.normal(0.0, 0.5)

                snr_list.append(max(snr_fake, 0.0))

        if missing and simulate:
            print(
                f"[{label}] Using {len(real_data)} real points and "
                f"{len(missing)} simulated fill-in points."
            )

        # ---------------------------------------------------------------------
        # Convert to arrays
        # ---------------------------------------------------------------------
        feh = np.asarray(feh_list, dtype=float)
        co = np.asarray(co_list, dtype=float)
        snr = np.asarray(snr_list, dtype=float)

        if len(feh) == 0:
            raise RuntimeError(f"No valid data points found for {label}")

        print(
            f"[{label}] Loaded {len(feh)} points, "
            f"S/N range = {np.nanmin(snr):.2f}--{np.nanmax(snr):.2f}"
        )

        # ---------------------------------------------------------------------
        # Peak detectability summary
        # ---------------------------------------------------------------------
        imax = np.nanargmax(snr)

        peak_summary[label] = {
            "snr_max": float(snr[imax]),
            "feh_at_max": float(feh[imax]),
            "co_at_max": float(co[imax])
        }

        print(
            f"[{label}] Peak S/N = {snr[imax]:.2f} "
            f"at [Fe/H] = {feh[imax]:+.2f}, C/O = {co[imax]:.2f}"
        )

        # ---------------------------------------------------------------------
        # Interpolation grid
        # ---------------------------------------------------------------------
        feh_g = np.linspace(feh.min(), feh.max(), 280)
        co_g = np.linspace(co.min(), co.max(), 280)

        FEH, CO = np.meshgrid(feh_g, co_g)

        SNR_lin = griddata(
            (feh, co),
            snr,
            (FEH, CO),
            method="linear"
        )

        SNR_near = griddata(
            (feh, co),
            snr,
            (FEH, CO),
            method="nearest"
        )

        SNRg = np.where(np.isnan(SNR_lin), SNR_near, SNR_lin)
        SNRg = gaussian_filter(SNRg, sigma=0.4)
        SNRg = np.nan_to_num(SNRg, nan=np.nanmin(snr))

        # ---------------------------------------------------------------------
        # Colour scale per panel
        # ---------------------------------------------------------------------
        vmin = 0.0
        vmax = max(1.0, np.nanmax(snr))
        levels = np.linspace(vmin, vmax, 30)

        cf = ax.contourf(
            FEH,
            CO,
            SNRg,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend="both",
            alpha=0.95,
            antialiased=False
        )

        # ---------------------------------------------------------------------
        # Truth marker
        # ---------------------------------------------------------------------
        if truths is not None:

            truth_feh, truth_co = truths

            ax.scatter(
                truth_feh,
                truth_co,
                marker="*",
                s=1500,
                facecolor="none",
                edgecolor="k",
                linewidth=1,
                zorder=40
            )

            ax.scatter(
                truth_feh,
                truth_co,
                marker="*",
                s=1500,
                facecolor="gold",
                edgecolor="k",
                linewidth=1.4,
                zorder=41
            )

        # ---------------------------------------------------------------------
        # Detection contours
        # ---------------------------------------------------------------------
        snr_max = float(np.nanmax(SNRg))

        contour_levels = _get_contour_levels(
            i=i,
            label=label,
            snr_max=snr_max
        )

        if contour_levels.size > 0:

            color = "k"

            CS = ax.contour(
                FEH,
                CO,
                SNRg,
                levels=contour_levels,
                colors=color,
                linestyles="--",
                linewidths=contour_linewidth
            )

            fmt = {
                lev: contour_label_fmt.format(lev)
                for lev in contour_levels
            }

            texts = ax.clabel(
                CS,
                levels=contour_levels,
                fmt=fmt,
                inline=contour_inline,
                inline_spacing=8,
                fontsize=contour_fontsize,
                colors=color,
                rightside_up=True
            )

            # Ensure no label background box is drawn.
            for txt in texts:
                txt.set_bbox(dict(facecolor="none", edgecolor="none", pad=0.0))

        # ---------------------------------------------------------------------
        # Labels and styling
        # ---------------------------------------------------------------------
        ax.set_title(label, fontsize=20)
        ax.set_xlabel("[Fe/H]", fontsize=20)

        if i % ncols == 0:
            ax.set_ylabel("C/O", fontsize=20)
        else:
            ax.set_yticklabels([])

        ax.set_xlim(feh.min(), feh.max())
        ax.set_ylim(co.min(), co.max())

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=16,
            width=1.2,
            length=6
        )

        # ---------------------------------------------------------------------
        # Individual colorbar
        # ---------------------------------------------------------------------
        vmax_int = int(np.floor(vmax))

        if vmax_int <= 10:
            step = 1
        elif vmax_int <= 20:
            step = 2
        else:
            step = 5

        cbar_ticks = np.arange(0, vmax_int + 1, step)

        if cbar_ticks.size == 0:
            cbar_ticks = np.array([0, vmax_int])

        if cbar_ticks[-1] != vmax_int:
            cbar_ticks = np.append(cbar_ticks, vmax_int)

        cbar = fig.colorbar(
            cf,
            ax=ax,
            orientation="horizontal",
            pad=0.14,
            fraction=0.06,
            ticks=cbar_ticks
        )

        cbar.set_label("S/N", fontsize=18)
        cbar.ax.tick_params(labelsize=17)

    # -------------------------------------------------------------------------
    # Hide unused axes
    # -------------------------------------------------------------------------
    for j in range(n_maps, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    return fig, axes, peak_summary


