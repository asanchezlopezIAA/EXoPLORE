"""
exoplore.analysis.stats
=======================

Statistical analysis tools for high-resolution cross-correlation spectroscopy.
"""


# Role in the calculation
# Plot statistical summaries and correlations, and convert evidence ratios.
# Night-study calculations and Pearson summaries come from ccf/statistics.py.
# SYSREM pass selection comes from pipelines/sysrem.py.


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, shutil

from exoplore.ccf.statistics import statistical_study, get_corr_coeff
# Per-order pass selection uses the SYSREM implementation shared by the runner.
from exoplore.pipelines.sysrem import get_SYSREM_its_ordbyord
from exoplore.io.utils import format_number, bootstrap_corrcoeffs
from exoplore.ccf.compute import get_shifted_ccf_matrix


def bayes_factor_to_sigma(bayes_factor):
    """Map a Bayes factor ``B = Z_model / Z_null`` (>1) to a frequentist
    detection significance σ, following the **minimum-Bayes-factor bound**.

    This is a generic Bayesian→frequentist calibration, not specific to any one
    analysis: the underlying result is Sellke, Bayarri & Berger (2001). It was
    introduced to exoplanet-atmosphere retrievals by Benneke & Seager (2013) and
    written in the now-standard form by Welbanks & Madhusudhan (2021, Eq. 17)
    (the conversion Cheverall et al. 2026 use to quote "≤3.9σ"). Because it is a
    *lower* bound on the Bayes factor at a given p-value, the implied σ is an
    **upper bound** on the detection significance.

    The bound relates B to the minimum false-alarm probability p via

        B = -1 / (e · p · ln p)        (valid for B > 1, p < 1/e)

    solved for p with the lower branch of the Lambert-W function
    (``ln p = W_{-1}(-1/(eB))``), then p → two-tailed Gaussian σ via
    ``σ = √2 · erfcinv(p)``.

    Parameters
    ----------
    bayes_factor : float
        B = Z_model / Z_null (linear, not ln).  B ≤ 1 returns σ = 0.

    Returns
    -------
    (sigma, p) : tuple of float
        Detection significance in σ and the corresponding p-value.
    """
    from scipy.special import lambertw, erfcinv
    B = float(bayes_factor)
    if not np.isfinite(B) or B <= 1.0:
        return 0.0, 1.0
    c = -1.0 / (np.e * B)                            # p ln p = c, c in (-1/e, 0)
    p = float(np.real(np.exp(lambertw(c, k=-1))))    # small-p (W_{-1}) branch
    p = min(max(p, 1e-300), 1.0)
    sigma = float(np.sqrt(2.0) * erfcinv(p))
    return sigma, p



def plot_stats(
    stats,
    kp_lim_inf,
    kp_lim_sup,
    kp_step,
    vrest_lim_inf,
    vrest_lim_sup,
    vrest_step,
    sn_lim_inf,
    sn_lim_sup,
    sn_lim_step,
    binwidth_sn,
    binwidth_kp,
    binwidth_v_rest,
    significance_metric,
    v_rest,
    auto_lims=False,
    show_SN_quantile=False,
    shade_true_region=False,
    mark_true_values=False,
    kp_shade_width=None,
    vrest_shade_width=None,
    show_dist_CC_values=True,
    show_plot=False,
    save_plot=True,
    CCF_Noise=False,
    *,
    reference_kp_kms,
    reference_vrest_kms,
    plots_dir,
    simulation_name,
    kp_max_kms,
    velocity_step_kms,
    systemic_velocity_kms
):
    """
    Corner plot of statistics related to simulating n nights of
    observation.

    Args:
        stats: Numpy array of shape (n_samples, 3) containing 
               the statistics.
        plot_name: Name of the plot file to be saved.
        kp_lim_inf: Lower limit of the Kp range for the plots.
        kp_lim_sup: Upper limit of the Kp range for the plots.
        kp_step: Step size for Kp values.
        vrest_lim_inf: Lower limit of the Vrest range for the plots.
        vrest_lim_sup: Upper limit of the Vrest range for the plots.
        vrest_step: Step size for Vrest values.
        sn_lim_inf: Lower limit of the S/N range for the plots.
        sn_lim_sup: Upper limit of the S/N range for the plots.
        sn_lim_step: Step size for S/N values.
        binwidth_sn: Bin width for S/N histograms.
        binwidth_kp: Bin width for Kp histograms.
        binwidth_v_rest: Bin width for Vrest histograms.
        kp_shade_width: Width of the shaded region for Kp.
        vrest_shade_width: Width of the shaded region for Vrest.
        show_dist_CC_values: Boolean flag indicating whether to plot 
                             distributions of CC values at the true 
                             values and away from them and from the 
                             tellurics
    Returns:
        None

    """
    # Lazy imports: gridspec and seaborn are only needed for this corner plot,
    # so they are not imported at module level (keeping seaborn optional for the
    # rest of the analysis module).
    from matplotlib import gridspec
    import seaborn as sns

    plt.close()
    gs = gridspec.GridSpec(3, 3)

    fig = plt.figure(figsize=(12,8))

    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    # Histograms with or without quantile        
    if show_SN_quantile:
        a = sns.ecdfplot(stats[:,0], complementary = True, color='gold', 
                         alpha = 0.)
        l1 = a.lines[0]
        x1 = l1.get_xydata()[:,0]
        y1 = l1.get_xydata()[:,1]
        ax.fill_between(x1,y1, color="gold", alpha=0.5, label='Quantile')
        ax.plot(x1,y1, color='gold', marker='o', markersize=0.2, alpha=0.1)
        sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                     stat='density', label='Max. S/N', alpha = 0.7)
    else:
        sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                     stat='density', label='Max. S/N')     
    ax.set_title(f"S/N = {np.round(np.mean(stats[:,0]), 1)}", fontsize = 20)
    ax.set_ylabel('', fontsize = 17)
    ax.legend(prop={'size': 10})
    if not auto_lims:
        xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
        ax.set_xticks(xticks)
        ax.set_xlim([sn_lim_inf, sn_lim_sup])
    else:
        xticks = np.arange(np.floor(stats[:,0].min()), np.ceil(stats[:,0].max()) + 2, 2)
        ax.set_xticks(xticks)
        ax.set_xlim([stats[:,0].min()-1.5, stats[:,0].max()+1.5])
    ax.grid(True, which='both')
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    
    ax = plt.subplot(gs[1, 1]) # row 1, col 1
    sns.histplot(stats[:,1], binwidth = binwidth_kp, color = 'black',
                 stat='density',
                 label='Max. $K_P$')
    ax.set_title(f"K$_P$ = {np.round(np.mean(stats[:,1]), 1)}", fontsize = 20)
    ax.grid(True, which='both')
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
    ax.set_xticks(xticks)
    ax.set_xlim([kp_lim_inf, kp_lim_sup])
    
    if shade_true_region: 
        # Shade the region of interest
        plt.axvspan(reference_kp_kms - kp_shade_width,
                    reference_kp_kms + kp_shade_width,
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = reference_kp_kms, color = 'r', linestyle = '--',
                    linewidth = 2)

    ax = plt.subplot(gs[2, 2]) # row 2, col 2
    # Identical recovered velocities occupy a single histogram bin.
    if np.all(stats[:,2] == reference_vrest_kms) or all(element == stats[:,2][0] for element in stats[:,2]):
        hist, bins = np.histogram(stats[:,2], bins=[0, 1])
        plt.bar(bins[:-1], hist, width=1)
    else:
        sns.histplot(stats[:,2], binwidth = binwidth_v_rest, color='black',
                      stat='density',
                      label='Max. $V_{rest}$')
    ax.set_title(f"V$rest$ = {np.round(np.mean(stats[:,2]), 1)}", fontsize = 20)
    ax.set_xlabel('V$_{rest}$ (km/s)', fontsize = 17)
    ax.grid(True, which='both')
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    if not auto_lims:
        xticks = np.arange(vrest_lim_inf, vrest_lim_sup + vrest_step, 
                           vrest_step)
        ax.set_xticks(xticks)
        ax.set_xlim([vrest_lim_inf, vrest_lim_sup])
    else:
        xticks = np.arange(np.floor(stats[:,2].min()/5)*5, np.ceil(stats[:,2].max()/5)*5+5, 5)
        ax.set_xticks(xticks)
        ax.set_xlim([np.round(reference_vrest_kms)-15, np.round(reference_vrest_kms)+15])
    
    if shade_true_region: 
        # Shade the region of interest
        plt.axvspan(reference_vrest_kms - vrest_shade_width,
                    reference_vrest_kms + vrest_shade_width,
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = reference_vrest_kms, color = 'r', linestyle = '--',
                    linewidth = 2)

    
    # Scatter plots
    ax = plt.subplot(gs[1, 0]) # row 1, col 0
    plt.scatter(stats[:, 0], stats[:, 1],c=stats[:,0], 
                cmap='gray', s = 17)  
    ax.set_ylabel('K$_P$ (km/s)', fontsize = 17)
    ax.grid(True, which='both')
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    if not auto_lims:
        xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
        ax.set_xticks(xticks)
        ax.set_xlim([sn_lim_inf, sn_lim_sup])
        yticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_yticks(yticks)
        ax.set_ylim([kp_lim_inf, kp_lim_sup])
    else:
        xticks = np.arange(np.floor(stats[:,0].min()), np.ceil(stats[:,0].max()) + 2, 2)
        ax.set_xticks(xticks)
        ax.set_xlim([stats[:,0].min()-1.5, stats[:,0].max()+1.5])
        yticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_yticks(yticks)
        ax.set_ylim([kp_lim_inf, kp_lim_sup])
        
    
    if shade_true_region: 
        # Shade the region of interest
        plt.fill_betweenx([reference_kp_kms - kp_shade_width,
                           reference_kp_kms + kp_shade_width],
                          sn_lim_inf,  sn_lim_sup, color='coral', 
                          alpha=0.4)
    elif mark_true_values:
        plt.axhline(y = reference_kp_kms, color = 'r', linestyle = '--',
                    linewidth = 2)

    ax = plt.subplot(gs[2,0]) # row 2, col 0
    plt.scatter(stats[:, 0], stats[:, 2],c=stats[:, 0], 
                cmap='gray', s = 17)
    ax.set_ylabel("V$_{rest}$ (km/s)", fontsize = 17)
    ax.set_xlabel('S/N', fontsize = 20)
    ax.grid(True, which='both')
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    if not auto_lims:
        xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
        ax.set_xticks(xticks)
        ax.set_xlim([sn_lim_inf, sn_lim_sup])
        yticks = np.arange(vrest_lim_inf, vrest_lim_sup+vrest_step, vrest_step)
        ax.set_yticks(yticks)
        ax.set_ylim([vrest_lim_inf, vrest_lim_sup])
    else:
        xticks = np.arange(np.floor(stats[:,0].min()), np.ceil(stats[:,0].max()) + 2, 2)
        ax.set_xticks(xticks)
        ax.set_xlim([stats[:,0].min()-1.5, stats[:,0].max()+1.5])
        yticks = np.arange(np.floor(stats[:,2].min()/5)*5, np.ceil(stats[:,2].max()/5)*5+5, 5)
        ax.set_yticks(yticks)
        ax.set_ylim([np.round(reference_vrest_kms)-15, np.round(reference_vrest_kms)+15])
    
    if shade_true_region: 
        # Shade the region of interest
        plt.fill_betweenx([reference_vrest_kms - vrest_shade_width,
                           reference_vrest_kms + vrest_shade_width],
                          sn_lim_inf,  sn_lim_sup, color='coral', 
                          alpha=0.4)
    elif mark_true_values:
        plt.axhline(y = reference_vrest_kms, color = 'r', linestyle = '--',
                    linewidth = 2)


    ax = plt.subplot(gs[2,1]) # row 2, col 1
    plt.scatter(stats[:, 1], stats[:, 2],c=stats[:, 0], 
                cmap='gray', s = 17)
    ax.set_xlabel('K$_P$ (km/s)', fontsize = 17)
    ax.grid(True, which='both')
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    if not auto_lims:
        xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_xticks(xticks)
        ax.set_xlim([kp_lim_inf, kp_lim_sup])
        yticks = np.arange(vrest_lim_inf, vrest_lim_sup+vrest_step, vrest_step)
        ax.set_yticks(yticks)
        ax.set_ylim([vrest_lim_inf, vrest_lim_sup])
    else:
        xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_xticks(xticks)
        ax.set_xlim([kp_lim_inf, kp_lim_sup])
        yticks = np.arange(np.floor(stats[:,2].min()/5)*5, np.ceil(stats[:,2].max()/5)*5+5, 5)
        ax.set_yticks(yticks)
        ax.set_ylim([np.round(reference_vrest_kms)-15, np.round(reference_vrest_kms)+15])
    if shade_true_region: 
        # Shade the region of interest
        x_values = plt.gca().get_xlim()
        plt.fill_betweenx([reference_vrest_kms - vrest_shade_width,
                           reference_vrest_kms + vrest_shade_width],
                          x_values[0], x_values[1],
                          color='coral', 
                          alpha=0.4)
        y_values = plt.gca().get_ylim()
        plt.fill_betweenx([y_values[0], y_values[1]],
              reference_kp_kms - kp_shade_width,
              reference_kp_kms + kp_shade_width,
              color='coral',
              alpha=0.4)
    elif mark_true_values:
        plt.axhline(y = reference_vrest_kms, color = 'r', linestyle = '--',
                    linewidth = 2.5)
        plt.axvline(x = reference_kp_kms, color = 'r', linestyle = '--',
                    linewidth = 2.5)

    fig.tight_layout()
    
    # Save it and show it
    if save_plot and not CCF_Noise: 
        plt.savefig(f"{plots_dir}Corner_plot_{simulation_name}.pdf")
        plt.savefig(f"{plots_dir}Corner_plot_{simulation_name}.png", transparent = True)
    elif save_plot and CCF_Noise:
        plt.savefig(f"{plots_dir}Corner_plot_noise_{simulation_name}.pdf")
    if show_plot: plt.show()
    plt.close()
    
    
    
    
    
    # And now the dist. of CC values, if specified by user
    if show_dist_CC_values:
        gs = gridspec.GridSpec(1, 4)
        fig = plt.figure(figsize=(16,4))
        

        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        true_area = significance_metric[np.argwhere(v_rest == reference_vrest_kms)[0][0] - 5 : np.argwhere(v_rest == reference_vrest_kms)[0][0] + 5,
                                      int(reference_kp_kms+kp_max_kms+1) - 40 : int(reference_kp_kms+kp_max_kms+1) + 60, :]
        
        sns.histplot(np.ndarray.flatten(true_area), 
                     color = 'black', stat='density')
        ax.set_title('Area around true', fontsize = 16)
        if not auto_lims:
            xticks = np.arange(-6, 12.1, 3)
            ax.set_xticks(xticks)
            ax.grid(True, which='both')
            #ax.legend(prop={'size': 10})
            ax.set_ylabel('', fontsize = 17)
            ax.set_xlabel('S/N', fontsize = 17)
            ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                           labelsize=16)
            ax.set_xlim([sn_lim_inf, sn_lim_sup])
        else:
            xticks = np.arange(np.floor(true_area.min()/2)*2, np.ceil(true_area.max()/2)*2+2, 2)
            ax.set_xticks(xticks)
            ax.grid(True, which='both')
            #ax.legend(prop={'size': 10})
            ax.set_ylabel('', fontsize = 17)
            ax.set_xlabel('S/N', fontsize = 17)
            ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                           labelsize=16)
            ax.set_xlim([np.floor(true_area.min())-2, np.ceil(true_area.max())+2])
            
        
        
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        plot_variable = np.ndarray.flatten(significance_metric[np.argwhere(v_rest == reference_vrest_kms)[0][0], int(reference_kp_kms+kp_max_kms+1), :])
        sns.histplot(
            plot_variable,
            color = 'k', stat='density'
            )
        ax.set_title('True $K_p-V_{rest}$', fontsize = 16)
        xticks = np.arange(
            np.floor(plot_variable.min()/2)*2, 
            np.ceil(plot_variable.max()/2)*2+2, 2
            )
        ax.set_xticks(xticks)
        ax.grid(True, which='both')
        #ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.set_xlabel('S/N', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        ax.set_xlim([np.floor(plot_variable.min())-2, np.ceil(plot_variable.max())+2])
        
        
        
        ax = plt.subplot(gs[0, 2]) # row 0, col 1
        
        telluric_area = np.ndarray.flatten(
            significance_metric[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) + 15),
                                320 - 30 : 320 + 30, :]
            )
        sns.histplot(telluric_area, 
                     color = 'black', stat='density')
        ax.set_title('Area around tellurics', fontsize = 16)
        xticks = np.arange(
            np.floor(telluric_area.min()/2)*2, 
            np.ceil(telluric_area.max()/2)*2+2, 2
            )
        ax.set_xticks(xticks)
        ax.set_xlim([np.floor(telluric_area.min())-2, np.ceil(telluric_area.max())+2])
        ax.grid(True, which='both')
        #ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.set_xlabel('S/N', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        
        ax = plt.subplot(gs[0, 3]) # row 0, col 2
        
        # Removing tellurics
        # Removing tellurics
        away_from_signal_and_tellurics = np.delete(
            significance_metric, 
            np.s_[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) + 15)],
            axis=0
            )
        
        away_from_signal_and_tellurics = np.delete(
            significance_metric, np.s_[320-40:320+40], 
            axis=1
            )
        # Removing planet signal
        away_from_signal_and_tellurics = np.delete(
            significance_metric, 
            np.s_[np.argwhere(v_rest == reference_vrest_kms)[0][0]-5:np.argwhere(v_rest == reference_vrest_kms)[0][0]+5],
            axis=0
            )
        
        away_from_signal_and_tellurics = np.delete(
            significance_metric, 
            np.s_[int(reference_kp_kms+kp_max_kms+1) - 40:int(reference_kp_kms+kp_max_kms+1) + 40],
            axis=1
            )
        
        sns.histplot(np.ndarray.flatten(away_from_signal_and_tellurics), 
                     color = 'black', stat='density')
        ax.set_title('Away from signal and tellurics', fontsize = 16)
        ax.grid(True, which='both')
        xticks = np.arange(
            np.floor(away_from_signal_and_tellurics.min()/2)*2, 
            np.ceil(away_from_signal_and_tellurics.max()/2)*2+2, 2
            )
        ax.set_xticks(xticks)
        ax.set_xlim([np.floor(away_from_signal_and_tellurics.min())-2, np.ceil(away_from_signal_and_tellurics.max())+2])
        #ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.set_xlabel('S/N', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        fig.tight_layout()
        
        # Save it and show it
        if save_plot and not CCF_Noise: 
            plt.savefig(f"{plots_dir}CC_distributions_{simulation_name}.pdf")
        elif save_plot and CCF_Noise:
            plt.savefig(f"{plots_dir}CC_distributions_noise_{simulation_name}.pdf")
        if show_plot: plt.show()
        plt.close()
    
    return






def get_CCvalues_dist(
    ccf_matrix,
    v_ccf,
    v_rest,
    in_trail_pix,
    night1,
    night2,
    with_signal,
    kp_range,
    phase,
    pixels_left_right,
    ccf_v_step,
    Kp,
    name,
    save_plot,
    *,
    systemic_velocity_kms,
    berv_kms,
    plots_dir,
    significant_eccentricity,
    eccentricity,
    argument_of_periastron_deg,
    sysrem_iterations
):
    """Plot distributions of in-trail and out-of-trail CCF values for two nights.

    Shifts the CCF matrix into the planet rest frame at trial orbital velocity
    ``Kp`` for two specified nights, then separates pixels into an in-trail
    distribution (velocity pixels within ±in_trail_pix/2 of the planet rest
    frame zero) and an out-of-trail distribution.  Overplotting the two nights
    reveals whether the in-trail signal is consistently brighter than the noise.

    Parameters
    ----------
    systemic_velocity_kms, berv_kms
        Systemic velocity and barycentric correction in km/s.
    plots_dir : str
        Output directory including its trailing path separator.
    significant_eccentricity, eccentricity, argument_of_periastron_deg
        Orbit settings used to align the exposure CCFs.
    sysrem_iterations : int
        Stored SYSREM pass count for optimisation arrays.
    ccf_matrix : ndarray, shape (n_v_rest, n_kp, n_nights)
        Co-added CCF significance map.
    v_ccf : ndarray
        Earth-frame CCF velocity grid in km/s.
    v_rest : ndarray
        Planet-rest-frame velocity axis in km/s.
    in_trail_pix : int
        Width of the in-trail window in pixels (must be odd).
    night1, night2 : int
        Night indices to compare (e.g., the best and worst nights).
    with_signal : ndarray
        Indices of in-transit exposures.
    kp_range : ndarray
        Grid of trial Kp values in km/s.
    phase : ndarray
        Orbital phase array.
    pixels_left_right : int
        Half-width of the rest-frame shift window in pixels.
    ccf_v_step : float
        CCF velocity step in km/s.
    Kp : int
        Kp index (offset from kp_range centre) at which to evaluate the map.
    name : str
        Label used in the saved figure filenames.
    save_plot : bool
        If True, write figures to the simulation output directory.
    """
    from matplotlib import gridspec
    import seaborn as sns

    if in_trail_pix % 2 == 0:
        raise ValueError("The width of the in-trail distribution\
                         should be an odd number")
    # Get the shifted matrices at the desired Kp, no winds considered
    ccf_matrix_shifted1 = get_shifted_ccf_matrix(
        with_signal,
        v_rest,
        v_ccf,
        kp_range,
        phase,
        systemic_velocity_kms,
        berv_kms,
        pixels_left_right,
        ccf_v_step,
        ccf_matrix[:, :, night1],
        significant_eccentricity=significant_eccentricity,
        eccentricity=eccentricity,
        argument_of_periastron_deg=argument_of_periastron_deg,
        sysrem_iterations=sysrem_iterations,
    )[:, :, Kp+int(np.floor(len(kp_range)/2))]
    ccf_matrix_shifted2 = get_shifted_ccf_matrix(
        with_signal,
        v_rest,
        v_ccf,
        kp_range,
        phase,
        systemic_velocity_kms,
        berv_kms,
        pixels_left_right,
        ccf_v_step,
        ccf_matrix[:, :, night2],
        significant_eccentricity=significant_eccentricity,
        eccentricity=eccentricity,
        argument_of_periastron_deg=argument_of_periastron_deg,
        sysrem_iterations=sysrem_iterations,
    )[:, :, Kp+int(np.floor(len(kp_range)/2))]

    # Create the array of velocity indices centred in exoplanet signal
    # (in the planet's rest-frame, that is 0) for the in-trail distribution
    left_right = in_trail_pix // 2
    zero_vel_idx = np.where(v_rest == 0)[0][0]
    intrail_idx = np.arange(
        zero_vel_idx-left_right, zero_vel_idx+left_right+1, 1
        )
    outtrail_idx = np.delete(np.arange(len(v_rest)), intrail_idx)
    
    # In-trail distribution of CC values
    in_trail_data1 = np.ndarray.flatten(
                        ccf_matrix_shifted1[intrail_idx, :]
                        )
    in_trail_data2 = np.ndarray.flatten(
                        ccf_matrix_shifted2[intrail_idx, :]
                        )
    # Out-of-trail distributions of CC values
    out_trail_data1 = np.ndarray.flatten(
                        ccf_matrix_shifted1[outtrail_idx, :]
                        )
    out_trail_data2 = np.ndarray.flatten(
                        ccf_matrix_shifted2[outtrail_idx, :]
                        )
    
    # Plot
    plt.close()
    count_out1, bins_out1, ignored_out1 = plt.hist(
        out_trail_data1, 28, alpha = 0.5, color = 'k', 
        histtype = 'bar', linewidth = 1.6, label = f"Night {night1}",
        )
    count_out2, bins_out2, ignored_out2 = plt.hist(
        out_trail_data2, 28, alpha = 0.5, color = 'gold', 
        histtype = 'bar', linewidth = 1.6, label = f"Night {night2}",
        )
    if save_plot:
        os.makedirs(f"{plots_dir}statistical/ccf_minmax", exist_ok=True)
    plt.title('Out-of-trail Cross correlation values', fontsize = 15)
    plt.xlabel("Cross correlation value", fontsize = 17)
    plt.ylabel("Frequency", fontsize = 17)
    plt.tick_params(axis = 'both', width = 1.8, direction = 'in',
            labelsize=15)
    plt.legend(loc='best', prop={'size': 12})

    if save_plot:
        plt.savefig(
            f"{plots_dir}statistical/"
            f"ccf_minmax/{name}_out_trail_dist.png",
            bbox_inches='tight')

    plt.show()
    plt.close()
    
    # Plot
    plt.close()
    count_in1, bins_in1, ignored_in1 = plt.hist(
        in_trail_data1, 28, alpha = 0.5, color = 'k', 
        histtype = 'bar', linewidth = 1.6, label = f"Night {night1}"
        )
    count_in2, bin_in2, ignored_in2 = plt.hist(
        in_trail_data2, 28, alpha = 0.5, color = 'gold', 
        histtype = 'bar', linewidth = 1.6, label = f"Night {night2}"
        )
    plt.title('In-trail Cross correlation values', fontsize = 15)
    plt.xlabel("Cross correlation value", fontsize = 17)
    plt.ylabel("Frequency", fontsize = 17)
    plt.tick_params(axis = 'both', width = 1.8, direction = 'in', 
            labelsize=15)
    plt.legend(loc='best', prop={'size': 12})
    
    if save_plot:
        plt.savefig(
            f"{plots_dir}statistical/"
            f"ccf_minmax/{name}_in_trail_dist.png",
            bbox_inches='tight')

    plt.show()
    plt.close()
            
    return


def compare_empirical_SN(
    matrix,
    n_pixels,
    night1,
    night2,
    name,
    save_plot,
    *,
    n_orders,
    order_indices,
    plots_dir
):
    """Plot and compare per-order empirical S/N between two observing nights.

    For each spectral order computes the empirical signal-to-noise ratio as
    ``mean_spectrum / std_spectrum`` (across exposures), then plots the
    mean S/N per order for two selected nights on the same axes.  This
    diagnostic is used to identify orders or nights with anomalous noise
    properties before combining.

    Parameters
    ----------
    matrix : ndarray, shape (n_nights, n_orders, n_spectra, n_pixels)
        Spectral matrix cube.
    n_orders
        Number of selected spectral orders.
    order_indices
        Physical spectral-order indices.
    plots_dir
        Output directory including its trailing path separator.
    n_pixels : int
        Number of wavelength pixels per order.
    night1, night2 : int
        Night indices to compare.
    name : str
        Label used in saved figure filenames.
    save_plot : bool
        If True, save the figure to the simulation output directory.
    """
    import seaborn as sns

    sn_night1 = np.zeros((n_orders, n_pixels))
    sn_night2 = np.zeros((n_orders, n_pixels))
    for h in range(n_orders):
        master_spectrum1 = np.mean(matrix[night1, h, :, :], axis = 0)
        master_spectrum2 = np.mean(matrix[night2, h, :, :], axis = 0)
        std_noise1 = np.std(matrix[night1, h, :, :], axis = 0)
        std_noise2 = np.std(matrix[night2, h, :, :], axis = 0)
        sn_night1[h,:] = master_spectrum1 / std_noise1
        sn_night2[h,:] = master_spectrum2 / std_noise2
        
    # Mean S/N per order
    sn_night1_mean = np.mean(sn_night1, axis = 1)
    sn_night2_mean = np.mean(sn_night2, axis = 1)
    
    if save_plot:
        os.makedirs(f"{plots_dir}statistical", exist_ok=True)
    # Plot
    plt.plot(order_indices, sn_night1_mean, 'ko-', label = f"Night {night1}")
    plt.plot(order_indices, sn_night2_mean, 'ro-', label = f"Night {night2}")
    plt.xlabel("Empirical S/N", fontsize = 17)
    plt.ylabel("Spectral order", fontsize = 17)
    plt.tick_params(axis = 'both', width = 1.8, direction = 'in', 
            labelsize=15)
    plt.legend(loc='best', prop={'size': 12})
    
    if save_plot:
        plt.savefig(
            f"{plots_dir}statistical/"
            f"{name}_empirical_SN.png",
            bbox_inches='tight')

    plt.show()
    plt.close()
    
    return 


def plot_std_errors(save_plot, error, prop_error, stats, *, n_nights, plots_dir):
    """Plot the per-night standard deviations of pipeline noise residuals.

    Computes the standard deviation of the residual error and propagated
    noise cubes across orders, spectra, and pixels for each night, then
    plots both quantities alongside the per-night peak CCF S/N from
    ``stats``.  Useful for diagnosing nights where the noise model
    disagrees with the actual pipeline residuals.

    Parameters
    ----------
    n_nights
        Number of observing nights.
    plots_dir
        Output directory including its trailing path separator.
    save_plot : bool
        If True, save the figure as a PDF.
    error : ndarray, shape (n_nights, n_orders, n_spectra, n_pixels)
        Residual error array (data - model) from the pipeline.
    prop_error : ndarray, shape (n_nights, n_orders, n_spectra, n_pixels)
        Propagated noise (σ) array from the pipeline.
    stats : ndarray, shape (n_nights, 3)
        Per-night detection statistics. column 0 is the peak CCF S/N.
    """
    std_error = np.zeros((n_nights))
    std_prop_error = np.zeros((n_nights))
    for n in range(n_nights):
        std_prop_error[n] = np.std(prop_error[n,:,:,:])
        std_error[n] = np.std(error[n,:,:,:])
        
    if save_plot:
        os.makedirs(f"{plots_dir}statistical", exist_ok=True)
    # Original uncertainties
    plt.close()
    plt.plot(std_error[1:],stats[1:,0], marker = 'o', color = 'k',linewidth = 0)
    plt.xlabel(r'Mean stddev($\epsilon_{\lambda}$) per night', fontsize=17)
    plt.ylabel('S/N', fontsize=17)
    plt.tick_params(axis='both', width=1.5, direction='in', 
                    labelsize=17)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    if save_plot:
        plt.savefig(
            f"{plots_dir}statistical/"
            f"std_original_additive_error.png",
            bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    # Propagated uncertainties
    plt.plot(std_prop_error[1:],stats[1:,0],marker = 'o', color =  'goldenrod', linewidth = 0)
    plt.xlabel(r'Mean stddev($R(\sigma)$) per night', fontsize=17)
    plt.ylabel('S/N', fontsize=17)
    plt.tick_params(axis='both', width=1.5, direction='in', 
                    labelsize=17)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_axisbelow(True)  # Ensure grid is behind the data

    if save_plot:
        plt.savefig(
            f"{plots_dir}statistical/"
            f"std_propagated_error.png",
            bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    
    return 


def compare_KpVr_dist(
    v_rest,
    ccf_matrix,
    night1,
    night2,
    saveplot,
    SNR=True,
    *,
    reference_kp_kms,
    kp_max_kms,
    reference_vrest_kms,
    velocity_step_kms,
    systemic_velocity_kms
):
    """Compare the Kp-Vsys signal distributions between two nights.

    Produces a four-panel diagnostic figure showing: the Kp-Vsys maps for
    ``night1`` and ``night2`` zoomed around the expected planet position,
    a scatter plot of signal values in the planet region between the two
    nights (useful for spotting outliers), and a histogram of the
    per-velocity CCF values for each night.  Helps identify nights with
    anomalously strong or weak detections.

    Parameters
    ----------
    reference_kp_kms
        Configured orbital velocity amplitude in km/s.
    kp_max_kms
        Maximum trial Kp in km/s.
    reference_vrest_kms
        Configured reference velocity offset in km/s.
    velocity_step_kms
        Velocity grid spacing in km/s.
    systemic_velocity_kms
        Systemic velocity in km/s.
    v_rest : ndarray, shape (n_v_rest,)
        Planet rest-frame velocity axis in km/s.
    ccf_matrix : ndarray, shape (n_v_rest, n_kp_steps, n_nights)
        Kp-Vsys significance map.
    night1, night2 : int
        Indices of the two nights to compare (0-based).
    saveplot : bool
        If True, save the figure as a PDF.
    SNR : bool
        If True, treat ``ccf_matrix`` values as S/N. If False, as raw CCF.
    """
    from matplotlib import gridspec
    import seaborn as sns

    plt.close()
    gs = gridspec.GridSpec(1, 4)
    fig = plt.figure(figsize=(16,4))
    

    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    signal_area1 = ccf_matrix[np.argwhere(v_rest == reference_vrest_kms)[0][0] - 5 : np.argwhere(v_rest == reference_vrest_kms)[0][0] + 5,
                                  int(reference_kp_kms+kp_max_kms+1) - 40 : int(reference_kp_kms+kp_max_kms+1) + 60, night1]
    signal_area2 = ccf_matrix[np.argwhere(v_rest == reference_vrest_kms)[0][0] - 5 : np.argwhere(v_rest == reference_vrest_kms)[0][0] + 5,
                                  int(reference_kp_kms+kp_max_kms+1) - 40 : int(reference_kp_kms+kp_max_kms+1) + 60, night2]

    
    sns.histplot(np.ndarray.flatten(signal_area1), 
                 color = 'black', stat='density', 
                 label='Night_max', alpha = 0.6)
    sns.histplot(np.ndarray.flatten(signal_area2), 
                 color = 'goldenrod', stat='density', 
                 label='Night_min', alpha = 0.6)
    if SNR: 
        xticks = np.arange(-6, 12.1, 3)
        ax.set_xticks(xticks)
        ax.set_title('S/N area \naround injected signal')
        ax.set_xlabel('S/N', fontsize = 17)
    else:
        ax.set_title('CC in area \naround injected signal')
        ax.set_xlabel('CC values', fontsize = 17)
    ax.grid(True, which='both')
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    
    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    telluric_area1 = np.ndarray.flatten(
        ccf_matrix[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) + 15),
                   320 - 30 : 320 + 30, night1]
        )
    telluric_area2 = np.ndarray.flatten(
        ccf_matrix[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) + 15),
                   320 - 40 : 320 + 40, night2]
        )
    sns.histplot(telluric_area1, 
                 color = 'black', stat='density', 
                 label='Night_max', alpha = 0.6)
    sns.histplot(telluric_area2, 
                 color = 'goldenrod', stat='density', 
                 label='Night_min', alpha = 0.6)
    if SNR:
        xticks = np.arange(-6, 6.1, 2)
        ax.set_xticks(xticks)
        ax.set_xlim(-5,5)
        ax.set_title('S/N tellurics \n($K_p=V_{rest}$=0 km/s)')
        ax.set_xlabel('S/N', fontsize = 17)
    else:
        ax.set_title('CC tellurics \n($K_p=V_{rest}$=0 km/s)')
        ax.set_xlabel('CC values', fontsize = 17)
    ax.grid(True, which='both')
    
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    
    ax = plt.subplot(gs[0, 2]) # row 0, col 2
    
    # Removing tellurics
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, np.s_[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(systemic_velocity_kms) / velocity_step_kms) + 15)],
        axis=0
        )
    
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, np.s_[320-40:320+40], 
        axis=1
        )
    # Removing planet signal
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, 
        np.s_[np.argwhere(v_rest == reference_vrest_kms)[0][0]-5:np.argwhere(v_rest == reference_vrest_kms)[0][0]+5],
        axis=0
        )
    
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, np.s_[int(reference_kp_kms+kp_max_kms+1) - 40:int(reference_kp_kms+kp_max_kms+1) + 40],
        axis=1
        )
    
    away_from_signal_and_tellurics1 = away_from_signal_and_tellurics[:, :, night1]
    away_from_signal_and_tellurics2 = away_from_signal_and_tellurics[:, :, night2]
    stddev1 = np.round(np.std(away_from_signal_and_tellurics1), 2)
    stddev2 = np.round(np.std(away_from_signal_and_tellurics2), 2)
    amp1 = np.round(np.ptp(away_from_signal_and_tellurics1), 2)
    amp2 = np.round(np.ptp(away_from_signal_and_tellurics2), 2)

    sns.histplot(np.ndarray.flatten(away_from_signal_and_tellurics1), 
                 color = 'black', stat='density', 
                 label=f'Night_max\n stddev = {stddev1},\n Amplitude = {amp1}', alpha = 0.6)
    sns.histplot(np.ndarray.flatten(away_from_signal_and_tellurics2), 
                 color = 'goldenrod', stat='density', 
                 label=f'Night_min\n stddev = {stddev2},\n Amplitude = {amp2}', alpha = 0.6)
    ax.grid(True, which='both')
    
    if SNR:
        xticks = np.arange(-6, 6.1, 2)
        ax.set_xticks(xticks)
        ax.set_xlim(-5,5)
        ax.set_title('S/N Away \nfrom signal \nand tellurics')
        ax.set_xlabel('S/N', fontsize = 17)
    else:
        ax.set_title('CC Away \nfrom signal \nand tellurics')
        ax.set_xlabel('CC values', fontsize = 17)
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    fig.tight_layout()
    
    # Save it in PDF and png
    if saveplot:
        #plt.savefig(f"KpVr_distributions.pdf")
        plt.savefig("KpVr_distributions.png", transparent=True)
        plt.show()
        plt.close

    return






def compare_correlations(
    corr_x,
    corr_y,
    filename_flag,
    plotname,
    xlabel,
    ylabel,
    title="",
    plot_lims=None,
    show_plot=True,
    save_plot=True,
    *,
    correlations_dir,
    simulation_name
):
    """Scatter-plot two per-night arrays and compute their Pearson correlation.

    Plots ``corr_y`` vs ``corr_x`` with a linear best-fit line and annotates
    with the Pearson correlation coefficient and its bootstrap standard error.
    Typical use is to compare the S/N distribution from the noise-only CCF
    against that from the signal+noise CCF to assess how strongly noise
    properties predict detection significance.

    Parameters
    ----------
    correlations_dir
        Directory for correlation diagnostics.
    simulation_name
        Name used in output filenames.
    corr_x, corr_y : ndarray, shape (n_nights,)
        The two arrays to correlate and plot.
    filename_flag : str
        String tag (e.g., noise-scaling factor) appended to output filenames.
    plotname : str
        Base name for the saved PDF figure.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Figure title.
    plot_lims : list of float or None
        ``[min, max]`` used for both axes.  If None, defaults to ±0.25.
    show_plot, save_plot : bool
        Display and/or save the figure.

    Returns
    -------
    original_correlation : float
        Pearson correlation coefficient between ``corr_x`` and ``corr_y``.
    standard_error : float
        Bootstrap standard error of the correlation coefficient.
    """
    # Compute the Pearson correlation coefficient
    original_correlation = np.corrcoef(corr_x, corr_y)[0, 1]
    standard_error = bootstrap_corrcoeffs(corr_x, corr_y)
    
   
    plt.close()
    plt.scatter(corr_x, corr_y, 
                color = 'k', 
                marker='o', edgecolors='k')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if plot_lims == None:
        plt.xlim([-0.25, 0.3])
        plt.ylim([-0.25, 0.3])
    else: 
        plt.xlim([plot_lims[0], plot_lims[1]])
        plt.ylim([plot_lims[0], plot_lims[1]])
    plt.grid()
    plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
    # Calculate the slope and intercept for the line
    slope = original_correlation * np.std(corr_y) / np.std(corr_x)
    intercept = np.mean(corr_y) - slope * np.mean(corr_x)

    # Create the line using the equation of a line (y = mx + b)
    line_x = np.array([corr_x.min(), corr_x.max()])
    line_y = slope * line_x + intercept
    # Plot the line
    plt.plot(line_x, line_y, color='red', linestyle='--', 
             label=f"Correlation: {original_correlation:.3f} ± {standard_error:.3f}") 
    plt.axvline(x = 0, color = 'k', zorder = -1)
    plt.axhline(y = 0, color = 'k', zorder = -1)
    plt.title(title)
    plt.legend()
    if save_plot:
        plt.savefig(f"{correlations_dir}{simulation_name}/{plotname}.pdf")
    if show_plot: plt.show()
    plt.close()
    return original_correlation, standard_error




def perform_correlations_with_noise(
    stats,
    stats_tvalue,
    stats_pvalue,
    stats_planet_pos,  # stats_cc_values,
    # stats_cc_values_planet_pos, stats_cc_values_std,
    # stats_cc_values_std_planet_pos,
    stats_noise,
    stats_tvalue_noise,
    stats_pvalue_noise,
    stats_planet_pos_noise,
    show_plot=False,
    save_plot=True,
    etiqueta="",
    *,
    welch_ttest,
    noise_scaling_factor,
    all_significance_metrics,
    ccf_snr,
    correlations_dir,
    simulation_name
):
    """Correlate signal-detection statistics with noise-only statistics across nights.

    For each significance metric (S/N maximum, Welch t-value, Welch p-value,
    and signal at the true planet position) this function calls
    :func:`compare_correlations` to measure the Pearson correlation between
    the per-night statistics from the noise-only CCF and those from the
    signal+noise CCF.  A high correlation indicates that the nightly noise
    properties dominate the detection variability. A low correlation suggests
    the signal itself drives night-to-night scatter.

    Results are collected into an ``outputs`` dict and saved as a compressed
    NPZ file in the correlations directory.

    Parameters
    ----------
    welch_ttest
        Use the Welch t-test statistic.
    noise_scaling_factor
        Configured multiplicative noise scale.
    all_significance_metrics
        Save separate products for each significance metric.
    ccf_snr
        Use the normalised map S/N statistic.
    correlations_dir
        Directory for correlation diagnostics.
    simulation_name
        Name used in output filenames.
    stats : ndarray, shape (n_nights, 3)
        Signal+noise detection statistics. column 0 is the S/N maximum.
    stats_tvalue, stats_pvalue : ndarray or None
        Welch t-value and p-value statistics arrays, same shape.
    stats_planet_pos : ndarray, shape (n_nights, 3)
        S/N (or σ) at the true planet Kp-Vsys position per night.
    stats_noise : ndarray, shape (n_nights, 3)
        Noise-only detection statistics matching the shape of ``stats``.
    stats_tvalue_noise, stats_pvalue_noise : ndarray or None
        Noise-only t-value and p-value arrays.
    stats_planet_pos_noise : ndarray, shape (n_nights, 3)
        Noise-only S/N at the true planet position.
    show_plot, save_plot : bool
        Display and/or save scatter plots for each metric.
    etiqueta : str
        Optional suffix added to the output NPZ filename when
        ``"All_significance_metrics"`` is True.
    """
    from scipy.stats import norm
    mean_SNR, mean_SNR_error = norm.fit(stats[:,0])
    if not ccf_snr and welch_ttest:
        mean_SNR_tvalue, mean_SNR_error_tvalue = norm.fit(stats_tvalue[:,0])
        mean_SNR_pvalue, mean_SNR_error_pvalue = norm.fit(stats_pvalue[:,0])
        stats_tvalue = stats_tvalue[:, 0]
        stats_pvalue = stats_pvalue[:, 0]
        stats_tvalue_noise = stats_tvalue_noise[:, 0]
        stats_pvalue_noise = stats_pvalue_noise[:, 0]
    elif ccf_snr and not welch_ttest:
        mean_SNR_tvalue = None
        mean_SNR_error_tvalue = None
        mean_SNR_pvalue = None
        mean_SNR_error_pvalue = None
    mean_SNR_planet_pos, mean_SNR_error_planet_pos = norm.fit(stats_planet_pos[:,0])
    stats = stats[:, 0]
    
    stats_noise = stats_noise[:, 0]
    
    stats_planet_pos = stats_planet_pos[:, 0]
    stats_planet_pos_noise = stats_planet_pos_noise[:, 0]
    #mean_cc_values, mean_cc_values_error = norm.fit(stats_cc_values[:,0])
    #mean_cc_values_planet_pos, mean_cc_values_planet_pos_error = norm.fit(stats_cc_values_planet_pos[:,0])
    #mean_cc_std, mean_cc_std_error = norm.fit(stats_cc_values_std[:,0])
    #mean_cc_std_planet_pos, mean_cc_std_planet_pos_error = norm.fit(stats_cc_values_std_planet_pos[:,0])

    filename_flag = format_number(noise_scaling_factor)
    pearson_coeff_SNR_max, pearson_coeff_SNR_error_max = compare_correlations(
        stats_noise,
        stats,
        filename_flag,
        plotname=f"scatter_MAX_{filename_flag}",
        xlabel="",
        ylabel="",
        title="MSS",
        plot_lims=[np.amin([stats_noise, stats]) - 1, np.amax([stats_noise, stats]) + 1],
        show_plot=show_plot,
        save_plot=save_plot,
        correlations_dir=correlations_dir,
        simulation_name=simulation_name,
    )

    if welch_ttest:
        pearson_coeff_SNR_max_tvalue, pearson_coeff_SNR_error_max_tvalue = compare_correlations(
            stats_tvalue_noise,
            stats_tvalue,
            filename_flag,
            plotname=f"scatter_MAX_tvalue_{filename_flag}",
            xlabel="",
            ylabel="",
            title="MSS",
            plot_lims=[
                np.amin([stats_tvalue_noise, stats_tvalue]) - 1,
                np.amax([stats_tvalue_noise, stats_tvalue]) + 1,
            ],
            show_plot=show_plot,
            save_plot=save_plot,
            correlations_dir=correlations_dir,
            simulation_name=simulation_name,
        )
    
        pearson_coeff_SNR_max_pvalue, pearson_coeff_SNR_error_max_pvalue = compare_correlations(
            stats_pvalue_noise,
            stats_pvalue,
            filename_flag,
            plotname=f"scatter_MAX_pvalue_{filename_flag}",
            xlabel="",
            ylabel="",
            title="MSS",
            plot_lims=[
                np.amin([stats_pvalue_noise, stats_pvalue]) - 1,
                np.amax([stats_pvalue_noise, stats_pvalue]) + 1,
            ],
            show_plot=show_plot,
            save_plot=save_plot,
            correlations_dir=correlations_dir,
            simulation_name=simulation_name,
        )
    else:
        pearson_coeff_SNR_max_tvalue = None
        pearson_coeff_SNR_error_max_tvalue = None
        pearson_coeff_SNR_max_pvalue = None
        pearson_coeff_SNR_error_max_pvalue = None
    
    pearson_coeff_SNR_planet_pos, pearson_coeff_SNR_error_planet_pos = compare_correlations(
        stats_planet_pos_noise,
        stats_planet_pos,
        filename_flag,
        plotname=f"scatter_PLANETPOS_{filename_flag}",
        xlabel="",
        ylabel="",
        title="planet_pos",
        plot_lims=[
            np.amin([stats_planet_pos_noise, stats_planet_pos]) - 1,
            np.amax([stats_planet_pos_noise, stats_planet_pos]) + 1,
        ],
        show_plot=show_plot,
        save_plot=save_plot,
        correlations_dir=correlations_dir,
        simulation_name=simulation_name,
    )
    
    # Save the plotting variables in a dictionary
    
    outputs = {}
    outputs['scaling_factors_noise'] = noise_scaling_factor
    outputs['pearson_coeff_SNR'] = pearson_coeff_SNR_max
    outputs['pearson_coeff_SNR_tvalue'] = pearson_coeff_SNR_max_tvalue
    outputs['pearson_coeff_SNR_pvalue'] = pearson_coeff_SNR_max_pvalue
    outputs['pearson_coeff_SNR_planet_pos'] = pearson_coeff_SNR_planet_pos
    outputs['pearson_coeff_SNR_error'] = pearson_coeff_SNR_error_max
    outputs['pearson_coeff_SNR_error_tvalue'] = pearson_coeff_SNR_error_max_tvalue
    outputs['pearson_coeff_SNR_error_pvalue'] = pearson_coeff_SNR_error_max_pvalue
    outputs['pearson_coeff_SNR_error_planet_pos'] = pearson_coeff_SNR_error_planet_pos
    #outputs['corr_coeff_data_NTC'] = corr_coeff_data_NTC
    #outputs['corr_coeff_data_TC'] = corr_coeff_data_TC
    #outputs['corr_coeff_noise'] = corr_coeff_noise
    outputs['mean_SNR'] = mean_SNR
    outputs['mean_SNR_tvalue'] = mean_SNR_tvalue
    outputs['mean_SNR_pvalue'] = mean_SNR_pvalue
    outputs['mean_SNR_planet_pos'] = mean_SNR_planet_pos
    outputs['mean_SNR_error'] = mean_SNR_error
    outputs['mean_SNR_error_tvaluealue'] = mean_SNR_error_tvalue
    outputs['mean_SNR_error_pvalue'] = mean_SNR_error_pvalue
    outputs['mean_SNR_error_planet_pos'] = mean_SNR_error_planet_pos
    #outputs['mean_cc_values'] = mean_cc_values
    #outputs['mean_cc_values_error'] = mean_cc_values_error
    #outputs['mean_cc_values_planet_pos'] = mean_cc_values_planet_pos
    #outputs['mean_cc_values_planet_pos_error'] = mean_cc_values_planet_pos_error
    #outputs['mean_cc_std'] = mean_cc_std
    #outputs['mean_cc_std_planet_pos'] = mean_cc_std_planet_pos
    
    
    # Save the data in a file
    if not all_significance_metrics:
        filename = f"{correlations_dir}/outputs_{simulation_name}"
        np.savez_compressed(filename, a = outputs)
    else:
        filename = f"{correlations_dir}/outputs_{simulation_name}_{etiqueta}"
        np.savez_compressed(filename, a = outputs)
    return
