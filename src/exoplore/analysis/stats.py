"""
exoplore.analysis.stats
=======================

Statistical analysis tools for high-resolution cross-correlation spectroscopy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, shutil

from exoplore.ccf.statistics import Welch_ttest_map, Combine_Nights
from exoplore.io.utils import find_nearest, format_number
from exoplore.observation.velocity import get_V
from exoplore.ccf.compute import get_max_CCF_peak, get_shifted_ccf_matrix


def bayes_factor_to_sigma(bayes_factor):
    """Map a Bayes factor ``B = Z_model / Z_null`` (>1) to a frequentist
    detection significance σ, following the **minimum-Bayes-factor bound**.

    This is a generic Bayesian→frequentist calibration, not specific to any one
    analysis: the underlying result is Sellke, Bayarri & Berger (2001); it was
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


def _planet_area_stats(stat_map, v_rest_arr, kp, kp_max, v_wind, kp_range, b):
    """Extract peak S/N and its location from a Kp-Vsys map near the planet.

    Searches a ±5 velocity / ±40 Kp window around the injected planet
    position (V_wind, Kp) and returns the peak value, its Kp index offset
    from the map centre, and the corresponding Vsys.

    Used to compute ``stats_planet_area`` for both the CCF_SNR and
    Welch t-test branches, which differ only in which stat array and
    velocity grid they use.
    """
    v_idx    = np.argwhere(v_rest_arr == v_wind)[0][0]
    kp_ctr   = int(kp + kp_max)
    region   = stat_map[v_idx - 5 : v_idx + 5,
                        kp_ctr - 40 : kp_ctr + 41, b]
    peak_val = np.max(region)
    kp_off   = (np.where(stat_map[v_idx - 5 : v_idx + 5, :, b] == peak_val)[1][0]
                - len(kp_range) // 2)
    v_peak   = v_rest_arr[
        np.where(stat_map[:, kp_ctr - 40 : kp_ctr + 41, b] == peak_val)[0][0]
    ]
    return peak_val, kp_off, v_peak

def plot_stats(stats, kp_lim_inf, kp_lim_sup, 
               kp_step, vrest_lim_inf, vrest_lim_sup, vrest_step,
               sn_lim_inf, sn_lim_sup, sn_lim_step, 
               binwidth_sn, binwidth_kp, binwidth_v_rest, 
               significance_metric, inp_dat, v_rest, auto_lims = False,
               show_SN_quantile = False, shade_true_region = False, 
               mark_true_values = False, 
               kp_shade_width = None, vrest_shade_width = None,
               show_dist_CC_values = True, 
               show_plot = False, save_plot = True,
               CCF_Noise = False):
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
        plt.axvspan(inp_dat['K_p'] - kp_shade_width, 
                    inp_dat['K_p'] + kp_shade_width, 
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = inp_dat['K_p'], color = 'r', linestyle = '--',
                    linewidth = 2)

    ax = plt.subplot(gs[2, 2]) # row 2, col 2
    # If the injected signal is found clearly, V_rest is 
    # always = inp_dat['V_wind'] and a histogram cannot be shown as before
    if np.all(stats[:,2] == inp_dat['V_wind']) or all(element == stats[:,2][0] for element in stats[:,2]):
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
        ax.set_xlim([np.round(inp_dat["V_wind"])-15, np.round(inp_dat["V_wind"])+15])
    
    if shade_true_region: 
        # Shade the region of interest
        plt.axvspan(inp_dat['V_wind'] - vrest_shade_width, 
                    inp_dat['V_wind'] + vrest_shade_width, 
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = inp_dat['V_wind'], color = 'r', linestyle = '--',
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
        plt.fill_betweenx([inp_dat['K_p'] - kp_shade_width, 
                           inp_dat['K_p'] + kp_shade_width],
                          sn_lim_inf,  sn_lim_sup, color='coral', 
                          alpha=0.4)
    elif mark_true_values:
        plt.axhline(y = inp_dat['K_p'], color = 'r', linestyle = '--',
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
        ax.set_ylim([np.round(inp_dat["V_wind"])-15, np.round(inp_dat["V_wind"])+15])
    
    if shade_true_region: 
        # Shade the region of interest
        plt.fill_betweenx([inp_dat['V_wind'] - vrest_shade_width, 
                           inp_dat['V_wind'] + vrest_shade_width],
                          sn_lim_inf,  sn_lim_sup, color='coral', 
                          alpha=0.4)
    elif mark_true_values:
        plt.axhline(y = inp_dat['V_wind'], color = 'r', linestyle = '--',
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
        ax.set_ylim([np.round(inp_dat["V_wind"])-15, np.round(inp_dat["V_wind"])+15])
    if shade_true_region: 
        # Shade the region of interest
        x_values = plt.gca().get_xlim()
        plt.fill_betweenx([inp_dat['V_wind'] - vrest_shade_width, 
                           inp_dat['V_wind'] + vrest_shade_width],
                          x_values[0], x_values[1],
                          color='coral', 
                          alpha=0.4)
        y_values = plt.gca().get_ylim()
        plt.fill_betweenx([y_values[0], y_values[1]],
              inp_dat['K_p'] - kp_shade_width,
              inp_dat['K_p'] + kp_shade_width,
              color='coral',
              alpha=0.4)
    elif mark_true_values:
        plt.axhline(y = inp_dat['V_wind'], color = 'r', linestyle = '--',
                    linewidth = 2.5)
        plt.axvline(x = inp_dat['K_p'], color = 'r', linestyle = '--',
                    linewidth = 2.5)

    fig.tight_layout()
    
    # Save it and show it
    if save_plot and not CCF_Noise: 
        plt.savefig(f"{inp_dat['plots_dir']}Corner_plot_{inp_dat['Simulation_name']}.pdf")
        plt.savefig(f"{inp_dat['plots_dir']}Corner_plot_{inp_dat['Simulation_name']}.png", transparent = True)
    elif save_plot and CCF_Noise:
        plt.savefig(f"{inp_dat['plots_dir']}Corner_plot_noise_{inp_dat['Simulation_name']}.pdf")
    if show_plot: plt.show()
    plt.close()
    
    """
    # Create 3D subplot with projections   
    X = stats[:,0]
    Y = stats[:,1]
    Z = stats[:,2]
    
    # Mock plot to get the axes consistent afterwards
    plt.figure()
    ax1 = plt.subplot(111,  projection='3d')

    ax1.scatter(X, Y, Z, c='b', marker='.', alpha=0.2)

    plt.figure()
    ax2 = plt.subplot(111,  projection='3d')

    cx = np.ones_like(X) * ax1.get_xlim3d()[0]
    cy = np.ones_like(X) * ax1.get_ylim3d()[1]
    cz = np.ones_like(Z) * ax1.get_zlim3d()[0]

    ax2.scatter(X, Y, cz.min(), color = 'grey',  marker='.', lw=0, alpha=0.8)
    ax2.scatter(X, cy.min(), Z, color = 'grey', marker='.', lw=0, alpha=0.8)
    ax2.scatter(cx.min(), Y, Z, color = 'grey',  marker='.', lw=0, alpha=0.8)

    ax2.scatter(X, Y, Z, c='navy', marker='.', alpha=0.8)
    
    # Fixed Y value for the vertical line
    true_Kp = inp_dat['K_p']
    true_vrest = inp_dat['V_wind']
    
    # Plot a vertical line at the fixed Y value in the Y-Z plane
    ax2.plot([cx.min(), cx.min()], [true_Kp, true_Kp], [Z.min(), Z.max()], color='r')
    ax2.plot([cx.min(), cx.min()], [Y.min(), Y.max()], [true_vrest, true_vrest], color='r')
    ax2.plot([X.min(), X.max()], [true_Kp, true_Kp], [cz.min(), cz.min()], color='r')
    ax2.plot([X.min(), X.max()], [cy.min(), cy.min()], [true_vrest, true_vrest], color='r')


    ax2.set_xlim3d(ax1.get_xlim3d())
    ax2.set_ylim3d(ax1.get_ylim3d())
    ax2.set_zlim3d(ax1.get_zlim3d())
    
    # Customize plot appearance
    ax2.set_xlabel('S/N', fontsize=12)
    ax2.set_ylabel('$K_p$', fontsize=12)
    ax2.set_zlabel('$V_{rest}$', fontsize=12)
    ax2.xaxis.set_tick_params(labelsize=10)
    ax2.yaxis.set_tick_params(labelsize=10)
    ax2.zaxis.set_tick_params(labelsize=10)
    #ax.legend(loc = 'best', prop={'size': 10})

    
    plt.tight_layout()
    
    # Save it in PDF and png and show
    plt.savefig(f"{plot_name}_3D.pdf")
    plt.show()
    """
    
    """
    # Histograms 
    gs = gridspec.GridSpec(1, 1)

    fig = plt.figure(figsize=(8,8))

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
    ax.set_ylabel('', fontsize = 17)
    ax.legend(prop={'size': 10})
    xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
    ax.set_xticks(xticks)
    ax.set_xlim([sn_lim_inf, sn_lim_sup])
    ax.grid(True, which='both')
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    
    # Save it in PDF and png
    plt.savefig(f"{plot_name}_SNR_distribution.pdf")
    plt.savefig(f"{plot_name}_SNR_distribution.png", transparent=True)
    plt.show()
    
    
    # Histograms 
    gs = gridspec.GridSpec(1, 1)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(gs[0,0]) # row 1, col 1
    sns.histplot(stats[:,1], binwidth = binwidth_kp, color = 'black',
                 stat='density',
                 label='Max. $K_P$')
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
        plt.axvspan(inp_dat['K_p'] - kp_shade_width, 
                    inp_dat['K_p'] + kp_shade_width, 
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = inp_dat['K_p'], color = 'goldenrod', linestyle = '--',
                    linewidth = 2)
        
    # Save it in PDF and png
    plt.savefig(f"{plot_name}_Kp_distribution.pdf")
    plt.savefig(f"{plot_name}_Kp_distribution.png", transparent=True)
    plt.show()

    # Histograms 
    gs = gridspec.GridSpec(1, 1)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(gs[0,0]) # row 2, col 2
    # If the injected signal is found clearly, V_rest is 
    # always = inp_dat['V_wind'] and a histogram cannot be shown as before
    if np.all(stats[:,2] == inp_dat['V_wind']):
        hist, bins = np.histogram(stats[:,2], bins=[0, 1])
        plt.bar(bins[:-1], hist, width=1)
    else:
        sns.histplot(stats[:,2], binwidth = binwidth_v_rest, color='black',
                      stat='density',
                      label='Max. $V_{rest}$')
    ax.set_xlabel('V$_{rest}$ (km/s)', fontsize = 17)
    ax.grid(True, which='both')
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    xticks = np.arange(vrest_lim_inf, vrest_lim_sup + vrest_step, 
                       vrest_step)
    ax.set_xticks(xticks)
    ax.set_xlim([vrest_lim_inf, vrest_lim_sup])
    if shade_true_region: 
        # Shade the region of interest
        plt.axvspan(inp_dat['V_wind'] - vrest_shade_width, 
                    inp_dat['V_wind'] + vrest_shade_width, 
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = inp_dat['V_wind'], color = 'goldenrod', linestyle = '--',
                    linewidth = 2)
        
    # Save it in PDF and png
    plt.savefig(f"{plot_name}_Vrest_distribution.pdf")
    plt.savefig(f"{plot_name}_Vrest_distribution.png", transparent=True)
    plt.show()
    """
    
    """
    # Histograms but all in a column
    gs = gridspec.GridSpec(3, 1)

    fig = plt.figure(figsize=(10,16))

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
    ax.set_ylabel('', fontsize = 17)
    ax.legend(prop={'size': 10})
    xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
    ax.set_xticks(xticks)
    ax.set_xlim([sn_lim_inf, sn_lim_sup])
    ax.grid(True, which='both')
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    
    
    # Histograms 
    ax = plt.subplot(gs[1,0]) # row 1, col 1
    sns.histplot(stats[:,1], binwidth = binwidth_kp, color = 'black',
                 stat='density',
                 label='Max. $K_P$')
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
        plt.axvspan(inp_dat['K_p'] - kp_shade_width, 
                    inp_dat['K_p'] + kp_shade_width, 
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = inp_dat['K_p'], color = 'goldenrod', linestyle = '--',
                    linewidth = 2)

    # Histograms 
    ax = plt.subplot(gs[2,0]) # row 2, col 2
    # If the injected signal is found clearly, V_rest is 
    # always = inp_dat['V_wind'] and a histogram cannot be shown as before
    if np.all(stats[:,2] == inp_dat['V_wind']):
        hist, bins = np.histogram(stats[:,2], bins=[0, 1])
        plt.bar(bins[:-1], hist, width=1)
    else:
        sns.histplot(stats[:,2], binwidth = binwidth_v_rest, color='black',
                      stat='density',
                      label='Max. $V_{rest}$')
    ax.set_xlabel('V$_{rest}$ (km/s)', fontsize = 17)
    ax.grid(True, which='both')
    ax.legend(prop={'size': 10})
    ax.set_ylabel('', fontsize = 17)
    ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                   labelsize=16)
    xticks = np.arange(vrest_lim_inf, vrest_lim_sup + vrest_step, 
                       vrest_step)
    ax.set_xticks(xticks)
    ax.set_xlim([vrest_lim_inf, vrest_lim_sup])
    if shade_true_region: 
        # Shade the region of interest
        plt.axvspan(inp_dat['V_wind'] - vrest_shade_width, 
                    inp_dat['V_wind'] + vrest_shade_width, 
                    facecolor='coral', alpha=0.4)
    elif mark_true_values:
        plt.axvline(x = inp_dat['V_wind'], color = 'goldenrod', linestyle = '--',
                    linewidth = 2)
        
    # Save it in PDF and png
    plt.savefig(f"{plot_name}_columnHistograms.pdf")
    plt.savefig(f"{plot_name}_columnHistograms.png", transparent=True)
    plt.show()
    """
    
    
    # And now the dist. of CC values, if specified by user
    if show_dist_CC_values:
        gs = gridspec.GridSpec(1, 4)
        fig = plt.figure(figsize=(16,4))
        

        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        true_area = significance_metric[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                      int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 60, :]
        
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
            
        
        """
        ax = plt.subplot(gs[0, 2]) # row 0, col 2
        sns.histplot(np.ndarray.flatten(ccf_tot_sn_stat[120, 300,:]), 
                     color = 'black', stat='density', alpha = 0.8, 
                     label='S/N random \n$K_p-V_{rest}$')
        xticks = np.arange(-3, 3.1, 1)
        ax.set_xticks(xticks)
        ax.grid(True, which='both')
        ax.legend()
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        """
        
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        plot_variable = np.ndarray.flatten(significance_metric[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], int(inp_dat['K_p']+inp_dat['kp_max']+1), :])
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
            significance_metric[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15), 
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
            np.s_[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15)], 
            axis=0
            )
        
        away_from_signal_and_tellurics = np.delete(
            significance_metric, np.s_[320-40:320+40], 
            axis=1
            )
        # Removing planet signal
        away_from_signal_and_tellurics = np.delete(
            significance_metric, 
            np.s_[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]-5:np.argwhere(v_rest == inp_dat['V_wind'])[0][0]+5], 
            axis=0
            )
        
        away_from_signal_and_tellurics = np.delete(
            significance_metric, 
            np.s_[int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40:int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40],
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
            plt.savefig(f"{inp_dat['plots_dir']}CC_distributions_{inp_dat['Simulation_name']}.pdf")
        elif save_plot and CCF_Noise:
            plt.savefig(f"{inp_dat['plots_dir']}CC_distributions_noise_{inp_dat['Simulation_name']}.pdf")
        if show_plot: plt.show()
        plt.close()
    
    return


def statistical_study(inp_dat, ccf_v_step, ccf_stat, kp_range, phase,
        v_ccf, v_rest, with_signal, pixels_left_right, sysrem_it_opt,
        ccf_iterations, in_trail_pix, auto_lims, input_stats = None,
        input_stats_tvalue = None, input_stats_pvalue = None,
        previous_shuffle = None, verbose = True, show_plot = False,
        save_plot = True, CCF_Noise = False, ccf_SSIM = False
        ):
    """Build cumulative Kp-Vsys significance maps and extract detection statistics.

    This is the central analysis loop for the Monte-Carlo statistical study.
    For each simulated night it shifts the CCF matrix into the planet rest
    frame, co-adds in time, applies the chosen significance metric (S/N or
    Welch t-test), and records the peak S/N and its Kp-Vsys coordinates.

    The loop iterates over nights, accumulating ``stats`` which allows the
    user to study how detection significance grows with co-added nights and
    to identify best/worst nights from the resulting distribution.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used include ``"n_nights"``,
        ``"CCF_SNR"``, ``"Welch_ttest"``, ``"Stack_Group_Size"``,
        ``"Opt_PCA_its_ord_by_ord"``, ``"K_p"``, ``"V_wind"``,
        ``"kp_max"``, and ``"CCF_V_STEP"``.
    ccf_v_step : float
        Velocity step of the CCF grid in km/s.
    ccf_stat : ndarray
        CCF matrix cube from the simulator.  Shape depends on the pipeline
        mode: ``(n_orders, n_nights, n_lags, n_spectra)`` (standard) or
        ``(n_orders, n_nights, n_lags, n_spectra, 2, sysrem_its)``
        (order-by-order SYSREM optimisation).
    kp_range : ndarray, shape (n_kp_steps,)
        Grid of trial Kp values in km/s over which the map is evaluated.
    phase : ndarray
        Orbital phase array (or list of arrays for multi-night simulations).
    v_ccf : ndarray
        Earth-frame CCF velocity grid in km/s.
    v_rest : ndarray
        Planet-rest-frame velocity axis in km/s.
    with_signal : ndarray or list of ndarray
        Indices of in-transit exposures.
    pixels_left_right : int
        Half-width (in pixels) of the velocity window centred on the
        planet's instantaneous velocity for the rest-frame shift.
    sysrem_it_opt : ndarray
        Optimal SYSREM iteration indices per order and night, shape
        ``(n_orders, n_nights, 2)``.  Only used when
        ``inp_dat["Opt_PCA_its_ord_by_ord"]`` is True.
    ccf_iterations : int
        Total number of CCF lag steps.
    in_trail_pix : int
        Width of the in-trail velocity window (pixels; must be odd) used
        by the Welch t-test significance metric.
    auto_lims : bool
        If True, axis limits in the corner plot are determined automatically.
    input_stats, input_stats_tvalue, input_stats_pvalue : ndarray or None
        Pre-existing statistics arrays from a previous run (used when
        computing noise-only CCF significance for comparison).
    previous_shuffle : ndarray or None
        Shuffled night order from a previous call (for reproducibility of
        the stacking sequence).
    verbose : bool
        If True, print progress messages every 10 nights.
    show_plot, save_plot : bool
        Display and/or save the corner plot.
    CCF_Noise : bool
        If True, process the noise-only CCF instead of the signal CCF.
    ccf_SSIM : bool
        Reserved for structural similarity metric (not yet implemented).

    Returns
    -------
    tuple
        ``(ccf_tot_stat, significance_metric, significance_metric2,
        significance_metric3, stats, stats_tvalue, stats_pvalue,
        stats_planet_pos, stats_planet_area, stats_cc_values,
        stats_cc_values_planet_pos, stats_cc_values_std,
        stats_cc_values_std_planet_pos, ccf_complete_stat,
        ccf_values_shift_stat, shuffled_nights, v_rest_sigma)``

        * ``ccf_tot_stat``: co-added CCF map, shape
          ``(n_v_rest, n_kp, n_nights)``.
        * ``significance_metric``: S/N or σ map, same shape.
        * ``stats``: peak detection statistics per night, shape
          ``(n_nights, 3)``, columns are [S/N, Kp, Vrest].
    """

    # Co-adding of orders in each night with NO WEIGHTS
    if np.logical_or(not inp_dat["Opt_PCA_its_ord_by_ord"], CCF_Noise):
        if len(ccf_stat.shape) == 4:
            ccf_complete_stat = np.sum(ccf_stat, 0)
        else: ccf_complete_stat = ccf_stat
        
        # Analyse n_nigts individually or study co-addings? This
        # allows us to study how the varibility in signal's
        # significance is (hopefully) reduced when co-adding nights.
        if inp_dat["Stack_Group_Size"] is not None and inp_dat["Stack_Group_Size"] > 1:
            ccf_complete_stat, shuffled_nights= Combine_Nights(
                inp_dat, ccf_complete_stat, CCF_Noise, previous_shuffle
                )
        else: shuffled_nights = None
    else:
        # We can already set the criterion selected by the user
        if inp_dat["Opt_crit"] == "Maximum":
            crit_choice = 0
        elif inp_dat["Opt_crit"] == "Max_Diff":
            crit_choice = 1
   
            
    # Define planet velocity vector as a function of Kp
    vp = np.zeros((kp_range.shape[0], len(phase)))
    for k in range(len(kp_range)): 
        vp[k, :] = get_V(
            kp_range[k], phase, inp_dat['BERV'],
            inp_dat['V_sys'], inp_dat['V_wind']
            )
    
    for b in range(inp_dat["n_nights"]):
        
        # If optimising SYSREM its., then the actual ccf_complete_stat
        # has to be computed for each night's selection iterations 
        # for each order
       
        if inp_dat["Opt_PCA_its_ord_by_ord"] and not CCF_Noise:
            
            if sysrem_it_opt.shape[0] == inp_dat["n_orders"] and len(ccf_stat.shape) == 6:
                # Extract the relevant data based on sysrem_it_opt
                # Create a new matrix for storing the selected data
                ccf_complete_stat = np.zeros(
                    (ccf_stat.shape[:4]), float
                    )
                
                # Iterate over dimensions to select data based on sysrem_it_opt
                for h in range(inp_dat["n_orders"]):
                    #ccf_complete_stat = ccf_stat[:,:,:,:,0,2]
                    for n in range(ccf_stat.shape[3]): # Loop in spectra
                        sysrem_index = sysrem_it_opt[h, b, crit_choice]
                        ccf_complete_stat[h, b, :, n] = ccf_stat[h, b, :, n, 0, sysrem_index]
                
                # Co-adding the orders with the selected iterations
                ccf_complete_stat = np.sum(ccf_complete_stat, axis=0)
            # Now we make sure the shuffling and co-adding of nights 
            # gets done only once
            if b == 0:
                if inp_dat["Stack_Group_Size"] is not None and inp_dat["Stack_Group_Size"] > 1:
                    ccf_complete_stat, shuffled_nights= Combine_Nights(
                        inp_dat, ccf_complete_stat, CCF_Noise, previous_shuffle
                        )
                else: shuffled_nights = None
                
        # The rest of the loop is the same regardless 
        # of SYSREM optimisation
        if inp_dat["n_nights"] > 20 and (b % 10 == 0) and verbose:
            print('STATISTICAL STUDY: Co-adding night ' + str(b+1) + '/'
              + str(inp_dat["n_nights"]))

        ######################################################################
        
        ######################################################################

        # Variable that stores all shifts as a function of Kp
        if b == 0:
            left_right = in_trail_pix #// 2
            ccf_values_shift_stat = np.zeros((len(v_rest), len(with_signal),
                                              kp_range.shape[0]), float)
            ccf_tot_stat = np.zeros((len(v_rest), kp_range.shape[0], 
                                     inp_dat["n_nights"]), 
                                    float) 
            if inp_dat['CCF_SNR']:
                ccf_tot_sn_stat = np.zeros_like(ccf_tot_stat)
            elif inp_dat["Welch_ttest"]:
                ccf_tot_sigma_stat = np.zeros(
                    (v_rest.shape[0] - 2 * left_right, len(kp_range), 
                     inp_dat["n_nights"])
                    )
                ccf_tot_t_stat = np.zeros(
                    (v_rest.shape[0] - 2 * left_right, len(kp_range), 
                     inp_dat["n_nights"])
                    )
                ccf_tot_p_stat = np.zeros(
                    (v_rest.shape[0] - 2 * left_right, len(kp_range), 
                     inp_dat["n_nights"])
                    )
                
            stats = np.zeros((inp_dat["n_nights"], 3))
            stats_tvalue = np.zeros((inp_dat["n_nights"], 3))
            stats_pvalue = np.zeros((inp_dat["n_nights"], 3))
            stats_planet_pos = np.zeros((inp_dat["n_nights"], 3))
            stats_planet_area = np.zeros((inp_dat["n_nights"], 3))
            stats_cc_values = np.zeros((inp_dat["n_nights"], 3))
            stats_cc_values_planet_pos = np.zeros((inp_dat["n_nights"], 3))
            stats_cc_values_std = np.zeros((inp_dat["n_nights"], 3))
            stats_cc_values_std_planet_pos = np.zeros((inp_dat["n_nights"], 3))
            
            v_aux = np.zeros((len(with_signal), len(v_rest), kp_range.shape[0]))
       
        # Loop over the signal frames and velocity values
        for idx, i in enumerate(with_signal):
            # Loop over the planetary velocities
            for k_idx in range(len(kp_range)):
                # We create a velocity array centered in the pixel with signal vp[i]
                v_aux[idx, :, k_idx] = np.linspace(
                    vp[k_idx, i] - pixels_left_right * ccf_v_step, 
                    vp[k_idx, i] + pixels_left_right * ccf_v_step, 
                    num=2*pixels_left_right+1
                    )
                
                # CCF centered at the bin where the planet signal should be
                ccf_values_shift_stat[:, idx, k_idx] = np.interp(
                    v_aux[idx, :, k_idx], v_ccf, ccf_complete_stat[b, :, i]
                    )   
        # Co-adding in time
        ccf_tot_stat[:, :, b] = np.sum(
            ccf_values_shift_stat, axis=1, out=ccf_tot_stat[:, :, b]
            )
        
        
        if inp_dat['CCF_SNR'] and not CCF_Noise:
            ccf_tot_sn_stat[:,:,b], max_sig, max_kp_idx, max_v_rest, cc_values_std = \
                get_max_CCF_peak(
                    inp_dat=inp_dat, ccf_tot=ccf_tot_stat[:,:,b], v_rest=v_rest, kp_range=kp_range, 
                    b = None, stats = None,
                    sysrem_opt = False,
                    CCF_Noise = False,
                    )
            significance_metric = ccf_tot_sn_stat
            significance_metric2 = None
            significance_metric3 = None
            v_rest_sigma = None
        elif inp_dat['CCF_SNR'] and CCF_Noise:
            ccf_tot_sn_stat[:,:,b], max_sig_noise, max_kp_noise_idx, max_v_rest_noise, cc_values_std_noise = \
                get_max_CCF_peak(
                    inp_dat, ccf_tot_stat[:,:,b], v_rest, kp_range, 
                    b, input_stats, False,
                    CCF_Noise = True,
                    )
            significance_metric = ccf_tot_sn_stat
            significance_metric2 = None
            significance_metric3 = None
            v_rest_sigma = None
        elif not inp_dat['CCF_SNR'] and inp_dat["Welch_ttest"] and not CCF_Noise:
            #custom_start_time = time.time()
            ccf_tot_sigma_stat[:,:,b], ccf_tot_t_stat[:,:,b], ccf_tot_p_stat[:,:,b], v_rest_sigma , max_sig, max_kp_idx, max_v_rest, max_t_value, max_kp_idx_t, max_v_rest_t, max_p_value, max_kp_idx_p, max_v_rest_p = \
                Welch_ttest_map(
                    ccf_values_shift_stat, v_rest, kp_range,
                    inp_dat, CCF_Noise = CCF_Noise, plotting = show_plot
                    )
            significance_metric = ccf_tot_sigma_stat
            significance_metric2 = ccf_tot_t_stat
            significance_metric3 = ccf_tot_p_stat
        elif not inp_dat['CCF_SNR'] and inp_dat["Welch_ttest"] and CCF_Noise:
            #custom_start_time = time.time()
            ccf_tot_sigma_stat[:,:,b], ccf_tot_t_stat[:,:,b], ccf_tot_p_stat[:,:,b], v_rest_sigma, max_sig_noise, max_kp_noise_idx, max_v_rest_noise, max_t_value_noise, max_kp_idx_t_noise, max_v_rest_t_noise, max_p_value_noise, max_kp_idx_p_noise, max_v_rest_p_noise  = \
                Welch_ttest_map(
                    ccf_values_shift_stat, v_rest, kp_range,
                    inp_dat, stats=input_stats, stats_tvalue=input_stats_tvalue, stats_pvalue=input_stats_pvalue, b=b, CCF_Noise = CCF_Noise, 
                    plotting = show_plot
                    )
            significance_metric = ccf_tot_sigma_stat
            significance_metric2 = ccf_tot_t_stat
            significance_metric3 = ccf_tot_p_stat
            # Record the end time
            #end_time = time.time()
            
            # Calculate the elapsed time
            #elapsed_time = end_time - custom_start_time
            
            #print(f"Time elapsed: {elapsed_time:.4f} seconds")
            
        ######################################################################
        ######################################################################
        """
        Now we will search the maximum SNR around the expected position (where
        we put the fake planet originally) and store its value to
        do the plot.
        
        Playing with the stats variable below allows us to see Kp-Vrest maps
        for each night, to check for the best and worst nights, e.g.
        np.where(stats[:,0] < 3.8)[0] and then re-use the code 
        for Kp-Vrest plot with ccf_tot_sn_stat[:, :,
                                               np.where(stats[:,0] < 3.8)[0]]
        """
        ######################################################################
        ######################################################################
        
        if not CCF_Noise:
            stats[b, 0] = max_sig
            stats[b, 1] = max_kp_idx - (len(kp_range) // 2)
            stats[b, 2] = max_v_rest
            
            # The stats for maximum value in cc maps (no S/N)
            stats_cc_values[b, 0] = ccf_tot_stat[np.argwhere(v_rest == max_v_rest)[0][0], 
                                                 max_kp_idx, 
                                                 b]
            stats_cc_values[b, 1] = max_kp_idx - (len(kp_range) // 2)
            stats_cc_values[b, 2] = max_v_rest
            
            if inp_dat['CCF_SNR']:
                stats_cc_values_std[b, 0] = cc_values_std[np.argwhere(v_rest == max_v_rest)[0][0], 
                                                          max_kp_idx]
                stats_cc_values_std[b, 1] = max_kp_idx - (len(kp_range) // 2)
                stats_cc_values_std[b, 2] = max_v_rest
            else:
                stats_cc_values_std = None
            
            # And now the stats at exactly the Kp-Vrest of the planet
            if inp_dat['CCF_SNR']:
                stats_planet_pos[b, 0] = ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                         int(np.ceil(inp_dat['K_p'])+len(kp_range)//2), 
                                                         b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
            
                stats_tvalue = None
                stats_pvalue = None
            elif inp_dat["Welch_ttest"]:
                
                stats_tvalue[b, 0] = max_t_value
                stats_tvalue[b, 1] = max_kp_idx_t - (len(kp_range) // 2)
                stats_tvalue[b, 2] = max_v_rest_t
                
                stats_pvalue[b, 0] = max_p_value
                stats_pvalue[b, 1] = max_kp_idx_p - (len(kp_range) // 2)
                stats_pvalue[b, 2] = max_v_rest_p
                
                stats_planet_pos[b, 0] = ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0], 
                                                         int(np.ceil(inp_dat['K_p'])+inp_dat['kp_max']), 
                                                         b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest_sigma[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0]]
            
            # Stats at around the Kp-Vrest of the planet
            if inp_dat['CCF_SNR']:
                (stats_planet_area[b, 0],
                 stats_planet_area[b, 1],
                 stats_planet_area[b, 2]) = _planet_area_stats(
                    ccf_tot_sn_stat, v_rest,
                    inp_dat['K_p'], inp_dat['kp_max'],
                    inp_dat['V_wind'], kp_range, b)

            elif inp_dat["Welch_ttest"]:
                (stats_planet_area[b, 0],
                 stats_planet_area[b, 1],
                 stats_planet_area[b, 2]) = _planet_area_stats(
                    ccf_tot_sigma_stat, v_rest_sigma,
                    inp_dat['K_p'], inp_dat['kp_max'],
                    inp_dat['V_wind'], kp_range, b)
                
                
                
            stats_cc_values_planet_pos[b, 0] = ccf_tot_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                     int(np.ceil(inp_dat['K_p'])+inp_dat['kp_max']), 
                                                     b]
            stats_cc_values_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
            stats_cc_values_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
            
            if inp_dat['CCF_SNR']:
                stats_cc_values_std_planet_pos[b, 0] = cc_values_std[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                         int(np.ceil(inp_dat['K_p'])+inp_dat['kp_max'])]
                stats_cc_values_std_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_cc_values_std_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
            else:
                stats_cc_values_std_planet_pos = None
        else:
            stats[b, 0] = max_sig_noise
            stats[b, 1] = max_kp_noise_idx - (len(kp_range) // 2)
            stats[b, 2] = max_v_rest_noise
            
            # And now the stats at exactly the Kp-Vrest of the planet
            if inp_dat['CCF_SNR']:
                stats_planet_pos[b, 0] = ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                     int(np.ceil(inp_dat['K_p'])+len(kp_range)//2), 
                                                     b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
            
                stats_tvalue = None
                stats_pvalue = None
            elif inp_dat["Welch_ttest"]: 
                
                stats_tvalue[b, 0] = max_t_value_noise
                stats_tvalue[b, 1] = max_kp_idx_t_noise - (len(kp_range) // 2)
                stats_tvalue[b, 2] = max_v_rest_t_noise
                
                stats_pvalue[b, 0] = max_p_value_noise
                stats_pvalue[b, 1] = max_kp_idx_p_noise - (len(kp_range) // 2)
                stats_pvalue[b, 2] = max_v_rest_p_noise
                
                stats_planet_pos[b, 0] = ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0], 
                                                         int(np.ceil(inp_dat['K_p'])+len(kp_range)//2), 
                                                         b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest_sigma[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0]]
            
            
    return ccf_tot_stat, significance_metric, significance_metric2, significance_metric3, stats, stats_tvalue, stats_pvalue, stats_planet_pos, stats_planet_area,\
           stats_cc_values, stats_cc_values_planet_pos,\
           stats_cc_values_std, stats_cc_values_std_planet_pos,\
           ccf_complete_stat, ccf_values_shift_stat, shuffled_nights,\
           v_rest_sigma


def get_SYSREM_its_ordbyord(inp_dat, ccf_store, v_rest, with_signal, phase, berv, v_sys,
        pixels_left_right, ccf_v_step, v_erf
        ):
    """Select the optimal SYSREM iteration count order-by-order via signal injection.

    For each spectral order and each night, shifts the CCF into the planet
    rest frame using the known injection velocity (``inp_dat['K_p']``,
    ``inp_dat['V_wind']``) and evaluates a detection criterion (S/N or
    maximum CCF value) as a function of SYSREM iteration count.  The
    iteration index that maximises the criterion is stored in
    ``sysrem_it_opt``.

    .. warning::
        This optimisation is performed using the *injected* signal and
        should only be applied to the noiseless or signal-injected dataset.
        Applying it to a real detection will inflate the reported S/N because
        SYSREM is tuned to maximise a known signal rather than a blind search.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``"n_orders"``,
        ``"n_nights"``, ``"sysrem_its"``, ``"K_p"``, ``"V_wind"``,
        ``"kp_max"``, ``"CCF_V_STEP"``, ``"Opt_crit"``.
    ccf_store : ndarray, shape (n_orders, n_nights, n_lags, n_spectra, 2, sysrem_its)
        Per-order CCF cubes for each SYSREM iteration count.
    v_rest : ndarray, shape (n_v_rest,)
        Planet rest-frame velocity axis in km/s.
    with_signal : ndarray or list
        In-transit exposure indices per night.
    phase : ndarray or list
        Orbital phase arrays.
    berv : float or ndarray
        Barycentric Earth RV correction in km/s.
    v_sys : float
        Systemic velocity in km/s.
    pixels_left_right : int
        Half-width of the in-trail integration window in pixels.
    ccf_v_step : float
        CCF velocity step size in km/s.
    v_erf : ndarray
        Planet-frame velocity grid used for the ERF (rest-frame) plot axis.

    Returns
    -------
    ndarray, shape (n_orders, n_nights, 2)
        Optimal SYSREM iteration indices, one per order per night.
        The third axis holds [index_criterion_0, index_criterion_1].
    """
    ccf_values_shift = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"], len(v_rest), len(with_signal), 
         2, inp_dat["sysrem_its"]), float
        )
    
    # Calculate injected-planetary velocities during the night
    vp = get_V(
        inp_dat["Kp_Vrest_inj"][0], phase, berv,
        v_sys, inp_dat["Kp_Vrest_inj"][1]
        )
    
    # Move all matrices to INJECTION REST-FRAME
    for idx, i in enumerate(with_signal):
        # We create a velocity array centered in the 
        # pixel with signal vp[i]
        v_inj_prf = np.linspace(
            vp[i] - pixels_left_right * ccf_v_step, 
            vp[i] + pixels_left_right * ccf_v_step, 
            num=2*pixels_left_right+1
            )
        for b in range(inp_dat["n_nights"]):
            for h in range(inp_dat["n_orders"]):
                for n in range(2):
                    for l in range(inp_dat["sysrem_its"]):
                        ccf_values_shift[h, b, :, idx, n, l] = np.interp(
                            v_inj_prf, v_erf, ccf_store[h, b, :, idx, n, l]
                            )
                        
    # Co-adding in time. The new matrix ccf_tot has a shape of
    # (inp_dat["n_orders"], inp_dat["n_nights"], len(v_rest),
    #  2, inp_dat["sysrem_its"])
    ccf_tot = np.sum(ccf_values_shift, axis = 3)
    
    # Now we extract the value of the CCFs with and without injection
    # at 0 (the V_wind of the injected signal)
    injection_v = np.argwhere(v_rest == find_nearest(
        v_rest, 0
        )
        )[0][0]
    ccf_maxinj_pos = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"],
         2, inp_dat["sysrem_its"]), float
        )
    v_maxinj_pos = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"],
         inp_dat["sysrem_its"]), int
        )
    
    for b in range(inp_dat["n_nights"]):
        for h in range(inp_dat["n_orders"]):
            for l in range(inp_dat["sysrem_its"]):
                v_maxinj_pos[h,b,l] = np.where(
                    ccf_tot[h,b,:,1,l] == np.amax(ccf_tot[h, b, injection_v-20:injection_v+21, 1, l])
                    )[0][0]
                ccf_maxinj_pos[h, b, 1, l] = ccf_tot[h, b, v_maxinj_pos[h,b,l], 1, l]
                ccf_maxinj_pos[h, b, 0, l] = ccf_tot[h, b, v_maxinj_pos[h,b,l], 0, l]
    
    # Now we store both which iteration maximises the recovery
    # of the injected signal and the CCF difference between 
    # injected and non-injected cases
    sysrem_opt = np.zeros(
        (inp_dat["n_orders"], inp_dat["n_nights"], 2), float
        )
    for b in range(inp_dat["n_nights"]):
        for h in range(inp_dat["n_orders"]):
            diff = ccf_maxinj_pos[h, b, 1, :] - ccf_maxinj_pos[h, b, 0, :]
            # I esclude the first SYSREM iteration because it is
            # really bad and it messes up results (i.e. sometimes
            # the maximum recovery or difference is reached in that
            # iteration because it still has strong residuals)
            sysrem_opt[h, b, 0] = int(np.where(ccf_maxinj_pos[h, b, 1, 2:] == np.amax(ccf_maxinj_pos[h, b, 1, 2:]))[0][0] + 2)
            sysrem_opt[h, b, 1] = int(np.where(diff[2:] == np.amax(diff[2:]))[0] + 2)
    return sysrem_opt


def get_CCvalues_dist(inp_dat, ccf_matrix, v_ccf, v_rest, in_trail_pix,
                      night1, night2, with_signal, kp_range, phase, pixels_left_right,
                      ccf_v_step, Kp, name, save_plot):
    """Plot distributions of in-trail and out-of-trail CCF values for two nights.

    Shifts the CCF matrix into the planet rest frame at trial orbital velocity
    ``Kp`` for two specified nights, then separates pixels into an in-trail
    distribution (velocity pixels within ±in_trail_pix/2 of the planet rest
    frame zero) and an out-of-trail distribution.  Overplotting the two nights
    reveals whether the in-trail signal is consistently brighter than the noise.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary (``"V_sys"``, ``"BERV"``, ``"CCF_V_STEP"``).
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
    if in_trail_pix % 2 == 0:
        raise ValueError("The width of the in-trail distribution\
                         should be an odd number")
    # Get the shifted matrices at the desired Kp, no winds considered
    ccf_matrix_shifted1 = get_shifted_ccf_matrix(
        with_signal, v_rest, v_ccf, kp_range, phase, inp_dat['V_sys'], 
        inp_dat['BERV'], pixels_left_right, ccf_v_step, 
        ccf_matrix[:, :, night1]
        )[:, :, Kp+int(np.floor(len(kp_range)/2))]
    ccf_matrix_shifted2 = get_shifted_ccf_matrix(
        with_signal, v_rest, v_ccf, kp_range, phase, inp_dat['V_sys'], 
        inp_dat['BERV'], pixels_left_right, ccf_v_step, 
        ccf_matrix[:, :, night2]
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
        os.makedirs(f"{inp_dat['plots_dir']}statistical/ccf_minmax", exist_ok=True)
    plt.title('Out-of-trail Cross correlation values', fontsize = 15)
    plt.xlabel("Cross correlation value", fontsize = 17)
    plt.ylabel("Frequency", fontsize = 17)
    plt.tick_params(axis = 'both', width = 1.8, direction = 'in',
            labelsize=15)
    plt.legend(loc='best', prop={'size': 12})

    if save_plot:
        plt.savefig(
            f"{inp_dat['plots_dir']}statistical/"
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
            f"{inp_dat['plots_dir']}statistical/"
            f"ccf_minmax/{name}_in_trail_dist.png",
            bbox_inches='tight')

    plt.show()
    plt.close()
            
    return


def compare_empirical_SN(matrix, inp_dat, n_pixels, night1, night2,
                         name, save_plot):
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
    inp_dat : dict
        Simulation input dictionary (``"n_orders"``, ``"order_selection"``).
    n_pixels : int
        Number of wavelength pixels per order.
    night1, night2 : int
        Night indices to compare.
    name : str
        Label used in saved figure filenames.
    save_plot : bool
        If True, save the figure to the simulation output directory.
    """
    sn_night1 = np.zeros((inp_dat['n_orders'], n_pixels))
    sn_night2 = np.zeros((inp_dat['n_orders'], n_pixels))
    for h in range(inp_dat['n_orders']):
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
        os.makedirs(f"{inp_dat['plots_dir']}statistical", exist_ok=True)
    # Plot
    plt.plot(inp_dat['order_selection'], sn_night1_mean, 'ko-', label = f"Night {night1}")
    plt.plot(inp_dat['order_selection'], sn_night2_mean, 'ro-', label = f"Night {night2}")
    plt.xlabel("Empirical S/N", fontsize = 17)
    plt.ylabel("Spectral order", fontsize = 17)
    #plt.xticks(np.arange(0, inp_dat['n_orders'], 1))
    plt.tick_params(axis = 'both', width = 1.8, direction = 'in', 
            labelsize=15)
    plt.legend(loc='best', prop={'size': 12})
    
    if save_plot:
        plt.savefig(
            f"{inp_dat['plots_dir']}statistical/"
            f"{name}_empirical_SN.png",
            bbox_inches='tight')

    plt.show()
    plt.close()
    
    return 


def plot_std_errors(inp_dat, save_plot, error, prop_error, stats
        ):
    """Plot the per-night standard deviations of pipeline noise residuals.

    Computes the standard deviation of the residual error and propagated
    noise cubes across orders, spectra, and pixels for each night, then
    plots both quantities alongside the per-night peak CCF S/N from
    ``stats``.  Useful for diagnosing nights where the noise model
    disagrees with the actual pipeline residuals.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary; uses ``'n_nights'``, ``'plots_dir'``,
        ``'Simulation_name'``.
    save_plot : bool
        If True, save the figure as a PDF.
    error : ndarray, shape (n_nights, n_orders, n_spectra, n_pixels)
        Residual error array (data - model) from the pipeline.
    prop_error : ndarray, shape (n_nights, n_orders, n_spectra, n_pixels)
        Propagated noise (σ) array from the pipeline.
    stats : ndarray, shape (n_nights, 3)
        Per-night detection statistics; column 0 is the peak CCF S/N.
    """
    std_error = np.zeros((inp_dat['n_nights']))
    std_prop_error = np.zeros((inp_dat['n_nights']))
    #axs[0, 0].set_title(f"Order {inp_dat['order_selection
    for n in range(inp_dat['n_nights']): 
        std_prop_error[n] = np.std(prop_error[n,:,:,:])
        std_error[n] = np.std(error[n,:,:,:])
        
    if save_plot:
        os.makedirs(f"{inp_dat['plots_dir']}statistical", exist_ok=True)
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
            f"{inp_dat['plots_dir']}statistical/"
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
            f"{inp_dat['plots_dir']}statistical/"
            f"std_propagated_error.png",
            bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    
    return 


def compare_KpVr_dist(inp_dat, v_rest, ccf_matrix, night1,
                      night2, saveplot, SNR = True):
    """Compare the Kp-Vsys signal distributions between two nights.

    Produces a four-panel diagnostic figure showing: the Kp-Vsys maps for
    ``night1`` and ``night2`` zoomed around the expected planet position,
    a scatter plot of signal values in the planet region between the two
    nights (useful for spotting outliers), and a histogram of the
    per-velocity CCF values for each night.  Helps identify nights with
    anomalously strong or weak detections.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary; uses ``'V_wind'``, ``'K_p'``,
        ``'kp_max'``, ``'plots_dir'``, ``'Simulation_name'``.
    v_rest : ndarray, shape (n_v_rest,)
        Planet rest-frame velocity axis in km/s.
    ccf_matrix : ndarray, shape (n_v_rest, n_kp_steps, n_nights)
        Kp-Vsys significance map.
    night1, night2 : int
        Indices of the two nights to compare (0-based).
    saveplot : bool
        If True, save the figure as a PDF.
    SNR : bool
        If True, treat ``ccf_matrix`` values as S/N; if False, as raw CCF.
    """
    plt.close()
    gs = gridspec.GridSpec(1, 4)
    fig = plt.figure(figsize=(16,4))
    

    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    signal_area1 = ccf_matrix[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                  int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 60, night1]
    signal_area2 = ccf_matrix[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                  int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 60, night2]

    
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
        ccf_matrix[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15), 
                   320 - 30 : 320 + 30, night1]
        )
    telluric_area2 = np.ndarray.flatten(
        ccf_matrix[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15), 
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
        ccf_matrix, np.s_[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15)], 
        axis=0
        )
    
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, np.s_[320-40:320+40], 
        axis=1
        )
    # Removing planet signal
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, 
        np.s_[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]-5:np.argwhere(v_rest == inp_dat['V_wind'])[0][0]+5], 
        axis=0
        )
    
    away_from_signal_and_tellurics = np.delete(
        ccf_matrix, np.s_[int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40:int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40],
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


def get_corr_coeff(inp_dat, with_signal, data, model, color_variable,
        h, stats, title, night_max, night_min, phase, plotname,
        CC_2D = True, show_plot = False, save_plot = True
        ):
    """Compute and plot the correlation between spectral residuals and a model.

    For each night, computes the Pearson correlation coefficient between the
    pipeline residual data and the injected atmospheric model spectrum in the
    in-transit frames.  The per-night correlation is plotted against
    ``color_variable`` (typically orbital phase, airmass, or S/N) to look for
    trends that would indicate systematic effects or signal leakage.

    In 2D mode (``CC_2D=True``) the correlation is computed as a function of
    both order and night, producing a 2D diagnostic.  In 1D mode a single
    correlation coefficient per night is returned.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.
    with_signal : ndarray or list
        In-transit exposure indices per night.
    data : ndarray, shape (n_orders, n_nights, n_spectra, n_pixels)
        Pipeline residual data.
    model : ndarray, shape (n_orders, n_nights, n_spectra, n_pixels)
        Injected atmospheric model spectra.
    color_variable : ndarray, shape (n_nights,)
        Per-night scalar variable used to colour the scatter plot
        (e.g., mean airmass, BERV, or peak S/N).
    h : int
        Spectral order index to use for the 1D correlation.
    stats : ndarray, shape (n_nights, 3)
        Per-night detection statistics; column 0 is peak S/N.
    title : str
        Figure title string.
    night_max, night_min : int
        Indices of the best and worst nights (from
        :func:`find_nights_with_extrema`), highlighted in the plot.
    phase : ndarray or list
        Orbital phase arrays.
    plotname : str
        Output filename suffix for the saved PDF.
    CC_2D : bool
        If True, compute the 2D (order × night) correlation matrix.
    show_plot, save_plot : bool
        Display and/or save the figure.
    """
    
    if inp_dat['first_night_noiseless']: 
        stats_0 = stats[1:, 0]
    else: stats_0 = stats[:, 0]

    if CC_2D:
        corr_coeff=np.zeros(
            (inp_dat['n_nights']-1), float
            )
        #standard_error=np.zeros(
        #    (inp_dat['n_nights']-1), float
        #    )
        for n in range(1,inp_dat['n_nights']):
            X = data[h, n, with_signal, :].flatten()
            Y = model[h, with_signal,:].flatten()
            corr_coeff[n-1] = np.corrcoef(
                X, 
                Y
                )[0,1]
            #standard_error[n-1] = bootstrap_corrcoeffs(X, Y)
    else:
        corr_coeff=np.zeros(
            (inp_dat['n_nights']-1, len(with_signal)), float
            )
        for n in range(1,inp_dat['n_nights']):
            for idx, i in enumerate(with_signal):
                corr_coeff[n-1, idx] = np.corrcoef(
                    data[h, n, i, :], model[h, i, :]
                    )[0,1]
    
    # Now look for a correlation between higher correlations
    # and higher S/N of the MSS in the canonical analysis

    # Calculate Pearson correlation coefficient and p-value
    if CC_2D:
        X = stats_0
        Y = corr_coeff
        pearson_coeff = sc.pearsonr(X, Y)[0]
        standard_error = bootstrap_corrcoeffs(X, Y)
        
    else:
        pearson_coeff = sc.pearsonr(stats_0, 
                                    np.sum(corr_coeff, axis = 1))
        
    #print(f"Pearson coeff & p-value = {pearson_coeff}")
    
    if show_plot:
        plt.close()
        plt.figure(figsize=(8, 6))
        if CC_2D:
            plt.scatter(stats_0, corr_coeff, 
                        c=color_variable, cmap='viridis', 
                        marker='o', s = 70, edgecolors='k',
                        label=f"Pearson coeff & p-value = "
                        f"{np.round(pearson_coeff, 5)}")
            colorbar = plt.colorbar()

            # Set the fontsize for the colorbar labels and tick labels
            colorbar.ax.tick_params(labelsize=14)
            colorbar.set_label(label = 'Night index', fontsize=17)
        else:
            plt.plot(stats_0, np.sum(corr_coeff, axis = 1), 
                     'k', marker = 'o', linewidth = 0,
                     label=f"Pearson coeff & p-value = "
                     f"{np.round(pearson_coeff, 5)}")
            
        plt.tick_params(axis='both', width=1.5, direction='in',
                        labelsize=16)
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Corr. Coeff.', fontsize=17)
        plt.title(title, fontsize=17)
        plt.legend()
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        if save_plot:
            plt.savefig(f"{inp_dat['plots_dir']}{plotname}.pdf")
        plt.show()
        plt.close()
    
    if CC_2D:
        return pearson_coeff, standard_error
    else: return np.sum(corr_coeff, axis = 1)


def bootstrap_corrcoeffs(X, Y, samples = 1000):
    """Estimate the standard error of the Pearson correlation coefficient by bootstrapping.

    Resamples (X, Y) with replacement ``samples`` times and computes the
    Pearson correlation coefficient for each resample.  The standard deviation
    of the resulting distribution is returned as the standard error of the
    original correlation.

    Parameters
    ----------
    X, Y : array-like, shape (n,)
        Two arrays of equal length to correlate (e.g., per-night S/N values
        and per-night model-data correlation coefficients).
    samples : int
        Number of bootstrap resamples (default 1000).

    Returns
    -------
    float
        Bootstrap standard error of the Pearson correlation coefficient.
    """
    # Number of bootstrap samples
    num_samples = samples

    # Store the calculated correlation coefficients
    bootstrap_corrcoeffs = []

    # Perform bootstrapping
    for _ in range(num_samples):
        # Resample with replacement
        resampled_x = np.random.choice(X, size=len(X), replace=True)
        resampled_y = np.random.choice(Y, size=len(Y), replace=True)

        # Calculate Pearson correlation coefficient for the resampled data
        correlation = np.corrcoef(resampled_x, resampled_y)[0, 1]

        bootstrap_corrcoeffs.append(correlation)

    # Calculate the standard error of the correlation coefficients
    return np.std(bootstrap_corrcoeffs)


def compare_correlations(inp_dat, corr_x, corr_y, filename_flag, plotname,
        xlabel, ylabel, title="", plot_lims = None,
        show_plot = True, save_plot = True
        ):
    """Scatter-plot two per-night arrays and compute their Pearson correlation.

    Plots ``corr_y`` vs ``corr_x`` with a linear best-fit line and annotates
    with the Pearson correlation coefficient and its bootstrap standard error.
    Typical use is to compare the S/N distribution from the noise-only CCF
    against that from the signal+noise CCF to assess how strongly noise
    properties predict detection significance.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary (``"correlations_dir"``,
        ``"Simulation_name"``).
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
        plt.savefig(f"{inp_dat['correlations_dir']}{inp_dat['Simulation_name']}/{plotname}.pdf")
    if show_plot: plt.show()
    plt.close()
    return original_correlation, standard_error


def find_nights_with_extrema(stats, first_night_noiseless):
    """Return the indices of the best and worst nights by peak CCF S/N.

    Scans ``stats[:, 0]`` (the per-night peak detection S/N) and returns the
    indices of the minimum and maximum values.  When the first night is
    noiseless (``first_night_noiseless=True``), it is excluded from the
    maximum search so that the noiseless reference night does not trivially
    win.

    Parameters
    ----------
    stats : ndarray, shape (n_nights, 3)
        Per-night statistics array; column 0 is peak CCF S/N, column 1 is
        best-fit Kp (km/s), column 2 is best-fit Vrest (km/s).
    first_night_noiseless : bool
        If True, exclude night index 0 from the maximum search.

    Returns
    -------
    night_min : int
        Index of the night with the lowest peak S/N.
    night_max : int
        Index of the night with the highest peak S/N (excluding night 0 when
        ``first_night_noiseless=True``).
    """
    night_min = np.where(stats[:, 0] == stats[:, 0].min())[0][0]
    if first_night_noiseless:
        night_max = np.where(stats[1:, 0] == stats[1:, 0].max())[0][0] + 1
    else:
        night_max = np.where(stats[:, 0] == stats[:, 0].max())[0][0]
    return night_min, night_max


def perform_correlations_with_noise(inp_dat, stats, stats_tvalue, stats_pvalue,
        stats_planet_pos, #stats_cc_values,
        #stats_cc_values_planet_pos, stats_cc_values_std,
        #stats_cc_values_std_planet_pos,
        stats_noise, stats_tvalue_noise, stats_pvalue_noise,
        stats_planet_pos_noise,
        show_plot = False, save_plot = True, etiqueta = "",
        ):
    """Correlate signal-detection statistics with noise-only statistics across nights.

    For each significance metric (S/N maximum, Welch t-value, Welch p-value,
    and signal at the true planet position) this function calls
    :func:`compare_correlations` to measure the Pearson correlation between
    the per-night statistics from the noise-only CCF and those from the
    signal+noise CCF.  A high correlation indicates that the nightly noise
    properties dominate the detection variability; a low correlation suggests
    the signal itself drives night-to-night scatter.

    Results are collected into an ``outputs`` dict and saved as a compressed
    NPZ file in the correlations directory.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary (``"CCF_SNR"``, ``"Welch_ttest"``,
        ``"All_significance_metrics"``, ``"Simulation_name"``,
        ``"Noise_scaling_factor"``, ``"correlations_dir"``).
    stats : ndarray, shape (n_nights, 3)
        Signal+noise detection statistics; column 0 is the S/N maximum.
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
    if not inp_dat["CCF_SNR"] and inp_dat["Welch_ttest"]:
        mean_SNR_tvalue, mean_SNR_error_tvalue = norm.fit(stats_tvalue[:,0])
        mean_SNR_pvalue, mean_SNR_error_pvalue = norm.fit(stats_pvalue[:,0])
        stats_tvalue = stats_tvalue[:, 0]
        stats_pvalue = stats_pvalue[:, 0]
        stats_tvalue_noise = stats_tvalue_noise[:, 0]
        stats_pvalue_noise = stats_pvalue_noise[:, 0]
    elif inp_dat["CCF_SNR"] and not inp_dat["Welch_ttest"]: 
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

    filename_flag = format_number(inp_dat["Noise_scaling_factor"])
    pearson_coeff_SNR_max, pearson_coeff_SNR_error_max = compare_correlations(
        inp_dat, stats_noise, stats, filename_flag,
        plotname = f"scatter_MAX_{filename_flag}",
        xlabel="", ylabel="", title="MSS",
        plot_lims=[np.amin([stats_noise,stats])-1,
                   np.amax([stats_noise,stats])+1], 
        show_plot = show_plot, save_plot = save_plot
        )
    
    if inp_dat["Welch_ttest"]:
        pearson_coeff_SNR_max_tvalue, pearson_coeff_SNR_error_max_tvalue = compare_correlations(
            inp_dat, stats_tvalue_noise, stats_tvalue, filename_flag,
            plotname = f"scatter_MAX_tvalue_{filename_flag}",
            xlabel="", ylabel="", title="MSS",
            plot_lims=[np.amin([stats_tvalue_noise,stats_tvalue])-1,
                       np.amax([stats_tvalue_noise,stats_tvalue])+1], 
            show_plot = show_plot, save_plot = save_plot
            )
    
        pearson_coeff_SNR_max_pvalue, pearson_coeff_SNR_error_max_pvalue = compare_correlations(
            inp_dat, stats_pvalue_noise, stats_pvalue, filename_flag,
            plotname = f"scatter_MAX_pvalue_{filename_flag}",
            xlabel="", ylabel="", title="MSS",
            plot_lims=[np.amin([stats_pvalue_noise,stats_pvalue])-1,
                       np.amax([stats_pvalue_noise,stats_pvalue])+1], 
            show_plot = show_plot, save_plot = save_plot
            )
    else:
        pearson_coeff_SNR_max_tvalue = None
        pearson_coeff_SNR_error_max_tvalue = None
        pearson_coeff_SNR_max_pvalue = None
        pearson_coeff_SNR_error_max_pvalue = None
    
    pearson_coeff_SNR_planet_pos, pearson_coeff_SNR_error_planet_pos = compare_correlations(
        inp_dat, stats_planet_pos_noise, stats_planet_pos, filename_flag,
        plotname = f"scatter_PLANETPOS_{filename_flag}",
        xlabel="", ylabel="", title="planet_pos",
        plot_lims=[np.amin([stats_planet_pos_noise,stats_planet_pos])-1,
                   np.amax([stats_planet_pos_noise,stats_planet_pos])+1], 
        show_plot = show_plot, save_plot = save_plot
        )
    
    # Save the plotting variables in a dictionary
    
    outputs = {}
    outputs['scaling_factors_noise'] = inp_dat["Noise_scaling_factor"]
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
    if not inp_dat["All_significance_metrics"]:
        filename = f"{inp_dat['correlations_dir']}/outputs_{inp_dat['Simulation_name']}" 
        np.savez_compressed(filename, a = outputs)
    else:
        filename = f"{inp_dat['correlations_dir']}/outputs_{inp_dat['Simulation_name']}_{etiqueta}" 
        np.savez_compressed(filename, a = outputs)
    return
