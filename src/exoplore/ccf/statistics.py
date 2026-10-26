"""
exoplore.ccf.statistics
========================

CCF-based detection statistics.

Functions
---------
Welch_ttest_map
    Welch's t-test significance map on the Kp-Vsys grid.
Combine_Nights
    Randomly co-add groups of simulated nights for significance studies.
statistical_study
    Full simulation-based significance study (n_nights bootstrap runs).
"""

from __future__ import annotations

from exoplore.ccf.models import NightStudyInputs

# Role in the calculation
# Calculate Welch detection maps and night-combination statistics from CCFs. The
# selected in-trail and out-of-trail samples define each statistic.


import random

import numpy as np
from scipy.stats import norm, ttest_ind


def Welch_ttest_map(
    ccf_values_shift,
    v_rest,
    kp_range,
    stats=None,
    stats_tvalue=None,
    stats_pvalue=None,
    b=None,
    CCF_Noise=False,
    plotting=False,
    *,
    in_trail_left_right,
    snr_exclude_kms,
    velocity_step_kms
):
    """Compute a Welch's t-test significance map on the Kp-Vsys grid.

    For each (Kp, v_rest) position, an in-trail and out-of-trail sample
    are defined and a Welch t-test is performed.  The resulting sigma
    map indicates the significance of any excess in-trail signal.

    Parameters
    ----------
    ccf_values_shift : ndarray, shape (n_v, n_transit, n_kp)
        CCF matrix shifted to the planet rest frame.
    v_rest : ndarray
        Rest-frame velocity grid (km/s).
    kp_range : ndarray
        Trial Kp values (km/s).
    in_trail_left_right
        Number of velocity pixels on either side of the in-trail centre.
    snr_exclude_kms
        Velocity interval excluded from the out-of-trail sample in km/s.
    velocity_step_kms
        Velocity grid spacing in km/s.
    stats : ndarray or None
        Bootstrap statistics (used only when ``CCF_Noise=True``).
    stats_tvalue : ndarray or None
        Bootstrap t-value statistics (noise mode).
    stats_pvalue : ndarray or None
        Bootstrap p-value statistics (noise mode).
    b : int or None
        Bootstrap index (noise mode).
    CCF_Noise : bool
        If ``True``, report significance at the position given by ``stats[b]``.

    Returns
    -------
    sigma_values : ndarray, shape (n_v - 2*left_right, n_kp)
    t_values : ndarray, shape (n_v - 2*left_right, n_kp)
    p_values : ndarray, shape (n_v - 2*left_right, n_kp)
    v_rest_sigma : ndarray
    max_sigma_value : float
    max_kp_idx_sigma : int
    max_v_rest_sigma : float
    max_t_value : float
    max_kp_idx_t : int
    max_v_rest_t : float
    max_p_value : float
    max_kp_idx_p : int
    max_v_rest_p : float
    """
    left_right = in_trail_left_right
    max_ttest = 0
    max_kp_idx = 0
    max_v_rest_ttest = 0
    t_values = np.zeros(
        (len(v_rest) - 2 * left_right, len(kp_range)), float
        )
    p_values = np.zeros(
        (len(v_rest) - 2 * left_right, len(kp_range)), float
        )
    sigma_values = np.zeros(
        (len(v_rest) - 2 * left_right, len(kp_range)), float
        )
    v_rest_sigma = v_rest[left_right:-left_right]
    full_range_pts = np.arange(len(v_rest_sigma))

    safety_window = int(snr_exclude_kms / velocity_step_kms)
    for kp in range(len(kp_range)):
        for v_idx in range(len(v_rest_sigma)):
            in_trail_pts = np.arange(
                v_idx - left_right,
                v_idx + left_right + 1,
                1
                )
            start_exclude = max(0, np.min(in_trail_pts) - safety_window)
            end_exclude = min(len(v_rest_sigma),
                              np.max(in_trail_pts) + safety_window + 1)
            out_trail_pts = np.setdiff1d(
                full_range_pts,
                np.arange(start_exclude, end_exclude)
                )
            in_trail_data = np.ndarray.flatten(
                ccf_values_shift[in_trail_pts, :, kp]
                )
            out_trail_data = np.ndarray.flatten(
                ccf_values_shift[out_trail_pts, :, kp]
                )
            t_values[v_idx, kp], p_values[v_idx, kp] = ttest_ind(
                in_trail_data, out_trail_data, equal_var=False
                )
            sigma_values[v_idx, kp] = abs(norm.ppf(p_values[v_idx, kp] / 2))

    if not CCF_Noise:
        max_index_sigma = np.unravel_index(
            np.argmax(sigma_values, axis=None), sigma_values.shape
            )
        max_index_t = np.unravel_index(
            np.argmax(t_values, axis=None), t_values.shape
            )
        max_index_p = np.unravel_index(
            np.argmin(p_values, axis=None), p_values.shape
            )
        max_sigma_value = sigma_values[max_index_sigma]
        max_t_value = t_values[max_index_t]
        max_p_value = p_values[max_index_p]
        max_kp_idx_sigma = max_index_sigma[1]
        max_kp_idx_t = max_index_t[1]
        max_kp_idx_p = max_index_p[1]
        max_v_rest_sigma = v_rest_sigma[max_index_sigma[0]]
        max_v_rest_t = v_rest_sigma[max_index_t[0]]
        max_v_rest_p = v_rest_sigma[max_index_p[0]]

    elif CCF_Noise:
        max_kp_idx_sigma = int(stats[b, 1] + len(kp_range) // 2)
        max_v_rest_sigma = stats[b, 2]
        max_sigma_value = sigma_values[
            np.argwhere(v_rest_sigma == stats[b, 2])[0][0],
            max_kp_idx
            ]
        max_kp_idx_t = int(stats_tvalue[b, 1] + len(kp_range) // 2)
        max_v_rest_t = stats_tvalue[b, 2]
        max_t_value = t_values[
            np.argwhere(v_rest_sigma == stats_tvalue[b, 2])[0][0],
            max_kp_idx_t
            ]
        max_kp_idx_p = int(stats_pvalue[b, 1] + len(kp_range) // 2)
        max_v_rest_p = stats_pvalue[b, 2]
        max_p_value = p_values[
            np.argwhere(v_rest_sigma == stats_pvalue[b, 2])[0][0],
            max_kp_idx_p
            ]

    return (sigma_values, t_values, p_values, v_rest_sigma,
            max_sigma_value, max_kp_idx_sigma, max_v_rest_sigma,
            max_t_value, max_kp_idx_t, max_v_rest_t,
            max_p_value, max_kp_idx_p, max_v_rest_p)


def Combine_Nights(ccf, CCF_Noise, previous_shuffle, *, stack_group_size, n_nights):
    """Randomly co-add groups of simulated nights.

    Produces ``n_nights`` random combinations, each of size
    ``stack_group_size``.  When ``CCF_Noise=True`` the
    shuffling from the "real" data run is reused for consistency.

    Parameters
    ----------
    stack_group_size
        Number of nights in each random combination.
    n_nights
        Number of available nights.
    ccf : ndarray, shape (n_nights, ...)
        CCF array indexed by night along axis 0.
    CCF_Noise : bool
        If ``True``, reuse the indices in ``previous_shuffle``.
    previous_shuffle : list or None
        Shuffled night indices from the paired "real" data run (noise mode).

    Returns
    -------
    combined_ccf : ndarray
        Co-added CCF, same shape as ``ccf``.
    shuffled_nights : list or None
        The random combination used (``None`` in noise mode).
    """
    if stack_group_size > n_nights:
        return "Stack_Group_Size is greater than n_nights"

    combined_ccf = np.zeros_like(ccf)

    if CCF_Noise:
        for i in range(n_nights):
            combined_ccf[i, :] = np.sum(
                ccf[previous_shuffle[i], :], axis=0
                )
        return combined_ccf, None

    observed_nights = np.arange(n_nights)
    shuffled_nights = list()
    for i in range(n_nights):
        shuffled_nights.append(
            random.sample(list(observed_nights), stack_group_size)
            )

    for i in range(n_nights):
        combined_ccf[i, :] = np.sum(ccf[shuffled_nights[i], :], axis=0)

    return combined_ccf, shuffled_nights


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


def statistical_study(
    inputs: NightStudyInputs,
    ccf_v_step,
    ccf_stat,
    kp_range,
    phase,
    v_ccf,
    v_rest,
    with_signal,
    pixels_left_right,
    sysrem_it_opt,
    ccf_iterations,
    in_trail_pix,
    auto_lims=True,
    input_stats=None,
    input_stats_tvalue=None,
    input_stats_pvalue=None,
    previous_shuffle=None,
    verbose=True,
    show_plot=False,
    save_plot=True,
    CCF_Noise=False,
    ccf_SSIM=False,
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
    inputs : NightStudyInputs
        Statistic, optimisation choices, night counts and reference motion.
    ccf_v_step : float
        Velocity step of the CCF grid in km/s.
    ccf_stat : ndarray
        CCF matrix cube from the simulator.  Shape depends on the pipeline
        mode: ``(n_orders, n_nights, n_lags, n_spectra)`` (standard) or
        ``(n_orders, n_nights, n_lags, n_spectra, 2, sysrem_its)``
        (order-by-order SYSREM optimisation).
    kp_range : ndarray, shape (n_kp_steps,)
        Grid of trial Kp values in km/s over which the map is evaluated.
    Phase : ndarray
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
        ``inputs.pipeline.optimize_sysrem_order_by_order`` is True.
    ccf_iterations : int
        Total number of CCF lag steps.
    in_trail_pix : int
        Width of the in-trail velocity window (pixels, must be odd) used
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
        If True, process the noise-only CCF.
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
    from exoplore.ccf.compute import get_max_CCF_peak
    from exoplore.observation.velocity import get_V


    # Co-adding of orders in each night with NO WEIGHTS
    if np.logical_or(not inputs.pipeline.optimize_sysrem_order_by_order, CCF_Noise):
        if len(ccf_stat.shape) == 4:
            ccf_complete_stat = np.sum(ccf_stat, 0)
        else: ccf_complete_stat = ccf_stat

        # Analyse n_nigts individually or study co-addings? This
        # allows us to study how the varibility in signal's
        # significance is (hopefully) reduced when co-adding nights.
        if inputs.stack_group_size is not None and inputs.stack_group_size > 1:
            ccf_complete_stat, shuffled_nights= Combine_Nights(
                ccf_complete_stat,
                CCF_Noise,
                previous_shuffle,
                stack_group_size=inputs.stack_group_size,
                n_nights=inputs.n_nights,
            )
        else: shuffled_nights = None
    else:
        # We can already set the criterion selected by the user
        if inputs.pipeline.optimize_criterion == "Maximum":
            crit_choice = 0
        elif inputs.pipeline.optimize_criterion == "Max_Diff":
            crit_choice = 1


    # Define planet velocity vector as a function of Kp
    vp = np.zeros((kp_range.shape[0], len(phase)))
    for k in range(len(kp_range)):
        vp[k, :] = get_V(
            kp_range[k], phase, inputs.berv_kms,
            inputs.systemic_velocity_kms, inputs.reference_vrest_kms
            )

    for b in range(inputs.n_nights):

        # If optimising SYSREM its., then the actual ccf_complete_stat
        # has to be computed for each night's selection iterations
        # for each order

        if inputs.pipeline.optimize_sysrem_order_by_order and not CCF_Noise:

            if sysrem_it_opt.shape[0] == inputs.n_orders and len(ccf_stat.shape) == 6:
                # Extract the relevant data based on sysrem_it_opt
                # Create a new matrix for storing the selected data
                ccf_complete_stat = np.zeros(
                    (ccf_stat.shape[:4]), float
                    )

                # Iterate over dimensions to select data based on sysrem_it_opt
                for h in range(inputs.n_orders):
                    #ccf_complete_stat = ccf_stat[:,:,:,:,0,2]
                    for n in range(ccf_stat.shape[3]): # Loop in spectra
                        sysrem_index = sysrem_it_opt[h, b, crit_choice]
                        ccf_complete_stat[h, b, :, n] = ccf_stat[h, b, :, n, 0, sysrem_index]

                # Co-adding the orders with the selected iterations
                ccf_complete_stat = np.sum(ccf_complete_stat, axis=0)
            # Now we make sure the shuffling and co-adding of nights
            # gets done only once
            if b == 0:
                if inputs.stack_group_size is not None and inputs.stack_group_size > 1:
                    ccf_complete_stat, shuffled_nights= Combine_Nights(
                        ccf_complete_stat,
                        CCF_Noise,
                        previous_shuffle,
                        stack_group_size=inputs.stack_group_size,
                        n_nights=inputs.n_nights,
                    )
                else: shuffled_nights = None

        # The rest of the loop is the same regardless
        # of SYSREM optimisation
        if inputs.n_nights > 20 and (b % 10 == 0) and verbose:
            print('STATISTICAL STUDY: Co-adding night ' + str(b+1) + '/'
              + str(inputs.n_nights))

        ######################################################################

        ######################################################################

        # Variable that stores all shifts as a function of Kp
        if b == 0:
            left_right = in_trail_pix #// 2
            ccf_values_shift_stat = np.zeros((len(v_rest), len(with_signal),
                                              kp_range.shape[0]), float)
            ccf_tot_stat = np.zeros((len(v_rest), kp_range.shape[0],
                                     inputs.n_nights),
                                    float)
            if inputs.cross_correlation.ccf_snr:
                ccf_tot_sn_stat = np.zeros_like(ccf_tot_stat)
            elif inputs.cross_correlation.welch_ttest:
                ccf_tot_sigma_stat = np.zeros(
                    (v_rest.shape[0] - 2 * left_right, len(kp_range),
                     inputs.n_nights)
                    )
                ccf_tot_t_stat = np.zeros(
                    (v_rest.shape[0] - 2 * left_right, len(kp_range),
                     inputs.n_nights)
                    )
                ccf_tot_p_stat = np.zeros(
                    (v_rest.shape[0] - 2 * left_right, len(kp_range),
                     inputs.n_nights)
                    )

            stats = np.zeros((inputs.n_nights, 3))
            stats_tvalue = np.zeros((inputs.n_nights, 3))
            stats_pvalue = np.zeros((inputs.n_nights, 3))
            stats_planet_pos = np.zeros((inputs.n_nights, 3))
            stats_planet_area = np.zeros((inputs.n_nights, 3))
            stats_cc_values = np.zeros((inputs.n_nights, 3))
            stats_cc_values_planet_pos = np.zeros((inputs.n_nights, 3))
            stats_cc_values_std = np.zeros((inputs.n_nights, 3))
            stats_cc_values_std_planet_pos = np.zeros((inputs.n_nights, 3))

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


        if inputs.cross_correlation.ccf_snr and not CCF_Noise:
            ccf_tot_sn_stat[:,:,b], max_sig, max_kp_idx, max_v_rest, cc_values_std = \
                get_max_CCF_peak(
                    ccf_tot=ccf_tot_stat[:, :, b],
                    v_rest=v_rest,
                    kp_range=kp_range,
                    b=None,
                    stats=None,
                    sysrem_opt=False,
                    CCF_Noise=False,
                    snr_exclude_kms=inputs.cross_correlation.snr_exclude_kms,
                    sysrem_iterations=inputs.sysrem_iterations,
                    exclude_around=inputs.exclude_around,
                    noise_source=inputs.cross_correlation.snr_noise_source,
                    noise_kp_exclude_kms=inputs.cross_correlation.snr_noise_kp_exclude_kms,
                )
            significance_metric = ccf_tot_sn_stat
            significance_metric2 = None
            significance_metric3 = None
            v_rest_sigma = None
        elif inputs.cross_correlation.ccf_snr and CCF_Noise:
            ccf_tot_sn_stat[:,:,b], max_sig_noise, max_kp_noise_idx, max_v_rest_noise, cc_values_std_noise = \
                get_max_CCF_peak(
                    ccf_tot_stat[:, :, b],
                    v_rest,
                    kp_range,
                    b,
                    input_stats,
                    False,
                    CCF_Noise=True,
                    snr_exclude_kms=inputs.cross_correlation.snr_exclude_kms,
                    sysrem_iterations=inputs.sysrem_iterations,
                    exclude_around=inputs.exclude_around,
                    noise_source=inputs.cross_correlation.snr_noise_source,
                    noise_kp_exclude_kms=inputs.cross_correlation.snr_noise_kp_exclude_kms,
                )
            significance_metric = ccf_tot_sn_stat
            significance_metric2 = None
            significance_metric3 = None
            v_rest_sigma = None
        elif not inputs.cross_correlation.ccf_snr and inputs.cross_correlation.welch_ttest and not CCF_Noise:
            #custom_start_time = time.time()
            ccf_tot_sigma_stat[:,:,b], ccf_tot_t_stat[:,:,b], ccf_tot_p_stat[:,:,b], v_rest_sigma , max_sig, max_kp_idx, max_v_rest, max_t_value, max_kp_idx_t, max_v_rest_t, max_p_value, max_kp_idx_p, max_v_rest_p = \
                Welch_ttest_map(
                    ccf_values_shift_stat,
                    v_rest,
                    kp_range,
                    CCF_Noise=CCF_Noise,
                    plotting=show_plot,
                    in_trail_left_right=inputs.cross_correlation.in_trail_left_right,
                    snr_exclude_kms=inputs.cross_correlation.snr_exclude_kms,
                    velocity_step_kms=inputs.cross_correlation.velocity_step_kms,
                )
            significance_metric = ccf_tot_sigma_stat
            significance_metric2 = ccf_tot_t_stat
            significance_metric3 = ccf_tot_p_stat
        elif not inputs.cross_correlation.ccf_snr and inputs.cross_correlation.welch_ttest and CCF_Noise:
            #custom_start_time = time.time()
            ccf_tot_sigma_stat[:,:,b], ccf_tot_t_stat[:,:,b], ccf_tot_p_stat[:,:,b], v_rest_sigma, max_sig_noise, max_kp_noise_idx, max_v_rest_noise, max_t_value_noise, max_kp_idx_t_noise, max_v_rest_t_noise, max_p_value_noise, max_kp_idx_p_noise, max_v_rest_p_noise  = \
                Welch_ttest_map(
                    ccf_values_shift_stat,
                    v_rest,
                    kp_range,
                    stats=input_stats,
                    stats_tvalue=input_stats_tvalue,
                    stats_pvalue=input_stats_pvalue,
                    b=b,
                    CCF_Noise=CCF_Noise,
                    plotting=show_plot,
                    in_trail_left_right=inputs.cross_correlation.in_trail_left_right,
                    snr_exclude_kms=inputs.cross_correlation.snr_exclude_kms,
                    velocity_step_kms=inputs.cross_correlation.velocity_step_kms,
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

            if inputs.cross_correlation.ccf_snr:
                stats_cc_values_std[b, 0] = cc_values_std[np.argwhere(v_rest == max_v_rest)[0][0],
                                                          max_kp_idx]
                stats_cc_values_std[b, 1] = max_kp_idx - (len(kp_range) // 2)
                stats_cc_values_std[b, 2] = max_v_rest
            else:
                stats_cc_values_std = None

            # And now the stats at exactly the Kp-Vrest of the planet
            if inputs.cross_correlation.ccf_snr:
                stats_planet_pos[b, 0] = ccf_tot_sn_stat[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0],
                                                         int(np.ceil(inputs.reference_kp_kms)+len(kp_range)//2),
                                                         b]
                stats_planet_pos[b, 1] = np.ceil(inputs.reference_kp_kms)
                stats_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0]]

                stats_tvalue = None
                stats_pvalue = None
            elif inputs.cross_correlation.welch_ttest:

                stats_tvalue[b, 0] = max_t_value
                stats_tvalue[b, 1] = max_kp_idx_t - (len(kp_range) // 2)
                stats_tvalue[b, 2] = max_v_rest_t

                stats_pvalue[b, 0] = max_p_value
                stats_pvalue[b, 1] = max_kp_idx_p - (len(kp_range) // 2)
                stats_pvalue[b, 2] = max_v_rest_p

                stats_planet_pos[b, 0] = ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inputs.reference_vrest_kms)[0][0],
                                                         int(np.ceil(inputs.reference_kp_kms)+inputs.cross_correlation.kp_max_kms),
                                                         b]
                stats_planet_pos[b, 1] = np.ceil(inputs.reference_kp_kms)
                stats_planet_pos[b, 2] = v_rest_sigma[np.argwhere(v_rest_sigma == inputs.reference_vrest_kms)[0][0]]

            # Stats at around the Kp-Vrest of the planet
            if inputs.cross_correlation.ccf_snr:
                (stats_planet_area[b, 0],
                 stats_planet_area[b, 1],
                 stats_planet_area[b, 2]) = _planet_area_stats(
                    ccf_tot_sn_stat, v_rest,
                    inputs.reference_kp_kms, inputs.cross_correlation.kp_max_kms,
                    inputs.reference_vrest_kms, kp_range, b)

            elif inputs.cross_correlation.welch_ttest:
                (stats_planet_area[b, 0],
                 stats_planet_area[b, 1],
                 stats_planet_area[b, 2]) = _planet_area_stats(
                    ccf_tot_sigma_stat, v_rest_sigma,
                    inputs.reference_kp_kms, inputs.cross_correlation.kp_max_kms,
                    inputs.reference_vrest_kms, kp_range, b)



            stats_cc_values_planet_pos[b, 0] = ccf_tot_stat[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0],
                                                     int(np.ceil(inputs.reference_kp_kms)+inputs.cross_correlation.kp_max_kms),
                                                     b]
            stats_cc_values_planet_pos[b, 1] = np.ceil(inputs.reference_kp_kms)
            stats_cc_values_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0]]

            if inputs.cross_correlation.ccf_snr:
                stats_cc_values_std_planet_pos[b, 0] = cc_values_std[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0],
                                                         int(np.ceil(inputs.reference_kp_kms)+inputs.cross_correlation.kp_max_kms)]
                stats_cc_values_std_planet_pos[b, 1] = np.ceil(inputs.reference_kp_kms)
                stats_cc_values_std_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0]]
            else:
                stats_cc_values_std_planet_pos = None
        else:
            stats[b, 0] = max_sig_noise
            stats[b, 1] = max_kp_noise_idx - (len(kp_range) // 2)
            stats[b, 2] = max_v_rest_noise

            # And now the stats at exactly the Kp-Vrest of the planet
            if inputs.cross_correlation.ccf_snr:
                stats_planet_pos[b, 0] = ccf_tot_sn_stat[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0],
                                                     int(np.ceil(inputs.reference_kp_kms)+len(kp_range)//2),
                                                     b]
                stats_planet_pos[b, 1] = np.ceil(inputs.reference_kp_kms)
                stats_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inputs.reference_vrest_kms)[0][0]]

                stats_tvalue = None
                stats_pvalue = None
            elif inputs.cross_correlation.welch_ttest:

                stats_tvalue[b, 0] = max_t_value_noise
                stats_tvalue[b, 1] = max_kp_idx_t_noise - (len(kp_range) // 2)
                stats_tvalue[b, 2] = max_v_rest_t_noise

                stats_pvalue[b, 0] = max_p_value_noise
                stats_pvalue[b, 1] = max_kp_idx_p_noise - (len(kp_range) // 2)
                stats_pvalue[b, 2] = max_v_rest_p_noise

                stats_planet_pos[b, 0] = ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inputs.reference_vrest_kms)[0][0],
                                                         int(np.ceil(inputs.reference_kp_kms)+len(kp_range)//2),
                                                         b]
                stats_planet_pos[b, 1] = np.ceil(inputs.reference_kp_kms)
                stats_planet_pos[b, 2] = v_rest_sigma[np.argwhere(v_rest_sigma == inputs.reference_vrest_kms)[0][0]]


    return ccf_tot_stat, significance_metric, significance_metric2, significance_metric3, stats, stats_tvalue, stats_pvalue, stats_planet_pos, stats_planet_area,\
           stats_cc_values, stats_cc_values_planet_pos,\
           stats_cc_values_std, stats_cc_values_std_planet_pos,\
           ccf_complete_stat, ccf_values_shift_stat, shuffled_nights,\
           v_rest_sigma


# ---------------------------------------------------------------------------
# v0.24 additions
# ---------------------------------------------------------------------------

def get_corr_coeff(
    with_signal,
    data,
    model,
    color_variable,
    h,
    stats,
    title,
    plotname,
    CC_2D=True,
    show_plot=False,
    save_plot=True,
    *,
    first_night_noiseless,
    n_nights,
    plots_dir
):
    """
    Compute the Pearson correlation coefficient between data and model
    across nights for a given spectral order.

    Parameters
    ----------
    first_night_noiseless
        The first stored night is the noiseless reference.
    n_nights
        Number of observing nights.
    plots_dir
        Output directory including its trailing path separator.
    with_signal : array_like of int
        In-transit exposure indices.
    Data : numpy.ndarray, shape (n_orders, n_nights, n_spectra, n_pixels)
        Observed data cube.
    Model : numpy.ndarray, shape (n_orders, n_spectra, n_pixels)
        Model template.
    color_variable : array_like
        Variable used to colour the scatter plot (e.g. night index).
    h : int
        Spectral order index.
    stats : numpy.ndarray
        Statistics array. column 0 contains the S/N per night.
    title, plotname : str
        Plot title and output filename stem.
    CC_2D : bool, optional
        If ``True``, compute 2-D (per-night, order-averaged) correlation.
        Default ``True``.
    show_plot, save_plot : bool, optional
        Control matplotlib output.  Default ``False``/``True``.

    Returns
    -------
    pearson_coeff : float  (when ``CC_2D=True``)
        Pearson correlation coefficient between night-averaged S/N and
        per-night correlation coefficients.
    standard_error : float  (when ``CC_2D=True``)
        Bootstrap standard error of *pearson_coeff*.
    sum_corr_coeff : numpy.ndarray  (when ``CC_2D=False``)
        Sum of per-exposure correlation coefficients across in-transit frames.
    """
    from scipy import stats as sc
    from exoplore.io.utils import bootstrap_corrcoeffs

    if first_night_noiseless:
        stats_0 = stats[1:, 0]
    else:
        stats_0 = stats[:, 0]

    if CC_2D:
        corr_coeff = np.zeros((n_nights - 1), float)
        for n in range(1, n_nights):
            X = data[h, n, with_signal, :].flatten()
            Y = model[h, with_signal, :].flatten()
            corr_coeff[n - 1] = np.corrcoef(X, Y)[0, 1]
    else:
        corr_coeff = np.zeros(
            (n_nights - 1, len(with_signal)), float
        )
        for n in range(1, n_nights):
            for idx, i in enumerate(with_signal):
                corr_coeff[n - 1, idx] = np.corrcoef(
                    data[h, n, i, :], model[h, i, :]
                )[0, 1]

    # Calculate Pearson correlation coefficient and p-value
    if CC_2D:
        X = stats_0
        Y = corr_coeff
        pearson_coeff = sc.pearsonr(X, Y)[0]
        standard_error = bootstrap_corrcoeffs(X, Y)
    else:
        pearson_coeff = sc.pearsonr(stats_0, np.sum(corr_coeff, axis=1))

    if show_plot:
        import matplotlib.pyplot as plt
        plt.close()
        plt.figure(figsize=(8, 6))
        if CC_2D:
            plt.scatter(stats_0, corr_coeff,
                        c=color_variable, cmap='viridis',
                        marker='o', s=70, edgecolors='k',
                        label=f"Pearson coeff & p-value = "
                              f"{np.round(pearson_coeff, 5)}")
            colorbar = plt.colorbar()
            colorbar.ax.tick_params(labelsize=14)
            colorbar.set_label(label='Night index', fontsize=17)
        else:
            plt.plot(stats_0, np.sum(corr_coeff, axis=1),
                     'k', marker='o', linewidth=0,
                     label=f"Pearson coeff & p-value = "
                           f"{np.round(pearson_coeff, 5)}")

        plt.tick_params(axis='both', width=1.5, direction='in', labelsize=16)
        plt.xlabel('S/N', fontsize=17)
        plt.ylabel('Corr. Coeff.', fontsize=17)
        plt.title(title, fontsize=17)
        plt.legend()
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.gca().set_axisbelow(True)
        if save_plot:
            plt.savefig(
                f"{plots_dir}{plotname}.pdf"
            )
        plt.show()
        plt.close()

    if CC_2D:
        return pearson_coeff, standard_error
    else:
        return np.sum(corr_coeff, axis=1)
