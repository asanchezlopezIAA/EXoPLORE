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

import random

import numpy as np
from scipy.stats import norm, ttest_ind


def Welch_ttest_map(
        ccf_values_shift, v_rest, kp_range,
        inp_dat, stats=None, stats_tvalue=None, stats_pvalue=None, b=None,
        CCF_Noise=False, plotting=False
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
    inp_dat : dict
        Simulation input dictionary.  Keys used:
        ``"in_trail_left_right"``, ``"CCF_SNR_exclude"``,
        ``"CCF_V_STEP"``.
    stats : ndarray or None
        Bootstrap statistics (used only when ``CCF_Noise=True``).
    stats_tvalue : ndarray or None
        Bootstrap t-value statistics (noise mode).
    stats_pvalue : ndarray or None
        Bootstrap p-value statistics (noise mode).
    b : int or None
        Bootstrap index (noise mode).
    CCF_Noise : bool
        If ``True``, report significance at the position given by
        ``stats[b]`` rather than at the true maximum.

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
    left_right = inp_dat["in_trail_left_right"]
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

    safety_window = int(inp_dat['CCF_SNR_exclude'] / inp_dat["CCF_V_STEP"])
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


def Combine_Nights(inp_dat, ccf, CCF_Noise, previous_shuffle):
    """Randomly co-add groups of simulated nights.

    Produces ``n_nights`` random combinations, each of size
    ``inp_dat["Stack_Group_Size"]``.  When ``CCF_Noise=True`` the
    shuffling from the "real" data run is reused for consistency.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used: ``"Stack_Group_Size"``,
        ``"n_nights"``.
    ccf : ndarray, shape (n_nights, ...)
        CCF array indexed by night along axis 0.
    CCF_Noise : bool
        If ``True``, use ``previous_shuffle`` instead of a fresh random draw.
    previous_shuffle : list or None
        Shuffled night indices from the paired "real" data run (noise mode).

    Returns
    -------
    combined_ccf : ndarray
        Co-added CCF, same shape as ``ccf``.
    shuffled_nights : list or None
        The random combination used (``None`` in noise mode).
    """
    if inp_dat["Stack_Group_Size"] > inp_dat["n_nights"]:
        return "Stack_Group_Size is greater than n_nights"

    combined_ccf = np.zeros_like(ccf)

    if CCF_Noise:
        for i in range(inp_dat["n_nights"]):
            combined_ccf[i, :] = np.sum(
                ccf[previous_shuffle[i], :], axis=0
                )
        return combined_ccf, None

    observed_nights = np.arange(inp_dat["n_nights"])
    shuffled_nights = list()
    for i in range(inp_dat["n_nights"]):
        shuffled_nights.append(
            random.sample(list(observed_nights), inp_dat["Stack_Group_Size"])
            )

    for i in range(inp_dat["n_nights"]):
        combined_ccf[i, :] = np.sum(ccf[shuffled_nights[i], :], axis=0)

    return combined_ccf, shuffled_nights


def statistical_study(
        inp_dat, ccf_v_step, ccf_stat, kp_range, phase,
        v_ccf, v_rest, with_signal, pixels_left_right, sysrem_it_opt,
        ccf_iterations, in_trail_pix, input_stats=None,
        input_stats_tvalue=None, input_stats_pvalue=None,
        previous_shuffle=None, verbose=True, show_plot=False,
        CCF_Noise=False
        ):
    """Run a simulation-based significance study over n_nights draws.

    For each simulated night, the CCF is co-added in the planet rest
    frame for a grid of trial Kp values and the significance of the
    resulting peak is recorded.  Supports SNR, Welch t-test, and SSIM
    significance metrics.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.
    ccf_v_step : float
        CCF velocity step (km/s).
    ccf_stat : ndarray
        CCF array used for the study.
    kp_range : ndarray
        Trial Kp values (km/s).
    phase : ndarray
        Orbital phases.
    v_ccf : ndarray
        Earth-frame CCF velocity grid.
    v_rest : ndarray
        Rest-frame velocity grid.
    with_signal : ndarray
        Indices of in-transit exposures.
    pixels_left_right : int
        Half-width of the in-trail window.
    sysrem_it_opt : ndarray or None
        Per-order SYSREM iteration choices (order-by-order mode).
    ccf_iterations : int
        Number of CCF lag positions.
    in_trail_pix : int
        In-trail half-width in pixels.
    input_stats, input_stats_tvalue, input_stats_pvalue : ndarray or None
        Reference statistics for the paired noise run.
    previous_shuffle : list or None
        Night shuffling from the paired data run.
    verbose : bool
        Print progress messages.
    show_plot : bool
        If ``True``, enable matplotlib display in called sub-routines.
    CCF_Noise : bool
        If ``True``, compute significance at the reference position.

    Returns
    -------
    Tuple of:
        ccf_tot_stat, significance_metric, significance_metric2,
        significance_metric3, stats, stats_tvalue, stats_pvalue,
        stats_planet_pos, stats_planet_area, stats_cc_values,
        stats_cc_values_planet_pos, stats_cc_values_std,
        stats_cc_values_std_planet_pos, ccf_complete_stat,
        ccf_values_shift_stat, shuffled_nights, v_rest_sigma
    """
    from exoplore.observation.velocity import get_V
    from exoplore.ccf.compute import get_max_CCF_peak

    # Co-adding of orders in each night with NO WEIGHTS
    if np.logical_or(not inp_dat["Opt_PCA_its_ord_by_ord"], CCF_Noise):
        if len(ccf_stat.shape) == 4:
            ccf_complete_stat = np.sum(ccf_stat, 0)
        else:
            ccf_complete_stat = ccf_stat

        if inp_dat["Stack_Group_Size"] is not None and inp_dat["Stack_Group_Size"] > 1:
            ccf_complete_stat, shuffled_nights = Combine_Nights(
                inp_dat, ccf_complete_stat, CCF_Noise, previous_shuffle
                )
        else:
            shuffled_nights = None
    else:
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

        if inp_dat["Opt_PCA_its_ord_by_ord"] and not CCF_Noise:

            if sysrem_it_opt.shape[0] == inp_dat["n_orders"] and len(ccf_stat.shape) == 6:
                ccf_complete_stat = np.zeros(
                    (ccf_stat.shape[:4]), float
                    )
                for h in range(inp_dat["n_orders"]):
                    for n in range(ccf_stat.shape[3]):
                        sysrem_index = sysrem_it_opt[h, b, crit_choice]
                        ccf_complete_stat[h, b, :, n] = ccf_stat[h, b, :, n, 0, sysrem_index]
                ccf_complete_stat = np.sum(ccf_complete_stat, axis=0)

            if b == 0:
                if inp_dat["Stack_Group_Size"] is not None and inp_dat["Stack_Group_Size"] > 1:
                    ccf_complete_stat, shuffled_nights = Combine_Nights(
                        inp_dat, ccf_complete_stat, CCF_Noise, previous_shuffle
                        )
                else:
                    shuffled_nights = None

        if inp_dat["n_nights"] > 20 and (b % 10 == 0) and verbose:
            print('STATISTICAL STUDY: Co-adding night ' + str(b + 1) + '/'
                  + str(inp_dat["n_nights"]))

        if b == 0:
            left_right = in_trail_pix
            ccf_values_shift_stat = np.zeros(
                (len(v_rest), len(with_signal), kp_range.shape[0]), float
                )
            ccf_tot_stat = np.zeros(
                (len(v_rest), kp_range.shape[0], inp_dat["n_nights"]), float
                )
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

        for idx, i in enumerate(with_signal):
            for k_idx in range(len(kp_range)):
                v_aux[idx, :, k_idx] = np.linspace(
                    vp[k_idx, i] - pixels_left_right * ccf_v_step,
                    vp[k_idx, i] + pixels_left_right * ccf_v_step,
                    num=2 * pixels_left_right + 1
                    )
                ccf_values_shift_stat[:, idx, k_idx] = np.interp(
                    v_aux[idx, :, k_idx], v_ccf, ccf_complete_stat[b, :, i]
                    )

        ccf_tot_stat[:, :, b] = np.sum(
            ccf_values_shift_stat, axis=1, out=ccf_tot_stat[:, :, b]
            )

        if inp_dat['CCF_SNR'] and not CCF_Noise:
            ccf_tot_sn_stat[:, :, b], max_sig, max_kp_idx, max_v_rest, cc_values_std = \
                get_max_CCF_peak(
                    inp_dat=inp_dat, ccf_tot=ccf_tot_stat[:, :, b],
                    v_rest=v_rest, kp_range=kp_range,
                    b=None, stats=None, sysrem_opt=False, CCF_Noise=False,
                    )
            significance_metric = ccf_tot_sn_stat
            significance_metric2 = None
            significance_metric3 = None
            v_rest_sigma = None
        elif inp_dat['CCF_SNR'] and CCF_Noise:
            ccf_tot_sn_stat[:, :, b], max_sig_noise, max_kp_noise_idx, max_v_rest_noise, cc_values_std_noise = \
                get_max_CCF_peak(
                    inp_dat, ccf_tot_stat[:, :, b], v_rest, kp_range,
                    b, input_stats, False, CCF_Noise=True,
                    )
            significance_metric = ccf_tot_sn_stat
            significance_metric2 = None
            significance_metric3 = None
            v_rest_sigma = None
        elif not inp_dat['CCF_SNR'] and inp_dat["Welch_ttest"] and not CCF_Noise:
            (ccf_tot_sigma_stat[:, :, b], ccf_tot_t_stat[:, :, b],
             ccf_tot_p_stat[:, :, b], v_rest_sigma,
             max_sig, max_kp_idx, max_v_rest,
             max_t_value, max_kp_idx_t, max_v_rest_t,
             max_p_value, max_kp_idx_p, max_v_rest_p) = \
                Welch_ttest_map(
                    ccf_values_shift_stat, v_rest, kp_range,
                    inp_dat, CCF_Noise=CCF_Noise
                    )
            significance_metric = ccf_tot_sigma_stat
            significance_metric2 = ccf_tot_t_stat
            significance_metric3 = ccf_tot_p_stat
        elif not inp_dat['CCF_SNR'] and inp_dat["Welch_ttest"] and CCF_Noise:
            (ccf_tot_sigma_stat[:, :, b], ccf_tot_t_stat[:, :, b],
             ccf_tot_p_stat[:, :, b], v_rest_sigma,
             max_sig_noise, max_kp_noise_idx, max_v_rest_noise,
             max_t_value_noise, max_kp_idx_t_noise, max_v_rest_t_noise,
             max_p_value_noise, max_kp_idx_p_noise, max_v_rest_p_noise) = \
                Welch_ttest_map(
                    ccf_values_shift_stat, v_rest, kp_range,
                    inp_dat,
                    stats=input_stats, stats_tvalue=input_stats_tvalue,
                    stats_pvalue=input_stats_pvalue, b=b, CCF_Noise=CCF_Noise,
                    plotting=show_plot
                    )
            significance_metric = ccf_tot_sigma_stat
            significance_metric2 = ccf_tot_t_stat
            significance_metric3 = ccf_tot_p_stat

        if not CCF_Noise:
            stats[b, 0] = max_sig
            stats[b, 1] = max_kp_idx - (len(kp_range) // 2)
            stats[b, 2] = max_v_rest

            stats_cc_values[b, 0] = ccf_tot_stat[
                np.argwhere(v_rest == max_v_rest)[0][0], max_kp_idx, b
                ]
            stats_cc_values[b, 1] = max_kp_idx - (len(kp_range) // 2)
            stats_cc_values[b, 2] = max_v_rest

            if inp_dat['CCF_SNR']:
                stats_cc_values_std[b, 0] = cc_values_std[
                    np.argwhere(v_rest == max_v_rest)[0][0], max_kp_idx
                    ]
                stats_cc_values_std[b, 1] = max_kp_idx - (len(kp_range) // 2)
                stats_cc_values_std[b, 2] = max_v_rest
            else:
                stats_cc_values_std = None

            if inp_dat['CCF_SNR']:
                stats_planet_pos[b, 0] = ccf_tot_sn_stat[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0],
                    int(np.ceil(inp_dat['K_p']) + len(kp_range) // 2),
                    b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0]
                    ]
                stats_tvalue = None
                stats_pvalue = None
            elif inp_dat["Welch_ttest"]:
                stats_tvalue[b, 0] = max_t_value
                stats_tvalue[b, 1] = max_kp_idx_t - (len(kp_range) // 2)
                stats_tvalue[b, 2] = max_v_rest_t
                stats_pvalue[b, 0] = max_p_value
                stats_pvalue[b, 1] = max_kp_idx_p - (len(kp_range) // 2)
                stats_pvalue[b, 2] = max_v_rest_p
                stats_planet_pos[b, 0] = ccf_tot_sigma_stat[
                    np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0],
                    int(np.ceil(inp_dat['K_p']) + inp_dat['kp_max']),
                    b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest_sigma[
                    np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0]
                    ]

            if inp_dat['CCF_SNR']:
                stats_planet_area[b, 0] = np.max(ccf_tot_sn_stat[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5:
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                    int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                    int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40,
                    b])
                stats_planet_area[b, 1] = np.where(
                    ccf_tot_sn_stat[
                        np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5:
                        np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                        :, b] == np.max(ccf_tot_sn_stat[
                            np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5:
                            np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                            int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                            int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40,
                            b])
                    )[1][0] - (len(kp_range) // 2)
                stats_planet_area[b, 2] = v_rest[np.where(
                    ccf_tot_sn_stat[
                        :,
                        int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                        int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40, b]
                    == np.max(ccf_tot_sn_stat[
                        np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5:
                        np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                        int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                        int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40,
                        b]))[0][0]]
            elif inp_dat["Welch_ttest"]:
                stats_planet_area[b, 0] = np.max(ccf_tot_sigma_stat[
                    np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5:
                    np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5,
                    int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                    int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40,
                    b])
                stats_planet_area[b, 1] = np.where(
                    ccf_tot_sigma_stat[
                        np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5:
                        np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5,
                        :, b] == np.max(ccf_tot_sigma_stat[
                            np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5:
                            np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5,
                            int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                            int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40,
                            b])
                    )[1][0] - (len(kp_range) // 2)
                stats_planet_area[b, 2] = v_rest[np.where(
                    ccf_tot_sigma_stat[
                        :,
                        int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                        int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40, b]
                    == np.max(ccf_tot_sigma_stat[
                        np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5:
                        np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5,
                        int(inp_dat['K_p'] + inp_dat['kp_max']) - 40:
                        int(inp_dat['K_p'] + inp_dat['kp_max'] + 1) + 40,
                        b]))[0][0]]

            stats_cc_values_planet_pos[b, 0] = ccf_tot_stat[
                np.argwhere(v_rest == inp_dat['V_wind'])[0][0],
                int(np.ceil(inp_dat['K_p']) + inp_dat['kp_max']),
                b]
            stats_cc_values_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
            stats_cc_values_planet_pos[b, 2] = v_rest[
                np.argwhere(v_rest == inp_dat['V_wind'])[0][0]
                ]

            if inp_dat['CCF_SNR']:
                stats_cc_values_std_planet_pos[b, 0] = cc_values_std[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0],
                    int(np.ceil(inp_dat['K_p']) + inp_dat['kp_max'])]
                stats_cc_values_std_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_cc_values_std_planet_pos[b, 2] = v_rest[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0]
                    ]
            else:
                stats_cc_values_std_planet_pos = None
        else:
            stats[b, 0] = max_sig_noise
            stats[b, 1] = max_kp_noise_idx - (len(kp_range) // 2)
            stats[b, 2] = max_v_rest_noise

            if inp_dat['CCF_SNR']:
                stats_planet_pos[b, 0] = ccf_tot_sn_stat[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0],
                    int(np.ceil(inp_dat['K_p']) + len(kp_range) // 2),
                    b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest[
                    np.argwhere(v_rest == inp_dat['V_wind'])[0][0]
                    ]
                stats_tvalue = None
                stats_pvalue = None
            elif inp_dat["Welch_ttest"]:
                stats_tvalue[b, 0] = max_t_value_noise
                stats_tvalue[b, 1] = max_kp_idx_t_noise - (len(kp_range) // 2)
                stats_tvalue[b, 2] = max_v_rest_t_noise
                stats_pvalue[b, 0] = max_p_value_noise
                stats_pvalue[b, 1] = max_kp_idx_p_noise - (len(kp_range) // 2)
                stats_pvalue[b, 2] = max_v_rest_p_noise
                stats_planet_pos[b, 0] = ccf_tot_sigma_stat[
                    np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0],
                    int(np.ceil(inp_dat['K_p']) + len(kp_range) // 2),
                    b]
                stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_planet_pos[b, 2] = v_rest_sigma[
                    np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0]
                    ]

    return (ccf_tot_stat, significance_metric, significance_metric2,
            significance_metric3, stats, stats_tvalue, stats_pvalue,
            stats_planet_pos, stats_planet_area,
            stats_cc_values, stats_cc_values_planet_pos,
            stats_cc_values_std, stats_cc_values_std_planet_pos,
            ccf_complete_stat, ccf_values_shift_stat, shuffled_nights,
            v_rest_sigma)


# ---------------------------------------------------------------------------
# v0.24 additions
# ---------------------------------------------------------------------------

def get_corr_coeff(inp_dat, with_signal, data, model, color_variable,
                   h, stats, title, plotname,
                   CC_2D=True, show_plot=False, save_plot=True):
    """
    Compute the Pearson correlation coefficient between data and model
    across nights for a given spectral order.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``'first_night_noiseless'``,
        ``'n_nights'``, ``'plots_dir'``, ``'Simulation_name'``.
    with_signal : array_like of int
        In-transit exposure indices.
    data : numpy.ndarray, shape (n_orders, n_nights, n_spectra, n_pixels)
        Observed data cube.
    model : numpy.ndarray, shape (n_orders, n_spectra, n_pixels)
        Model template.
    color_variable : array_like
        Variable used to colour the scatter plot (e.g. night index).
    h : int
        Spectral order index.
    stats : numpy.ndarray
        Statistics array; column 0 contains the S/N per night.
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

    if inp_dat['first_night_noiseless']:
        stats_0 = stats[1:, 0]
    else:
        stats_0 = stats[:, 0]

    if CC_2D:
        corr_coeff = np.zeros((inp_dat['n_nights'] - 1), float)
        for n in range(1, inp_dat['n_nights']):
            X = data[h, n, with_signal, :].flatten()
            Y = model[h, with_signal, :].flatten()
            corr_coeff[n - 1] = np.corrcoef(X, Y)[0, 1]
    else:
        corr_coeff = np.zeros(
            (inp_dat['n_nights'] - 1, len(with_signal)), float
        )
        for n in range(1, inp_dat['n_nights']):
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
                f"{inp_dat['plots_dir']}{plotname}.pdf"
            )
        plt.show()
        plt.close()

    if CC_2D:
        return pearson_coeff, standard_error
    else:
        return np.sum(corr_coeff, axis=1)
