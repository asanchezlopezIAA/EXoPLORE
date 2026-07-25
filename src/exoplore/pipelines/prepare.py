"""
exoplore.pipelines.prepare
==========================

Main pipeline orchestrator and injection utilities.

This module provides:

- :func:`preparing_pipeline`, the main data-preparation dispatcher that
  selects and runs the correct pipeline (BL19, Blain24, ASL19, Gibson22)
  based on ``inp_dat['preparing_pipeline']``.

- :func:`injection`, injects a synthetic planetary signal into the
  observed data matrix.

- :func:`init_pipeline_outputs`, initialises the output arrays for a
  pipeline run.
"""

from __future__ import annotations

import copy

import numpy as np


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def preparing_pipeline(
        inp_dat, data, noise,
        wave, useful_spectral_points, mask, airmass,
        phase, without_signal, sysrem_pass, data_inj=None,
        max_fit_BL19=False,
        masks=True, correct_uncertainties=True,
        retrieval=False, mask_inter_retrieval=None,
        useful_spectral_points_inter_retrieval=None,
        tell_mask_threshold_Blain24=0.8,
        sysrem_division=False,
):
    """Prepare pipeline.

    Dispatches to the pipeline selected by
    ``inp_dat['preparing_pipeline']``:

    - ``'BL19'``, Brogi & Line (2019)
    - ``'Blain24'``, Blain, Sánchez-López & Mollière (2024)
    - ``'ASL19'``, ASL 2019 (BL19 normalisation + SYSREM)
    - ``'Gibson22'``, Gibson 2022 (normalisation + SYSREM)

    When ``inp_dat['telluric_variation']`` is False, falls back to
    :func:`~exoplore.pipelines.tellurics.pipeline_fixedTellurics`.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Relevant keys include
        ``'preparing_pipeline'``, ``'telluric_variation'``,
        ``'sysrem_its'``, ``'SYSREM_robust_halt'``,
        ``'Opt_PCA_its_ord_by_ord'``, ``'Use_real_data'``.
    data : ndarray
        Spectral matrix, shape ``(n_spectra, n_pixels)``.
    noise : ndarray
        Noise matrix, same shape.
    wave : ndarray
        Wavelength array, shape ``(n_pixels,)``.
    useful_spectral_points : ndarray
        Integer indices of good columns.
    mask : ndarray
        Integer mask of bad columns.
    airmass : ndarray
        Airmass per spectrum, shape ``(n_spectra,)``.
    phase : ndarray
        Orbital phase per spectrum, shape ``(n_spectra,)``.
    without_signal : ndarray
        Integer indices of out-of-transit exposures.
    sysrem_pass : int
        Pre-determined SYSREM pass count (used only when
        ``SYSREM_robust_halt=True`` and not re-determining).
    data_inj : ndarray or None
        Injected data matrix for ASL19 sysrem_opt mode.
    max_fit_BL19 : bool
        If ``True``, use a maximum-fit normalisation in the BL19 branch.
    masks : bool
        If ``True``, apply telluric and column-scatter masking.
    correct_uncertainties : bool
        If ``True``, propagate and correct uncertainties through the pipeline.
    retrieval : bool
        If ``True``, apply the inter-retrieval mask overrides.
    mask_inter_retrieval : ndarray or None
        Mask to apply when ``retrieval=True``.
    useful_spectral_points_inter_retrieval : ndarray or None
        Good-pixel indices to apply when ``retrieval=True``.
    tell_mask_threshold_Blain24 : float, optional
        Telluric transmittance threshold used in the Blain24 branch when
        masking telluric lines.  A wavelength column is masked if the fitted
        telluric transmittance ``exp T(t, lambda)`` drops below this value in
        any exposure.  Default is ``0.8`` (Blain, Sanchez-Lopez & Molliere
        2024, Sec. 4.1; matches petitRADTRANS ``polyfit`` preparing pipeline).
    sysrem_division : bool, optional
        If ``True``, SYSREM corrects by division instead of subtraction.
        Default is ``False`` (subtraction mode).

    Returns
    -------
    Varies by branch (see the per-branch code for the full return signature).
    """
    # Lazy imports to avoid circular dependencies
    from exoplore.pipelines.bl19 import (
        pipeline_BL19_norm, pipeline_BL19_tellcorr,
        pipeline_pseudocontinuum_norm,
    )
    from exoplore.pipelines.cheverall26 import (
        chev26_rescale,
        chev26_outlier_removal,
        chev26_normalise,
        chev26_normalise_maxima,
        chev26_normalise_polyfit,
    )
    from exoplore.pipelines.blain24 import (
        remove_throughput_fit, remove_telluric_lines_fit,
    )
    from exoplore.pipelines.masking import (
        mask_tellurics, mask_tellurics_window, mask_columns,
        _merge_masks, Robust_Outlier_Removal,
    )
    from exoplore.pipelines.sysrem import sysrem as _sysrem
    from exoplore.pipelines.pca import apply_pca as _apply_pca

    # Detrending operator shared by every pipeline branch: "sysrem" (default,
    # inverse-variance weighted) or "pca" (unweighted SVD component removal).
    _detrend = inp_dat.get('detrend_method', 'sysrem')
    from exoplore.pipelines.tellurics import pipeline_fixedTellurics

    try:
        from scipy.ndimage import median_filter, gaussian_filter
    except ImportError:
        median_filter = gaussian_filter = None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_spectra, n_pixels = data.shape
    data_prepared = np.ones_like(data)
    propag_noise = np.ones_like(data)

    # Local aliases, inp_dat is accessed throughout; naming key values
    # once avoids repeating long dict lookups and aids readability.
    _pipeline      = inp_dat['preparing_pipeline']
    _has_tellurics = inp_dat['telluric_variation']
    _n_sysrem      = inp_dat['sysrem_its']
    _robust_halt   = inp_dat['SYSREM_robust_halt']
    _opt_ord       = inp_dat['Opt_PCA_its_ord_by_ord']
    _use_real      = inp_dat['Use_real_data']

    if _has_tellurics:
        if _pipeline == 'BL19':

            aux, aux2 = pipeline_BL19_norm(
                wave, data, noise, useful_spectral_points,
                max_fit=max_fit_BL19
            )
            data_prepared[:, useful_spectral_points] = aux
            propag_noise[:, useful_spectral_points] = aux2

            if masks:
                mask, useful_spectral_points = mask_tellurics(
                    inp_dat, data_prepared, mask,
                )
                inter_mask = np.copy(mask)
                inter_useful = np.copy(useful_spectral_points)

            if retrieval:
                mask = np.copy(mask_inter_retrieval)
                useful_spectral_points = np.copy(
                    useful_spectral_points_inter_retrieval
                )

            aux, aux2 = pipeline_BL19_tellcorr(
                data_prepared, propag_noise, useful_spectral_points
            )
            data_prepared[:, useful_spectral_points] = aux
            propag_noise[:, useful_spectral_points] = aux2

            if masks and mask.shape != (0,):
                mask, useful_spectral_points = mask_columns(
                    data_prepared, mask
                )
                mask = np.asarray(mask, dtype=int)
                inter_mask = np.asarray(inter_mask, dtype=int)
                data_prepared[:, mask] = 1
                propag_noise[:, mask] = 1

            cor = None
            U = None

        elif _pipeline == 'Blain24':

            # Step 1: throughput removal (polynomial over wavelength)
            # Step 2: telluric removal (polynomial over airmass on ln(F_X))
            # Paper (Blain et al. 2024 Eq. 13-14): Step 2 must receive the
            # throughput-corrected data_prepared from Step 1, NOT the
            # original data.
            if masks and correct_uncertainties:
                data_prepared, propag_noise = remove_throughput_fit(
                    data, mask, useful_spectral_points, wave,
                    correct_uncertainties,
                    uncertainties=noise, mask_threshold=1e-16,
                    polynomial_fit_degree=2,
                    uncertainties_as_weights=False
                )
                data_prepared, propag_noise, mask, useful_spectral_points = \
                    remove_telluric_lines_fit(
                        data_prepared, airmass, mask, useful_spectral_points,
                        True, uncertainties=propag_noise,
                        masking=True,
                        mask_threshold=tell_mask_threshold_Blain24,
                        polynomial_fit_degree=2,
                        uncertainties_as_weights=False
                    )
            else:
                data_prepared, _ = remove_throughput_fit(
                    data, mask, useful_spectral_points, wave,
                    correct_uncertainties,
                    uncertainties=None, mask_threshold=1e-16,
                    polynomial_fit_degree=2,
                    uncertainties_as_weights=False
                )
                data_prepared, _, _, _ = remove_telluric_lines_fit(
                    data_prepared, airmass, mask, useful_spectral_points,
                    False, uncertainties=None,
                    masking=False,
                    mask_threshold=tell_mask_threshold_Blain24,
                    polynomial_fit_degree=2,
                    uncertainties_as_weights=False
                )

            mask = mask.astype(int)
            if mask.shape != (0,):
                data_prepared[:, mask] = 1

            # Blain24 has a single combined masking step, the intermediate
            # and final masks are identical.
            inter_mask   = np.copy(mask)
            inter_useful = np.copy(useful_spectral_points)
            cor = None
            U   = None

        elif _pipeline == 'ASL19':
            if not _opt_ord:
                data_prepared = np.ones_like(data)
                propag_noise = np.ones_like(data)

                if masks and correct_uncertainties:
                    data_prepared[:, useful_spectral_points], \
                        propag_noise[:, useful_spectral_points] = \
                        pipeline_pseudocontinuum_norm(
                            wave, data, noise, useful_spectral_points)

                    if masks:
                        mask, useful_spectral_points = mask_tellurics_window(
                            inp_dat, data_prepared, mask
                        )
                        inter_mask = np.copy(mask)
                        inter_useful = np.copy(useful_spectral_points)

                    if not _robust_halt:
                        cor = None
                        U   = np.zeros((n_spectra, _n_sysrem))
                        for l in range(_n_sysrem):
                            data_prepared[:, useful_spectral_points], _, a_l, _ = \
                                _sysrem(
                                    data_prepared, propag_noise,
                                    useful_spectral_points
                                )
                            U[:, l] = a_l[:, 0]
                    else:
                        if _n_sysrem < 15:
                            print(
                                f"Only {_n_sysrem} SYSREM passes "
                                f"selected. Increasing to 20."
                            )
                            _n_sysrem = 20

                        data_prepared_iterations = np.zeros(
                            (_n_sysrem,
                             data_prepared.shape[0],
                             data_prepared.shape[1])
                        )
                        sysrem_runner = data_prepared

                        for l in range(_n_sysrem):
                            sysrem_runner[:, useful_spectral_points], _, U, _ = \
                                _sysrem(
                                    sysrem_runner, propag_noise,
                                    useful_spectral_points
                                )
                            data_prepared_iterations[
                                l, :, useful_spectral_points
                            ] = sysrem_runner[:, useful_spectral_points].T
                        del sysrem_runner

                        std_dev_res = np.zeros((_n_sysrem))
                        for l in range(_n_sysrem):
                            std_dev_res[l] = np.std(
                                data_prepared_iterations[
                                    l, :, useful_spectral_points
                                ]
                            )

                        delta_stddev = np.zeros((_n_sysrem - 1))
                        for l in range(1, _n_sysrem):
                            delta_stddev[l - 1] = (
                                (std_dev_res[l - 1] - std_dev_res[l])
                                / std_dev_res[l - 1]
                            )

                        dy_dx = np.gradient(
                            delta_stddev,
                            np.arange(_n_sysrem - 1)
                        )
                        dy2_dx2 = np.gradient(
                            dy_dx,
                            np.arange(_n_sysrem - 1)
                        )

                        threshold = 0.02
                        plateau_index = np.where(
                            np.abs(dy_dx) < threshold
                        )[0][0]

                        plt.close()
                        plt.figure(figsize=(10, 6))
                        plt.plot(
                            np.arange(_n_sysrem - 1),
                            delta_stddev, marker='o', linestyle='-',
                            label=r'$\Delta \sigma$ (%)', color='k'
                        )
                        plt.plot(
                            np.arange(_n_sysrem - 1),
                            dy_dx, color='violet', marker='o',
                            label='Derivative Curve', linewidth=2
                        )
                        plt.plot(
                            np.arange(_n_sysrem - 1),
                            dy2_dx2, color='g', marker='o',
                            label='2nd Derivative Curve', linewidth=2
                        )
                        plt.plot(
                            np.arange(_n_sysrem - 1)[plateau_index],
                            delta_stddev[plateau_index],
                            marker='*', markersize=25, color='k',
                            linewidth=0, label='Selected'
                        )
                        plt.ylabel(r'$\Delta \sigma$ (%)', fontsize=16)
                        plt.title(
                            'Plateau in $\\Delta \\sigma$ from the derivative '
                            f'and using threshold = {threshold}',
                            fontsize=16
                        )
                        plt.legend(fontsize=16)
                        xticks = np.arange(_n_sysrem - 1)
                        plt.xticks(xticks)
                        plt.tick_params(
                            axis='both', width=1.5,
                            direction='out', labelsize=16
                        )
                        plt.grid(True, color='gray',
                                 linestyle='--', linewidth=0.5)
                        plt.close()

                        sysrem_pass = (
                            np.arange(_n_sysrem - 1)[plateau_index]
                            + 1
                        )
                        data_prepared = data_prepared_iterations[
                            sysrem_pass, :
                        ]

                    data_prepared[:, mask] = 1
                    cor = None
                else:
                    data_prepared[:, useful_spectral_points], _ = \
                        pipeline_pseudocontinuum_norm(
                            wave, data, noise, useful_spectral_points)

                    if not _robust_halt:
                        for l in range(_n_sysrem):
                            data_prepared[:, useful_spectral_points], _, _, _ = \
                                _sysrem(
                                    data_prepared, noise,
                                    useful_spectral_points
                                )
                    else:
                        for l in range(sysrem_pass):
                            data_prepared[:, useful_spectral_points], _, _, _ = \
                                _sysrem(
                                    data_prepared, noise,
                                    useful_spectral_points
                                )
                    cor = None
                    U = None
            else:
                _criterion = inp_dat.get('Opt_criterion', 'DeltaSigma')

                if _criterion == 'DeltaSigma':
                    # Model-independent per-order optimisation
                    # (Parker et al. 2025, MNRAS 538, 3263;
                    #  Peláez-Torres et al. 2026, A&A 705, A256).
                    # Halts when fractional σ-drop < threshold.
                    _threshold = inp_dat.get('sysrem_delta_sigma_threshold', 0.01)
                    data_prepared = np.ones_like(data)
                    propag_noise  = np.ones_like(data)

                    if masks and correct_uncertainties:
                        data_prepared[:, useful_spectral_points], \
                            propag_noise[:, useful_spectral_points] = \
                            pipeline_pseudocontinuum_norm(
                                wave, data, noise, useful_spectral_points)
                        if masks:
                            mask, useful_spectral_points = mask_tellurics_window(
                                inp_dat, data_prepared, mask)
                            inter_mask  = np.copy(mask)
                            inter_useful = np.copy(useful_spectral_points)

                    _std_prev = np.std(data_prepared[:, useful_spectral_points])
                    _U_buf = np.zeros((n_spectra, _n_sysrem))
                    _n_used = _n_sysrem
                    cor = None
                    for l in range(_n_sysrem):
                        data_prepared[:, useful_spectral_points], _, _al, _ = \
                            _sysrem(data_prepared, propag_noise,
                                    useful_spectral_points)
                        _U_buf[:, l] = _al[:, 0]
                        _std_curr = np.std(
                            data_prepared[:, useful_spectral_points])
                        _ds = ((_std_prev - _std_curr) / _std_prev
                               if _std_prev > 0 else 0.0)
                        if _ds < _threshold:
                            _n_used = l + 1
                            break
                        _std_prev = _std_curr
                    U = _U_buf[:, :_n_used]

                else:
                    # Model-dependent injection-recovery optimisation
                    # (Maximum / Max_Diff).  Kept for backward compatibility
                    # and testing; known to introduce biases (Cabot et al.
                    # 2019; Cheverall et al. 2023).
                    _n_used = _n_sysrem
                    data_prepared = np.ones(
                        (n_spectra, n_pixels, 2, _n_sysrem), float
                    )
                    propag_noise = np.ones_like(data)

                    for i in range(2):
                        if i == 0:
                            data_prepared[:, useful_spectral_points, i, 0], \
                                propag_noise[:, useful_spectral_points] = \
                                pipeline_pseudocontinuum_norm(
                                    wave, data, noise, useful_spectral_points)
                        else:
                            data_prepared[:, useful_spectral_points, i, 0], _ = \
                                pipeline_pseudocontinuum_norm(
                                    wave, data_inj, noise, useful_spectral_points)

                    mask, useful_spectral_points = mask_tellurics_window(
                        inp_dat, data_prepared[:, :, 0, 0], mask
                    )
                    # Intermediate mask/useful-point snapshot, required so the
                    # 9-value return signature is satisfied (inter_mask,
                    # inter_useful); mirrors the DeltaSigma and Cheverall26 paths.
                    inter_mask   = np.copy(mask)
                    inter_useful = np.copy(useful_spectral_points)

                    for i in range(2):
                        for l in range(_n_sysrem):
                            if l == 0:
                                syrem_runner = data_prepared[:, :, i, 0]
                            else:
                                syrem_runner = data_prepared[:, :, i, l - 1]
                            data_prepared[:, useful_spectral_points, i, l], _, _, _ = \
                                _sysrem(syrem_runner, propag_noise,
                                        useful_spectral_points)

                    data_prepared[:, mask, :, :] = 1
                    cor = None
                    U   = None

        elif _pipeline == 'Cheverall26':
            # ------------------------------------------------------------------
            # Cheverall et al. (2026) pipeline for IGRINS real-data analysis.
            #
            # Step 1, Rescale each spectrum by its median.
            # Step 2, Sigma-clip outliers per wavelength column (5σ, 2 passes).
            # Step 3, Pseudo-continuum normalisation via 2nd-order polynomial
            #           fit to the maxima of 80 wavelength bins (same module used
            #           by ASL19 / BL19 with max_fit=True).
            # Step 4, SYSREM detrending with Max_Diff / ΔCCF optimisation:
            #           inject H2S model at offset velocity (+19 km/s from
            #           Vsys), run SYSREM for N=1..max, select N where marginal
            #           signal recovery (ΔCCF first difference) is maximised.
            #           Follows Holmberg & Madhusudhan (2022) and Cheverall et
            #           al. (2023; 2026).
            # ------------------------------------------------------------------

            _n_used = _n_sysrem

            # Continuum-normalisation estimator:
            #   "cm2024"  (default), following Cheverall & Madhusudhan 2024: sliding
            #             31-px 95th-percentile envelope + 2nd-order fit (the
            #             recipe Cheverall+26 inherit).
            #   "maxima", ASL19 80-bin maxima + 2nd-order fit.
            #   "polyfit", literal per-exposure 2nd-order continuum polynomial.
            _cm = inp_dat.get('continuum_method') or 'cm2024'
            if _cm == 'polyfit':
                _norm_fn = chev26_normalise_polyfit
            elif _cm == 'maxima':
                _norm_fn = chev26_normalise_maxima
            else:
                _norm_fn = chev26_normalise

            if data_inj is None:
                # ----------------------------------------------------------
                # Global fixed-N mode: a single PCA/SYSREM component count
                # common to all orders (the per-species convention of
                # Cheverall et al. 2026, e.g. Table 3), rather than a per-order
                # optimum.  With no injected track there is nothing to
                # optimise: run steps 1-3 then exactly _n_sysrem SYSREM passes
                # on the data and keep the final residual, returning the
                # standard (n_spectra, n_pixels) matrix so the non-optimisation
                # CCF path consumes it (as for ASL19).  cor/U are accumulated
                # so the SYSREM basis is available for later model reprocessing.
                # ----------------------------------------------------------
                data_prepared = np.ones_like(data)
                propag_noise  = np.ones_like(data)
                d_i, n_i = chev26_rescale(
                    data, noise, useful_spectral_points
                )
                d_i, n_i, _ = chev26_outlier_removal(
                    d_i, n_i, useful_spectral_points
                )
                norm_out, noise_out = _norm_fn(
                    wave, d_i, n_i, useful_spectral_points
                )
                data_prepared[:, useful_spectral_points] = \
                    norm_out[:, useful_spectral_points]
                propag_noise[:, useful_spectral_points] = \
                    noise_out[:, useful_spectral_points]

                mask, useful_spectral_points = mask_tellurics_window(
                    inp_dat, data_prepared, mask
                )
                inter_mask   = np.copy(mask)
                inter_useful = np.copy(useful_spectral_points)

                cor = np.zeros_like(data)
                U   = np.zeros((n_spectra, _n_sysrem))
                _criterion_ch = inp_dat.get('Opt_criterion', 'Max_Diff')
                if _opt_ord and _criterion_ch == 'DeltaSigma':
                    # Model-independent halt: subtract components until the
                    # marginal reduction in the residual scatter falls below a
                    # threshold (Gibson et al. 2022 style), choosing N per order
                    # from the data alone (no injected model).
                    _threshold = inp_dat.get(
                        'sysrem_delta_sigma_threshold', 0.01)
                    _std_prev = np.std(
                        data_prepared[:, useful_spectral_points])
                    _n_used = _n_sysrem
                    for l in range(_n_sysrem):
                        data_prepared[:, useful_spectral_points], \
                            aux_cor, a_l, _ = _sysrem(
                                data_prepared, propag_noise,
                                useful_spectral_points)
                        cor[:, useful_spectral_points] += aux_cor
                        U[:, l] = a_l[:, 0]
                        _std_curr = np.std(
                            data_prepared[:, useful_spectral_points])
                        _ds = ((_std_prev - _std_curr) / _std_prev
                               if _std_prev > 0 else 0.0)
                        if _ds < _threshold:
                            _n_used = l + 1
                            break
                        _std_prev = _std_curr
                    U = U[:, :_n_used]
                elif _detrend == 'pca':
                    # Unweighted PCA: remove the top _n_sysrem common modes in
                    # one SVD (Cheverall et al. 2023). Component count is the
                    # same knob (sysrem_iterations); cor/U mirror the SYSREM
                    # outputs for downstream model reprocessing.
                    _d_clean, _pca_cors, _pca_U = _apply_pca(
                        data_prepared, _n_sysrem, useful_spectral_points)
                    data_prepared[:, useful_spectral_points] = _d_clean
                    for _c in _pca_cors:
                        cor[:, useful_spectral_points] += _c
                    U[:, :_pca_U.shape[1]] = _pca_U
                    _n_used = _pca_U.shape[1]
                else:
                    for l in range(_n_sysrem):
                        data_prepared[:, useful_spectral_points], \
                            aux_cor, a_l, _ = _sysrem(
                                data_prepared, propag_noise,
                                useful_spectral_points)
                        cor[:, useful_spectral_points] += aux_cor
                        U[:, l] = a_l[:, 0]

                if masks and mask.shape != (0,):
                    data_prepared[:, mask] = 1
                    propag_noise[:, mask]  = 1

            else:
                # ----------------------------------------------------------
                # Order-by-order ΔCCF / Max_Diff optimisation: build both the
                # uninjected (track 0) and injected (track 1) time-series and
                # all SYSREM iterations; the per-order optimum N is selected
                # downstream at CCF time.  data_inj carries the model injected
                # at inp_dat['kp_vrest_injection'].
                # ----------------------------------------------------------
                data_prepared = np.ones(
                    (n_spectra, n_pixels, 2, _n_sysrem), float
                )
                propag_noise  = np.ones_like(data)

                for i, src in enumerate([data, data_inj]):
                    d_i, n_i = chev26_rescale(
                        src, noise, useful_spectral_points
                    )
                    d_i, n_i, _ = chev26_outlier_removal(
                        d_i, n_i, useful_spectral_points
                    )
                    norm_out, noise_out = _norm_fn(
                        wave, d_i, n_i, useful_spectral_points
                    )
                    data_prepared[:, useful_spectral_points, i, 0] = \
                        norm_out[:, useful_spectral_points]
                    if i == 0:
                        propag_noise[:, useful_spectral_points] = \
                            noise_out[:, useful_spectral_points]

                mask, useful_spectral_points = mask_tellurics_window(
                    inp_dat, data_prepared[:, :, 0, 0], mask
                )

                # Intermediate mask/useful-point snapshot (post-telluric-
                # masking), mirroring the BL19/Blain24/ASL19 branches so the
                # 9-value return signature is satisfied (inter_mask, inter_useful)
                # and is reused later for model reprocessing / retrievals.
                inter_mask   = np.copy(mask)
                inter_useful = np.copy(useful_spectral_points)

                # Step 4, SYSREM for each iteration on both tracks
                for i in range(2):
                    for l in range(_n_sysrem):
                        runner = data_prepared[:, :, i, 0] if l == 0 \
                            else data_prepared[:, :, i, l - 1]
                        data_prepared[:, useful_spectral_points, i, l], \
                            _, _, _ = _sysrem(
                                runner, propag_noise, useful_spectral_points)

                data_prepared[:, mask, :, :] = 1
                cor = None
                U   = None

        elif _pipeline == 'Gibson22':

            if _use_real:
                # Real data: divide all spectra by the out-of-transit
                # wavelength-dependent median spectrum (Gibson 2022, Sect. 2.1).
                data_prepared[:, useful_spectral_points] = (
                    data[:, useful_spectral_points]
                    / np.median(
                        data[without_signal][:, useful_spectral_points],
                        axis=0
                    )
                )
            else:
                # Simulation: divide every spectrum by its own spectral median
                # so that all spectra (in- and out-of-transit) are placed on a
                # common continuum ~1.  Leaving out-of-transit as flat ones
                # discards their stellar/telluric content and produces a step
                # discontinuity that corrupts SYSREM.
                for ii in range(n_spectra):
                    med = np.median(data[ii, useful_spectral_points])
                    if med != 0:
                        data_prepared[ii, useful_spectral_points] = (
                            data[ii, useful_spectral_points] / med
                        )

            propag_noise[:, useful_spectral_points] = (
                data_prepared[:, useful_spectral_points]
                * (noise[:, useful_spectral_points]
                   / data[:, useful_spectral_points])
            )

            if masks:
                mask, useful_spectral_points = mask_tellurics(
                    inp_dat, data_prepared, mask,
                )
                inter_mask = np.copy(mask)
                inter_useful = np.copy(useful_spectral_points)

            if retrieval:
                mask = np.copy(mask_inter_retrieval)
                useful_spectral_points = np.copy(
                    useful_spectral_points_inter_retrieval
                )

            if not retrieval:
                _criterion_g22 = inp_dat.get('Opt_criterion', 'DeltaSigma')
                if (_opt_ord and _criterion_g22 == 'DeltaSigma'):
                    # Model-independent per-order halt for Gibson22.
                    _threshold_g22 = inp_dat.get(
                        'sysrem_delta_sigma_threshold', 0.01)
                    _std_prev = np.std(
                        data_prepared[:, useful_spectral_points])
                    _U_buf = np.zeros((n_spectra, _n_sysrem))
                    _n_used = _n_sysrem
                    cor = np.zeros_like(data)
                    for l in range(_n_sysrem):
                        data_prepared[:, useful_spectral_points], \
                            aux_cor, _al, _ = _sysrem(
                                data_prepared, propag_noise,
                                useful_spectral_points)
                        cor[:, useful_spectral_points] += aux_cor
                        _U_buf[:, l] = _al[:, 0]
                        _std_curr = np.std(
                            data_prepared[:, useful_spectral_points])
                        _ds = ((_std_prev - _std_curr) / _std_prev
                               if _std_prev > 0 else 0.0)
                        if _ds < _threshold_g22:
                            _n_used = l + 1
                            break
                        _std_prev = _std_curr
                    U = _U_buf[:, :_n_used]
                else:
                    _n_used = _n_sysrem
                    cor = np.zeros_like(data)
                    U = np.zeros((n_spectra, _n_sysrem))
                    for l in range(_n_sysrem):
                        data_prepared[:, useful_spectral_points], \
                            aux_cor, a_l, c_l = _sysrem(
                                data_prepared, propag_noise,
                                useful_spectral_points)
                        cor[:, useful_spectral_points] += aux_cor
                        U[:, l] = a_l[:, 0]

            if masks and mask.shape != (0,):
                data_prepared[:, mask] = 1
                propag_noise[:, mask] = 1

        elif _pipeline == 'Nortmann26':
            # -----------------------------------------------------------------
            # Nortmann et al. (2026) CRIRES+ pipeline, per wavelength segment:
            #   1. two step common blaze normalisation (Sec 3.1);
            #   2. deepest telluric line mask, below 20% of the continuum
            #      (Sec 3.1), on top of the NaN / column scatter mask;
            #   3. SYSREM correcting by DIVISION (Sec 3.2), keeping the per
            #      iteration time basis U for the Gibson 2022 model filter.
            # The theoretical telluric transmittance from molecfit is used for
            # the fit masks and the deep mask when the engine supplies it
            # (inp_dat['nortmann_telluric_order']); otherwise the deep mask is
            # taken from the out of transit master flux, the data driven
            # equivalent of the below 20% continuum rule.
            # -----------------------------------------------------------------
            from exoplore.pipelines.nortmann26 import (
                nortmann_normalise, nortmann_column_mask,
                nortmann_sysrem_division,
            )
            _n_used = _n_sysrem
            usp = np.asarray(useful_spectral_points, dtype=int)

            # Pre scale to order unity for SYSREM numerical stability.
            _scale = np.nanmedian(data[:, usp]) if usp.size else 1.0
            _scale = _scale if _scale and np.isfinite(_scale) else 1.0
            _d = data / _scale
            _nz = noise / _scale

            _tell = inp_dat.get('nortmann_telluric_order', None)
            dn, nn, flagged = nortmann_normalise(
                wave, _d, _nz, without_signal, telluric_transmittance=_tell,
            )
            di, colmask = nortmann_column_mask(flagged, dn)
            # SYSREM inverse variance weighting needs strictly positive, finite
            # uncertainties; reduction masked / edge pixels can carry zero or non
            # finite noise.  Give them a very large uncertainty so they take
            # effectively zero weight (equivalent to masking them).
            _npos = nn[np.isfinite(nn) & (nn > 0)]
            _fill = (np.median(_npos) * 1e6) if _npos.size else 1e6
            nn = np.where(np.isfinite(nn) & (nn > 0), nn, _fill)
            di = np.where(np.isfinite(di), di, 1.0)

            if _tell is not None:
                _tt = np.asarray(_tell)
                _deep = (_tt.min(axis=0) if _tt.ndim == 2 else _tt) < 0.20
            else:
                _master = np.nanmedian(dn[without_signal], axis=0)
                _mmed = np.nanmedian(_master[usp]) if usp.size else 1.0
                _master = _master / (_mmed if _mmed else 1.0)
                _deep = _master < 0.20

            _bad = np.zeros(n_pixels, dtype=bool)
            _bad[np.asarray(colmask, dtype=bool)] = True
            _bad |= _deep
            if mask is not None and np.asarray(mask).size:
                _bad[np.asarray(mask, dtype=int)] = True
            good = np.setdiff1d(usp, np.where(_bad)[0])

            mask = np.where(_bad)[0].astype(int)
            useful_spectral_points = good
            inter_mask = np.copy(mask)
            inter_useful = np.copy(useful_spectral_points)

            _res = nortmann_sysrem_division(
                di, nn, n_iterations=_n_sysrem, good_pixels=good,
            )
            data_prepared = _res["residual"]
            propag_noise = _res["noise"]
            cor = _res["model"]
            U = _res["U"]

            if mask.size:
                data_prepared[:, mask] = 1
                propag_noise[:, mask] = 1

        else:
            raise ValueError(
                f"Unknown pipeline '{_pipeline}'. Valid values: 'BL19', "
                "'Blain24', 'ASL19', 'Gibson22', 'Cheverall26', 'Nortmann26'."
            )

        # n_passes_used: actual SYSREM iterations consumed (slot 4).
        # DeltaSigma / Gibson22-DeltaSigma set _n_used during the loop;
        # injection-based paths use the full _n_sysrem; non-opt paths use 0.
        _n_passes_used = locals().get('_n_used', 0)

        if masks and correct_uncertainties:
            if not _robust_halt:
                if 'inter_mask' in locals() and 'inter_useful' in locals():
                    return (data_prepared, propag_noise,
                            useful_spectral_points, mask, _n_passes_used,
                            inter_mask, inter_useful, cor, U)
                else:
                    return (data_prepared, propag_noise,
                            useful_spectral_points, mask, None, None, None)
            else:
                return (data_prepared, propag_noise,
                        useful_spectral_points, mask, sysrem_pass)
        else:
            return data_prepared

    else:
        data_prepared, propag_noise = pipeline_fixedTellurics(
            phase, wave, data, noise,
            useful_spectral_points, mask,
            np.array([], dtype=int)   # mask_snr: no separate SNR mask at this point
        )
        return data_prepared, propag_noise


# ---------------------------------------------------------------------------
# Signal injection
# ---------------------------------------------------------------------------


def injection(
        inp_dat, wave_ins, mat_og, wave_pRT, syn_spec,
        with_signal, without_signal, fraction, phase, mat_star,
        T_0, syn_jd
):
    """Inject a synthetic planetary signal into the data matrix.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Keys used:
        ``"Kp_Vrest_inj"``, ``"BERV"``, ``"V_sys"``.
    wave_ins : ndarray
        Instrument wavelength grid.
    mat_og : ndarray
        Original (noise-free) data matrix to inject into.
    wave_pRT : ndarray
        pRT wavelength grid.
    syn_spec : ndarray
        Synthetic planetary spectrum.
    with_signal : ndarray
        In-transit exposure indices.
    without_signal : ndarray
        Out-of-transit exposure indices.
    fraction : ndarray
        Transit blocking fraction per exposure.
    phase : ndarray
        Orbital phase per exposure.
    mat_star : ndarray
        Stellar spectrum matrix.
    T_0 : float
        Mid-transit time (BJD).
    syn_jd : ndarray
        Synthetic Julian dates.

    Returns
    -------
    ndarray
        Updated data matrix with injected signal.
    """
    from exoplore.observation.velocity import get_V
    from exoplore.core.observation import spec_to_mat_fraction

    v_p_inj = get_V(
        inp_dat["Kp_Vrest_inj"][0], phase, inp_dat['BERV'],
        inp_dat['V_sys'], inp_dat["Kp_Vrest_inj"][1]
    )

    spec_mat_inj, spec_mat_shift_inj = spec_to_mat_fraction(
        inp_dat, syn_jd, T_0, v_p_inj, wave_ins, wave_pRT, syn_spec,
        mat_star, with_signal, without_signal, fraction,
        injection_setup=True
    )

    return mat_og * spec_mat_inj


# ---------------------------------------------------------------------------
# Pipeline output initialisation
# ---------------------------------------------------------------------------


def init_pipeline_outputs(spectrum, reduction_matrix, uncertainties):
    """Initialise output arrays for a pipeline run.

    Parameters
    ----------
    spectrum : ndarray or MaskedArray
        Input spectral matrix.
    reduction_matrix : ndarray or None
        Existing reduction matrix (or None to create a new one).
    uncertainties : ndarray or None
        Per-pixel uncertainties.

    Returns
    -------
    spectral_data_corrected : ndarray or MaskedArray
    reduction_matrix : ndarray
    pipeline_uncertainties : ndarray or None
    """
    if reduction_matrix is None:
        reduction_matrix = np.ma.ones(spectrum.shape)
        reduction_matrix.mask = np.zeros(spectrum.shape, dtype=bool)

    if isinstance(spectrum, np.ma.core.MaskedArray):
        spectral_data_corrected = np.ma.zeros(spectrum.shape)
        spectral_data_corrected.mask = copy.deepcopy(spectrum.mask)

        if uncertainties is not None:
            pipeline_uncertainties = np.ma.masked_array(
                copy.deepcopy(uncertainties)
            )
            pipeline_uncertainties.mask = copy.deepcopy(spectrum.mask)
        else:
            pipeline_uncertainties = None
    else:
        spectral_data_corrected = np.zeros(spectrum.shape)

        if uncertainties is not None:
            pipeline_uncertainties = copy.deepcopy(uncertainties)
        else:
            pipeline_uncertainties = None

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


# ---------------------------------------------------------------------------
# v0.24 additions
# ---------------------------------------------------------------------------

def remove_throughput_fit_og(spectrum, reduction_matrix, wavelengths, mask,
                              uncertainties=None, mask_threshold=1e-16,
                              polynomial_fit_degree=2,
                              correct_uncertainties=True,
                              uncertainties_as_weights=True):
    """Remove variable throughput with a polynomial fit.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectral data to correct, shape ``(n_spectra, n_pixels)``.
    reduction_matrix : numpy.ndarray
        Matrix storing all operations applied to reduce the data.
    wavelengths : array_like, shape (n_pixels,)
        Wavelengths of the data.
    mask : numpy.ndarray of int
        Indices of masked (bad) pixels.
    uncertainties : numpy.ndarray or None, optional
        Uncertainties on *spectrum*.  Default ``None``.
    mask_threshold : float, optional
        Mask wavelengths where the throughput fit falls below this value.
        Default ``1e-16``.
    polynomial_fit_degree : int, optional
        Degree of the polynomial fit.  Default 2.
    correct_uncertainties : bool, optional
        Apply the fitting-bias correction to uncertainties.  Default ``True``.
    uncertainties_as_weights : bool, optional
        (Reserved for future use; kept for API parity.)  Default ``True``.

    Returns
    -------
    spectral_data_corrected : numpy.ndarray
    reduction_matrix : numpy.ndarray
    pipeline_uncertainties : numpy.ndarray or None
    mask : numpy.ndarray
    useful_spectral_points : numpy.ndarray
    """
    from exoplore.pipelines.masking import _merge_masks as merge_masks

    # Initialization
    degrees_of_freedom = polynomial_fit_degree + 1

    if spectrum.shape[1] <= degrees_of_freedom:
        print(f"not enough points in wavelengths axis ({spectrum.shape[1]}) "
              f"for a meaningful correction with the requested fit degree "
              f"({polynomial_fit_degree}). "
              f"At least {polynomial_fit_degree + 2} wavelengths axis points "
              f"are required. Increase the number of wavelengths axis points "
              f"to decrease correction bias, or decrease the polynomial fit "
              f"degree.")

    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = (
        init_pipeline_outputs(spectrum, reduction_matrix, uncertainties)
    )

    weights = np.ones(spectrum.shape)

    if mask.shape != (0,):
        weights[:, mask] = 0
        spectrum[:, mask] = 1  # ensure no invalid values where weight = 0

    throughput_fits = np.zeros(spectral_data_corrected.shape)

    if np.ndim(wavelengths) == 3:
        print('Assuming same wavelength solution for each observation, '
              'taking wavelengths of observation 0')

    # Correction
    for j, exposure in enumerate(spectrum):
        # The "old" way >5× faster than np.polynomial.Polynomial.fit
        fit_parameters = np.polyfit(
            x=wavelengths, y=exposure, deg=polynomial_fit_degree,
            w=weights[j, :]
        )
        fit_function = np.poly1d(fit_parameters)
        throughput_fits[j, :] = fit_function(wavelengths)

    # Apply mask where estimate is lower than the threshold
    mask_tp = throughput_fits < mask_threshold
    mask_tp = np.any(mask_tp, axis=0)
    mask, useful_spectral_points = merge_masks(mask, mask_tp, spectrum.shape[1])
    if mask.shape != (0,):
        throughput_fits[:, mask] = 1

    # Apply correction
    spectral_data_corrected[:, :] = spectrum
    spectral_data_corrected[:, :] /= throughput_fits[:, :]
    reduction_matrix[:, :] /= throughput_fits[:, :]

    # Propagation of uncertainties
    if uncertainties is not None:
        pipeline_uncertainties /= np.abs(throughput_fits)

        if correct_uncertainties:
            valid_points = (wavelengths.size - int(len(mask))
                            - degrees_of_freedom)

            # Correct from fitting effect, bias the uncertainties so they
            # truly reflect the data standard deviation (see Wikipedia:
            # Weighted arithmetic mean § Weighted sample variance)
            pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 1, 0)
            pipeline_uncertainties *= np.sqrt(valid_points / wavelengths.size)

            # Mask values less than or equal to 0
            mask_uncertainties = pipeline_uncertainties <= 0
            mask_uncertainties = np.any(mask_uncertainties, axis=0)
            mask, useful_spectral_points = merge_masks(
                mask, mask_uncertainties, spectrum.shape[1]
            )
            pipeline_uncertainties[:, mask_uncertainties] = 0
            # Move axis back
            pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 0, 1)

    return (spectral_data_corrected, reduction_matrix,
            pipeline_uncertainties, mask, useful_spectral_points)


# ---------------------------------------------------------------------------
# v0.25 additions
# ---------------------------------------------------------------------------


def remove_telluric_lines_fit_og(spectrum, reduction_matrix, airmass, mask,
                                  uncertainties=None, mask_threshold=1e-16,
                                  polynomial_fit_degree=2,
                                  correct_uncertainties=True,
                                  uncertainties_as_weights=True):
    """Remove telluric lines with a polynomial function.

    The telluric transmittance can be written as::

        T = exp(-airmass * optical_depth)

    hence the log of the transmittance can be written as a first-order
    polynomial::

        log(T) ~ b * airmass + a

    Using a 1st-order polynomial might not be enough, as the atmospheric
    composition can change slowly over time.  A second-order polynomial::

        log(T) ~ c * airmass**2 + b * airmass + a

    may be safer.

    Parameters
    ----------
    spectrum : ndarray, shape (n_spectra, n_pixels)
        Spectral data to correct.
    reduction_matrix : ndarray or None
        Matrix storing all operations made to reduce the data.
    airmass : ndarray, shape (n_spectra,)
        Airmass of the data.
    mask : ndarray of int
        Indices of masked (bad) pixels.
    uncertainties : ndarray or None
        Uncertainties on *spectrum*.
    mask_threshold : float
        Mask wavelengths where the Earth atmospheric transmittance
        estimate is below this value.
    polynomial_fit_degree : int
        Degree of the polynomial fit.
    correct_uncertainties : bool
        Apply the fitting-bias correction to uncertainties.
    uncertainties_as_weights : bool
        (Reserved for future use; kept for API parity.)

    Returns
    -------
    spectral_data_corrected : ndarray
    reduction_matrix : ndarray
    pipeline_uncertainties : ndarray or None
    mask : ndarray
    useful_spectral_points : ndarray
    """
    from exoplore.pipelines.masking import _merge_masks as merge_masks

    # Initialization
    degrees_of_freedom = polynomial_fit_degree + 1

    if spectrum.shape[0] <= degrees_of_freedom:
        print(f"not enough points in airmass axis ({spectrum.shape[1]}) "
                      f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                      f"At least {polynomial_fit_degree + 2} airmass axis points are required. "
                      f"Increase the number of airmass axis points to decrease correction bias, "
                      f"or decrease the polynomial fit degree.")

    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    weights = np.ones(spectrum.shape)
    if mask.shape != (0,):
        weights[:, mask] = 0
        spectrum[:, mask] = 1  # ensure no invalid values are hidden where weight = 0

    # Default: all non-masked pixels are useful (reassigned by merge_masks when masks grow)
    useful_spectral_points = np.setdiff1d(np.arange(spectrum.shape[1]), mask)

    telluric_lines_fits = np.zeros_like(spectrum)

    # Mask wavelength columns where at least one value is lower or equal to 0, to avoid invalid log values
    mask_log = np.any(spectrum <= 0, axis=0)
    mask_log = np.where(mask_log)[0]
    if mask_log.shape != (0,):
        mask, useful_spectral_points = merge_masks(mask, mask_log, spectrum.shape[1])
    if mask.shape != (0,):
        weights[:, mask] = 0
        spectrum[:, mask] = 1  # ensure no invalid values are hidden where weight = 0
    # Fit each wavelength column
    for k in range(spectrum.shape[1]):
        if k in mask:
            telluric_lines_fits[:, k] = 0
            continue
        if weights[np.nonzero(weights[:, k]), k].size > degrees_of_freedom:
            # np.polyfit is >5× faster than np.polynomial.Polynomial.fit
            fit_parameters = np.polyfit(
                x=airmass, y=np.log(spectrum[:, k]), deg=polynomial_fit_degree, w=weights[:, k]
            )
            fit_function = np.poly1d(fit_parameters)

            telluric_lines_fits[:, k] = fit_function(airmass)
        else:
            telluric_lines_fits[:, k] = 0

            print("not all columns have enough valid points for fitting")

        # Calculate telluric transmittance estimate
        telluric_lines_fits = np.exp(telluric_lines_fits)

        # Apply mask where estimate is lower than the threshold, as well as the data mask
        mask_tel = np.any(telluric_lines_fits <= mask_threshold, axis=0)
        mask_tel = np.where(mask_tel)[0]
        if mask_tel.shape != (0,):
            mask, useful_spectral_points = merge_masks(mask, mask_tel, spectrum.shape[1])
        if mask.shape != (0,): telluric_lines_fits[:, mask] = 1

        # Apply correction
        spectral_data_corrected = np.copy(spectrum)
        spectral_data_corrected /= telluric_lines_fits
        reduction_matrix /= telluric_lines_fits

    # Propagation of uncertainties
    if uncertainties is not None:
        pipeline_uncertainties = uncertainties / np.abs(telluric_lines_fits)

        if correct_uncertainties:
            degrees_of_freedom = 1 + polynomial_fit_degree

            # Count number of non-masked points minus degrees of freedom in each time axes
            valid_points = airmass.size - degrees_of_freedom
            #valid_points[np.less(valid_points, 0)] = 0

            # Correct from fitting effect
            # Uncertainties are assumed unbiased, but fitting induces a bias, so here the uncertainties are voluntarily
            # biased (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
            # This way the uncertainties truly reflect the standard deviation of the data
            pipeline_uncertainties *= np.sqrt(valid_points / airmass.size)
            # Mask values less than or equal to 0
            mask_uncertainties = pipeline_uncertainties <= 0
            mask_uncertainties = np.any(mask_uncertainties, axis=0)
            if mask_uncertainties.shape != (0,):
                mask, useful_spectral_points = merge_masks(mask, mask_uncertainties, spectrum.shape[1])
    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties, mask, useful_spectral_points
