"""
exoplore.analysis.multi_night
==============================

Multi-night / multi-instrument combination and comparison plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

def combine_nights_and_plot_3params(base_dir_template,
        retrieval_name,
        simulation_name,
        night_indices,
        prior_bounds_1d,
        truths=None,
        param_names=None,
        out_dir="./combined_posterior",
        max_samples_per_night=20000,
        candidate_pool_max=60000,
        nsamples_combined=20000,
        kde_bw_method=None,
        overlay_per_night=True,
        per_night_levels=(0.68, 0.95),
        per_night_contour_res=3000,
        per_night_alpha=0.9,
        cmap_name="tab10",
        rng_seed=42,
        show_plot=True,
        # --- NEW options for automatic axis adjustment ---
        auto_adjust_axes=True,              # True | False | 'x' | 'y' | 'both'
        limit_percentiles=(0.5, 99.5),     # percentiles used to clip extremes before padding
        axis_pad_fraction=0.06,            # fractional padding added to (high-low)
        # --- NEW: allow user to supply explicit per-parameter limits ---
        user_param_limits=None             # e.g. [[z_lo,z_hi],[p_lo,p_hi],[beta_lo,beta_hi], ...] or None
    ):
    """
    Combine per-night posteriors, create combined samples, and plot a corner figure.

    New behavior:
      - `auto_adjust_axes` controls automatic axis limits (default True).
      - `user_param_limits`: if provided, must be a list with one entry per parameter.
           Each entry can be [lo, hi] (numbers) or None to keep auto-computed limits for that param.
           If `user_param_limits` is None, automatic limits are used for all params (default).
    Call like:
    ---------------
    user_limits = [[-2, 3], [-9, 0], [0.7973, 0.8005]]
    combined_samples, fig = exosims.combine_nights_and_plot_3params(
        base_dir_template='/Users/alexsl/Documents/Simulador/CARMENES_NIR/GJ436b/transit/matrices/matrices_Gibson22_withsignal_1nights_SNR_comb1_realdata_noiseless_stdnoisex1/retrieval_night_{night_index}',
        retrieval_name='retrieval',
        simulation_name='simname',
        night_indices=[0,1,2,3,4],
        prior_bounds_1d=[(-8.0,0.0),(85.0,200.0),(400.0,1500.0),(-25.0,25.0)],
        truths=[-3.0,118.30643476372705,686.0,0.0],
        param_names=["$log_{10}(Z)$", "$log_{10}(p_{c})$ (bar)", "$\\beta$"],
        show_plot = False,
        overlay_per_night=False, user_param_limits=None
    )
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import corner
    import warnings
    import matplotlib.cm as cm
    from scipy import ndimage

    # reproducibility
    if rng_seed is not None:
        np.random.seed(rng_seed)

    os.makedirs(out_dir, exist_ok=True)

    # ------------------ Helpers to read Multinest output ------------------
    def try_use_pymultinest_analyzer(base_dir, night, n_params_hint=None):
        """Load posterior samples for one night via the pymultinest Analyzer API.

        Attempts to read the MultiNest output files under ``base_dir`` using
        ``pymultinest.Analyzer``.  The Analyzer expects files named
        ``<retrieval_name>_night_0_*`` (MultiNest convention).

        Parameters
        ----------
        base_dir : str
            Directory containing the MultiNest output files for this night.
        night : int
            Night index (used for logging; the basename currently hardcodes 0).
        n_params_hint : int or None
            Number of free parameters, inferred from the stats JSON when
            available.  Passed directly to ``Analyzer(n_params=...)``.

        Returns
        -------
        dat : np.ndarray, shape (n_samples, n_params)
            Posterior parameter samples.
        weights : np.ndarray, shape (n_samples,)
            Importance weights from column 0 of the Analyzer data array.
        mask_points : np.ndarray of bool, shape (n_samples,)
            Boolean mask selecting samples with weight > 1e-4.
        stats : dict
            Statistics dict returned by ``Analyzer.get_stats()``.

        Raises
        ------
        ImportError
            If ``pymultinest`` is not installed.
        RuntimeError
            If the Analyzer returns an empty data array.
        """
        try:
            from pymultinest import Analyzer
        except Exception as e:
            raise ImportError("pymultinest not available") from e

        basename = os.path.join(base_dir, f"{retrieval_name}_night_0_")
        an = Analyzer(n_params=n_params_hint, outputfiles_basename=basename)
        stats = an.get_stats()
        data = an.get_data()
        if data is None or data.size == 0:
            raise RuntimeError("Analyzer returned empty data")
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] >= 3:
            weights = data[:, 0].astype(float)
            dat = data[:, 2:].astype(float)
        else:
            if n_params_hint is not None and data.shape[1] >= n_params_hint:
                dat = data[:, -n_params_hint:].astype(float)
                weights = data[:, 0].astype(float) if data.shape[1] > n_params_hint else np.ones(dat.shape[0])
            else:
                dat = data.astype(float)
                weights = np.ones(dat.shape[0], dtype=float)
        mask_points = weights > 1e-4
        return dat, weights, mask_points, stats

    def try_read_post_equal_weights(base_dir, night, n_params_hint=None):
        """Load posterior samples for one night by reading the post-equal-weights file.

        Tries a prioritised list of candidate filenames produced by MultiNest
        (``post_equal_weights.dat``, ``.txt``, ``IS.points``, ``live.points``,
        ``IS.ptprob``, and a gzip variant).  Column 0 is treated as the importance
        weight and columns 2 onward as the physical parameter values.

        Parameters
        ----------
        base_dir : str
            Directory containing the MultiNest output files for this night.
        night : int
            Night index, used to construct candidate filenames.
        n_params_hint : int or None
            Expected number of free parameters.  Used to identify which columns
            contain parameter values when the file has fewer than 3 columns.

        Returns
        -------
        dat : np.ndarray, shape (n_samples, n_params)
            Posterior parameter samples.
        weights : np.ndarray, shape (n_samples,)
            Importance weights (ones if all samples are equal-weight).
        mask_points : np.ndarray of bool, shape (n_samples,)
            Boolean mask selecting samples with weight > 1e-4.
        None
            Placeholder for the stats dict (unavailable from flat files).

        Raises
        ------
        FileNotFoundError
            If no readable file is found in any candidate location.
        """
        candidate_fnames = [
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_post_equal_weights.dat"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_post_equal_weights.txt"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_IS.points"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_live.points"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_IS.ptprob"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_post_equal_weights.dat.gz"),
        ]
        for fn in candidate_fnames:
            if not os.path.exists(fn):
                continue
            try:
                arr = np.loadtxt(fn, comments='#', ndmin=2)
            except Exception:
                data_list = []
                with open(fn, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        try:
                            vals = [float(p) for p in parts]
                        except Exception:
                            continue
                        data_list.append(vals)
                if len(data_list) == 0:
                    continue
                arr = np.array(data_list, dtype=float)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 3:
                weights = arr[:, 0].astype(float)
                dat = arr[:, 2:].astype(float)
            else:
                if n_params_hint is not None and arr.shape[1] >= n_params_hint:
                    dat = arr[:, -n_params_hint:].astype(float)
                    weights = arr[:, 0].astype(float) if arr.shape[1] > n_params_hint else np.ones(dat.shape[0])
                else:
                    dat = arr.astype(float)
                    weights = np.ones(dat.shape[0], dtype=float)
            mask_points = weights > 1e-4
            return dat, weights, mask_points, None
        raise FileNotFoundError("No suitable post-equal-weights / points file found")

    def infer_nparams_from_stats_json(stats_fn):
        """Infer the number of free parameters from a MultiNest stats JSON file.

        Reads the ``*_stats.json`` file written by MultiNest/pymultinest and
        counts the parameters by inspecting the ``'marginals'`` list or the
        ``'parameters'`` dict.  The inferred count is passed as ``n_params_hint``
        to the loader helpers so the correct columns are extracted.

        Parameters
        ----------
        stats_fn : str
            Path to the ``*_stats.json`` file.

        Returns
        -------
        n_params : int or None
            Number of free parameters, or ``None`` if the file is missing,
            unreadable, or lacks a recognisable structure.
        stats : dict or None
            The parsed JSON object, or ``None`` on failure.
        """
        try:
            with open(stats_fn, 'r') as fh:
                st = json.load(fh)
            if isinstance(st, dict):
                if 'marginals' in st and isinstance(st['marginals'], (list, tuple)):
                    return len(st['marginals']), st
                if 'parameters' in st and isinstance(st['parameters'], dict):
                    return len(st['parameters']), st
        except Exception:
            pass
        return None, None

    # ------------------ Load nights (robustly) ------------------
    nights_data = []
    nparams = None
    for night in night_indices:
        base_dir = base_dir_template.format(night_index=night)
        stats_fn = os.path.join(base_dir, f"{retrieval_name}_night_{night}_stats.json")
        nparams_hint = None
        stats_json_obj = None
        if os.path.exists(stats_fn):
            nph, st = infer_nparams_from_stats_json(stats_fn)
            if nph is not None:
                nparams_hint = nph
                stats_json_obj = st

        loaded = False
        last_exc = None
        try:
            dat, weights, mask_points, stats_obj = try_use_pymultinest_analyzer(base_dir, night, n_params_hint=nparams_hint)
            if stats_obj is None and stats_json_obj is not None:
                stats_obj = stats_json_obj
            loaded = True
            print(f"Loaded night {night} via pymultinest.Analyzer: samples {dat.shape}, weights sum {np.sum(weights):.6g}")
        except Exception as e:
            last_exc = e
            try:
                dat, weights, mask_points, stats_obj = try_read_post_equal_weights(base_dir, night, n_params_hint=nparams_hint)
                loaded = True
                if stats_obj is None and os.path.exists(stats_fn):
                    try:
                        with open(stats_fn, 'r') as fh:
                            stats_obj = json.load(fh)
                    except Exception:
                        stats_obj = stats_json_obj
                print(f"Loaded night {night} via file fallback: samples {dat.shape}, weights sum {np.sum(weights):.6g}")
            except Exception as e2:
                last_exc = e2

        if not loaded:
            raise FileNotFoundError(f"Could not load samples for night {night}: last error: {last_exc}")

        dat = np.asarray(dat)
        if dat.ndim == 1:
            dat = dat.reshape(-1, 1)

        nrows, npar = dat.shape
        if nparams is None:
            nparams = npar
        else:
            if npar != nparams:
                raise RuntimeError(f"Mismatch in parameter count across nights (night {night} has {npar}, expected {nparams})")

        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != nrows:
            if weights.size < nrows:
                neww = np.ones(nrows, dtype=float)
                neww[:weights.size] = weights
                weights = neww
            else:
                weights = weights[:nrows]
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            warnings.warn(f"Weights for night {night} sum to zero or negative; using uniform weights")
            weights = np.ones_like(weights, dtype=float)
        weights = weights / np.sum(weights)

        try:
            mask = np.asarray(mask_points).astype(bool)
            if mask.ndim != 1 or mask.shape[0] != nrows:
                mask = (mask_points > 0).reshape(-1)[:nrows]
        except Exception:
            mask = np.ones(nrows, dtype=bool)

        sel = mask
        if sel.sum() == 0:
            warnings.warn(f"Mask for night {night} selected 0 points; using all")
            sel = np.ones(nrows, dtype=bool)

        dat_sel = dat[sel, :]
        w_sel = weights[sel].astype(float)
        w_sel = np.maximum(w_sel, 0.0)
        if w_sel.sum() <= 0:
            w_sel = np.ones_like(w_sel)
        w_sel = w_sel / np.sum(w_sel)

        if dat_sel.shape[0] > max_samples_per_night:
            idx = np.random.choice(np.arange(dat_sel.shape[0]), size=max_samples_per_night, replace=False, p=w_sel)
            dat_sel = dat_sel[idx, :]
            w_sel = w_sel[idx]
            w_sel = w_sel / w_sel.sum()

        nights_data.append({'night_index': night, 'dat': dat_sel, 'weights': w_sel, 'stats': stats_obj})
        print(f"Prepared night {night}: samples {dat_sel.shape}, weights sum {w_sel.sum():.3f}")

    # default param names
    if param_names is None:
        if nparams == 3:
            param_names = [r"$\log_{10}(Z)$", r"$\log_{10}(p_{cloud})$ (bar)", r"$\beta$"]
        else:
            param_names = [f"p{i}" for i in range(nparams)]

    if truths is not None and len(truths) != nparams:
        warnings.warn("truths length != n_params; ignoring truths")
        truths = None

    print("Parameter names used:", param_names)
    if truths is not None:
        print("Truths provided:", truths)

    # ------------------ Fit KDEs (full multivariate for combination) ------------------
    kdes = []
    for nd in nights_data:
        data = nd['dat']
        w = nd['weights']
        try:
            kde = gaussian_kde(dataset=data.T, weights=w, bw_method=kde_bw_method)
        except Exception as e:
            jitter = 1e-8 * np.std(data, axis=0)
            jitter = np.where(jitter == 0, 1e-8, jitter)
            data_j = data + np.random.normal(scale=jitter, size=data.shape)
            kde = gaussian_kde(dataset=data_j.T, weights=w, bw_method=kde_bw_method)
            warnings.warn(f"KDE fit jitter fallback for night {nd['night_index']}: {e}")
        kdes.append(kde)
    print("Fitted full multivariate KDEs for all nights.")

    # ------------------ Candidate pool (improved: sample from KDEs + data + prior) ------------------
    pool = np.vstack([nd['dat'] for nd in nights_data])

    # fractions for candidate composition (tweak here if you like)
    frac_from_kdes = 0.65
    frac_from_prior = 0.10
    frac_from_data = 0.20
    total_frac = frac_from_kdes + frac_from_prior + frac_from_data
    if total_frac > 1.0:
        frac_from_kdes /= total_frac
        frac_from_prior /= total_frac
        frac_from_data /= total_frac

    target_ncand = int(candidate_pool_max)
    if target_ncand < 1000:
        target_ncand = max(1000, target_ncand)

    n_kde_total = int(target_ncand * frac_from_kdes)
    n_prior = int(target_ncand * frac_from_prior)
    n_data = int(target_ncand * frac_from_data)
    n_remain = target_ncand - (n_kde_total + n_prior + n_data)
    if n_remain < 0:
        n_remain = 0

    candidates_list = []

    # 1) sample from each per-night KDE (spread across nights)
    if n_kde_total > 0:
        per_kde = max(1, n_kde_total // len(kdes))
        extras = n_kde_total - per_kde * len(kdes)
        for i, kde in enumerate(kdes):
            n_this = per_kde + (1 if i < extras else 0)
            try:
                s = kde.resample(n_this).T  # shape (n_this, nparams)
                candidates_list.append(s)
            except Exception:
                # fallback: jittered resample from data
                d = nights_data[i]['dat']
                jitter = 1e-6 * np.std(d, axis=0)
                jitter = np.where(jitter == 0, 1e-6, jitter)
                s = d[np.random.choice(d.shape[0], size=n_this, replace=True)] + np.random.normal(scale=jitter, size=(n_this, d.shape[1]))
                candidates_list.append(s)

    # 2) uniform draws from the prior bounds (to cover edges)
    if n_prior > 0:
        prior_samples = np.zeros((n_prior, nparams))
        for j in range(nparams):
            a, b = prior_bounds_1d[j]
            prior_samples[:, j] = np.random.uniform(low=a, high=b, size=n_prior)
        candidates_list.append(prior_samples)

    # 3) some draws from the pooled original samples (real posterior points)
    if n_data > 0:
        if pool.shape[0] > n_data:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_data, replace=False)
        else:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_data, replace=True)
        candidates_list.append(pool[idx, :])

    # 4) fill remainder with pooled samples (keeps backward compatibility)
    if n_remain > 0:
        if pool.shape[0] > n_remain:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_remain, replace=False)
        else:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_remain, replace=True)
        candidates_list.append(pool[idx, :])

    # assemble candidates
    if len(candidates_list) == 0:
        candidates = pool.copy()
    else:
        candidates = np.vstack(candidates_list)

    # final safeguard: limit to candidate_pool_max by subsampling (but keep diversity)
    if candidates.shape[0] > candidate_pool_max:
        idx = np.random.choice(np.arange(candidates.shape[0]), size=candidate_pool_max, replace=False)
        candidates = candidates[idx, :]

    ncand = candidates.shape[0]
    print("Candidate pool built: total candidates =", ncand, "(from KDEs/prior/data/pool fractions:",
          frac_from_kdes, frac_from_prior, frac_from_data, ")")

    # ------------------ Evaluate log densities (per-night) ------------------
    tiny = 1e-300
    log_ps = np.zeros((len(kdes), ncand))
    for i, kde in enumerate(kdes):
        try:
            vals = kde(candidates.T)
        except Exception:
            jitter = 1e-12 * np.std(candidates, axis=0)
            jitter = np.where(jitter == 0, 1e-12, jitter)
            cand_j = candidates + np.random.normal(scale=jitter, size=candidates.shape)
            vals = kde(cand_j.T)
        vals = np.maximum(vals, tiny)
        log_ps[i, :] = np.log(vals)

    # evaluate prior on candidates
    def prior_pdf(theta, priors_per_param):
        """Evaluate the joint prior probability density for a parameter vector.

        Computes the product of independent uniform prior densities for each
        parameter.  Any parameter that falls outside its allowed range returns
        density 0 immediately (hard rejection).  Used to regularise the
        combined-posterior reweighting so that the product of per-night
        posteriors is divided by ``(N_nights - 1)`` copies of the prior,
        recovering the correctly normalised combination.

        Parameters
        ----------
        theta : array-like, length n_params
            Physical parameter values.
        priors_per_param : list of dict
            One entry per parameter; each dict must have keys ``'min'`` and
            ``'max'`` (float), or ``None`` to skip that parameter.

        Returns
        -------
        float
            Joint prior density ``∏ 1/(b-a)`` if all parameters are within
            their bounds, else 0.
        """
        pdf = 1.0
        for j, val in enumerate(theta):
            p = priors_per_param[j]
            if p is None:
                continue
            a, b = p['min'], p['max']
            if val < a or val > b:
                return 0.0
            pdf *= 1.0 / (b - a)
        return pdf

    priors_per_param = [{'type': 'uniform', 'min': float(a), 'max': float(b)} for (a, b) in prior_bounds_1d]
    lp = np.zeros(ncand)
    for j in range(ncand):
        pr = prior_pdf(candidates[j, :], priors_per_param)
        lp[j] = np.maximum(pr, tiny)
    log_prior = np.log(lp)

    # ------------------ Combine in log-space ------------------
    Nn = len(kdes)
    log_combined = np.sum(log_ps, axis=0) - (Nn - 1) * log_prior
    log_combined -= np.max(log_combined)
    combined_unnorm = np.exp(log_combined)
    prob = combined_unnorm / np.sum(combined_unnorm)

    if np.any(np.isinf(log_combined)) or np.sum(prob) <= 0:
        raise RuntimeError("Combined posterior failed (all zero or infinite)")

    # ------------------ Resample combined posterior ------------------
    pick_idx = np.random.choice(np.arange(ncand), size=nsamples_combined, replace=True, p=prob)
    combined_samples = candidates[pick_idx, :]

    # compute summary
    medians = np.median(combined_samples, axis=0)
    lo68 = np.percentile(combined_samples, 16, axis=0)
    hi68 = np.percentile(combined_samples, 84, axis=0)
    sig = 0.5 * (hi68 - lo68)

    print("\nCombined posterior summary (median +- 1-sigma):")
    for i, name in enumerate(param_names):
        s = f"{medians[i]: .6g} +- {sig[i]: .6g}"
        if truths is not None:
            s += f"    truth={truths[i]: .6g}"
        print(f"  {name:20s} {s}")

    # ------------------ Save combined samples & summary ------------------
    os.makedirs(out_dir, exist_ok=True)
    out_samples_fn = os.path.join(out_dir, "combined_samples.npz")
    np.savez_compressed(out_samples_fn, samples=combined_samples, param_names=param_names, truths=truths)
    summary = {
        'nights_combined': [nd['night_index'] for nd in nights_data],
        'n_params': nparams,
        'param_names': param_names,
        'nsamples_combined': nsamples_combined,
        'medians': medians.tolist(),
        'sigma_1sig': sig.tolist(),
        'truths': truths
    }
    with open(os.path.join(out_dir, "combined_summary.json"), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print("Saved combined samples and summary to:", out_dir)
    if not show_plot: 
        return combined_samples, None
    else:
        
        # ------------------ Make corner plot for combined samples ------------------
        plt.close('all')
        corner_kwargs = dict(
            labels=param_names,
            plot_datapoints=True,
            show_titles=False,
            title_fmt=".3g",
            quantiles=[0.16, 0.5, 0.84],
            label_kwargs={"fontsize": 12},
            title_kwargs={"fontsize": 12},
            truth_color='firebrick',
            color="darkslateblue"
        )
        if truths is not None:
            fig = corner.corner(combined_samples, truths=truths, **corner_kwargs)
        else:
            fig = corner.corner(combined_samples, **corner_kwargs)

        # try to reshape axes into nparams x nparams (corner produces a full grid, upper triangle axes hidden)
        try:
            axes = np.array(fig.get_axes()).reshape((nparams, nparams))
        except Exception:
            axes = None
            warnings.warn("Cannot reshape corner axes; skipping per-night overlay and auto-axis adjustments.")

        # ------------------ AUTO-ADJUST AXES (OPTION 1: default) ------------------
        def compute_param_limits_from_samples(samples, percentiles=(0.5, 99.5), pad_frac=0.06):
            """Compute per-parameter axis limits from a combined posterior sample array.

            Clips the distribution at the given percentiles to remove outliers,
            then adds a fractional pad on each side.  Used to set the axis limits
            of the corner plot panels so the bulk of the posterior is always visible
            regardless of long tails or prior-boundary artefacts.

            Parameters
            ----------
            samples : np.ndarray, shape (n_samples, n_params)
                Combined posterior samples.
            percentiles : tuple of float
                Lower and upper percentile thresholds (default ``(0.5, 99.5)``).
            pad_frac : float
                Fractional padding added to ``(hi - lo)`` on each side.

            Returns
            -------
            list of tuple
                One ``(lo, hi)`` pair per parameter.
            """
            mins = np.percentile(samples, percentiles[0], axis=0)
            maxs = np.percentile(samples, percentiles[1], axis=0)
            limits = []
            for lo, hi in zip(mins, maxs):
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = np.nanmin(samples[:, 0]), np.nanmax(samples[:, 0])  # fallback (unlikely)
                if hi <= lo:
                    delta = np.abs(lo) * 1e-3 if lo != 0 else 1e-3
                    lo -= delta
                    hi += delta
                pad = (hi - lo) * pad_frac
                limits.append((lo - pad, hi + pad))
            return limits

        # parse auto_adjust_axes flag (accept True/False/'x'/'y'/'both')
        adjust_x = adjust_y = False
        if auto_adjust_axes is True or auto_adjust_axes == 'both':
            adjust_x = adjust_y = True
        elif auto_adjust_axes == 'x':
            adjust_x = True
        elif auto_adjust_axes == 'y':
            adjust_y = True
        elif auto_adjust_axes in (False, None):
            adjust_x = adjust_y = False
        else:
            adjust_x = adjust_y = True

        # compute auto limits
        auto_param_limits = None
        if (adjust_x or adjust_y) and axes is not None:
            try:
                auto_param_limits = compute_param_limits_from_samples(combined_samples, percentiles=limit_percentiles, pad_frac=axis_pad_fraction)
            except Exception as e:
                warnings.warn(f"Failed to compute auto param limits: {e}")
                auto_param_limits = None

        # ------------------ MERGE user_param_limits with auto limits ------------------
        final_param_limits = None
        if user_param_limits is None:
            final_param_limits = auto_param_limits
        else:
            try:
                upl = list(user_param_limits)
                if len(upl) != nparams:
                    warnings.warn(f"user_param_limits length {len(upl)} != nparams {nparams}; ignoring user input and using auto limits")
                    final_param_limits = auto_param_limits
                else:
                    final_param_limits = []
                    for i, entry in enumerate(upl):
                        if entry is None:
                            if auto_param_limits is not None:
                                final_param_limits.append(auto_param_limits[i])
                            else:
                                lo = np.percentile(combined_samples[:, i], limit_percentiles[0])
                                hi = np.percentile(combined_samples[:, i], limit_percentiles[1])
                                if hi <= lo:
                                    hi = lo + 1e-3
                                pad = (hi - lo) * axis_pad_fraction
                                final_param_limits.append((lo - pad, hi + pad))
                        else:
                            try:
                                a, b = float(entry[0]), float(entry[1])
                                if b <= a:
                                    warnings.warn(f"user_param_limits[{i}] has hi<=lo ({a},{b}); swapping values")
                                    a, b = min(a, b), max(a, b)
                                final_param_limits.append((a, b))
                            except Exception:
                                warnings.warn(f"Invalid entry for user_param_limits[{i}] = {entry}; using auto limit for this param")
                                if auto_param_limits is not None:
                                    final_param_limits.append(auto_param_limits[i])
                                else:
                                    lo = np.percentile(combined_samples[:, i], limit_percentiles[0])
                                    hi = np.percentile(combined_samples[:, i], limit_percentiles[1])
                                    pad = (hi - lo) * axis_pad_fraction
                                    final_param_limits.append((lo - pad, hi + pad))
            except Exception as e:
                warnings.warn(f"Failed to parse user_param_limits ({e}); falling back to auto limits")
                final_param_limits = auto_param_limits

        # ------------------ APPLY final_param_limits to axes ----------------    --
        if axes is not None and final_param_limits is not None and (adjust_x or adjust_y):
            try:
                for i in range(nparams):
                    for j in range(nparams):
                        ax = axes[i, j]
                        if i == j:
                            lo, hi = final_param_limits[i]
                            if adjust_x:
                                try:
                                    ax.set_xlim(lo, hi)
                                except Exception:
                                    pass
                        elif i > j:
                            if adjust_x:
                                try:
                                    lo_x, hi_x = final_param_limits[j]
                                    ax.set_xlim(lo_x, hi_x)
                                except Exception:
                                    pass
                            if adjust_y:
                                try:
                                    lo_y, hi_y = final_param_limits[i]
                                    ax.set_ylim(lo_y, hi_y)
                                except Exception:
                                    pass
                print("Applied final parameter limits to corner axes.")
            except Exception as e:
                warnings.warn(f"Applying final_param_limits failed: {e}")

        # ----- Cosmetic tweak: invert y-axis for p_cloud subplot (row 1, col 0) -----
        p_cloud_index = 1  # your pressure parameter index
        try:
            if axes is not None and 0 <= p_cloud_index < nparams:
                ax_to_invert = axes[p_cloud_index, 0]
                ymin, ymax = ax_to_invert.get_ylim()
                if ymin < ymax:
                    ax_to_invert.set_ylim(ymax, ymin)
                    print(f"Inverted y-axis for subplot [{p_cloud_index}, 0] (p_cloud).")
        except Exception as e:
            warnings.warn(f"Could not invert p_cloud axis at axes[{p_cloud_index},0]: {e}")

        # ------------------ Per-night overlays (optional) ------------------
        # upgraded kde->grid->upsample function for smooth contours
        def kde_2d_levels_on_grid_up(kde2d, X, Y, prob_levels, upsample=3, min_pdf_val=1e-300):
            """Evaluate a 2D KDE on a grid and return credible-interval iso-density levels.

            Evaluates ``kde2d`` on the coarse meshgrid ``(X, Y)``, optionally
            upsamples the resulting PDF via cubic interpolation (``ndimage.zoom``),
            then computes the PDF threshold corresponding to each requested
            probability level (e.g., 0.68 for 1-sigma).  The thresholds are
            passed directly to ``ax.contour``/``ax.contourf`` to draw credible
            contours on the corner subplots.

            Parameters
            ----------
            kde2d : scipy.stats.gaussian_kde
                Fitted 2D KDE for a single pair of parameters.
            X, Y : np.ndarray, shape (ny, nx)
                Meshgrid coordinate arrays (output of ``np.meshgrid``).
            prob_levels : sequence of float
                Enclosed probability fractions (e.g., ``[0.68, 0.95]``).
            upsample : int or None
                Upsampling factor applied via cubic ``ndimage.zoom``.
                ``None`` or ``1`` skips upsampling.
            min_pdf_val : float
                Floor applied to the PDF before computing cumulative sums to
                avoid zero-weight cells causing numerical issues.

            Returns
            -------
            X_out, Y_out : np.ndarray
                Coordinate meshgrids (upsampled if ``upsample > 1``).
            pdf_out : np.ndarray
                PDF values on the (upsampled) grid.
            levels : list of float
                PDF iso-density thresholds, one per entry in ``prob_levels``.
            """
            coords = np.vstack([X.ravel(), Y.ravel()])
            pdf = kde2d(coords).reshape(X.shape)
            pdf = np.maximum(pdf, min_pdf_val)

            # coarse area element
            dx = (X[0, 1] - X[0, 0])
            dy = (Y[1, 0] - Y[0, 0])
            area = dx * dy

            if upsample is None or int(upsample) <= 1:
                # compute thresholds on coarse grid
                pf = pdf.ravel()
                idx = np.argsort(pf)[::-1]
                pf_sorted = pf[idx]
                cumsum = np.cumsum(pf_sorted) * area
                levels = []
                for p in prob_levels:
                    mask = cumsum >= p
                    if np.any(mask):
                        levels.append(pf_sorted[mask][-1])
                    else:
                        levels.append(pf_sorted.max() * 0.0)
                return X, Y, pdf, levels

            # upsample pdf array
            pdf_up = ndimage.zoom(pdf, zoom=(upsample, upsample), order=3)
            ny_up, nx_up = pdf_up.shape

            # build new coords
            x_min, x_max = X[0, 0], X[0, -1]
            y_min, y_max = Y[0, 0], Y[-1, 0]
            xs_up = np.linspace(x_min, x_max, nx_up)
            ys_up = np.linspace(y_min, y_max, ny_up)
            X_up, Y_up = np.meshgrid(xs_up, ys_up)

            dx_up = xs_up[1] - xs_up[0] if nx_up > 1 else (x_max - x_min)
            dy_up = ys_up[1] - ys_up[0] if ny_up > 1 else (y_max - y_min)
            area_up = dx_up * dy_up

            pf = pdf_up.ravel()
            idx = np.argsort(pf)[::-1]
            pf_sorted = pf[idx]
            cumsum = np.cumsum(pf_sorted) * area_up

            levels = []
            for p in prob_levels:
                mask = cumsum >= p
                if np.any(mask):
                    levels.append(pf_sorted[mask][-1])
                else:
                    levels.append(pf_sorted.max() * 0.0)

            return X_up, Y_up, pdf_up, levels

        if overlay_per_night and axes is not None:
            cmap = cm.get_cmap(cmap_name)
            colors = [cmap(i % cmap.N) for i in range(len(nights_data))]

            for ix, nd in enumerate(nights_data):
                data = nd['dat']
                w = nd['weights']
                color = colors[ix]
                per_night_medians = np.median(data, axis=0)

                for i in range(nparams):
                    axd = axes[i, i]
                    axd.axvline(per_night_medians[i], color=color, alpha=per_night_alpha, lw=1.0, linestyle='--')

                for i in range(nparams):
                    for j in range(i):
                        ax = axes[i, j]
                        pair_data = data[:, [j, i]]
                        try:
                            kde2d = gaussian_kde(dataset=pair_data.T, weights=w, bw_method=kde_bw_method)
                        except Exception:
                            jitter = 1e-8 * np.std(pair_data, axis=0)
                            jitter = np.where(jitter == 0, 1e-8, jitter)
                            pair_j = pair_data + np.random.normal(scale=jitter, size=pair_data.shape)
                            kde2d = gaussian_kde(dataset=pair_j.T, weights=w, bw_method=kde_bw_method)

                        # use axis limits as base grid
                        x_min, x_max = ax.get_xlim()
                        y_min, y_max = ax.get_ylim()

                        # small guard in case limits are degenerate
                        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                            x_min, x_max = np.percentile(pair_data[:, 0], [0.5, 99.5])
                            if x_max <= x_min:
                                x_min -= 1e-3
                                x_max += 1e-3
                        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                            y_min, y_max = np.percentile(pair_data[:, 1], [0.5, 99.5])
                            if y_max <= y_min:
                                y_min -= 1e-3
                                y_max += 1e-3

                        xs = np.linspace(x_min, x_max, int(np.sqrt(per_night_contour_res)))
                        ys = np.linspace(y_min, y_max, int(np.sqrt(per_night_contour_res)))
                        Xc, Yc = np.meshgrid(xs, ys)

                        # evaluate coarse grid, then upsample for smoothness
                        upsample_factor = max(1, int(np.clip(4, 1, 12)))  # conservative default; adjust if needed
                        X_up, Y_up, pdf_up, levels = kde_2d_levels_on_grid_up(kde2d, Xc, Yc, per_night_levels, upsample=upsample_factor)

                        levels_plot = [l for l in levels if l > 0]
                        if len(levels_plot) > 0 and np.all(np.isfinite(pdf_up)):
                            try:
                                cs = ax.contour(X_up, Y_up, pdf_up, levels=levels_plot, colors=[color],
                                                alpha=per_night_alpha, linewidths=1.0)
                            except Exception:
                                # fallback to plotting without upsample (coarse grid)
                                pdf_coarse = kde2d(np.vstack([Xc.ravel(), Yc.ravel()])).reshape(Xc.shape)
                                pdf_coarse = np.maximum(pdf_coarse, 1e-300)
                                cs = ax.contour(Xc, Yc, pdf_coarse, levels=levels_plot, colors=[color],
                                                alpha=per_night_alpha, linewidths=1.0)

            handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(nights_data))]
            labels = [f"night {nd['night_index']}" for nd in nights_data]
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)

        # ------------------ ALWAYS DRAW H2O MODEL TRACKS ON p_cloud subplot ([p_cloud_index, 0]) ----
        try:
            if axes is None:
                warnings.warn("Corner axes unavailable: cannot draw H2O model tracks.")
            elif not (0 <= p_cloud_index < nparams):
                warnings.warn(f"p_cloud_index {p_cloud_index} out of range; cannot draw H2O model tracks.")
            else:
                ax = axes[p_cloud_index, 0]

                # paths (change if needed)
                path5 = '/Users/alexsl/Downloads/contour_h2o_5sn.npz'
                #path3 = '/Users/alexsl/Downloads/contour_h2o_3sn.npz'
                
                # load
                d5 = np.load(path5, allow_pickle=True)
                #d3 = np.load(path3, allow_pickle=True)
                
                coords5 = d5['coords']
                #coords3 = d3['coords']
                
                # helper: flatten to (N,2) and score which column looks like pressure
                refs = np.array([1000., 562., 316., 100., 31.6, 10., 3.16, 1.0])
                
                def _assign_Z_P(coords):
                    a = np.asarray(coords).reshape(-1, 2).astype(float).copy()
                    c0, c1 = a[:, 0], a[:, 1]
                    s0 = s1 = 0
                    for r in refs:
                        if np.any(np.isclose(c0, r, rtol=0.02, atol=1e-8)): s0 += 1
                        if np.any(np.isclose(c1, r, rtol=0.02, atol=1e-8)): s1 += 1
                    # if column1 matches pressures more -> (Z, P) = (c0, c1). If col0 matches more -> swap.
                    if s1 >= s0:
                        return c0, c1
                    else:
                        return c1, c0
                
                Z_h2o_sn5, P_cloud_h2o_sn5 = _assign_Z_P(coords5)
                #Z_h2o_sn3, P_cloud_h2o_sn3 = _assign_Z_P(coords3)

                P_cloud_h2o_sn5  *= 1e-3          
                #P_cloud_h2o_sn3  *= 1e-3          


                # convert to log10 to match your corner axes (parameters are log10)
                x5 = np.log10(Z_h2o_sn5)
                y5 = np.log10(P_cloud_h2o_sn5)
                #x3 = np.log10(Z_h2o_sn3)
                #y3 = np.log10(P_cloud_h2o_sn3)

                # plot with high zorder so they appear on top
                ln1 = ax.plot(x5, y5, 'o-', color='goldenrod', markersize=5, linewidth=1.4,
                              label='CCF S/N = 5', zorder=200)[0]
                #ln2 = ax.plot(x3, y3, '^--', color='royalblue', markersize=6, linewidth=1.4,
                #              label='CCF S/N = 3', zorder=201)[0]

                # ensure the model tracks are inside the axis limits (expand if necessary)
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                y_inverted = cur_ylim[0] > cur_ylim[1]

                x_min_model = x5.min() #min(x5.min(), x3.min())
                x_max_model = x5.max() #max(x5.max(), x3.max())
                y_min_model = y5.min() #min(y5.min(), y3.min())
                y_max_model = y5.max() #max(y5.max(), y3.max())

                pad_x = (x_max_model - x_min_model) * 0.05 if (x_max_model > x_min_model) else 0.1
                pad_y = (y_max_model - y_min_model) * 0.05 if (y_max_model > y_min_model) else 0.1

                new_x0 = min(cur_xlim[0], x_min_model - pad_x)
                new_x1 = max(cur_xlim[1], x_max_model + pad_x)
                ax.set_xlim(new_x0, new_x1)

                if y_inverted:
                    new_y_top = max(cur_ylim[0], y_max_model + pad_y)
                    new_y_bot = min(cur_ylim[1], y_min_model - pad_y)
                    ax.set_ylim(new_y_top, new_y_bot)
                else:
                    new_y_bot = min(cur_ylim[0], y_min_model - pad_y)
                    new_y_top = max(cur_ylim[1], y_max_model + pad_y)
                    ax.set_ylim(new_y_bot, new_y_top)

                # add a small legend on that subplot
                ax.legend(fontsize=10, loc='lower left', frameon=False)

                # debug prints (optional)
                print("Drew H2O model tracks on axes[{},{}]. x-range model [{:.3g}, {:.3g}], axis xlim [{:.3g}, {:.3g}]".format(
                      p_cloud_index, 0, x_min_model, x_max_model, new_x0, new_x1))
                print("Drew H2O model tracks on axes[{},{}]. y-range model [{:.3g}, {:.3g}], axis ylim [{:.3g}, {:.3g}] (inverted={})".format(
                      p_cloud_index, 0, y_min_model, y_max_model, ax.get_ylim()[0], ax.get_ylim()[1], y_inverted))

        except Exception as e:
            warnings.warn(f"Could not plot H2O model tracks on p_cloud subplot: {e}")

        # ------------------ Save and show ------------------
        out_corner = os.path.join(out_dir, "combined_corner.pdf")
        fig.savefig(out_corner, bbox_inches='tight')
        print("Saved corner plot to:", out_corner)

        if show_plot:
            plt.show()
        
        return combined_samples, fig


def combine_nights_and_make_ZPc_and_beta_panels(base_dir_template,
    retrieval_name,
    simulation_name,
    night_indices,
    prior_bounds_1d,
    truths=None,
    param_names=None,
    out_dir="./combined_posterior",
    kde_bw_method=None,
    nsamples_combined=20000,
    show_plot=True,
    # optional plotting limits
    z_pc_limits=None,        # e.g. ([-2,3], [-9,0])
    beta_combined_lim=None,  # e.g. [0.7973,0.8005]
    beta_night_lim=None,     # e.g. [1.6,1.7] OR list of per-night [(lo,hi),...]
    beta_night_xlims=None,   # NEW: explicit per-night xlims (list-of-tuples or dict{night: (lo,hi)})
    # NEW customization kwargs
    beta_titles=None,        # list or dict mapping night -> title (falls back to "Night {idx}")
    beta_night_colors=None,  # list or dict mapping night -> color; default: corner_color except last uses alt color
    alt_last_color='tab:orange',  # default color used for the last night if beta_night_colors not provided
    h2o_path5='/Users/alexsl/Downloads/contour_h2o_5sn.npz',
    corner_color='darkslateblue',
    truth_color='firebrick',
    beta_bins=60,
    ):
    """
    CALL LIKE
    -----------
    titles = {0:"CARM. 1st",1:"CARM. 2nd",2:"CARM. 3rd",3:"CARM. 4th",4:"CARM. 5th",5:"CRIRES$^+$"}
    colors = {5: 'firebrick'}  # make last night firebrick
    combined_primary, fig_corner = exosims.combine_nights_and_make_ZPc_and_beta_panels2(
        base_dir_template='/Users/alexsl/Documents/Simulador/CARMENES_NIR/GJ436b/transit/matrices/matrices_Gibson22_withsignal_1nights_SNR_comb1_realdata_noiseless_stdnoisex1/retrieval_night_{night_index}',
        retrieval_name='retrieval',
        simulation_name='simname',
        night_indices=[0,1,2,3,4,5],
        prior_bounds_1d=[(-8.0,0.0),(85.0,200.0),(400.0,1500.0),(-25.0,25.0)],
        beta_night_xlims=[(0.772, 0.78)] +[(0.779, 0.785)]+[(0.795, 0.802)]+[(0.7975, 0.801)]+[(0.7565, 0.761)] + [(1.642, 1.652)],
        beta_titles=titles,
        beta_night_colors=colors,
        show_plot=True,
    )
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import corner
    import warnings
    import math

    os.makedirs(out_dir, exist_ok=True)

    if param_names is None:
        param_names = [r"$\log_{10}(Z)$", r"$\log_{10}(p_{cloud})$ (bar)", r"$\beta$"]

    # Validate night indices
    if not isinstance(night_indices, (list, tuple, np.ndarray)) or len(night_indices) < 1:
        raise ValueError("Provide night_indices as a list/tuple/ndarray with at least one entry")

    # Primary nights = all but last; last = special single night
    if len(night_indices) >= 2:
        primary_nights = list(night_indices)[:-1]
        last_night = int(list(night_indices)[-1])
    else:
        primary_nights = list(night_indices)
        last_night = int(night_indices[0])

    # ---------------------- 1) Reduced corner Z vs p_cloud (all nights) ----------------------
    try:
        combined_all, _ = combine_nights_and_plot_3params(
            base_dir_template=base_dir_template,
            retrieval_name=retrieval_name,
            simulation_name=simulation_name,
            night_indices=night_indices,
            prior_bounds_1d=prior_bounds_1d,
            truths=truths,
            param_names=param_names,
            out_dir=out_dir,
            kde_bw_method=kde_bw_method,
            nsamples_combined=nsamples_combined,
            overlay_per_night=False,
            user_param_limits=None,
            show_plot=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to run combine_nights_and_plot_3params on all nights: {e}")

    combined_all = np.asarray(combined_all)
    if combined_all.ndim != 2 or combined_all.shape[1] < 2:
        raise RuntimeError("Combined result from all nights must have at least 2 parameters (Z,Pc)")

    # select Z (col 0) and p_cloud (col 1)
    samples_zpc = combined_all[:, [0, 1]]
    labels_zpc = [param_names[0], param_names[1]]
    truths_zpc = None
    if truths is not None and len(truths) >= 2:
        truths_zpc = [truths[0], truths[1]]

    plt.close("all")
    corner_kwargs = dict(
        labels=labels_zpc,
        plot_datapoints=True,
        show_titles=False,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 12},
        title_kwargs={"fontsize": 12},
        truth_color=truth_color,
        color=corner_color,
    )

    if truths_zpc is not None:
        fig_corner = corner.corner(samples_zpc, truths=truths_zpc, **corner_kwargs)
    else:
        fig_corner = corner.corner(samples_zpc, **corner_kwargs)

    # Add a global title for the corner plot
    try:
        fig_corner.suptitle("CARMENES $-$ CRIRES$^+$ Combined", fontsize=20, y=1.02)
    except Exception:
        pass

    # Try to reshape axes and apply z/p limits and invert pc y-axis
    try:
        axes = np.array(fig_corner.get_axes()).reshape((2, 2))
    except Exception:
        axes = None
        warnings.warn("Cannot reshape corner axes for Z-Pc; skipping axis tweaks and H2O track")

    # apply z_pc_limits if provided: z_pc_limits should be (zlim, pclim)
    if axes is not None and z_pc_limits is not None:
        try:
            zlim, pclim = z_pc_limits
            axes[0, 0].set_xlim(zlim)   # Z marginal
            axes[1, 1].set_xlim(pclim)  # p_cloud marginal
            axes[1, 0].set_xlim(zlim)
            axes[1, 0].set_ylim(pclim)
        except Exception:
            warnings.warn("Failed to apply z_pc_limits to corner axes")

    # invert p_cloud y-axis on subplot [1,0]
    try:
        if axes is not None:
            ax_to_invert = axes[1, 0]
            ymin, ymax = ax_to_invert.get_ylim()
            if ymin < ymax:
                ax_to_invert.set_ylim(ymax, ymin)
    except Exception as e:
        warnings.warn(f"Could not invert p_cloud axis for Z-Pc corner: {e}")

    # draw H2O model track same as bible
    try:
        if axes is not None:
            ax = axes[1, 0]
            d5 = np.load(h2o_path5, allow_pickle=True)
            coords5 = d5['coords']
            refs = np.array([1000., 562., 316., 100., 31.6, 10., 3.16, 1.0])

            def _assign_Z_P(coords):
                a = np.asarray(coords).reshape(-1, 2).astype(float).copy()
                c0, c1 = a[:, 0], a[:, 1]
                s0 = s1 = 0
                for r in refs:
                    if np.any(np.isclose(c0, r, rtol=0.02, atol=1e-8)): s0 += 1
                    if np.any(np.isclose(c1, r, rtol=0.02, atol=1e-8)): s1 += 1
                if s1 >= s0:
                    return c0, c1
                else:
                    return c1, c0

            Z_h2o_sn5, P_cloud_h2o_sn5 = _assign_Z_P(coords5)
            P_cloud_h2o_sn5 *= 1e-3
            x5 = np.log10(Z_h2o_sn5)
            y5 = np.log10(P_cloud_h2o_sn5)
            ax.plot(x5, y5, 'o-', color='goldenrod', markersize=5, linewidth=1.4, label='CCF S/N = 5', zorder=200)

            # expand axes if needed
            cur_xlim = ax.get_xlim(); cur_ylim = ax.get_ylim()
            y_inverted = cur_ylim[0] > cur_ylim[1]
            x_min_model = x5.min(); x_max_model = x5.max()
            y_min_model = y5.min(); y_max_model = y5.max()
            pad_x = (x_max_model - x_min_model) * 0.05 if (x_max_model > x_min_model) else 0.1
            pad_y = (y_max_model - y_min_model) * 0.05 if (y_max_model > y_min_model) else 0.1
            new_x0 = min(cur_xlim[0], x_min_model - pad_x)
            new_x1 = max(cur_xlim[1], x_max_model + pad_x)
            ax.set_xlim(new_x0, new_x1)
            if y_inverted:
                new_y_top = max(cur_ylim[0], y_max_model + pad_y)
                new_y_bot = min(cur_ylim[1], y_min_model - pad_y)
                ax.set_ylim(new_y_top, new_y_bot)
            else:
                new_y_bot = min(cur_ylim[0], y_min_model - pad_y)
                new_y_top = max(cur_ylim[1], y_max_model + pad_y)
                ax.set_ylim(new_y_bot, new_y_top)
            ax.legend(fontsize=10, loc='lower left', frameon=False)
    except Exception as e:
        warnings.warn(f"Could not plot H2O model track on Z-Pc: {e}")

    out_corner = os.path.join(out_dir, 'combined_corner_Z_Pcloud.pdf')
    try:
        fig_corner.savefig(out_corner, bbox_inches='tight')
        print('Saved reduced Z-Pc corner to:', out_corner)
    except Exception as e:
        warnings.warn(f"Failed to save corner plot: {e}")

    # ---------------------- 2) Prepare per-night beta samples ----------------------
    per_night_betas = []
    for night in night_indices:
        try:
            samp, _ = combine_nights_and_plot_3params(
                base_dir_template=base_dir_template,
                retrieval_name=retrieval_name,
                simulation_name=simulation_name,
                night_indices=[night],
                prior_bounds_1d=prior_bounds_1d,
                truths=truths,
                param_names=param_names,
                out_dir=out_dir,
                kde_bw_method=kde_bw_method,
                nsamples_combined=nsamples_combined,
                overlay_per_night=False,
                user_param_limits=None,
                show_plot=False
            )
            samp = np.asarray(samp)
            if samp.ndim == 1:
                if samp.size >= 3:
                    beta_vals = samp[2].reshape(-1)
                else:
                    beta_vals = samp.reshape(-1)
            elif samp.ndim == 2:
                if samp.shape[1] >= 3:
                    beta_vals = samp[:, 2]
                else:
                    beta_vals = samp[:, -1]
            else:
                beta_vals = None
            if beta_vals is None or (hasattr(beta_vals, 'size') and beta_vals.size == 0):
                raise RuntimeError("No beta samples found")
            per_night_betas.append({'night': night, 'beta': np.asarray(beta_vals)})
            print(f"Loaded beta for night {night}: {per_night_betas[-1]['beta'].shape[0]} samples")
        except Exception as e:
            warnings.warn(f"Could not load/prepare beta samples for night {night}: {e}")
            per_night_betas.append({'night': night, 'beta': None})

    # Also compute combined_primary (all but last) for reference median
    try:
        combined_primary, _ = combine_nights_and_plot_3params(
            base_dir_template=base_dir_template,
            retrieval_name=retrieval_name,
            simulation_name=simulation_name,
            night_indices=primary_nights,
            prior_bounds_1d=prior_bounds_1d,
            truths=truths,
            param_names=param_names,
            out_dir=out_dir,
            kde_bw_method=kde_bw_method,
            nsamples_combined=nsamples_combined,
            overlay_per_night=False,
            user_param_limits=None,
            show_plot=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to run combine_nights_and_plot_3params on primary nights: {e}")

    combined_primary = np.asarray(combined_primary)
    if combined_primary.ndim != 2 or combined_primary.shape[1] < 3:
        raise RuntimeError("Combined primary samples must include beta as third parameter")
    beta_comb = combined_primary[:, 2]

    # ---------------------- 3) Draw grid: 2 rows x 3 cols (or adapt if fewer nights) ----------------------
    n_nights = len(night_indices)
    if n_nights < 1:
        raise RuntimeError("No nights provided for beta plotting")

    # Layout: prefer 3 columns
    ncols = 3
    nrows = int(math.ceil(n_nights / ncols))

    # square size per subplot (inches). Adjust as you like.
    square_side = 2.2
    fig_w = square_side * ncols
    fig_h = square_side * nrows
    fig_beta = plt.figure(figsize=(fig_w, fig_h))

    # gridspec with nrows x ncols
    gs = fig_beta.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.22, hspace=0.35)

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            ax = fig_beta.add_subplot(gs[r, c])
            axes.append(ax)

    # ------------------ resolve per-night xlims using beta_night_xlims (dict/list) first
    per_night_xlims = [None] * n_nights
    if beta_night_xlims is not None:
        if isinstance(beta_night_xlims, dict):
            for k_idx, night in enumerate(night_indices):
                val = beta_night_xlims.get(night, None)
                if val is None:
                    per_night_xlims[k_idx] = None
                else:
                    try:
                        lo, hi = float(val[0]), float(val[1])
                        per_night_xlims[k_idx] = (lo, hi)
                    except Exception:
                        per_night_xlims[k_idx] = None
        elif isinstance(beta_night_xlims, (list, tuple)) and len(beta_night_xlims) == n_nights:
            for k_idx, val in enumerate(beta_night_xlims):
                if val is None:
                    per_night_xlims[k_idx] = None
                else:
                    try:
                        lo, hi = float(val[0]), float(val[1])
                        per_night_xlims[k_idx] = (lo, hi)
                    except Exception:
                        per_night_xlims[k_idx] = None
        else:
            warnings.warn("beta_night_xlims format not recognized; falling back to beta_night_lim / auto xlims")
            per_night_xlims = [None] * n_nights
    else:
        if beta_night_lim is not None:
            try:
                if (isinstance(beta_night_lim, (list, tuple)) and len(beta_night_lim) == n_nights
                        and all((isinstance(x, (list, tuple)) and len(x) == 2) for x in beta_night_lim)):
                    per_night_xlims = [tuple(x) for x in beta_night_lim]
                elif isinstance(beta_night_lim, (list, tuple)) and len(beta_night_lim) == 2 and np.isscalar(beta_night_lim[0]):
                    per_night_xlims = [tuple(beta_night_lim) for _ in range(n_nights)]
                else:
                    per_night_xlims = [None] * n_nights
            except Exception:
                per_night_xlims = [None] * n_nights
        else:
            per_night_xlims = [None] * n_nights

    # ------------------ resolve per-night titles (beta_titles) ------------------
    per_night_titles = [None] * n_nights
    if beta_titles is not None:
        if isinstance(beta_titles, dict):
            for k_idx, night in enumerate(night_indices):
                per_night_titles[k_idx] = beta_titles.get(night, None)
        elif isinstance(beta_titles, (list, tuple)) and len(beta_titles) == n_nights:
            per_night_titles = [str(x) if x is not None else None for x in beta_titles]
        else:
            warnings.warn("beta_titles format not recognized; falling back to default titles")
            per_night_titles = [None] * n_nights

    # ------------------ resolve per-night colors (beta_night_colors) ------------------
    per_night_colors = [None] * n_nights
    if beta_night_colors is not None:
        if isinstance(beta_night_colors, dict):
            for k_idx, night in enumerate(night_indices):
                per_night_colors[k_idx] = beta_night_colors.get(night, None)
        elif isinstance(beta_night_colors, (list, tuple)) and len(beta_night_colors) == n_nights:
            per_night_colors = [c for c in beta_night_colors]
        else:
            warnings.warn("beta_night_colors format not recognized; falling back to defaults")
            per_night_colors = [None] * n_nights

    # Fill defaults where color is None: use corner_color for all except last uses alt_last_color
    for idx in range(n_nights):
        if per_night_colors[idx] is None:
            if idx == (n_nights - 1):
                per_night_colors[idx] = alt_last_color
            else:
                per_night_colors[idx] = corner_color

    # helper to compute xlim from data
    def compute_xlim_from_data(arr, lowp=0.5, highp=99.5, pad_frac=0.06):
        """Compute a sensible x-axis range from a 1D data array.

        Clips the distribution to the given percentiles and adds a fractional
        pad on each side.  Used to set the x-limits of the per-night beta
        histogram panels so that the plotted range is driven by the data rather
        than by isolated outliers.

        Parameters
        ----------
        arr : array-like
            1D array of sample values (e.g., beta posterior draws).
        lowp, highp : float
            Lower and upper percentile thresholds for the core range.
        pad_frac : float
            Fractional padding added to ``(hi - lo)`` on each side.

        Returns
        -------
        tuple of float or None
            ``(lo - pad, hi + pad)``, or ``None`` if the array is empty or
            produces non-finite percentiles.
        """
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return None
            lo = np.percentile(a, lowp)
            hi = np.percentile(a, highp)
            if not np.isfinite(lo) or not np.isfinite(hi):
                return None
            if hi <= lo:
                delta = max(abs(lo) * 0.01, 1e-3)
                lo -= delta
                hi += delta
            pad = (hi - lo) * pad_frac
            if pad == 0:
                pad = max(abs(lo) * 0.01, 1e-3)
            return (lo - pad, hi + pad)
        except Exception:
            return None

    # draw function (now accepts color)
    def draw_beta(ax, data, xlim=None, truth_val=None, title_name="", color=None):
        """Draw a 1D marginal posterior histogram of beta onto a matplotlib Axes.

        Plots a stepped histogram of the beta (noise-scaling) parameter samples
        for one night, with dashed/dotted vertical lines at the 16th, 50th, and
        84th percentiles and an optional truth marker.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target subplot axis.
        data : array-like
            1D array of beta posterior samples for this night.
        xlim : tuple of float or None
            Explicit x-axis range ``(lo, hi)``.  If ``None`` the axis limits are
            left to matplotlib.
        truth_val : float or None
            Injected true value of beta; drawn as a solid vertical line in
            ``truth_color`` when provided.
        title_name : str
            Title string displayed above the subplot.
        color : matplotlib color spec or None
            Histogram edge colour.  Falls back to ``corner_color`` when ``None``.
        """
        if data is None or (hasattr(data, 'size') and data.size == 0):
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        data = np.asarray(data)
        hist_color = color if color is not None else corner_color

        # histogram
        ax.hist(data, bins=beta_bins, density=True, histtype='step', linewidth=1.4, color=hist_color)

        # kde overlay
        try:
            if np.unique(data).size > 1:
                kde = gaussian_kde(data, bw_method=kde_bw_method)
                xs = np.linspace(np.percentile(data, 0.5), np.percentile(data, 99.5), 1000)
                ys = kde(xs)
                #ax.plot(xs, ys, linewidth=1.15, linestyle='-', color=hist_color, alpha=0.9)
        except Exception:
            pass

        q16, q50, q84 = np.percentile(data, [16, 50, 84])
        ax.axvline(q50, color=hist_color, lw=1.6, linestyle='--')
        ax.axvline(q16, color=hist_color, lw=1.0, linestyle=':')
        ax.axvline(q84, color=hist_color, lw=1.0, linestyle=':')

        if truth_val is not None:
            ax.axvline(truth_val, color=truth_color, lw=1.6, linestyle='-')

        if xlim is not None:
            try:
                ax.set_xlim(xlim)
            except Exception:
                pass

        # title selection: prefer supplied title, else default "Night {index}"
        display_title = title_name if (title_name is not None and len(str(title_name)) > 0) else ""
        ax.set_title(display_title, fontsize=10, loc='center')
        ax.set_yticks([])

        try:
            ax.set_box_aspect(1.0)
        except Exception:
            pass

    # iterate and draw each subplot with its xlim & color & title
    for idx, nd in enumerate(per_night_betas):
        ax = axes[idx]
        night = nd['night']
        beta_vals = nd['beta']
        default_title = per_night_titles[idx] if per_night_titles[idx] is not None else f"Night {night}"
        truth_val = (truths[2] if truths is not None and len(truths) >= 3 else None)

        # determine xlim: explicit per-night (beta_night_xlims) already resolved into per_night_xlims
        xlim = per_night_xlims[idx]
        if xlim is None:
            xlim = compute_xlim_from_data(beta_vals)
        if xlim is None:
            xlim = compute_xlim_from_data(beta_comb)

        color = per_night_colors[idx]
        draw_beta(ax, beta_vals if beta_vals is not None else np.array([]), xlim=xlim, truth_val=truth_val, title_name=default_title, color=color)

        # set tick label size
        ax.tick_params(axis='x', which='major', labelsize=9)
        ax.tick_params(axis='y', which='major', labelsize=9)

    # hide any unused axes (if n_nights < nrows*ncols)
    total_axes = nrows * ncols
    if n_nights < total_axes:
        for j in range(n_nights, total_axes):
            try:
                axes[j].axis('off')
            except Exception:
                pass

    # add combined median reference line across all used subplots
    try:
        combined_median = np.median(beta_comb)
        for i in range(min(n_nights, len(axes))):
            axes[i].axvline(combined_median, color='gray', lw=1.0, linestyle='--', alpha=0.6)
    except Exception:
        pass

    # x-label only on bottom-center subplot to avoid clutter:
    try:
        bottom_row_start = (nrows - 1) * ncols
        mid_col_idx = bottom_row_start + (ncols // 2)
        label_ax = axes[mid_col_idx] if mid_col_idx < len(axes) else axes[-1]
        label_ax.set_xlabel(param_names[2], fontsize=16)
    except Exception:
        try:
            axes[-1].set_xlabel(param_names[2], fontsize=10)
        except Exception:
            pass

    out_two = os.path.join(out_dir, 'combined_beta_per_night_grid.pdf')
    try:
        fig_beta.savefig(out_two, bbox_inches='tight')
        print('Saved per-night beta grid plot to:', out_two)
    except Exception as e:
        warnings.warn(f"Could not save per-night beta figure: {e}")

    if show_plot:
        plt.show()

    return combined_primary, fig_corner


def combine_nights_and_plot_3params_2ins(base_dir_template,
        retrieval_name,
        simulation_name,
        night_indices,
        prior_bounds_1d,
        truths=None,
        param_names=None,
        out_dir="./combined_posterior",
        max_samples_per_night=20000,
        candidate_pool_max=60000,
        nsamples_combined=20000,
        kde_bw_method=None,
        overlay_per_night=True,
        per_night_levels=(0.68, 0.95),
        per_night_contour_res=3000,
        per_night_alpha=0.9,
        cmap_name="tab10",
        rng_seed=42,
        show_plot=True,
        # --- NEW options for automatic axis adjustment ---
        auto_adjust_axes=True,              # True | False | 'x' | 'y' | 'both'
        limit_percentiles=(0.5, 99.5),     # percentiles used to clip extremes before padding
        axis_pad_fraction=0.06,            # fractional padding added to (high-low)
        # --- NEW: allow user to supply explicit per-parameter limits ---
        user_param_limits=None             # e.g. [[z_lo,z_hi],[p_lo,p_hi],[beta_top_lo,beta_top_hi],[beta_bot_lo,beta_bot_hi]] or None
    ):
    """
    Combine per-night posteriors, create combined samples, and plot a corner figure.

    Same behaviour as before, but:
     - corner plot restricted to first two params (Z and p_cloud)
     - additional figure fig_betas: two stacked *square* panels with beta PDFs
       top: pooled CARMENES (nights 0..4), bottom: the "other" night (last non-CARMENES)
     - user_param_limits: if len >= 4, entries [2] and [3] set x-limits for top and bottom beta panels.
    Returns (combined_samples, fig, fig_betas)
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import corner
    import warnings
    import matplotlib.cm as cm
    from scipy import ndimage

    # reproducibility
    if rng_seed is not None:
        np.random.seed(rng_seed)

    os.makedirs(out_dir, exist_ok=True)

    # ------------------ Helpers to read Multinest output ------------------
    def try_use_pymultinest_analyzer(base_dir, night, n_params_hint=None):
        """Load posterior samples for one night via the pymultinest Analyzer API.

        Attempts to read the MultiNest output files under ``base_dir`` using
        ``pymultinest.Analyzer``.  The Analyzer expects files named
        ``<retrieval_name>_night_0_*`` (MultiNest convention).

        Parameters
        ----------
        base_dir : str
            Directory containing the MultiNest output files for this night.
        night : int
            Night index (used for logging; the basename currently hardcodes 0).
        n_params_hint : int or None
            Number of free parameters, inferred from the stats JSON when
            available.  Passed directly to ``Analyzer(n_params=...)``.

        Returns
        -------
        dat : np.ndarray, shape (n_samples, n_params)
            Posterior parameter samples.
        weights : np.ndarray, shape (n_samples,)
            Importance weights from column 0 of the Analyzer data array.
        mask_points : np.ndarray of bool, shape (n_samples,)
            Boolean mask selecting samples with weight > 1e-4.
        stats : dict
            Statistics dict returned by ``Analyzer.get_stats()``.

        Raises
        ------
        ImportError
            If ``pymultinest`` is not installed.
        RuntimeError
            If the Analyzer returns an empty data array.
        """
        try:
            from pymultinest import Analyzer
        except Exception as e:
            raise ImportError("pymultinest not available") from e

        basename = os.path.join(base_dir, f"{retrieval_name}_night_0_")
        an = Analyzer(n_params=n_params_hint, outputfiles_basename=basename)
        stats = an.get_stats()
        data = an.get_data()
        if data is None or data.size == 0:
            raise RuntimeError("Analyzer returned empty data")
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] >= 3:
            weights = data[:, 0].astype(float)
            dat = data[:, 2:].astype(float)
        else:
            if n_params_hint is not None and data.shape[1] >= n_params_hint:
                dat = data[:, -n_params_hint:].astype(float)
                weights = data[:, 0].astype(float) if data.shape[1] > n_params_hint else np.ones(dat.shape[0])
            else:
                dat = data.astype(float)
                weights = np.ones(dat.shape[0], dtype=float)
        mask_points = weights > 1e-4
        return dat, weights, mask_points, stats

    def try_read_post_equal_weights(base_dir, night, n_params_hint=None):
        """Load posterior samples for one night by reading the post-equal-weights file.

        Tries a prioritised list of candidate filenames produced by MultiNest
        (``post_equal_weights.dat``, ``.txt``, ``IS.points``, ``live.points``,
        ``IS.ptprob``, and a gzip variant).  Column 0 is treated as the importance
        weight and columns 2 onward as the physical parameter values.

        Parameters
        ----------
        base_dir : str
            Directory containing the MultiNest output files for this night.
        night : int
            Night index, used to construct candidate filenames.
        n_params_hint : int or None
            Expected number of free parameters.  Used to identify which columns
            contain parameter values when the file has fewer than 3 columns.

        Returns
        -------
        dat : np.ndarray, shape (n_samples, n_params)
            Posterior parameter samples.
        weights : np.ndarray, shape (n_samples,)
            Importance weights (ones if all samples are equal-weight).
        mask_points : np.ndarray of bool, shape (n_samples,)
            Boolean mask selecting samples with weight > 1e-4.
        None
            Placeholder for the stats dict (unavailable from flat files).

        Raises
        ------
        FileNotFoundError
            If no readable file is found in any candidate location.
        """
        candidate_fnames = [
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_post_equal_weights.dat"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_post_equal_weights.txt"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_IS.points"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_live.points"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_IS.ptprob"),
            os.path.join(base_dir, f"{retrieval_name}_night_{night}_post_equal_weights.dat.gz"),
        ]
        for fn in candidate_fnames:
            if not os.path.exists(fn):
                continue
            try:
                arr = np.loadtxt(fn, comments='#', ndmin=2)
            except Exception:
                data_list = []
                with open(fn, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        try:
                            vals = [float(p) for p in parts]
                        except Exception:
                            continue
                        data_list.append(vals)
                if len(data_list) == 0:
                    continue
                arr = np.array(data_list, dtype=float)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 3:
                weights = arr[:, 0].astype(float)
                dat = arr[:, 2:].astype(float)
            else:
                if n_params_hint is not None and arr.shape[1] >= n_params_hint:
                    dat = arr[:, -n_params_hint:].astype(float)
                    weights = arr[:, 0].astype(float) if arr.shape[1] > n_params_hint else np.ones(dat.shape[0])
                else:
                    dat = arr.astype(float)
                    weights = np.ones(dat.shape[0], dtype=float)
            mask_points = weights > 1e-4
            return dat, weights, mask_points, None
        raise FileNotFoundError("No suitable post-equal-weights / points file found")

    def infer_nparams_from_stats_json(stats_fn):
        """Infer the number of free parameters from a MultiNest stats JSON file.

        Reads the ``*_stats.json`` file written by MultiNest/pymultinest and
        counts the parameters by inspecting the ``'marginals'`` list or the
        ``'parameters'`` dict.  The inferred count is passed as ``n_params_hint``
        to the loader helpers so the correct columns are extracted.

        Parameters
        ----------
        stats_fn : str
            Path to the ``*_stats.json`` file.

        Returns
        -------
        n_params : int or None
            Number of free parameters, or ``None`` if the file is missing,
            unreadable, or lacks a recognisable structure.
        stats : dict or None
            The parsed JSON object, or ``None`` on failure.
        """
        try:
            with open(stats_fn, 'r') as fh:
                st = json.load(fh)
            if isinstance(st, dict):
                if 'marginals' in st and isinstance(st['marginals'], (list, tuple)):
                    return len(st['marginals']), st
                if 'parameters' in st and isinstance(st['parameters'], dict):
                    return len(st['parameters']), st
        except Exception:
            pass
        return None, None

    # ------------------ Load nights (robustly) ------------------
    nights_data = []
    nparams = None
    for night in night_indices:
        base_dir = base_dir_template.format(night_index=night)
        stats_fn = os.path.join(base_dir, f"{retrieval_name}_night_{night}_stats.json")
        nparams_hint = None
        stats_json_obj = None
        if os.path.exists(stats_fn):
            nph, st = infer_nparams_from_stats_json(stats_fn)
            if nph is not None:
                nparams_hint = nph
                stats_json_obj = st

        loaded = False
        last_exc = None
        try:
            dat, weights, mask_points, stats_obj = try_use_pymultinest_analyzer(base_dir, night, n_params_hint=nparams_hint)
            if stats_obj is None and stats_json_obj is not None:
                stats_obj = stats_json_obj
            loaded = True
            print(f"Loaded night {night} via pymultinest.Analyzer: samples {dat.shape}, weights sum {np.sum(weights):.6g}")
        except Exception as e:
            last_exc = e
            try:
                dat, weights, mask_points, stats_obj = try_read_post_equal_weights(base_dir, night, n_params_hint=nparams_hint)
                loaded = True
                if stats_obj is None and os.path.exists(stats_fn):
                    try:
                        with open(stats_fn, 'r') as fh:
                            stats_obj = json.load(fh)
                    except Exception:
                        stats_obj = stats_json_obj
                print(f"Loaded night {night} via file fallback: samples {dat.shape}, weights sum {np.sum(weights):.6g}")
            except Exception as e2:
                last_exc = e2

        if not loaded:
            raise FileNotFoundError(f"Could not load samples for night {night}: last error: {last_exc}")

        dat = np.asarray(dat)
        if dat.ndim == 1:
            dat = dat.reshape(-1, 1)

        nrows, npar = dat.shape
        if nparams is None:
            nparams = npar
        else:
            if npar != nparams:
                raise RuntimeError(f"Mismatch in parameter count across nights (night {night} has {npar}, expected {nparams})")

        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != nrows:
            if weights.size < nrows:
                neww = np.ones(nrows, dtype=float)
                neww[:weights.size] = weights
                weights = neww
            else:
                weights = weights[:nrows]
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            warnings.warn(f"Weights for night {night} sum to zero or negative; using uniform weights")
            weights = np.ones_like(weights, dtype=float)
        weights = weights / np.sum(weights)

        try:
            mask = np.asarray(mask_points).astype(bool)
            if mask.ndim != 1 or mask.shape[0] != nrows:
                mask = (mask_points > 0).reshape(-1)[:nrows]
        except Exception:
            mask = np.ones(nrows, dtype=bool)

        sel = mask
        if sel.sum() == 0:
            warnings.warn(f"Mask for night {night} selected 0 points; using all")
            sel = np.ones(nrows, dtype=bool)

        dat_sel = dat[sel, :]
        w_sel = weights[sel].astype(float)
        w_sel = np.maximum(w_sel, 0.0)
        if w_sel.sum() <= 0:
            w_sel = np.ones_like(w_sel)
        w_sel = w_sel / np.sum(w_sel)

        if dat_sel.shape[0] > max_samples_per_night:
            idx = np.random.choice(np.arange(dat_sel.shape[0]), size=max_samples_per_night, replace=False, p=w_sel)
            dat_sel = dat_sel[idx, :]
            w_sel = w_sel[idx]
            w_sel = w_sel / w_sel.sum()

        nights_data.append({'night_index': night, 'dat': dat_sel, 'weights': w_sel, 'stats': stats_obj})
        print(f"Prepared night {night}: samples {dat_sel.shape}, weights sum {w_sel.sum():.3f}")

    # default param names
    if param_names is None:
        if nparams == 3:
            param_names = [r"$\log_{10}(Z)$", r"$\log_{10}(p_{cloud})$ (bar)", r"$\beta$"]
        else:
            param_names = [f"p{i}" for i in range(nparams)]

    if truths is not None and len(truths) != nparams:
        warnings.warn("truths length != n_params; ignoring truths")
        truths = None

    print("Parameter names used:", param_names)
    if truths is not None:
        print("Truths provided:", truths)

    # ------------------ Fit KDEs (full multivariate for combination) ------------------
    kdes = []
    for nd in nights_data:
        data = nd['dat']
        w = nd['weights']
        try:
            kde = gaussian_kde(dataset=data.T, weights=w, bw_method=kde_bw_method)
        except Exception as e:
            jitter = 1e-8 * np.std(data, axis=0)
            jitter = np.where(jitter == 0, 1e-8, jitter)
            data_j = data + np.random.normal(scale=jitter, size=data.shape)
            kde = gaussian_kde(dataset=data_j.T, weights=w, bw_method=kde_bw_method)
            warnings.warn(f"KDE fit jitter fallback for night {nd['night_index']}: {e}")
        kdes.append(kde)
    print("Fitted full multivariate KDEs for all nights.")

    # ------------------ Candidate pool (improved: sample from KDEs + data + prior) ------------------
    pool = np.vstack([nd['dat'] for nd in nights_data])

    # fractions for candidate composition (tweak here if you like)
    frac_from_kdes = 0.65
    frac_from_prior = 0.10
    frac_from_data = 0.20
    total_frac = frac_from_kdes + frac_from_prior + frac_from_data
    if total_frac > 1.0:
        frac_from_kdes /= total_frac
        frac_from_prior /= total_frac
        frac_from_data /= total_frac

    target_ncand = int(candidate_pool_max)
    if target_ncand < 1000:
        target_ncand = max(1000, target_ncand)

    n_kde_total = int(target_ncand * frac_from_kdes)
    n_prior = int(target_ncand * frac_from_prior)
    n_data = int(target_ncand * frac_from_data)
    n_remain = target_ncand - (n_kde_total + n_prior + n_data)
    if n_remain < 0:
        n_remain = 0

    candidates_list = []

    # 1) sample from each per-night KDE (spread across nights)
    if n_kde_total > 0:
        per_kde = max(1, n_kde_total // len(kdes))
        extras = n_kde_total - per_kde * len(kdes)
        for i, kde in enumerate(kdes):
            n_this = per_kde + (1 if i < extras else 0)
            try:
                s = kde.resample(n_this).T  # shape (n_this, nparams)
                candidates_list.append(s)
            except Exception:
                # fallback: jittered resample from data
                d = nights_data[i]['dat']
                jitter = 1e-6 * np.std(d, axis=0)
                jitter = np.where(jitter == 0, 1e-6, jitter)
                s = d[np.random.choice(d.shape[0], size=n_this, replace=True)] + np.random.normal(scale=jitter, size=(n_this, d.shape[1]))
                candidates_list.append(s)

    # 2) uniform draws from the prior bounds (to cover edges)
    if n_prior > 0:
        prior_samples = np.zeros((n_prior, nparams))
        for j in range(nparams):
            a, b = prior_bounds_1d[j]
            prior_samples[:, j] = np.random.uniform(low=a, high=b, size=n_prior)
        candidates_list.append(prior_samples)

    # 3) some draws from the pooled original samples (real posterior points)
    if n_data > 0:
        if pool.shape[0] > n_data:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_data, replace=False)
        else:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_data, replace=True)
        candidates_list.append(pool[idx, :])

    # 4) fill remainder with pooled samples (keeps backward compatibility)
    if n_remain > 0:
        if pool.shape[0] > n_remain:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_remain, replace=False)
        else:
            idx = np.random.choice(np.arange(pool.shape[0]), size=n_remain, replace=True)
        candidates_list.append(pool[idx, :])

    # assemble candidates
    if len(candidates_list) == 0:
        candidates = pool.copy()
    else:
        candidates = np.vstack(candidates_list)

    # final safeguard: limit to candidate_pool_max by subsampling (but keep diversity)
    if candidates.shape[0] > candidate_pool_max:
        idx = np.random.choice(np.arange(candidates.shape[0]), size=candidate_pool_max, replace=False)
        candidates = candidates[idx, :]

    ncand = candidates.shape[0]
    print("Candidate pool built: total candidates =", ncand, "(from KDEs/prior/data/pool fractions:",
          frac_from_kdes, frac_from_prior, frac_from_data, ")")

    # ------------------ Evaluate log densities (per-night) ------------------
    tiny = 1e-300
    log_ps = np.zeros((len(kdes), ncand))
    for i, kde in enumerate(kdes):
        try:
            vals = kde(candidates.T)
        except Exception:
            jitter = 1e-12 * np.std(candidates, axis=0)
            jitter = np.where(jitter == 0, 1e-12, jitter)
            cand_j = candidates + np.random.normal(scale=jitter, size=candidates.shape)
            vals = kde(cand_j.T)
        vals = np.maximum(vals, tiny)
        log_ps[i, :] = np.log(vals)

    # evaluate prior on candidates
    def prior_pdf(theta, priors_per_param):
        """Evaluate the joint prior probability density for a parameter vector.

        Computes the product of independent uniform prior densities for each
        parameter.  Any parameter that falls outside its allowed range returns
        density 0 immediately (hard rejection).  Used to regularise the
        combined-posterior reweighting so that the product of per-night
        posteriors is divided by ``(N_nights - 1)`` copies of the prior,
        recovering the correctly normalised combination.

        Parameters
        ----------
        theta : array-like, length n_params
            Physical parameter values.
        priors_per_param : list of dict
            One entry per parameter; each dict must have keys ``'min'`` and
            ``'max'`` (float), or ``None`` to skip that parameter.

        Returns
        -------
        float
            Joint prior density ``∏ 1/(b-a)`` if all parameters are within
            their bounds, else 0.
        """
        pdf = 1.0
        for j, val in enumerate(theta):
            p = priors_per_param[j]
            if p is None:
                continue
            a, b = p['min'], p['max']
            if val < a or val > b:
                return 0.0
            pdf *= 1.0 / (b - a)
        return pdf

    priors_per_param = [{'type': 'uniform', 'min': float(a), 'max': float(b)} for (a, b) in prior_bounds_1d]
    lp = np.zeros(ncand)
    for j in range(ncand):
        pr = prior_pdf(candidates[j, :], priors_per_param)
        lp[j] = np.maximum(pr, tiny)
    log_prior = np.log(lp)

    # ------------------ Combine in log-space ------------------
    Nn = len(kdes)
    log_combined = np.sum(log_ps, axis=0) - (Nn - 1) * log_prior
    log_combined -= np.max(log_combined)
    combined_unnorm = np.exp(log_combined)
    prob = combined_unnorm / np.sum(combined_unnorm)

    if np.any(np.isinf(log_combined)) or np.sum(prob) <= 0:
        raise RuntimeError("Combined posterior failed (all zero or infinite)")

    # ------------------ Resample combined posterior ------------------
    pick_idx = np.random.choice(np.arange(ncand), size=nsamples_combined, replace=True, p=prob)
    combined_samples = candidates[pick_idx, :]

    # compute summary (for all params)
    medians = np.median(combined_samples, axis=0)
    lo68 = np.percentile(combined_samples, 16, axis=0)
    hi68 = np.percentile(combined_samples, 84, axis=0)
    sig = 0.5 * (hi68 - lo68)

    print("\nCombined posterior summary (median +- 1-sigma):")
    for i, name in enumerate(param_names):
        s = f"{medians[i]: .6g} +- {sig[i]: .6g}"
        if truths is not None:
            s += f"    truth={truths[i]: .6g}"
        print(f"  {name:20s} {s}")

    # ------------------ Save combined samples & summary ------------------
    os.makedirs(out_dir, exist_ok=True)
    out_samples_fn = os.path.join(out_dir, "combined_samples.npz")
    np.savez_compressed(out_samples_fn, samples=combined_samples, param_names=param_names, truths=truths)
    summary = {
        'nights_combined': [nd['night_index'] for nd in nights_data],
        'n_params': nparams,
        'param_names': param_names,
        'nsamples_combined': nsamples_combined,
        'medians': medians.tolist(),
        'sigma_1sig': sig.tolist(),
        'truths': truths
    }
    with open(os.path.join(out_dir, "combined_summary.json"), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print("Saved combined samples and summary to:", out_dir)

    # ------------------ Make corner plot for combined samples (ONLY FIRST 2 PARAMS) --
    plt.close('all')
    plot_inds = [0, 1]  # we only show the first two parameters: indices 0 and 1
    nplot = len(plot_inds)

    plotting_samples = combined_samples[:, plot_inds]
    plotting_param_names = [param_names[i] for i in plot_inds]
    plotting_truths = None
    if truths is not None:
        plotting_truths = [truths[i] for i in plot_inds]

    corner_kwargs = dict(
        labels=plotting_param_names,
        plot_datapoints=True,
        show_titles=False,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 12},
        title_kwargs={"fontsize": 12},
        truth_color='firebrick',
        color="darkslateblue"
    )
    if plotting_truths is not None:
        fig = corner.corner(plotting_samples, truths=plotting_truths, **corner_kwargs)
    else:
        fig = corner.corner(plotting_samples, **corner_kwargs)

    # try to reshape axes into nplot x nplot
    try:
        axes = np.array(fig.get_axes()).reshape((nplot, nplot))
    except Exception:
        axes = None
        warnings.warn("Cannot reshape corner axes; skipping per-night overlay and auto-axis adjustments.")

    # ------------------ AUTO-ADJUST AXES (applied to plotted params only) --------------
    def compute_param_limits_from_samples(samples, percentiles=(0.5, 99.5), pad_frac=0.06):
        mins = np.percentile(samples, percentiles[0], axis=0)
        maxs = np.percentile(samples, percentiles[1], axis=0)
        limits = []
        for lo, hi in zip(mins, maxs):
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = np.nanmin(samples[:, 0]), np.nanmax(samples[:, 0])  # fallback (unlikely)
            if hi <= lo:
                delta = np.abs(lo) * 1e-3 if lo != 0 else 1e-3
                lo -= delta
                hi += delta
            pad = (hi - lo) * pad_frac
            limits.append((lo - pad, hi + pad))
        return limits

    # parse auto_adjust_axes flag (accept True/False/'x'/'y'/'both')
    adjust_x = adjust_y = False
    if auto_adjust_axes is True or auto_adjust_axes == 'both':
        adjust_x = adjust_y = True
    elif auto_adjust_axes == 'x':
        adjust_x = True
    elif auto_adjust_axes == 'y':
        adjust_y = True
    elif auto_adjust_axes in (False, None):
        adjust_x = adjust_y = False
    else:
        adjust_x = adjust_y = True

    # compute auto limits for plotting dims using combined_samples subset
    auto_param_limits_plot = None
    if (adjust_x or adjust_y) and axes is not None:
        try:
            auto_param_limits_plot = compute_param_limits_from_samples(combined_samples[:, plot_inds], percentiles=limit_percentiles, pad_frac=axis_pad_fraction)
        except Exception as e:
            warnings.warn(f"Failed to compute auto param limits for plotting dims: {e}")
            auto_param_limits_plot = None

    # ------------------ MERGE user_param_limits with auto limits (compute full final_param_limits,
    # then map to plotting dims when applying) ------------------
    final_param_limits = None
    auto_param_limits_all = None
    if (adjust_x or adjust_y):
        try:
            auto_param_limits_all = compute_param_limits_from_samples(combined_samples, percentiles=limit_percentiles, pad_frac=axis_pad_fraction)
        except Exception:
            auto_param_limits_all = None

    if user_param_limits is None:
        final_param_limits = auto_param_limits_all
    else:
        try:
            upl = list(user_param_limits)
            if len(upl) != nparams and len(upl) < nparams:
                # allow case where user passed extra beta x-lims: len(upl) can be >= nparams
                if len(upl) < nparams:
                    warnings.warn(f"user_param_limits length {len(upl)} < nparams {nparams}; using auto limits for missing params")
                final_param_limits = auto_param_limits_all
            else:
                final_param_limits = []
                for i in range(nparams):
                    # if user provided explicit entry for parameter i, use it (if valid)
                    if i < len(upl) and (upl[i] is not None):
                        try:
                            a, b = float(upl[i][0]), float(upl[i][1])
                            if b <= a:
                                warnings.warn(f"user_param_limits[{i}] has hi<=lo ({a},{b}); swapping values")
                                a, b = min(a, b), max(a, b)
                            final_param_limits.append((a, b))
                            continue
                        except Exception:
                            warnings.warn(f"Invalid entry for user_param_limits[{i}] = {upl[i]}; will fall back to auto for this param")
                    # fallback
                    if auto_param_limits_all is not None:
                        final_param_limits.append(auto_param_limits_all[i])
                    else:
                        lo = np.percentile(combined_samples[:, i], limit_percentiles[0])
                        hi = np.percentile(combined_samples[:, i], limit_percentiles[1])
                        if hi <= lo:
                            hi = lo + 1e-3
                        pad = (hi - lo) * axis_pad_fraction
                        final_param_limits.append((lo - pad, hi + pad))
        except Exception as e:
            warnings.warn(f"Failed to parse user_param_limits ({e}); falling back to auto limits")
            final_param_limits = auto_param_limits_all

    # ------------------ APPLY final_param_limits to plotting axes (map indices) ----------------
    if axes is not None and final_param_limits is not None and (adjust_x or adjust_y):
        try:
            for ii in range(nplot):
                for jj in range(nplot):
                    ax = axes[ii, jj]
                    param_i = plot_inds[ii]
                    param_j = plot_inds[jj]
                    if ii == jj:
                        lo, hi = final_param_limits[param_i]
                        if adjust_x:
                            try:
                                ax.set_xlim(lo, hi)
                            except Exception:
                                pass
                    elif ii > jj:
                        if adjust_x:
                            try:
                                lo_x, hi_x = final_param_limits[param_j]
                                ax.set_xlim(lo_x, hi_x)
                            except Exception:
                                pass
                        if adjust_y:
                            try:
                                lo_y, hi_y = final_param_limits[param_i]
                                ax.set_ylim(lo_y, hi_y)
                            except Exception:
                                pass
            print("Applied final parameter limits to corner axes (for plotted dims).")
        except Exception as e:
            warnings.warn(f"Applying final_param_limits to plotted dims failed: {e}")

    # ----- Cosmetic tweak: invert y-axis for p_cloud subplot if p_cloud is plotted -----
    p_cloud_index = 1
    try:
        if axes is not None and (p_cloud_index in plot_inds):
            p_cloud_plot_idx = plot_inds.index(p_cloud_index)
            ax_to_invert = axes[p_cloud_plot_idx, 0]
            ymin, ymax = ax_to_invert.get_ylim()
            if ymin < ymax:
                ax_to_invert.set_ylim(ymax, ymin)
                print(f"Inverted y-axis for subplot [{p_cloud_plot_idx}, 0] (p_cloud).")
    except Exception as e:
        warnings.warn(f"Could not invert p_cloud axis at axes[{p_cloud_index},0]: {e}")

    # ------------------ Per-night overlays (optional) applied only on plotted dims -------------
    def kde_2d_levels_on_grid_up(kde2d, X, Y, prob_levels, upsample=3, min_pdf_val=1e-300):
        coords = np.vstack([X.ravel(), Y.ravel()])
        pdf = kde2d(coords).reshape(X.shape)
        pdf = np.maximum(pdf, min_pdf_val)

        dx = (X[0, 1] - X[0, 0])
        dy = (Y[1, 0] - Y[0, 0])
        area = dx * dy

        if upsample is None or int(upsample) <= 1:
            pf = pdf.ravel()
            idx = np.argsort(pf)[::-1]
            pf_sorted = pf[idx]
            cumsum = np.cumsum(pf_sorted) * area
            levels = []
            for p in prob_levels:
                mask = cumsum >= p
                if np.any(mask):
                    levels.append(pf_sorted[mask][-1])
                else:
                    levels.append(pf_sorted.max() * 0.0)
            return X, Y, pdf, levels

        pdf_up = ndimage.zoom(pdf, zoom=(upsample, upsample), order=3)
        ny_up, nx_up = pdf_up.shape

        x_min, x_max = X[0, 0], X[0, -1]
        y_min, y_max = Y[0, 0], Y[-1, 0]
        xs_up = np.linspace(x_min, x_max, nx_up)
        ys_up = np.linspace(y_min, y_max, ny_up)
        X_up, Y_up = np.meshgrid(xs_up, ys_up)

        dx_up = xs_up[1] - xs_up[0] if nx_up > 1 else (x_max - x_min)
        dy_up = ys_up[1] - ys_up[0] if ny_up > 1 else (y_max - y_min)
        area_up = dx_up * dy_up

        pf = pdf_up.ravel()
        idx = np.argsort(pf)[::-1]
        pf_sorted = pf[idx]
        cumsum = np.cumsum(pf_sorted) * area_up

        levels = []
        for p in prob_levels:
            mask = cumsum >= p
            if np.any(mask):
                levels.append(pf_sorted[mask][-1])
            else:
                levels.append(pf_sorted.max() * 0.0)

        return X_up, Y_up, pdf_up, levels

    if overlay_per_night and axes is not None:
        cmap = cm.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(len(nights_data))]

        for ix, nd in enumerate(nights_data):
            data = nd['dat']
            w = nd['weights']
            color = colors[ix]
            per_night_medians = np.median(data, axis=0)

            per_night_medians_plot = per_night_medians[plot_inds]

            for i_plot in range(nplot):
                axd = axes[i_plot, i_plot]
                axd.axvline(per_night_medians_plot[i_plot], color=color, alpha=per_night_alpha, lw=1.0, linestyle='--')

            for i_plot in range(nplot):
                for j_plot in range(i_plot):
                    ax = axes[i_plot, j_plot]
                    idx_j = plot_inds[j_plot]
                    idx_i = plot_inds[i_plot]
                    pair_data = data[:, [idx_j, idx_i]]
                    try:
                        kde2d = gaussian_kde(dataset=pair_data.T, weights=w, bw_method=kde_bw_method)
                    except Exception:
                        jitter = 1e-8 * np.std(pair_data, axis=0)
                        jitter = np.where(jitter == 0, 1e-8, jitter)
                        pair_j = pair_data + np.random.normal(scale=jitter, size=pair_data.shape)
                        kde2d = gaussian_kde(dataset=pair_j.T, weights=w, bw_method=kde_bw_method)

                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()

                    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                        x_min, x_max = np.percentile(pair_data[:, 0], [0.5, 99.5])
                        if x_max <= x_min:
                            x_min -= 1e-3
                            x_max += 1e-3
                    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                        y_min, y_max = np.percentile(pair_data[:, 1], [0.5, 99.5])
                        if y_max <= y_min:
                            y_min -= 1e-3
                            y_max += 1e-3

                    xs = np.linspace(x_min, x_max, int(np.sqrt(per_night_contour_res)))
                    ys = np.linspace(y_min, y_max, int(np.sqrt(per_night_contour_res)))
                    Xc, Yc = np.meshgrid(xs, ys)

                    upsample_factor = max(1, int(np.clip(4, 1, 12)))
                    X_up, Y_up, pdf_up, levels = kde_2d_levels_on_grid_up(kde2d, Xc, Yc, per_night_levels, upsample=upsample_factor)

                    levels_plot = [l for l in levels if l > 0]
                    if len(levels_plot) > 0 and np.all(np.isfinite(pdf_up)):
                        try:
                            cs = ax.contour(X_up, Y_up, pdf_up, levels=levels_plot, colors=[color],
                                            alpha=per_night_alpha, linewidths=1.0)
                        except Exception:
                            pdf_coarse = kde2d(np.vstack([Xc.ravel(), Yc.ravel()])).reshape(Xc.shape)
                            pdf_coarse = np.maximum(pdf_coarse, 1e-300)
                            cs = ax.contour(Xc, Yc, pdf_coarse, levels=levels_plot, colors=[color],
                                            alpha=per_night_alpha, linewidths=1.0)

        handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(nights_data))]
        labels = [f"night {nd['night_index']}" for nd in nights_data]
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)

    # ------------------ ALWAYS DRAW H2O MODEL TRACKS ON p_cloud subplot (if plotted) ----
    try:
        if axes is None:
            warnings.warn("Corner axes unavailable: cannot draw H2O model tracks.")
        elif not (0 <= p_cloud_index < nparams):
            warnings.warn(f"p_cloud_index {p_cloud_index} out of range; cannot draw H2O model tracks.")
        elif p_cloud_index not in plot_inds:
            warnings.warn("p_cloud parameter not among plotted dims; skipping H2O model tracks.")
        else:
            p_cloud_plot_idx = plot_inds.index(p_cloud_index)
            ax = axes[p_cloud_plot_idx, 0]

            path5 = '/Users/alexsl/Downloads/contour_h2o_5sn.npz'
            d5 = np.load(path5, allow_pickle=True)
            coords5 = d5['coords']
            refs = np.array([1000., 562., 316., 100., 31.6, 10., 3.16, 1.0])

            def _assign_Z_P(coords):
                a = np.asarray(coords).reshape(-1, 2).astype(float).copy()
                c0, c1 = a[:, 0], a[:, 1]
                s0 = s1 = 0
                for r in refs:
                    if np.any(np.isclose(c0, r, rtol=0.02, atol=1e-8)): s0 += 1
                    if np.any(np.isclose(c1, r, rtol=0.02, atol=1e-8)): s1 += 1
                if s1 >= s0:
                    return c0, c1
                else:
                    return c1, c0

            Z_h2o_sn5, P_cloud_h2o_sn5 = _assign_Z_P(coords5)
            P_cloud_h2o_sn5  *= 1e-3

            x5 = np.log10(Z_h2o_sn5)
            y5 = np.log10(P_cloud_h2o_sn5)

            ln1 = ax.plot(x5, y5, 'o-', color='goldenrod', markersize=5, linewidth=1.4,
                          label='CCF S/N = 5', zorder=200)[0]

            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            y_inverted = cur_ylim[0] > cur_ylim[1]

            x_min_model = x5.min()
            x_max_model = x5.max()
            y_min_model = y5.min()
            y_max_model = y5.max()

            pad_x = (x_max_model - x_min_model) * 0.05 if (x_max_model > x_min_model) else 0.1
            pad_y = (y_max_model - y_min_model) * 0.05 if (y_max_model > y_min_model) else 0.1

            new_x0 = min(cur_xlim[0], x_min_model - pad_x)
            new_x1 = max(cur_xlim[1], x_max_model + pad_x)
            ax.set_xlim(new_x0, new_x1)

            if y_inverted:
                new_y_top = max(cur_ylim[0], y_max_model + pad_y)
                new_y_bot = min(cur_ylim[1], y_min_model - pad_y)
                ax.set_ylim(new_y_top, new_y_bot)
            else:
                new_y_bot = min(cur_ylim[0], y_min_model - pad_y)
                new_y_top = max(cur_ylim[1], y_max_model + pad_y)
                ax.set_ylim(new_y_bot, new_y_top)

            ax.legend(fontsize=10, loc='lower left', frameon=False)

            print("Drew H2O model tracks on p_cloud plotted axes. x-range model [{:.3g}, {:.3g}]".format(x_min_model, x_max_model))

    except Exception as e:
        warnings.warn(f"Could not plot H2O model tracks on p_cloud subplot: {e}")

    # ------------------ Save corner and show ------------------
    out_corner = os.path.join(out_dir, "combined_corner.pdf")
    fig.savefig(out_corner, bbox_inches='tight')
    print("Saved corner plot to:", out_corner)

    if show_plot:
        plt.show()
        
    

    # ------------------ NEW: Build the second figure with stacked square beta panels ------------------
    def weighted_quantile(values, weights, quantiles):
        """Compute weighted quantiles of a 1D distribution.

        Sorts the values by magnitude, accumulates the normalised weights into a
        cumulative distribution, then interpolates to find the physical value at
        each requested quantile.  Used to compute credible-interval markers
        (e.g., 16th, 50th, 84th percentiles) on importance-weighted posterior
        samples where simple ``np.percentile`` would give incorrect results.

        Parameters
        ----------
        values : array-like
            1D array of sample values (e.g., beta draws).
        weights : array-like
            Non-negative importance weights; need not be normalised.
        quantiles : array-like
            Target quantile fractions in [0, 1], e.g. ``[0.16, 0.5, 0.84]``.

        Returns
        -------
        np.ndarray
            Interpolated quantile values, same length as ``quantiles``.
            Returns ``np.nan`` for each quantile if the input is empty or
            contains no finite values.
        """
        values = np.asarray(values).astype(float)
        weights = np.asarray(weights).astype(float)
        if values.size == 0:
            return np.array([np.nan for _ in quantiles])
        mask = np.isfinite(values) & np.isfinite(weights)
        values = values[mask]
        weights = weights[mask]
        if values.size == 0:
            return np.array([np.nan for _ in quantiles])
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
        cumw = np.cumsum(weights)
        if cumw[-1] == 0:
            return np.percentile(values, 100.0 * np.asarray(quantiles))
        cumw = cumw / cumw[-1]
        return np.interp(quantiles, cumw, values)

    beta_idx = None
    if nparams is not None:
        if nparams >= 3:
            beta_idx = 2
        else:
            beta_idx = None
    if param_names is not None:
        for i, nm in enumerate(param_names):
            if ('beta' in nm.lower()) or (r'\beta' in nm) or (nm.strip().lower() == 'beta'):
                beta_idx = i
                break

    # CARMENES nights target
    carmenes_target = [0, 1, 2, 3, 4]
    
    # locate which kdes correspond to carmenes_target (kdes are in same order as nights_data)
    kde_indices_carm = [i for i, nd in enumerate(nights_data) if nd['night_index'] in carmenes_target]
    if len(kde_indices_carm) == 0:
        print("No CARMENES kdes found among nights_data; beta top will be empty.")
    # ensure beta index defined
    if beta_idx is None or beta_idx >= nparams:
        warnings.warn("beta_idx not found or out of range; skipping beta panels.")
        fig_betas = None
    else:
        # --- Build combined posterior *for CARMENES nights only* using the same logic as above ---
        # Use the same candidate pool and per-night KDE log_ps already computed earlier.
        # `log_ps` was computed as shape (len(kdes), ncand)
        if 'log_ps' not in locals():
            # if your function did not compute log_ps earlier (should have), recompute for kdes/candidates
            tiny = 1e-300
            ncand = candidates.shape[0]
            log_ps = np.zeros((len(kdes), ncand))
            for i, kde in enumerate(kdes):
                vals = np.maximum(kde(candidates.T), tiny)
                log_ps[i, :] = np.log(vals)
    
        # indices in the log_ps array that correspond to CARMENES nights
        if len(kde_indices_carm) == 0:
            log_combined_carm = None
            carm_combined_samples = np.array([])
        else:
            log_ps_carm = log_ps[kde_indices_carm, :]   # shape (N_carm, ncand)
            Nn_carm = log_ps_carm.shape[0]
            # combine in log-space exactly as you did before
            log_combined_carm = np.sum(log_ps_carm, axis=0) - (Nn_carm - 1) * log_prior
            # stabilize & exponentiate
            log_combined_carm -= np.max(log_combined_carm)
            unnorm = np.exp(log_combined_carm)
            if np.sum(unnorm) <= 0:
                raise RuntimeError("CARMENES combined posterior is zero everywhere")
            prob_carm = unnorm / np.sum(unnorm)
            # resample to create samples that represent the combined posterior
            np.random.seed(rng_seed if rng_seed is not None else None)
            pick_idx_carm = np.random.choice(np.arange(candidates.shape[0]), size=nsamples_combined, replace=True, p=prob_carm)
            carm_combined_samples = candidates[pick_idx_carm, :]   # shape (nsamples_combined, nparams)
    
        # --- Build CRIRES (other) beta data: pick the last nights_data entry not in carmenes_target ---
        non_carm = [nd for nd in nights_data if nd['night_index'] not in carmenes_target]
        cri_vals = np.array([])
        cri_w = np.array([])
        if len(non_carm) > 0:
            nd_cri = non_carm[-1]
            if beta_idx is not None and beta_idx < nd_cri['dat'].shape[1]:
                cri_vals = nd_cri['dat'][:, beta_idx].ravel()
                cri_w = np.asarray(nd_cri['weights']).ravel()
                cri_w = np.maximum(cri_w, 0.0)
                if cri_w.sum() > 0:
                    cri_w = cri_w / cri_w.sum()
                else:
                    cri_w = np.ones_like(cri_vals) / cri_vals.size
    
        # --- Determine x-limits for the beta panels ---
        # user param limits entries 2 and 3 override if present
        beta_top_xlim = user_param_limits[2]
        beta_bot_xlim = user_param_limits[3]
        if user_param_limits is not None and len(user_param_limits) >= 4:
            try:
                beta_top_xlim = (float(user_param_limits[2][0]), float(user_param_limits[2][1]))
                beta_bot_xlim = (float(user_param_limits[3][0]), float(user_param_limits[3][1]))
                # make them canonical order
                if beta_top_xlim[1] <= beta_top_xlim[0]:
                    beta_top_xlim = (min(beta_top_xlim), max(beta_top_xlim))
                if beta_bot_xlim[1] <= beta_bot_xlim[0]:
                    beta_bot_xlim = (min(beta_bot_xlim), max(beta_bot_xlim))
            except Exception:
                beta_top_xlim = beta_bot_xlim = None
    
        # compute data-driven common xlim if user didn't force per-panel limits
        def _percentile_range(vals, qlo=0.5, qhi=99.5):
            if vals is None or vals.size == 0:
                return None
            try:
                lo = np.nanpercentile(vals, qlo)
                hi = np.nanpercentile(vals, qhi)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = np.nanpercentile(vals, [1.0, 99.0])
                return lo, hi
            except Exception:
                return None
    
        # build arrays to inspect
        carm_beta_samples = np.array([])    # will hold 1D beta samples from combined posterior
        if carm_combined_samples.size != 0:
            carm_beta_samples = carm_combined_samples[:, beta_idx].ravel()
    
        # choose final x-range:
        if beta_top_xlim is not None and beta_bot_xlim is not None:
            # user provided both: use union to allow comparison on same axis
            xmin = min(beta_top_xlim[0], beta_bot_xlim[0])
            xmax = max(beta_top_xlim[1], beta_bot_xlim[1])
            final_xlim = (xmin, xmax)
        else:
            # compute per-data ranges
            r1 = _percentile_range(carm_beta_samples) if carm_beta_samples.size>0 else None
            r2 = _percentile_range(cri_vals) if cri_vals.size>0 else None
            # prioritize user-specified single limits if present
            if beta_top_xlim is not None and (beta_bot_xlim is None):
                final_xlim = beta_top_xlim
            elif beta_bot_xlim is not None and (beta_top_xlim is None):
                final_xlim = beta_bot_xlim
            else:
                # union of data ranges with a small pad
                lo_candidates = []
                hi_candidates = []
                for rr in (r1, r2):
                    if rr is not None:
                        lo_candidates.append(rr[0])
                        hi_candidates.append(rr[1])
                if len(lo_candidates) == 0:
                    final_xlim = None
                else:
                    lo = min(lo_candidates); hi = max(hi_candidates)
                    pad = (hi - lo) * 0.06 if (hi > lo) else 0.1
                    final_xlim = (lo - pad, hi + pad)
    
        # --- plotting ---
        fig_betas = None
        try:
            W = 6.0
            # sharex=True so both panels have the same x-axis for direct comparison
            fig_betas, axes_b = plt.subplots(nrows=2, ncols=1, figsize=(W, 2 * W), sharex=True)
            ax_top, ax_bot = axes_b
            color = "darkslateblue"
    
            # a small helper to plot a 1D KDE (from unweighted samples or weighted arrays)
            def _plot_from_samples(ax, samples_1d, weights=None, label=None, xlim_user=None):
                samples_1d = np.asarray(samples_1d).ravel()
                if samples_1d.size == 0:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha='center', va='center')
                    return None, (np.nan, np.nan, np.nan)
    
                if weights is None or weights.size != samples_1d.size:
                    weights = np.ones_like(samples_1d) / samples_1d.size
                else:
                    weights = np.asarray(weights).ravel()
                    weights = np.maximum(weights, 0.0)
                    if weights.sum() <= 0:
                        weights = np.ones_like(samples_1d) / samples_1d.size
                    else:
                        weights = weights / weights.sum()
    
                # weighted percentiles
                sorter = np.argsort(samples_1d)
                vs = samples_1d[sorter]; ws = weights[sorter]
                cs = np.cumsum(ws)
                if cs[-1] <= 0:
                    cs = np.linspace(0, 1, len(ws))
                else:
                    cs = cs / cs[-1]
                q16 = np.interp(0.16, cs, vs)
                q50 = np.interp(0.50, cs, vs)
                q84 = np.interp(0.84, cs, vs)
    
                # determine x-range for density eval
                if xlim_user is not None:
                    x_min, x_max = float(xlim_user[0]), float(xlim_user[1])
                else:
                    lo = np.interp(0.005, cs, vs)
                    hi = np.interp(0.995, cs, vs)
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
                        lo, hi = np.nanpercentile(samples_1d, [0.5, 99.5])
                    pad = (hi - lo) * 0.08 if (hi > lo) else max(0.1, abs(lo)*0.05)
                    x_min, x_max = lo - pad, hi + pad
    
                x = np.linspace(x_min, x_max, 400)
    
                # build KDE from samples (weights supported)
                try:
                    kde1 = gaussian_kde(dataset=samples_1d.reshape(1, -1), weights=weights, bw_method=kde_bw_method)
                    pdf = kde1(x)
                    area = np.trapezoid(pdf, x)
                    if area > 0:
                        pdf = pdf / area
                    ax.plot(x, pdf, '-', color=color, lw=1.6, label=label)
                except Exception:
                    # fallback to weighted histogram
                    bins = 80
                    hist_vals, edges = np.histogram(samples_1d, bins=bins, weights=weights, density=False)
                    centers = 0.5*(edges[:-1] + edges[1:])
                    area = np.sum(hist_vals * np.diff(edges))
                    if area > 0:
                        hist_vals = hist_vals / area
                    ax.plot(centers, hist_vals, '-', color=color, lw=1.6, label=label)
                    pdf = None
    
                ax.axvline(q50, color='k', linestyle='--', lw=1.2)
                ax.axvline(q16, color='k', linestyle=':', lw=1.0)
                ax.axvline(q84, color='k', linestyle=':', lw=1.0)
                ax.set_yticks([])
                return kde1 if 'kde1' in locals() else None, (q16, q50, q84)
    
            # Top: combined posterior from CARMENES-only combined_samples (as produced above)
            if carm_beta_samples.size == 0:
                ax_top.text(0.5, 0.5, "No CARMENES combined beta data found", ha='center', va='center')
            else:
                _plot_from_samples(ax_top, carm_beta_samples, weights=None, label='CARMENES combined', xlim_user=final_xlim)
    
            ax_top.set_title("Combined β, CARMENES (nights 0..4)", fontsize=14)
    
            # Bottom: CRIRES individual night posterior (weighted)
            if cri_vals.size == 0:
                ax_bot.text(0.5, 0.5, "No non-CARMENES night beta data found", ha='center', va='center')
            else:
                _plot_from_samples(ax_bot, cri_vals, weights=cri_w, label='CRIRES (other)', xlim_user=final_xlim)
    
            ax_bot.set_title("β, Other night", fontsize=14)
            ax_bot.set_xlabel(r'$\beta$', fontsize=14)
    
            plt.tight_layout()
            out_betas = os.path.join(out_dir, "combined_betas.pdf")
            fig_betas.savefig(out_betas, bbox_inches='tight')
            print("Saved beta plot to:", out_betas)
            if show_plot:
                plt.show()
        except Exception as e:
            warnings.warn(f"Building/saving beta figure failed (new combined approach): {e}")
            fig_betas = None


        

    # ------------------ Return combined samples, main corner fig, and beta fig -------------
    return combined_samples, fig, fig_betas


