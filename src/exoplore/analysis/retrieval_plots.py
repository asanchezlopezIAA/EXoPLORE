"""
exoplore.analysis.retrieval_plots
===================================

Post-processing plots for Bayesian atmospheric retrievals (MultiNest output).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

def plot_live_posterior(base_dir, retrieval_name, night_index,
                n_params, inp_dat, truths=None, refresh=True):
    from pymultinest import Analyzer
    import corner
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    """
    Read the partial MultiNest output on disk, print parameter summaries,
    and plot a live corner plot.

    Parameters
    ----------
    base_dir : str
        Base output directory.
    retrieval_name : str
        Retrieval name prefix.
    night_index : int
        The `i` in your filename f"{retrieval_name}_night_{i}_".
    n_params : int
        Number of fitted parameters.
    inp_dat : dict
        Your input data dict (for defaults in `truths` if not provided).
    truths : list of float, optional
        True parameter values to overplot (default from inp_dat if None).
    refresh : bool, default=True
        If True, clears the current figure before plotting.
        
    USAGE:
        base_dir = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"
        retrieval_name = "retrieval"
        i=0
        n_params = 5
        
        exosims.plot_live_posterior(
            base_dir, retrieval_name, 0, n_params, inp_dat, 
            truths=[
                np.log10(inp_dat['vmr'][2]), 
                np.log10(inp_dat['vmr'][3]), 
                np.log10(inp_dat['vmr'][4]), 
                np.log10(inp_dat['vmr'][5]), 
                inp_dat['T_equ'], 1], refresh=True)
    """
    
    # build the basename exactly as used in run()
    basename = f"{base_dir}/{retrieval_name}_night_0_"

    # pull the partial posterior *and* weights
    a    = Analyzer(n_params=n_params, outputfiles_basename=basename)
    data = a.get_data()               # shape (N, 2 + n_params)
    weights = data[:, 1]              # the prior * weight column
    samples = data[:, 2:]             # the θ columns

    # apply your 1e-4 weight mask exactly as in the final plot
    mask = weights > 1e-4
    samples = samples[mask, :]
    weights = weights[mask]

    # print summary of just the heavy‐weight samples
    print("Live posterior summary (median ±1σ):")
    for j in range(n_params):
        p16, p50, p84 = np.percentile(samples[:, j], [16, 50, 84])
        lower = p50 - p16
        upper = p84 - p50
        print(f"  θ{j+1}: {p50:.3f} (+{upper:.3f}/-{lower:.3f})")

    # default truths if not provided
    if truths is None and 'vmr' in inp_dat:
        truths = [
            np.log10(inp_dat['vmr'][2]),
            inp_dat.get('K_p', None),
            inp_dat.get('T_equ', None),
            inp_dat.get('V_wind', None),
            1.0   # your beta truth
        ]

    # only plot if more masked samples than dimensions
    if samples.shape[0] <= n_params:
        print(f"Not enough live samples to plot corner "
              f"({samples.shape[0]} ≤ {n_params}); skipping plot.")
        return

    if refresh:
        plt.clf()

    # create the corner plot
    fig = corner.corner(
        samples,
        labels=[f"θ{i}" for i in range(1, n_params+1)],
        show_titles=True,
        title_fmt=".2f",
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 14}
    )
    plt.suptitle("Live posterior (partial run)", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    # optionally save
    plotdir = (
        inp_dat["home_dir"].rstrip("/") + "/plots_SNR"
        if inp_dat.get("All_significance_metrics", False)
        else inp_dat["plots_dir"].rstrip("/")
    )
    os.makedirs(plotdir, exist_ok=True)
    out = os.path.join(plotdir, f"{retrieval_name}_night_0_live_corner.pdf")
    fig.savefig(out)
    print(f"Saved live corner to {out}")


def plot_live_posterior2(output_basename: str,
        n_params: int,
        truths: list = None,
        mask_fn: callable = lambda w: w > 1e-4,
        interactive: bool = True,
        save_path: str = None,
        corner_kwargs: dict = None
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import corner
    import numpy as np
    from pymultinest import Analyzer
    """
    Read partial MultiNest output and produce a live-updating or saved corner plot.

    Safely handles incomplete parameter rows by adapting to available dims.
    Prints summaries and evidence to stdout.
    """
    corner_kwargs = corner_kwargs or {}

    # Read raw data
    analyzer = Analyzer(n_params=n_params, outputfiles_basename=output_basename)
    data = analyzer.get_data()          # shape (N, 2 + dims_written)
    print(f"Reading data from {output_basename}: shape {data.shape}")
    weights = data[:, 1]
    samples = data[:, 2:]

    # Determine actual dims present
    N, actual_dims = samples.shape
    if actual_dims < n_params:
        print(f"[Warning] Expected up to {n_params} dims, but found {actual_dims}. Using {actual_dims}.")
        n_params = actual_dims

    # Apply weight mask
    mask = mask_fn(weights)
    samples = samples[mask, :n_params]
    weights = weights[mask]
    N_masked = samples.shape[0]

    # Skip if insufficient
    if N_masked <= n_params:
        print(f"Not enough samples to plot ({N_masked} ≤ {n_params}); skipping.")
        return

    # Percentile summary
    print("Live posterior summary (16/50/84 percentiles):")
    for j in range(n_params):
        p16, p50, p84 = np.percentile(samples[:, j], [16, 50, 84])
        print(f"  θ{j+1}: median={p50:.3f} (+{p84-p50:.3f}/-{p50-p16:.3f})")

    # Weighted mean & std
    wmean = np.average(samples, axis=0, weights=weights)
    wvar = np.average((samples - wmean)**2, axis=0, weights=weights)
    wstd = np.sqrt(wvar)
    print("Live posterior summary (weighted mean ±1σ):")
    for j in range(n_params):
        print(f"  θ{j+1}: mean={wmean[j]:.3f} ±{wstd[j]:.3f}")

    # Evidence
    stats = analyzer.get_stats()
    logZ = stats.get('global evidence') or stats.get('nested sampling global log-evidence')
    if logZ is not None:
        print(f"Global evidence (logZ) = {logZ:.3f}")

    # Build title
    title_base = "Live posterior"
    if logZ is not None:
        title_base += f", logZ={logZ:.2f}"

    # Prepare labels/truths
    labels = [f"θ{i+1}" for i in range(n_params)]
    truths_plot = truths[:n_params] if truths else None

    # Plot
    if interactive:
        plt.ion()
        fig = plt.gcf()
        fig.clf()
        corner.corner(
            samples,
            labels=labels,
            truths=truths_plot,
            fig=fig,
            **corner_kwargs
        )
        plt.suptitle(title_base)
        plt.draw(); plt.pause(0.01)
    else:
        fig = corner.corner(
            samples,
            labels=labels,
            truths=truths_plot,
            **corner_kwargs
        )
        plt.suptitle(f"{title_base} (snapshot)")
        plt.tight_layout()

    # Save if requested
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        print(f"Saved corner plot to {path}")


def compare_retrieval_corners(base_dir,
    prefixes,
    param_names=None,
    labels=None,
    colors=None,
    smooth=1.0,
    user_ranges=None,
    truths=None,
    output_file='',
    pad_fraction=0.08,
    pad_user_ranges=True,
    use_extreme_percentiles=False,
    extreme_percentiles=(0.01, 99.99),
    show_titles=False,
    title_fmt=".3f",
    figsize=None,
    diag_y_pad_fraction=0.20,
    legend_loc=(0.98, 0.96)
    ):
    """
    Overplot multiple MultiNest posteriors using corner, with robust axis ranges
    and robust diagonal histogram scaling so contours and histogram peaks are
    fully encapsulated.

    Parameters
    ----------
    base_dir : str
        Directory where the retrieval files are located.

    prefixes : list of str
        List of filename prefixes (without '_post_equal_weights.dat').

    param_names : list of str, optional
        Names of parameters (must match dimensionality).

    labels : list of str, optional
        Labels for legend (same length as prefixes).

    colors : list of str, optional
        Colors for each retrieval.

    smooth : float, optional
        Smoothing factor for corner.

    user_ranges : list, optional
        Manual ranges, e.g. [[xmin, xmax], [ymin, ymax], ...].

    truths : list, optional
        Truth values to overplot.

    output_file : str, optional
        Path to save the figure.

    pad_fraction : float, optional
        Fractional padding added to each parameter x/y range when automatically computed.

    use_extreme_percentiles : bool, optional
        If True, automatic ranges are based on extreme percentiles instead of exact min/max.

    extreme_percentiles : tuple, optional
        Percentiles used if use_extreme_percentiles=True.

    show_titles : bool, optional
        Whether to show titles on the diagonal.

    title_fmt : str, optional
        Title format for corner.

    figsize : tuple, optional
        Matplotlib figure size.

    diag_y_pad_fraction : float, optional
        Extra padding added to diagonal histogram y-limits.

    legend_loc : tuple, optional
        Figure-coordinate location for legend anchor.

    EXAMPLE of use:
    ----------------
    base = "/Users/alexsl/Documents/Simulador/ANDES/HD189733b/transit/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/"

    prefixes = [
        "retrieval_night_0",
        "retrieval_night_0_1Datm"
    ]

    labels = [
        "pseudo-2D",
        "1D"
    ]

    param_names = ["C/O", "[(C+O)/H]"]

    exosims.compare_retrieval_corners(
        base_dir=base,
        prefixes=prefixes,
        param_names=param_names,
        labels=labels,
        user_ranges=[
            [0.32, 0.85],
            [0.49, 1.65]
        ],
        truths=[0.41, 0.53],
        output_file="/Users/alexsl/Documents/Simulador/ANDES/c_to_o_comparisons.pdf"
    )

    EXAMPLE for nested folders:
    ---------------------------
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_hd18_allmols_fulltransit_YJHband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_hd18_allmols_fulltransit_Kband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_hd18_allmols_fulltransit_YJHKband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0"
    ]

    labels = [
        "YJH bands",
        "K-band",
        "YJHK-band"
    ]

    param_names = [
        "log(X$_{H_2O}$)",
        "log(X$_{CH_4}$)",
        "log(X$_{NH_3}$)",
        "log(X$_{CO}$)",
        "log(X$_{H_2S}$)",
        "log(X$_{HCN}$)",
        "T$_{equ}$",
        "V$_{rest}$"
    ]

    exosims.compare_retrieval_corners(
        base_dir=base,
        prefixes=prefixes,
        param_names=param_names,
        labels=labels,
        output_file="/Users/alexsl/Documents/Simulador/ANDES/retrieval_bands_comparison_hd18.pdf",
        pad_fraction=0.18,
        pad_user_ranges=True
    )
    
    
    
    
    
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_w76b_allmols_fulltransit_YJHband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_w76b_allmols_fulltransit_Kband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_w76b_allmols_fulltransit_YJHKband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0"
    ]

    labels = [
        "YJH bands",
        "K-band",
        "YJHK-band"
    ]

    param_names = [
        "log(X$_{H_2O}$)",
        "log(X$_{CH_4}$)",
        "log(X$_{NH_3}$)",
        "log(X$_{CO}$)",
        "log(X$_{H_2S}$)",
        "log(X$_{HCN}$)",
        "T$_{equ}$",
        "V$_{rest}$"
    ]

    exosims.compare_retrieval_corners(
        base_dir=base,
        prefixes=prefixes,
        param_names=param_names,
        labels=labels,
        output_file="/Users/alexsl/Documents/Simulador/ANDES/retrieval_bands_comparison_w76b.pdf",
        pad_fraction=0.18,
        pad_user_ranges=True
    )
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import corner

    if labels is None:
        labels = prefixes

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(labels) != len(prefixes):
        raise ValueError("labels must have the same length as prefixes")

    samples_list = []
    ndim_ref = None

    # -------------------------
    # LOAD ALL DATA
    # -------------------------
    for prefix in prefixes:
        filepath = os.path.join(base_dir, prefix + "_post_equal_weights.dat")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)
        
        if data.shape[1] < 2:
            raise ValueError(f"Unexpected shape in {filepath}: {data.shape}")
        
        samples = data[:, :-1]
        
        # Safety: if array is accidentally transposed, fix it
        if samples.shape[0] < samples.shape[1]:
            # probably got (ndim, nsamples) instead of (nsamples, ndim)
            samples = samples.T
        
        # Final sanity check
        if samples.shape[0] <= samples.shape[1]:
            raise ValueError(
                f"Posterior in {filepath} appears to have too few samples for corner: "
                f"samples shape = {samples.shape}"
            )
            
        print(f"Loaded {filepath}")
        print(f"  raw data shape     = {data.shape}")
        print(f"  posterior shape    = {samples.shape}")

        if ndim_ref is None:
            ndim_ref = samples.shape[1]
        else:
            if samples.shape[1] != ndim_ref:
                raise ValueError(
                    f"Dimension mismatch: {prefix} has {samples.shape[1]} params, "
                    f"expected {ndim_ref}"
                )

        samples_list.append(samples)

        print(f"\n📊 Results for {prefix}:")
        for i in range(samples.shape[1]):
            
            param_vals = samples[:, i]
            med = np.median(param_vals)
            lo = np.percentile(param_vals, 16)
            hi = np.percentile(param_vals, 84)
            sigma = 0.5 * (hi - lo)

            if sigma == 0:
                ii = 3
            else:
                ii = max(0, int(-np.floor(np.log10(abs(sigma)))) + 1)

            fmt = f"%.{ii}f"
            pname = param_names[i] if param_names is not None else f"param_{i}"
            #if pname == "V$_{rest}$": 
            #    med -= 3.9
            #    param_vals -= 3.9
            print(f"  {pname:15s} {fmt % med} ± {fmt % sigma}")

    # -------------------------
    # CHECK PARAM NAMES
    # -------------------------
    if param_names is not None and len(param_names) != ndim_ref:
        raise ValueError(
            f"param_names length ({len(param_names)}) != ndim ({ndim_ref})"
        )

    # -------------------------
    # COMPUTE ROBUST GLOBAL RANGES
    # -------------------------
    if user_ranges is None:
        all_samples = np.vstack(samples_list)
        ranges = []

        for i in range(all_samples.shape[1]):
            vals = all_samples[:, i]

            if use_extreme_percentiles:
                lo = np.percentile(vals, extreme_percentiles[0])
                hi = np.percentile(vals, extreme_percentiles[1])
            else:
                lo = np.min(vals)
                hi = np.max(vals)

            width = hi - lo
            if width <= 0:
                center = lo
                width = max(1e-6, abs(center) * 1e-3, 1e-3)
                lo = center - 0.5 * width
                hi = center + 0.5 * width
            else:
                pad = pad_fraction * width
                lo -= pad
                hi += pad

            ranges.append((lo, hi))
    else:
        if len(user_ranges) != ndim_ref:
            raise ValueError(
                f"user_ranges length ({len(user_ranges)}) != ndim ({ndim_ref})"
            )
    
        ranges = []
    
        for r in user_ranges:
            lo, hi = float(r[0]), float(r[1])
            width = hi - lo
    
            if width <= 0:
                center = lo
                width = max(1e-6, abs(center) * 1e-3, 1e-3)
                lo = center - 0.5 * width
                hi = center + 0.5 * width
    
            if pad_user_ranges:
                pad = pad_fraction * width
                lo -= pad
                hi += pad
    
            ranges.append((lo, hi))

    # -------------------------
    # CREATE FIGURE
    # -------------------------
    plt.close("all")
    fig = plt.figure(figsize=figsize) if figsize is not None else None

    # -------------------------
    # PLOT
    # -------------------------
    for i, samples in enumerate(samples_list):
        fig = corner.corner(
            samples,
            fig=fig,
            labels=param_names if i == 0 else None,
            color=colors[i % len(colors)],
            plot_datapoints=False,
            fill_contours=True,
            truths=truths,
            truth_color="k",
            truth_kwargs={"linewidth": 1.5},
            alpha=0.35,
            levels=(0.68, 0.95),
            smooth=smooth,
            smooth1d=smooth,
            range=ranges,
            show_titles=show_titles,
            title_fmt=title_fmt,
            label_kwargs={"fontsize": 16},
            title_kwargs={"fontsize": 12},
            contour_kwargs={"linewidths": 2},
            hist_kwargs={"linewidth": 2}
        )

    # -------------------------
    # FORCE LIMITS ON ALL PANELS
    # -------------------------
    axes = np.array(fig.axes).reshape((ndim_ref, ndim_ref))

    for yi in range(ndim_ref):
        for xi in range(ndim_ref):
            ax = axes[yi, xi]
            if yi == xi:
                ax.set_xlim(ranges[xi])
            elif yi > xi:
                ax.set_xlim(ranges[xi])
                ax.set_ylim(ranges[yi])

    # -------------------------
    # FIX DIAGONAL HISTOGRAM CLIPPING
    # -------------------------
    for i in range(ndim_ref):
        ax = axes[i, i]
        ymax = 0.0

        for line in ax.lines:
            y = np.asarray(line.get_ydata())
            if y.size > 0 and np.all(np.isfinite(y)):
                ymax = max(ymax, np.nanmax(y))

        for patch in ax.patches:
            try:
                verts = patch.get_path().vertices
                if verts.size > 0:
                    ymax = max(ymax, np.nanmax(verts[:, 1]))
            except Exception:
                pass

        for coll in ax.collections:
            try:
                for path in coll.get_paths():
                    verts = path.vertices
                    if verts.size > 0:
                        ymax = max(ymax, np.nanmax(verts[:, 1]))
            except Exception:
                pass

        if ymax > 0:
            ax.set_ylim(0, ymax * (1.0 + diag_y_pad_fraction))

    # -------------------------
    # LEGEND IN TOP-RIGHT WHITE SPACE
    # -------------------------
    handles = [
        mlines.Line2D([], [], color=colors[i % len(colors)], lw=2, label=labels[i])
        for i in range(len(prefixes))
    ]

    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=legend_loc,
        fontsize=13,
        frameon=True
    )

    fig.subplots_adjust(left=0.07, bottom=0.07, right=0.84, top=0.96)

    if output_file:
        fig.savefig(output_file, bbox_inches="tight", dpi=300)

    plt.show()

    return fig, ranges


def compare_multinest_evidence(base_dir,
    prefixes,
    labels=None,
    prefix_suffixes_to_try=None,
    verbose=True
    ):
    """
    Compare Bayesian evidences (logZ) between two MultiNest retrievals.

    This function follows the same path logic as compare_retrieval_corners:
    each retrieval is specified through a common `base_dir` plus a retrieval
    `prefix`, where the retrieval files are expected to live at

        os.path.join(base_dir, prefix + <suffix>)

    Parameters
    ----------
    base_dir : str
        Common base directory containing the retrieval folders/files.

    prefixes : list of str, length 2
        Two retrieval prefixes, exactly as used in compare_retrieval_corners.
        These may include nested folders, e.g.
        "andes_w76b_allmols_fulltransit_Kband/matrices/.../retrieval_night_0"

    labels : list of str, optional
        Human-readable labels for the two models. If None, uses prefixes.

    prefix_suffixes_to_try : list of str, optional
        Candidate suffixes to append to each prefix when searching for the
        evidence file. If None, a robust default list is used.

    verbose : bool, optional
        If True, print extensive diagnostic output.

    Returns
    -------
    result : dict
        Dictionary containing:
            - logZ_1, err_1
            - logZ_2, err_2
            - delta_logZ
            - delta_logZ_err
            - preferred_model
            - strength
            - file_1, file_2

    Notes
    -----
    Interpretation is based on ΔlogZ = logZ(model 1) - logZ(model 2).

    Jeffreys-like scale used here:
        |ΔlogZ| < 1     : inconclusive
        1 to 2.5           : weak evidence
        2.5 to 5           : moderate evidence
        >5              : strong evidence

    Important:
    A higher logZ means the model is statistically preferred after accounting
    for the Occam penalty, but statistical preference does not automatically
    imply physical correctness.

    EXAMPLE of use
    --------------
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_hd18_allmols_fulltransit_Kband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_hd18_allmols_fulltransit_YJHKband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0"
   ]

    labels = [
        "K band",
        "YJHK bands"
    ]

    exosims.compare_multinest_evidence(
        base_dir=base,
        prefixes=prefixes,
        labels=labels
    )
    """
    import os
    import re
    import numpy as np

    if len(prefixes) != 2:
        raise ValueError("prefixes must contain exactly two retrieval prefixes")

    if labels is None:
        labels = prefixes

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries")

    # Robust list of common MultiNest / PyMultiNest stats filenames
    if prefix_suffixes_to_try is None:
        prefix_suffixes_to_try = [
            "_stats.dat",
            "stats.dat",
            "_summary.txt",
            "summary.txt",
            "_.txt",
            ".txt",
            "_post_separate.dat",
        ]

    def extract_logZ_from_text(text):
        """
        Try several regex patterns to locate the global evidence and its error.
        Returns (logZ, logZ_err) or (None, None) if not found.
        """
        patterns = [
            r"Nested Sampling Global Log-Evidence\s*[:=]?\s*([-\d.Ee+]+)\s*\+/-\s*([-\d.Ee+]+)",
            r"Global Evidence\s*[:=]?\s*([-\d.Ee+]+)\s*\+/-\s*([-\d.Ee+]+)",
            r"log\(Z\)\s*[:=]?\s*([-\d.Ee+]+)\s*\+/-\s*([-\d.Ee+]+)",
            r"ln\(Z\)\s*[:=]?\s*([-\d.Ee+]+)\s*\+/-\s*([-\d.Ee+]+)",
            r"Evidence\s*[:=]?\s*([-\d.Ee+]+)\s*\+/-\s*([-\d.Ee+]+)",
        ]

        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return float(m.group(1)), float(m.group(2))

        return None, None

    def read_logZ(base_dir, prefix):
        """
        Search for an evidence file corresponding to one retrieval prefix.
        """
        tried_files = []

        # First try the usual prefix-based filenames
        for suffix in prefix_suffixes_to_try:
            filepath = os.path.join(base_dir, prefix + suffix)
            tried_files.append(filepath)

            if os.path.exists(filepath):
                with open(filepath, "r", errors="ignore") as f:
                    text = f.read()

                logZ, err = extract_logZ_from_text(text)
                if logZ is not None:
                    return logZ, err, filepath, tried_files

        # If not found, also inspect the directory containing the prefix
        retrieval_dir = os.path.dirname(os.path.join(base_dir, prefix))
        retrieval_stub = os.path.basename(prefix)

        if os.path.isdir(retrieval_dir):
            for fname in sorted(os.listdir(retrieval_dir)):
                filepath = os.path.join(retrieval_dir, fname)

                # Only inspect text-like / stats-like files
                if not os.path.isfile(filepath):
                    continue

                lower = fname.lower()
                if not any(key in lower for key in ["stat", "summary", ".txt", "evidence"]):
                    continue

                # Prefer files related to the retrieval stub, but allow fallback
                if retrieval_stub not in fname and not any(key in lower for key in ["stat", "summary", "evidence"]):
                    continue

                tried_files.append(filepath)

                try:
                    with open(filepath, "r", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue

                logZ, err = extract_logZ_from_text(text)
                if logZ is not None:
                    return logZ, err, filepath, tried_files

        raise FileNotFoundError(
            f"No logZ found for prefix:\n  {prefix}\n"
            f"inside base_dir:\n  {base_dir}\n\n"
            f"Tried files:\n  " + "\n  ".join(tried_files)
        )

    # -------------------------
    # READ BOTH EVIDENCES
    # -------------------------
    logZ1, err1, file1, tried1 = read_logZ(base_dir, prefixes[0])
    logZ2, err2, file2, tried2 = read_logZ(base_dir, prefixes[1])

    # -------------------------
    # COMPARE
    # -------------------------
    dlogZ = logZ1 - logZ2
    dlogZ_err = np.sqrt(err1**2 + err2**2)
    abs_dlogZ = abs(dlogZ)

    if abs_dlogZ < 1.0:
        strength = "inconclusive"
    elif abs_dlogZ < 2.5:
        strength = "weak evidence"
    elif abs_dlogZ < 5.0:
        strength = "moderate evidence"
    else:
        strength = "strong evidence"

    preferred_model = labels[0] if dlogZ > 0 else labels[1]

    # -------------------------
    # PRINT
    # -------------------------
    if verbose:
        print("\n==============================")
        print("📊 BAYESIAN EVIDENCE COMPARISON")
        print("==============================\n")

        print(f"Base directory:")
        print(f"  {base_dir}\n")

        print(f"Model 1: {labels[0]}")
        print(f"  prefix  = {prefixes[0]}")
        print(f"  logZ    = {logZ1:.6f} ± {err1:.6f}")
        print(f"  source  = {file1}\n")

        print(f"Model 2: {labels[1]}")
        print(f"  prefix  = {prefixes[1]}")
        print(f"  logZ    = {logZ2:.6f} ± {err2:.6f}")
        print(f"  source  = {file2}\n")

        print("--------------------------------------------------")
        print(f"ΔlogZ = logZ({labels[0]}) - logZ({labels[1]})")
        print(f"      = {dlogZ:.6f} ± {dlogZ_err:.6f}")
        print("--------------------------------------------------\n")

        print("Jeffreys-style assessment:")
        print(f"  |ΔlogZ| = {abs_dlogZ:.6f}")
        print(f"  Strength = {strength}")
        print(f"  Preferred model = {preferred_model}\n")

        print("How to read this:")
        print("  - ΔlogZ > 0  => Model 1 is preferred")
        print("  - ΔlogZ < 0  => Model 2 is preferred")
        print("  - |ΔlogZ| < 1   : inconclusive")
        print("  - 1 to 2.5      : weak evidence")
        print("  - 2.5 to 5      : moderate evidence")
        print("  - > 5           : strong evidence\n")

        print("Important caveats:")
        print("  - logZ already includes an Occam penalty for extra parameters.")
        print("  - Statistical preference does NOT automatically imply physical realism.")
        print("  - In pseudo-2D vs 1D problems, extra species may improve fit quality")
        print("    by compensating for model inadequacy rather than representing")
        print("    genuine atmospheric detections.\n")

        # A more practical statement
        if abs_dlogZ < 1.0:
            print("Practical conclusion:")
            print("  The two models are statistically indistinguishable from the evidence alone.")
        else:
            print("Practical conclusion:")
            print(f"  The evidence favors '{preferred_model}' with {strength}.")

        print("\n==============================\n")

    return {
        "logZ_1": logZ1,
        "err_1": err1,
        "file_1": file1,
        "logZ_2": logZ2,
        "err_2": err2,
        "file_2": file2,
        "delta_logZ": dlogZ,
        "delta_logZ_err": dlogZ_err,
        "strength": strength,
        "preferred_model": preferred_model,
    }


def compare_CtoO_corners_from_multispecies(base_dir,
    prefixes,
    param_names,
    labels=None,
    colors=None,
    solar_CO_sum=8.5e-4,
    truths=None,
    smooth=1.0,
    user_ranges=None,
    pad_fraction=0.08,
    diag_y_pad_fraction=0.2,
    legend_loc=(0.98, 0.98),
    output_file=""
    ):
    """
    Overplot derived C/O and [(C+O)/H] corner plots from multiple
    MultiNest molecular retrievals (posterior algebra).

    This performs posterior algebra:
        C = CO + CH4 + HCN
        O = CO + H2O

    and plots:
        C/O
        log10((C+O)/solar)

    Parameters
    ----------
    base_dir : str
        Base directory.

    prefixes : list of str
        Retrieval prefixes (relative to base_dir).

    param_names : list
        Must include:
            log(X_H2O), log(X_CH4), log(X_CO), log(X_HCN)

    labels : list, optional
        Legend labels.

    colors : list, optional
        Plot colors.

    truths : list, optional
        [C/O, [(C+O)/H]]

    output_file : str, optional
        Save path.

    Returns
    -------
    fig, derived_list
    
    CAN BE USED WITH ONLY ONE RETRIEVAL. CAN ALSO COMPARE IFG YOU PASS SEVERAL FILES
    
    EXAMPLE OF USE:
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_hd18_allmols_fulltransit_YJHband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_hd18_allmols_fulltransit_Kband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_hd18_allmols_fulltransit_YJHKband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0" ]

    labels = [
        "YJH bands",
        "K-band",
        "YJHK bands"
    ]

    param_names = ["log(X$_{H_2O}$)", "log(X$_{CH_4}$)", "log(X$_{NH_3}$)", 
                  "log(X$_{CO}$)", "log(X$_{CO_2}$)", "log(X$_{HCN}$)", 
                  "T$_{equ}$", "V$_{rest}$"]

    exosims.compare_CtoO_corners_from_multispecies(
        base_dir=base,
        prefixes=prefixes,
        param_names=param_names,
        labels=labels,
        truths=[0.41, 0.53],
        output_file="/Users/alexsl/Documents/Simulador/ANDES/CtoO_band_comparison_w76b.pdf"
    )
    
    
    
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_w76b_allmols_fulltransit_YJHband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_w76b_allmols_fulltransit_Kband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_w76b_allmols_fulltransit_YJHKband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0" ]

    labels = [
        "YJH bands",
        "K-band",
        "YJHK bands"
    ]

    param_names = ["log(X$_{H_2O}$)", "log(X$_{CH_4}$)", "log(X$_{NH_3}$)", 
                  "log(X$_{CO}$)", "log(X$_{CO_2}$)", "log(X$_{HCN}$)", 
                  "T$_{equ}$", "V$_{rest}$"]

    exosims.compare_CtoO_corners_from_multispecies(
        base_dir=base,
        prefixes=prefixes,
        param_names=param_names,
        labels=labels,
        truths=[0.8, -1.47], #[0.41, 0.53],
        output_file="/Users/alexsl/Documents/Simulador/ANDES/CtoO_band_comparison_w76b.pdf"
    )
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import corner

    if labels is None:
        labels = prefixes

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # -------------------------
    # CHECK REQUIRED PARAMS
    # -------------------------
    required = [
        "log(X$_{H_2O}$)",
        "log(X$_{CH_4}$)",
        "log(X$_{CO}$)",
        "log(X$_{HCN}$)"
    ]
    for req in required:
        if req not in param_names:
            raise ValueError(f"Missing required parameter: {req}")

    idx = {name: i for i, name in enumerate(param_names)}

    derived_list = []

    # -------------------------
    # CONSTANTS
    # -------------------------
    # Molecular weights [amu].
    mu = {
        "H2": 2.01588,
        "He": 4.002602,
        "H2O": 18.01528,
        "CH4": 16.04246,
        "NH3": 17.03052,
        "CO": 28.01010,
        "CO2": 44.00950,
        "H2S": 34.08088,
        "HCN": 27.02534,
    }

    # Atomic content of each molecule.
    atoms = {
        "H2O": {"H": 2, "C": 0, "O": 1},
        "CH4": {"H": 4, "C": 1, "O": 0},
        "NH3": {"H": 3, "C": 0, "O": 0},
        "CO":  {"H": 0, "C": 1, "O": 1},
        "CO2": {"H": 0, "C": 1, "O": 2},
        "H2S": {"H": 2, "C": 0, "O": 0},
        "HCN": {"H": 1, "C": 1, "O": 0},
    }

    # Mapping between internal species names and your retrieval labels.
    pname = {
        "H2O": "log(X$_{H_2O}$)",
        "CH4": "log(X$_{CH_4}$)",
        "NH3": "log(X$_{NH_3}$)",
        "CO":  "log(X$_{CO}$)",
        "CO2": "log(X$_{CO_2}$)",
        "H2S": "log(X$_{H_2S}$)",
        "HCN": "log(X$_{HCN}$)",
    }

    # Fixed background assumed by the retrieval.
    # Change these values if the retrieval was run with different fixed
    # H2/He mass fractions.
    w_H2_fixed = 0.75
    w_He_fixed = 0.24

    # Hydrogen number abundance from the fixed H2 background.
    # The He value does not enter C/O or C+O/H directly, but is kept here
    # explicitly to document the retrieval convention.
    n_H_from_H2_fixed = 2.0 * w_H2_fixed / mu["H2"]

    # Retrieved species available in this run.
    available_species = [
        sp for sp, par in pname.items()
        if par in param_names
    ]

    # -------------------------
    # SMALL HELPERS
    # -------------------------
    def stats(x):
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        return q50, 0.5 * (q84 - q16)

    # -------------------------
    # LOAD + COMPUTE
    # -------------------------
    for prefix in prefixes:
    
        filepath = os.path.join(base_dir, prefix + "_post_equal_weights.dat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
    
        data = np.loadtxt(filepath)
        samples = data[:, :-1]

        n = {}

        for sp in available_species:
            par = pname[sp]

            # Retrieved quantities entered pRT as mass fractions.
            w_sp = 10.0 ** samples[:, idx[par]]

            # Convert mass fractions to number-abundance proxies.
            n[sp] = w_sp / mu[sp]

        # Atomic budgets from retrieved molecular carriers.
        # C/O includes all retrieved C- and O-bearing species.
        C = np.zeros(samples.shape[0])
        O = np.zeros(samples.shape[0])
        H_from_molecules = np.zeros(samples.shape[0])

        for sp in available_species:
            C += atoms[sp]["C"] * n[sp]
            O += atoms[sp]["O"] * n[sp]
            H_from_molecules += atoms[sp]["H"] * n[sp]

        CO_ratio = C / O

        # Retrieval-consistent C+O/H proxy.
        # H is the number of hydrogen nuclei, not the number of H2 molecules.
        # We include the fixed H2 background plus the hydrogen nuclei contained
        # in the retrieved molecules. H2S and NH3 affect this denominator if
        # they are retrieved, even though they do not enter the C+O numerator.
        n_H = n_H_from_H2_fixed + H_from_molecules

        COH = (C + O) / n_H
        metal_log = np.log10(COH / solar_CO_sum)
    
        derived = np.column_stack([CO_ratio, metal_log])
        derived_list.append(derived)

        co_med, co_err = stats(CO_ratio)
        met_med, met_err = stats(metal_log)

        print(f"\n📊 Derived results for {prefix}:")
        print(f"  Included species: {', '.join(available_species)}")
        print(f"  C/O             {co_med:.5f} ± {co_err:.5f}")
        print(f"  [(C+O)/H]_carriers   {met_med:.4f} ± {met_err:.4f}")

    # -------------------------
    # GLOBAL RANGES
    # -------------------------
    if user_ranges is None:
        all_samples = np.vstack(derived_list)

        ranges = []
        for i in range(2):
            vals = all_samples[:, i]
            lo, hi = np.min(vals), np.max(vals)

            width = hi - lo
            pad = pad_fraction * width

            ranges.append((lo - pad, hi + pad))
    else:
        ranges = user_ranges

    # -------------------------
    # PLOT
    # -------------------------
    plt.close("all")
    fig = None

    for i, samples_i in enumerate(derived_list):

        fig = corner.corner(
            samples_i,
            fig=fig,
            labels=[
                "C/O",
                r"$[(\mathrm{C+O})/\mathrm{H}]_{\rm mol}$"
            ] if i == 0 else None,
            color=colors[i % len(colors)],
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            smooth=smooth,
            smooth1d=smooth,
            range=ranges,
            truths=truths,
            truth_color="k",
            truth_kwargs={"linewidth": 1.5},
            alpha=0.4,
            contour_kwargs={"linewidths": 2},
            hist_kwargs={"linewidth": 2}
        )

    # -------------------------
    # FIX DIAGONAL CLIPPING
    # -------------------------
    axes = np.array(fig.axes).reshape((2, 2))
    
    for i in range(2):
        ax = axes[i, i]
    
        ymax = 0.0
    
        # Lines: normal corner 1D histograms and smoothed curves
        for line in ax.lines:
            ydata = line.get_ydata()
            if len(ydata) > 0:
                ydata = np.asarray(ydata, dtype=float)
                ydata = ydata[np.isfinite(ydata)]
                if len(ydata) > 0:
                    ymax = max(ymax, np.nanmax(ydata))
    
        # Patches: sometimes histograms are stored here
        for patch in ax.patches:
            try:
                verts = patch.get_path().vertices
                if verts.size > 0:
                    ydata = verts[:, 1]
                    ydata = ydata[np.isfinite(ydata)]
                    if len(ydata) > 0:
                        ymax = max(ymax, np.nanmax(ydata))
            except Exception:
                pass
    
        # Collections: filled/smoothed diagonal posteriors can end up here
        for coll in ax.collections:
            try:
                for path in coll.get_paths():
                    verts = path.vertices
                    if verts.size > 0:
                        ydata = verts[:, 1]
                        ydata = ydata[np.isfinite(ydata)]
                        if len(ydata) > 0:
                            ymax = max(ymax, np.nanmax(ydata))
            except Exception:
                pass
    
        if ymax > 0:
            ax.set_ylim(0, ymax * (1 + diag_y_pad_fraction))
    
        # Keep the requested x-range fixed
        ax.set_xlim(ranges[i])
        
    # -------------------------
    # LEGEND OUTSIDE
    # -------------------------
    handles = [
        mlines.Line2D([], [], color=colors[i], lw=2, label=labels[i])
        for i in range(len(prefixes))
    ]

    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=legend_loc,
        fontsize=13,
        frameon=True
    )

    fig.subplots_adjust(right=0.82)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, derived_list


def compare_CtoO_corners_from_multispecies_flexible(base_dir,
    prefixes,
    param_names_list,
    labels=None,
    colors=None,
    solar_CO_sum=8.5e-4,
    truths=None,
    smooth=1.0,
    user_ranges=None,
    pad_fraction=0.08,
    diag_y_pad_fraction=0.25,
    legend_loc=(0.98, 0.98),
    output_file=""
    ):
    """
    Overplot derived C/O and [(C+O)/H] corner plots from one or more retrievals.

    This flexible version works even if some molecules are not included
    in a given retrieval. Missing species are set to zero.

    Posterior algebra:
        C = CO + CH4 + HCN + CO2
        O = CO + H2O + 2*CO2

    If CH4, HCN, or CO2 are absent from the retrieval, their contribution
    is set to zero.

    Parameters
    ----------
    base_dir : str
        Base directory.

    prefixes : list of str
        Retrieval prefixes relative to base_dir, without "_post_equal_weights.dat".

    param_names_list : list
        Either:
            - one list of parameter names, used for all retrievals, or
            - a list of lists, one parameter-name list per retrieval.

    labels : list, optional
        Legend labels.

    colors : list, optional
        Plot colors.

    solar_CO_sum : float
        Solar normalization for C+O. Default: 8.5e-4.

    truths : list, optional
        [C/O, [(C+O)/H]]

    Returns
    -------
    fig, derived_list
    
    Example of use:
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_w76b_allmols_fulltransit_Kband_noCH4/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0"
    ]
    
    param_names_noCH4 = [
        "log(X$_{H_2O}$)",
        "log(X$_{NH_3}$)",
        "log(X$_{CO}$)",
        "log(X$_{H_2S}$)",
        "log(X$_{HCN}$)",
        "T$_{equ}$",
        "V$_{rest}$"
    ]
    
    exosims.compare_CtoO_corners_from_multispecies_flexible(
        base_dir=base,
        prefixes=prefixes,
        param_names_list=param_names_noCH4,
        labels=["K band, no CH4"],
        truths=[0.8, -1.47]
    )
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import corner

    if labels is None:
        labels = prefixes

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(labels) != len(prefixes):
        raise ValueError("labels must have the same length as prefixes")

    # Allow one param_names list for all retrievals
    if len(param_names_list) > 0 and isinstance(param_names_list[0], str):
        param_names_list = [param_names_list for _ in prefixes]

    if len(param_names_list) != len(prefixes):
        raise ValueError("param_names_list must be one list, or one list per prefix")

    derived_list = []

    def get_X(samples, idx, name):
        if name in idx:
            return 10.0**samples[:, idx[name]]
        return np.zeros(samples.shape[0])

    def stats(x):
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        return q50, 0.5 * (q84 - q16)

    for prefix, param_names in zip(prefixes, param_names_list):

        filepath = os.path.join(base_dir, prefix + "_post_equal_weights.dat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)

        samples = data[:, :-1]

        if samples.shape[0] < samples.shape[1]:
            samples = samples.T

        idx = {name: i for i, name in enumerate(param_names)}

        X_H2O = get_X(samples, idx, "log(X$_{H_2O}$)")
        X_CH4 = get_X(samples, idx, "log(X$_{CH_4}$)")
        X_CO  = get_X(samples, idx, "log(X$_{CO}$)")
        X_HCN = get_X(samples, idx, "log(X$_{HCN}$)")
        X_CO2 = get_X(samples, idx, "log(X$_{CO_2}$)")

        C = X_CO + X_CH4 + X_HCN + X_CO2
        O = X_CO + X_H2O + 2.0 * X_CO2

        CO_ratio = C / O
        metal_log = np.log10((C + O) / solar_CO_sum)

        derived = np.column_stack([CO_ratio, metal_log])
        derived_list.append(derived)

        co_med, co_err = stats(CO_ratio)
        met_med, met_err = stats(metal_log)

        print(f"\n📊 Derived results for {prefix}:")
        print("  Included carbon/oxygen carriers:")
        print(f"    H2O: {'yes' if 'log(X$_{H_2O}$)' in idx else 'no'}")
        print(f"    CO : {'yes' if 'log(X$_{CO}$)' in idx else 'no'}")
        print(f"    CH4: {'yes' if 'log(X$_{CH_4}$)' in idx else 'no'}")
        print(f"    HCN: {'yes' if 'log(X$_{HCN}$)' in idx else 'no'}")
        print(f"    CO2: {'yes' if 'log(X$_{CO_2}$)' in idx else 'no'}")
        print(f"  C/O             {co_med:.5f} ± {co_err:.5f}")
        print(f"  [(C+O)/H]       {met_med:.4f} ± {met_err:.4f}")

    if user_ranges is None:
        all_derived = np.vstack(derived_list)
        ranges = []

        for i in range(2):
            vals = all_derived[:, i]
            lo, hi = np.min(vals), np.max(vals)
            width = hi - lo

            if width <= 0:
                width = max(1e-6, abs(lo) * 1e-3)
                lo -= 0.5 * width
                hi += 0.5 * width
            else:
                pad = pad_fraction * width
                lo -= pad
                hi += pad

            ranges.append((lo, hi))
    else:
        ranges = user_ranges

    plt.close("all")
    fig = None

    for i, samples in enumerate(derived_list):
        fig = corner.corner(
            samples,
            fig=fig,
            labels=["C/O", "[(C+O)/H]"] if i == 0 else None,
            color=colors[i % len(colors)],
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            smooth=smooth,
            smooth1d=smooth,
            range=ranges,
            truths=truths,
            truth_color="k",
            truth_kwargs={"linewidth": 1.5},
            alpha=0.4,
            contour_kwargs={"linewidths": 2},
            hist_kwargs={"linewidth": 2}
        )

    axes = np.array(fig.axes).reshape((2, 2))

    for i in range(2):
        ax = axes[i, i]
        ymax = 0.0

        for line in ax.lines:
            y = np.asarray(line.get_ydata())
            if y.size > 0 and np.all(np.isfinite(y)):
                ymax = max(ymax, np.nanmax(y))

        if ymax > 0:
            ax.set_ylim(0, ymax * (1.0 + diag_y_pad_fraction))

    handles = [
        mlines.Line2D([], [], color=colors[i % len(colors)], lw=2, label=labels[i])
        for i in range(len(prefixes))
    ]

    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=legend_loc,
        fontsize=13,
        frameon=True
    )

    fig.subplots_adjust(right=0.82)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, derived_list


def compare_CtoO_corners_from_CO_H2O(base_dir,
    prefixes,
    param_names,
    labels=None,
    colors=None,
    solar_CO_sum=8.5e-4,
    truths=None,
    smooth=1.0,
    user_ranges=None,
    pad_fraction=0.08,
    diag_y_pad_fraction=0.25,
    legend_loc=(0.98, 0.98),
    output_file=""
):
    """
    Overplot derived C/O and [(C+O)/H] corner plots from multiple
    MultiNest retrievals using only CO and H2O.

    Posterior algebra used:
        C = CO
        O = CO + H2O

    Therefore:
        C/O = CO / (CO + H2O)
        log10((C+O)/solar) = log10((2*CO + H2O) / solar_CO_sum)

    Parameters
    ----------
    base_dir : str
        Base directory.

    prefixes : list of str
        Retrieval prefixes (relative to base_dir).

    param_names : list
        Must include:
            log(X_H2O), log(X_CO)
        using your exact parameter naming convention, e.g.:
            "log(X$_{H_2O}$)", "log(X$_{CO}$)"

    labels : list, optional
        Legend labels.

    colors : list, optional
        Plot colors.

    truths : list, optional
        [C/O, [(C+O)/H]]

    output_file : str, optional
        Save path.

    Returns
    -------
    fig, derived_list
    
    Example of use:
    
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/"

    prefixes = [
        "andes_w76b_fulltransit_Kband_onlyH2OCO/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0",
        "andes_w76b_fulltransit_Kband_onlyH2OCO/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/retrieval_night_0"
    ]

    labels = [
        "K-band",
        "K-band"
    ]

    param_names = [
        "log(X$_{H_2O}$)",
        "log(X$_{CO}$)",
        "T$_{equ}$",
        "V$_{rest}$"
    ]

    exosims.compare_CtoO_corners_from_CO_H2O(
        base_dir=base,
        prefixes=prefixes,
        param_names=param_names,
        labels=labels,
        truths=[0.8, -1.47],
        output_file="CtoO_CO_H2O_only.pdf"
    )
    
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import corner

    if labels is None:
        labels = prefixes

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # -------------------------
    # CHECK REQUIRED PARAMS
    # -------------------------
    required = [
        "log(X$_{H_2O}$)",
        "log(X$_{CO}$)"
    ]
    for req in required:
        if req not in param_names:
            raise ValueError(f"Missing required parameter: {req}")

    idx = {name: i for i, name in enumerate(param_names)}

    derived_list = []

    # -------------------------
    # LOAD + COMPUTE
    # -------------------------
    for prefix in prefixes:

        filepath = os.path.join(base_dir, prefix + "_post_equal_weights.dat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        data = np.loadtxt(filepath)
        samples = data[:, :-1]

        # abundances
        X_H2O = 10**samples[:, idx["log(X$_{H_2O}$)"]]
        X_CO  = 10**samples[:, idx["log(X$_{CO}$)"]]

        # derived elemental abundances
        C = X_CO
        O = X_CO + X_H2O

        CO_ratio = C / O
        metal_log = np.log10((C + O) / solar_CO_sum)

        derived = np.column_stack([CO_ratio, metal_log])
        derived_list.append(derived)

        # stats
        def stats(x):
            q16, q50, q84 = np.percentile(x, [16, 50, 84])
            return q50, 0.5 * (q84 - q16)

        co_med, co_err = stats(CO_ratio)
        met_med, met_err = stats(metal_log)

        print(f"\n📊 Derived results for {prefix}:")
        print(f"  C/O             {co_med:.5f} ± {co_err:.5f}")
        print(f"  [(C+O)/H]       {met_med:.4f} ± {met_err:.4f}")

    # -------------------------
    # GLOBAL RANGES
    # -------------------------
    if user_ranges is None:
        all_samples = np.vstack(derived_list)

        ranges = []
        for i in range(2):
            vals = all_samples[:, i]
            lo, hi = np.min(vals), np.max(vals)

            width = hi - lo
            pad = pad_fraction * width if width > 0 else pad_fraction * abs(lo if lo != 0 else 1.0)

            ranges.append((lo - pad, hi + pad))
    else:
        ranges = user_ranges

    # -------------------------
    # PLOT
    # -------------------------
    plt.close("all")
    fig = None

    for i, samples in enumerate(derived_list):

        fig = corner.corner(
            samples,
            fig=fig,
            labels=["C/O", "[(C+O)/H]"] if i == 0 else None,
            color=colors[i % len(colors)],
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            smooth=smooth,
            smooth1d=smooth,
            range=ranges,
            truths=truths,
            truth_color="k",
            truth_kwargs={"linewidth": 1.5},
            alpha=0.4,
            contour_kwargs={"linewidths": 2},
            hist_kwargs={"linewidth": 2}
        )

    # -------------------------
    # FIX DIAGONAL CLIPPING
    # -------------------------
    axes = np.array(fig.axes).reshape((2, 2))

    for i in range(2):
        ax = axes[i, i]

        ymax = 0
        for line in ax.lines:
            ydata = line.get_ydata()
            if len(ydata) > 0:
                ymax = max(ymax, np.max(ydata))

        if ymax > 0:
            ax.set_ylim(0, ymax * (1 + diag_y_pad_fraction))

    # -------------------------
    # LEGEND OUTSIDE
    # -------------------------
    handles = [
        mlines.Line2D([], [], color=colors[i % len(colors)], lw=2, label=labels[i])
        for i in range(len(prefixes))
    ]

    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=legend_loc,
        fontsize=13,
        frameon=True
    )

    fig.subplots_adjust(right=0.82)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, derived_list


def compute_CtoO_text(base_dir,
        prefix="retrieval_night_0",
        param_names=None,
        solar_CO_sum=8.5e-4
        ):
    """
    Full diagnostic of MultiNest retrieval:
    - Prints all parameter constraints
    - Classifies detections vs upper limits
    - Computes C/O and [(C+O)/H]
    
    USAGE:
        
    base = "/Users/alexsl/Documents/Simulador/ANDES/standards_tevol/andes_hd18_allmols_fulltransit_YJHband/matrices/matrices_BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1"

    param_names = [
        "log(X$_{H_2O}$)",
        "log(X$_{CH_4}$)",
        "log(X$_{NH_3}$)",
        "log(X$_{CO}$)",
        "log(X$_{H2S}$)",
        "log(X$_{HCN}$)",
        "T$_{equ}$",
        "V$_{rest}$"
    ]

    exosims.compute_CtoO(
        base_dir=base,
        prefix="retrieval_night_0",
        param_names=param_names
    )
    """

    # --- Load posterior samples ---
    data = np.loadtxt(f"{base_dir}/{prefix}_post_equal_weights.dat")
    samples = data[:, :-1]

    # --- Helper for stats ---
    def stats(x):
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        return q50, q16, q84, 0.5 * (q84 - q16)

    print("\n📊 Retrieved parameters:\n")

    results = {}

    # --- Loop over parameters ---
    for i, pname in enumerate(param_names):
        vals = samples[:, i]
        med, lo, hi, err = stats(vals)

        # --- Detection classification ---
        width = hi - lo

        if "log(X" in pname:
            # convert to linear
            lin_vals = 10**vals
            lin_med = 10**med

            # dynamic range check
            spread = np.percentile(lin_vals, 84) / np.percentile(lin_vals, 16)

            if spread < 3:
                status = "✅ well constrained"
            elif spread < 10:
                status = "⚠️ weak constraint"
            else:
                status = "⬇️ upper limit / unconstrained"

            print(f"{pname:20s} {med:.2f} ± {err:.2f}   ({status})")
            print(f"{'':20s} linear ~ {lin_med:.2e}")

        else:
            # non-log parameters
            if width < 0.2 * abs(med) if med != 0 else width < 1:
                status = "✅ well constrained"
            else:
                status = "⚠️ weak constraint"

            print(f"{pname:20s} {med:.2f} ± {err:.2f}   ({status})")

        results[pname] = (med, err)

    # --- Map indices ---
    idx = {name: i for i, name in enumerate(param_names)}

    # --- Extract abundances ---
    X_H2O = 10**samples[:, idx["log(X$_{H_2O}$)"]]
    X_CH4 = 10**samples[:, idx["log(X$_{CH_4}$)"]]
    X_CO  = 10**samples[:, idx["log(X$_{CO}$)"]]
    X_HCN = 10**samples[:, idx["log(X$_{HCN}$)"]]

    # --- Compute elemental budgets ---
    C = X_CO + X_CH4 + X_HCN
    O = X_CO + X_H2O

    CO_ratio = C / O
    metal = C + O
    metal_log = np.log10(metal / solar_CO_sum)

    # --- Stats ---
    CO_med, _, _, CO_err = stats(CO_ratio)
    met_med, _, _, met_err = stats(metal_log)

    print("\n📊 Derived elemental ratios:\n")
    print(f"C/O             {CO_med:.5f} ± {CO_err:.5f}")
    print(f"[(C+O)/H]       {met_med:.4f} ± {met_err:.4f}")

    return {
        "samples": samples,
        "C/O": CO_ratio,
        "metallicity": metal_log,
        "param_stats": results
    }


