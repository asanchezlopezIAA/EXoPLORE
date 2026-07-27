"""
exoplore.analysis.ccf_time
============================

Time-resolved 1D CCF analysis for high-resolution transit/eclipse spectroscopy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Dict, Tuple, Union, Set

from exoplore.atmosphere.prt import convolve
from exoplore.ccf.compute import get_max_CCF_peak
from exoplore.instruments.wavegrid import make_log_wave_grid

def plot_time_resolved_1D_CCFs(inp_dat,
    ccf_values_shift,
    v_rest,
    kp_range,
    max_kp_idx=None,
    ingress_n=None,
    egress_n=None,
    xlim=(-100, 100),
    output_file=None,
    show_plot=True,
    save_plot=True,
    sysrem_opt=False,
    colors=None,
    plot_halves=True,
    phase=None,
    with_signal=None,
    v_planet=None,
    sort_by_phase=False,
    diagnostic=True,

    # ------------------------------------------------------------------
    # Velocity-error options
    # ------------------------------------------------------------------
    velocity_error_method="snr_drop",
    snr_drop=1.0,
    velocity_error_floor="ccf_step",
    error_xlim=None,

    # ------------------------------------------------------------------
    # Bootstrap options, preserved from the old version
    # ------------------------------------------------------------------
    estimate_errors=False,
    n_bootstrap=500,
    bootstrap_seed=123,
    velocity_error_mode="global_kp",
    subpixel_peak=True
    ):
    """
    Plot time-resolved 1D CCFs for ingress, first half, second half, and egress.

    Parameters
    ----------
    inp_dat : dict
        Input dictionary.

    ccf_values_shift : ndarray
        Shifted CCF cube with shape:
            (n_vrest, n_spectra_with_signal, n_kp)

    v_rest : ndarray
        Velocity grid corresponding to axis 0 of ccf_values_shift.

    kp_range : ndarray
        Kp grid corresponding to axis 2 of ccf_values_shift.

    max_kp_idx : int, optional
        Global Kp index to use. If None, it is estimated from the full-transit CCF.

    ingress_n, egress_n : int, optional
        Number of in-transit spectra assigned to ingress/egress.

    xlim : tuple
        Velocity range shown in the plot.

    output_file : str, optional
        Path to save figure.

    show_plot, save_plot : bool
        Show/save figure.

    sysrem_opt : bool
        Passed to get_max_CCF_peak.

    colors : dict or list, optional
        Colors for plotted intervals.

    plot_halves : bool
        If False, only ingress and egress are shown, although all intervals
        are still computed.

    phase, with_signal, v_planet : ndarray, optional
        Optional diagnostics arrays.

    sort_by_phase : bool
        If True, define ingress/egress from sorted phase.

    diagnostic : bool
        If True, print detailed diagnostics.

    velocity_error_method : {"none", "snr_drop", "bootstrap", "both"}
        Method used to estimate velocity uncertainties.

        "snr_drop":
            Uses the interval where the CCF drops by `snr_drop` from the
            peak value.

        "bootstrap":
            Uses the old bootstrap-over-spectra method.

        "both":
            Computes both and stores both.

        "none":
            No velocity uncertainties are computed.

    snr_drop : float
        S/N drop used for the S/N-drop uncertainty interval.
        The default is 1.0.

    velocity_error_floor : None, float, "ccf_step", or "resolution"
        Lower limit imposed on one-sided velocity errors.

        None:
            No floor.

        float:
            Use this value in km/s.

        "ccf_step":
            Use the median spacing of v_rest.

        "resolution":
            Use c/R, where R is read from one of:
            inp_dat["Resolving_power"], inp_dat["R"], inp_dat["resolution"].
            If none is found, no resolution floor is applied.

    error_xlim : tuple, optional
        Velocity range used to search for peaks and errors.
        If None, uses xlim.

    estimate_errors : bool
        Backwards-compatible bootstrap switch. If True and
        velocity_error_method="none", the function switches to bootstrap.
        Otherwise, velocity_error_method has priority.

    n_bootstrap : int
        Number of bootstrap realizations.

    bootstrap_seed : int
        Random seed for bootstrap.

    velocity_error_mode : {"global_kp", "free_kp"}
        Bootstrap mode only.

    subpixel_peak : bool
        If True, refine peak velocities by local quadratic interpolation.

    Returns
    -------
    results : dict
        Dictionary containing CCFs, peak values, error estimates, diagnostics,
        and the figure.
        
    Example of use:
    
    HD18:
    
    results = exosims.plot_time_resolved_1D_CCFs(
        inp_dat=inp_dat,
        ccf_values_shift=ccf_values_shift,
        v_rest=v_rest,
        kp_range=kp_range,
        max_kp_idx=155 + 321,
        ingress_n=22,
        egress_n=22,
        xlim=(-100, 100),
        error_xlim=(-30, 30),
        output_file=f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf",
        show_plot=True,
        save_plot=True,
        sysrem_opt=inp_dat["Opt_PCA_its_ord_by_ord"],
        colors={
            "Ingress": "tab:cyan",
            "First half": "tab:cyan",
            "Second half": "tab:orange",
            "Egress": "tomato"
        },
        plot_halves=False,
        phase=phase,
        with_signal=with_signal,
        v_planet=v_planet,
        sort_by_phase=False,
        diagnostic=True,
        velocity_error_method="snr_drop",
        snr_drop=1.0,
        velocity_error_floor="ccf_step",
        subpixel_peak=True
    )
    
    W76b:
        
    results = exosims.plot_time_resolved_1D_CCFs(
        inp_dat=inp_dat,
        ccf_values_shift=ccf_values_shift,
        v_rest=v_rest,
        kp_range=kp_range,
        max_kp_idx=198 + 321,
        ingress_n=22,
        egress_n=22,
        xlim=(-100, 100),
        error_xlim=(-30, 30),
        output_file=f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf",
        show_plot=True,
        save_plot=True,
        sysrem_opt=inp_dat["Opt_PCA_its_ord_by_ord"],
        colors={
            "Ingress": "tab:cyan",
            "First half": "tab:cyan",
            "Second half": "tab:orange",
            "Egress": "tomato"
        },
        plot_halves=False,
        phase=phase,
        with_signal=with_signal,
        v_planet=v_planet,
        sort_by_phase=False,
        diagnostic=True,
        velocity_error_method="snr_drop",
        snr_drop=1.0,
        velocity_error_floor="ccf_step",
        subpixel_peak=True
    )
    
    W127b:
        
    results = exosims.plot_time_resolved_1D_CCFs(
        inp_dat=inp_dat,
        ccf_values_shift=ccf_values_shift,
        v_rest=v_rest,
        kp_range=kp_range,
        max_kp_idx=129 + 321,
        ingress_n=22,
        egress_n=22,
        xlim=(-100, 100),
        error_xlim=(-30, 30),
        output_file=f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf",
        show_plot=True,
        save_plot=True,
        sysrem_opt=inp_dat["Opt_PCA_its_ord_by_ord"],
        colors={
            "Ingress": "tab:cyan",
            "First half": "tab:cyan",
            "Second half": "tab:orange",
            "Egress": "tomato"
        },
        plot_halves=False,
        phase=phase,
        with_signal=with_signal,
        v_planet=v_planet,
        sort_by_phase=False,
        diagnostic=True,
        velocity_error_method="snr_drop",
        snr_drop=1.0,
        velocity_error_floor="ccf_step",
        subpixel_peak=True
    )
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------
    ccf_values_shift = np.asarray(ccf_values_shift)
    v_rest = np.asarray(v_rest)
    kp_range = np.asarray(kp_range)

    if ccf_values_shift.ndim != 3:
        raise ValueError(
            "ccf_values_shift must have shape "
            "(n_vrest, n_spectra_with_signal, n_kp)"
        )

    n_v, n_spec, n_kp = ccf_values_shift.shape

    if len(v_rest) != n_v:
        raise ValueError(
            f"len(v_rest) = {len(v_rest)} but ccf_values_shift has "
            f"{n_v} velocity points."
        )

    if len(kp_range) != n_kp:
        raise ValueError(
            f"len(kp_range) = {len(kp_range)} but ccf_values_shift has "
            f"{n_kp} Kp values."
        )

    if ingress_n is None:
        ingress_n = 41

    if egress_n is None:
        egress_n = 41

    ingress_n = int(ingress_n)
    egress_n = int(egress_n)

    if ingress_n <= 0:
        raise ValueError("ingress_n must be positive")

    if egress_n <= 0:
        raise ValueError("egress_n must be positive")

    if ingress_n + egress_n > n_spec:
        raise ValueError(
            f"ingress_n + egress_n = {ingress_n + egress_n}, "
            f"but only {n_spec} spectra are available."
        )

    if error_xlim is None:
        error_xlim = xlim

    allowed_error_methods = ["none", "snr_drop", "bootstrap", "both"]
    if velocity_error_method not in allowed_error_methods:
        raise ValueError(
            f"velocity_error_method must be one of {allowed_error_methods}"
        )

    if velocity_error_mode not in ["global_kp", "free_kp"]:
        raise ValueError(
            "velocity_error_mode must be either 'global_kp' or 'free_kp'"
        )

    # Backwards compatibility:
    # If someone still passes estimate_errors=True but asks for no method,
    # interpret that as the old bootstrap behaviour.
    if estimate_errors and velocity_error_method == "none":
        velocity_error_method = "bootstrap"

    do_snr_drop_errors = velocity_error_method in ["snr_drop", "both"]
    do_bootstrap_errors = velocity_error_method in ["bootstrap", "both"]

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------
    def _as_scalar(x, dtype=float):
        x = np.asarray(x)
        if x.size == 0:
            return np.nan
        return dtype(np.ravel(x)[0])

    def _as_int_scalar(x):
        return int(_as_scalar(x, dtype=float))

    def _select_in_transit_array(arr, name):
        """
        Return array with length n_spec.

        Accepts:
            - already in-transit array, length n_spec
            - full array + with_signal mask/indices
        """
        if arr is None:
            return None

        arr = np.asarray(arr)

        if arr.shape[0] == n_spec:
            return arr

        if with_signal is None:
            if diagnostic:
                print(
                    f"[diagnostic warning] {name} has length {arr.shape[0]}, "
                    f"but n_spec={n_spec} and with_signal was not provided. "
                    f"Skipping {name} diagnostics."
                )
            return None

        ws = np.asarray(with_signal)

        try:
            selected = arr[ws]
        except Exception as e:
            if diagnostic:
                print(
                    f"[diagnostic warning] Could not apply with_signal to {name}: {e}"
                )
            return None

        if selected.shape[0] != n_spec:
            if diagnostic:
                print(
                    f"[diagnostic warning] {name}[with_signal] has length "
                    f"{selected.shape[0]}, but n_spec={n_spec}. "
                    f"Skipping {name} diagnostics."
                )
            return None

        return selected

    def _get_velocity_floor():
        """
        Return velocity floor in km/s.
        """
        if velocity_error_floor is None:
            return 0.0

        if isinstance(velocity_error_floor, (int, float)):
            return float(velocity_error_floor)

        if isinstance(velocity_error_floor, str):
            mode = velocity_error_floor.lower()

            if mode == "ccf_step":
                dv = np.nanmedian(np.abs(np.diff(v_rest)))
                if np.isfinite(dv):
                    return float(dv)
                return 0.0

            if mode == "resolution":
                R = None
                for key in ["Resolving_power", "R", "resolution", "Resolution"]:
                    if key in inp_dat:
                        try:
                            R = float(inp_dat[key])
                            break
                        except Exception:
                            pass

                if R is not None and np.isfinite(R) and R > 0:
                    c_kms = 299792.458
                    return float(c_kms / R)

                if diagnostic:
                    print(
                        "[diagnostic warning] velocity_error_floor='resolution' "
                        "was requested, but no valid resolving power was found in inp_dat. "
                        "No resolution floor applied."
                    )
                return 0.0

        raise ValueError(
            "velocity_error_floor must be None, a number, 'ccf_step', or 'resolution'"
        )

    velocity_floor_value = _get_velocity_floor()

    def _peak_diagnostics(y, x, xlim_use=None, subpixel=True):
        """
        Peak information for a 1D CCF.

        Uses the maximum positive peak inside xlim_use.
        If subpixel=True, the peak velocity is refined with a local
        quadratic interpolation around the grid maximum.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if xlim_use is None:
            mask = np.ones_like(x, dtype=bool)
        else:
            mask = (x >= xlim_use[0]) & (x <= xlim_use[1])

        if not np.any(mask):
            return {
                "peak_snr": np.nan,
                "peak_vrest": np.nan,
                "peak_vrest_grid": np.nan,
                "peak_index_local": None,
                "peak_index_global": None,
                "min_snr": np.nan,
                "min_vrest": np.nan
            }

        idx_global = np.where(mask)[0]
        xx = x[mask]
        yy = y[mask]

        good = np.isfinite(xx) & np.isfinite(yy)

        if np.sum(good) < 3:
            return {
                "peak_snr": np.nan,
                "peak_vrest": np.nan,
                "peak_vrest_grid": np.nan,
                "peak_index_local": None,
                "peak_index_global": None,
                "min_snr": np.nan,
                "min_vrest": np.nan
            }

        idx_global = idx_global[good]
        xx = xx[good]
        yy = yy[good]

        imax = int(np.nanargmax(yy))
        imin = int(np.nanargmin(yy))

        peak_snr = float(yy[imax])
        peak_vrest_grid = float(xx[imax])
        peak_vrest = peak_vrest_grid

        if subpixel and imax > 0 and imax < len(xx) - 1:
            x3 = xx[imax - 1:imax + 2]
            y3 = yy[imax - 1:imax + 2]

            try:
                coeff = np.polyfit(x3, y3, 2)
                a, b, c = coeff

                if np.isfinite(a) and a < 0:
                    xv = -b / (2.0 * a)
                    yv = a * xv**2 + b * xv + c

                    if np.min(x3) <= xv <= np.max(x3):
                        peak_vrest = float(xv)
                        peak_snr = float(yv)

            except Exception:
                pass

        return {
            "peak_snr": peak_snr,
            "peak_vrest": peak_vrest,
            "peak_vrest_grid": peak_vrest_grid,
            "peak_index_local": imax,
            "peak_index_global": int(idx_global[imax]),
            "min_snr": float(yy[imin]),
            "min_vrest": float(xx[imin])
        }

    def _interpolate_crossing(x1, y1, x2, y2, y_target):
        """
        Linear interpolation for x where y crosses y_target.
        """
        if not np.isfinite(x1 + y1 + x2 + y2 + y_target):
            return np.nan

        if y2 == y1:
            return np.nan

        frac = (y_target - y1) / (y2 - y1)
        return float(x1 + frac * (x2 - x1))

    def _snr_drop_error(y, x, peak_info, xlim_use=None):
        """
        Estimate asymmetric velocity errors from the interval where the CCF
        drops by `snr_drop` from the peak value.

        The search is done on the grid CCF. The peak itself may be subpixel,
        but crossings are found by linear interpolation on the sampled CCF.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if xlim_use is None:
            mask = np.ones_like(x, dtype=bool)
        else:
            mask = (x >= xlim_use[0]) & (x <= xlim_use[1])

        if not np.any(mask):
            return {
                "method": "snr_drop",
                "snr_drop": float(snr_drop),
                "velocity_floor": float(velocity_floor_value),
                "v_left": np.nan,
                "v_right": np.nan,
                "err_minus_raw": np.nan,
                "err_plus_raw": np.nan,
                "err_minus": np.nan,
                "err_plus": np.nan,
                "err_symmetric": np.nan,
                "target_snr": np.nan
            }

        xx = x[mask]
        yy = y[mask]

        good = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[good]
        yy = yy[good]

        if len(xx) < 3:
            return {
                "method": "snr_drop",
                "snr_drop": float(snr_drop),
                "velocity_floor": float(velocity_floor_value),
                "v_left": np.nan,
                "v_right": np.nan,
                "err_minus_raw": np.nan,
                "err_plus_raw": np.nan,
                "err_minus": np.nan,
                "err_plus": np.nan,
                "err_symmetric": np.nan,
                "target_snr": np.nan
            }

        imax = int(np.nanargmax(yy))
        peak_snr = float(peak_info["peak_snr"])
        peak_v = float(peak_info["peak_vrest"])

        target = peak_snr - float(snr_drop)

        v_left = np.nan
        v_right = np.nan

        # Search left from the peak grid index
        for j in range(imax - 1, -1, -1):
            if yy[j] <= target:
                v_left = _interpolate_crossing(
                    xx[j], yy[j],
                    xx[j + 1], yy[j + 1],
                    target
                )
                break

        # Search right from the peak grid index
        for j in range(imax + 1, len(xx)):
            if yy[j] <= target:
                v_right = _interpolate_crossing(
                    xx[j - 1], yy[j - 1],
                    xx[j], yy[j],
                    target
                )
                break

        err_minus_raw = np.nan
        err_plus_raw = np.nan

        if np.isfinite(v_left):
            err_minus_raw = peak_v - v_left

        if np.isfinite(v_right):
            err_plus_raw = v_right - peak_v

        err_minus = err_minus_raw
        err_plus = err_plus_raw

        if np.isfinite(err_minus):
            err_minus = max(float(err_minus), float(velocity_floor_value))

        if np.isfinite(err_plus):
            err_plus = max(float(err_plus), float(velocity_floor_value))

        if np.isfinite(err_minus) and np.isfinite(err_plus):
            err_symmetric = 0.5 * (err_minus + err_plus)
        elif np.isfinite(err_minus):
            err_symmetric = err_minus
        elif np.isfinite(err_plus):
            err_symmetric = err_plus
        else:
            err_symmetric = np.nan

        return {
            "method": "snr_drop",
            "snr_drop": float(snr_drop),
            "velocity_floor": float(velocity_floor_value),
            "v_left": float(v_left) if np.isfinite(v_left) else np.nan,
            "v_right": float(v_right) if np.isfinite(v_right) else np.nan,
            "err_minus_raw": float(err_minus_raw) if np.isfinite(err_minus_raw) else np.nan,
            "err_plus_raw": float(err_plus_raw) if np.isfinite(err_plus_raw) else np.nan,
            "err_minus": float(err_minus) if np.isfinite(err_minus) else np.nan,
            "err_plus": float(err_plus) if np.isfinite(err_plus) else np.nan,
            "err_symmetric": float(err_symmetric) if np.isfinite(err_symmetric) else np.nan,
            "target_snr": float(target)
        }

    def _bootstrap_interval_errors(idx, fixed_kp_idx):
        """
        Bootstrap over spectra inside one interval.

        With velocity_error_mode='global_kp', this estimates errors on the
        same fixed-Kp CCF that is plotted.
        With velocity_error_mode='free_kp', this allows the Kp to vary in
        each bootstrap realization.
        """
        if (not do_bootstrap_errors) or n_bootstrap is None or n_bootstrap <= 1:
            return None

        idx = np.asarray(idx, dtype=int)
        n_this = len(idx)

        if n_this < 3:
            return None

        rng = np.random.default_rng(bootstrap_seed)

        boot_peak_v = []
        boot_peak_v_grid = []
        boot_peak_snr = []
        boot_kp = []
        boot_kp_idx = []

        for _ in range(int(n_bootstrap)):

            boot_idx = rng.choice(idx, size=n_this, replace=True)
            ccf_boot = np.sum(ccf_values_shift[:, boot_idx, :], axis=1)

            try:
                ccf_boot_sig, boot_max_sig, boot_max_kp_idx, boot_max_v_wind, _ = get_max_CCF_peak(
                    inp_dat,
                    ccf_boot,
                    v_rest,
                    kp_range,
                    b=None,
                    stats=None,
                    sysrem_opt=sysrem_opt,
                    CCF_Noise=False
                )

                boot_max_kp_idx = _as_int_scalar(boot_max_kp_idx)

                if velocity_error_mode == "global_kp":
                    use_kp_idx = fixed_kp_idx
                else:
                    use_kp_idx = boot_max_kp_idx

                if use_kp_idx < 0 or use_kp_idx >= n_kp:
                    continue

                y_boot = ccf_boot_sig[:, use_kp_idx]

                pinfo = _peak_diagnostics(
                    y_boot,
                    v_rest,
                    xlim_use=error_xlim,
                    subpixel=subpixel_peak
                )

                if np.isfinite(pinfo["peak_vrest"]):
                    boot_peak_v.append(pinfo["peak_vrest"])
                    boot_peak_v_grid.append(pinfo["peak_vrest_grid"])
                    boot_peak_snr.append(pinfo["peak_snr"])
                    boot_kp.append(kp_range[use_kp_idx])
                    boot_kp_idx.append(use_kp_idx)

            except Exception:
                continue

        boot_peak_v = np.asarray(boot_peak_v, dtype=float)
        boot_peak_v_grid = np.asarray(boot_peak_v_grid, dtype=float)
        boot_peak_snr = np.asarray(boot_peak_snr, dtype=float)
        boot_kp = np.asarray(boot_kp, dtype=float)
        boot_kp_idx = np.asarray(boot_kp_idx, dtype=int)

        out = {
            "method": "bootstrap",
            "n_bootstrap_requested": int(n_bootstrap),
            "n_bootstrap_used": int(boot_peak_v.size),
            "velocity_error_mode": velocity_error_mode,
            "error_xlim": error_xlim,
            "subpixel_peak": bool(subpixel_peak),
            "vrest_std": np.nan,
            "vrest_p16": np.nan,
            "vrest_p50": np.nan,
            "vrest_p84": np.nan,
            "vrest_grid_std": np.nan,
            "snr_std": np.nan,
            "snr_p16": np.nan,
            "snr_p50": np.nan,
            "snr_p84": np.nan,
            "kp_std": np.nan,
            "kp_p16": np.nan,
            "kp_p50": np.nan,
            "kp_p84": np.nan,
            "boot_peak_vrest": boot_peak_v,
            "boot_peak_vrest_grid": boot_peak_v_grid,
            "boot_peak_snr": boot_peak_snr,
            "boot_kp": boot_kp,
            "boot_kp_idx": boot_kp_idx
        }

        if boot_peak_v.size < max(20, int(0.1 * int(n_bootstrap))):
            return out

        out.update({
            "vrest_std": float(np.nanstd(boot_peak_v, ddof=1)),
            "vrest_p16": float(np.nanpercentile(boot_peak_v, 16)),
            "vrest_p50": float(np.nanpercentile(boot_peak_v, 50)),
            "vrest_p84": float(np.nanpercentile(boot_peak_v, 84)),
            "vrest_grid_std": float(np.nanstd(boot_peak_v_grid, ddof=1)),
            "snr_std": float(np.nanstd(boot_peak_snr, ddof=1)),
            "snr_p16": float(np.nanpercentile(boot_peak_snr, 16)),
            "snr_p50": float(np.nanpercentile(boot_peak_snr, 50)),
            "snr_p84": float(np.nanpercentile(boot_peak_snr, 84)),
            "kp_std": float(np.nanstd(boot_kp, ddof=1)),
            "kp_p16": float(np.nanpercentile(boot_kp, 16)),
            "kp_p50": float(np.nanpercentile(boot_kp, 50)),
            "kp_p84": float(np.nanpercentile(boot_kp, 84))
        })

        return out

    # ------------------------------------------------------------
    # Prepare optional diagnostic arrays
    # ------------------------------------------------------------
    phase_in = _select_in_transit_array(phase, "phase")
    v_planet_in = _select_in_transit_array(v_planet, "v_planet")

    # ------------------------------------------------------------
    # Determine global Kp index from the full transit if needed
    # ------------------------------------------------------------
    full_ccf_tot = np.sum(ccf_values_shift, axis=1)

    full_ccf_sig, full_max_sig, full_max_kp_idx, full_max_v_wind, _ = get_max_CCF_peak(
        inp_dat,
        full_ccf_tot,
        v_rest,
        kp_range,
        b=None,
        stats=None,
        sysrem_opt=sysrem_opt,
        CCF_Noise=False
    )

    full_max_sig = _as_scalar(full_max_sig)
    full_max_kp_idx = _as_int_scalar(full_max_kp_idx)
    full_max_v_wind = _as_scalar(full_max_v_wind)

    if max_kp_idx is None:
        max_kp_idx = full_max_kp_idx
    else:
        max_kp_idx = int(max_kp_idx)

    if max_kp_idx < 0 or max_kp_idx >= n_kp:
        raise ValueError(
            f"max_kp_idx={max_kp_idx} is outside allowed range 0--{n_kp - 1}"
        )

    # ------------------------------------------------------------
    # Define time intervals
    # ------------------------------------------------------------
    mid = n_spec // 2

    if sort_by_phase and phase_in is not None:
        order = np.argsort(phase_in)

        intervals = {
            "Ingress": order[:ingress_n],
            "First half": order[:mid],
            "Second half": order[mid:],
            "Egress": order[-egress_n:]
        }

        interval_definition = "phase-sorted"

    else:
        intervals = {
            "Ingress": np.arange(0, ingress_n),
            "First half": np.arange(0, mid),
            "Second half": np.arange(mid, n_spec),
            "Egress": np.arange(n_spec - egress_n, n_spec)
        }

        interval_definition = "array-order"

        if sort_by_phase and phase_in is None and diagnostic:
            print(
                "[diagnostic warning] sort_by_phase=True was requested, "
                "but no usable phase array was provided. Falling back to array-order intervals."
            )

    # ------------------------------------------------------------
    # Default colors
    # ------------------------------------------------------------
    default_colors = {
        "Ingress": "tab:blue",
        "First half": "tab:cyan",
        "Second half": "tab:orange",
        "Egress": "tab:red"
    }

    if colors is None:
        colors_dict = default_colors

    elif isinstance(colors, dict):
        colors_dict = default_colors.copy()
        colors_dict.update(colors)

    else:
        color_list = list(colors)
        interval_names = list(intervals.keys())

        if len(color_list) < len(interval_names):
            raise ValueError(
                "If colors is a list/tuple, it must contain at least "
                f"{len(interval_names)} colors."
            )

        colors_dict = {
            name: color_list[i]
            for i, name in enumerate(interval_names)
        }

    # ------------------------------------------------------------
    # Global diagnostics
    # ------------------------------------------------------------
    diagnostics = {
        "ccf_values_shift_shape": ccf_values_shift.shape,
        "n_vrest": int(n_v),
        "n_spectra_with_signal": int(n_spec),
        "n_kp": int(n_kp),
        "interval_definition": interval_definition,
        "sort_by_phase": bool(sort_by_phase),
        "full_max_sig": float(full_max_sig),
        "full_max_kp_idx": int(full_max_kp_idx),
        "full_max_kp": float(kp_range[full_max_kp_idx]),
        "full_max_v_wind": float(full_max_v_wind),
        "used_global_kp_idx": int(max_kp_idx),
        "used_global_kp": float(kp_range[max_kp_idx]),
        "velocity_error_method": velocity_error_method,
        "snr_drop": float(snr_drop),
        "velocity_error_floor": velocity_error_floor,
        "velocity_error_floor_value": float(velocity_floor_value),
        "n_bootstrap": int(n_bootstrap) if n_bootstrap is not None else None,
        "bootstrap_seed": int(bootstrap_seed),
        "error_xlim": error_xlim,
        "velocity_error_mode": velocity_error_mode,
        "subpixel_peak": bool(subpixel_peak)
    }

    if phase_in is not None:
        dphase = np.diff(phase_in)
        diagnostics["phase_first"] = float(phase_in[0])
        diagnostics["phase_last"] = float(phase_in[-1])
        diagnostics["phase_min"] = float(np.nanmin(phase_in))
        diagnostics["phase_max"] = float(np.nanmax(phase_in))
        diagnostics["phase_monotonic_increasing"] = bool(np.all(dphase >= 0))
        diagnostics["phase_monotonic_decreasing"] = bool(np.all(dphase <= 0))
        diagnostics["phase_n_negative_steps"] = int(np.sum(dphase < 0))
        diagnostics["phase_n_positive_steps"] = int(np.sum(dphase > 0))

    if v_planet_in is not None:
        diagnostics["v_planet_first"] = float(v_planet_in[0])
        diagnostics["v_planet_last"] = float(v_planet_in[-1])
        diagnostics["v_planet_min"] = float(np.nanmin(v_planet_in))
        diagnostics["v_planet_max"] = float(np.nanmax(v_planet_in))

    if diagnostic:
        print("\n" + "=" * 72)
        print("TIME-RESOLVED CCF DIAGNOSTICS")
        print("=" * 72)
        print(f"ccf_values_shift shape = {ccf_values_shift.shape}")
        print("Assumed axes: (Vrest, in-transit spectrum index, Kp)")
        print(f"len(v_rest)  = {len(v_rest)}")
        print(f"len(kp_range)= {len(kp_range)}")
        print(f"Interval definition = {interval_definition}")
        print(f"Full-transit best Kp index = {full_max_kp_idx}")
        print(f"Full-transit best Kp       = {kp_range[full_max_kp_idx]:.3f} km/s")
        print(f"Full-transit best Vrest    = {full_max_v_wind:.3f} km/s")
        print(f"Full-transit max S/N       = {full_max_sig:.3f}")
        print(f"Using global Kp index      = {max_kp_idx}")
        print(f"Using global Kp            = {kp_range[max_kp_idx]:.3f} km/s")
        print(f"Velocity error method      = {velocity_error_method}")
        print(f"S/N-drop value             = {snr_drop:.3f}")
        print(f"Velocity error floor       = {velocity_error_floor}")
        print(f"Velocity floor value       = {velocity_floor_value:.3f} km/s")
        print(f"Error peak xlim            = {error_xlim}")
        print(f"Subpixel peak refinement   = {subpixel_peak}")

        if do_bootstrap_errors:
            print(f"Bootstrap realizations     = {n_bootstrap}")
            print(f"Bootstrap mode             = {velocity_error_mode}")

        if phase_in is not None:
            print("\nPhase diagnostics:")
            print(f"  phase[0]  = {phase_in[0]: .8f}")
            print(f"  phase[-1] = {phase_in[-1]: .8f}")
            print(f"  phase min = {np.nanmin(phase_in): .8f}")
            print(f"  phase max = {np.nanmax(phase_in): .8f}")
            print(f"  monotonic increasing = {diagnostics['phase_monotonic_increasing']}")
            print(f"  monotonic decreasing = {diagnostics['phase_monotonic_decreasing']}")
            print(f"  negative phase steps = {diagnostics['phase_n_negative_steps']}")
            print(f"  positive phase steps = {diagnostics['phase_n_positive_steps']}")

            if not diagnostics["phase_monotonic_increasing"]:
                print(
                    "  WARNING: phase is not monotonically increasing. "
                    "Array-order ingress/egress labels may be wrong."
                )

        else:
            print(
                "\nPhase diagnostics unavailable. Pass phase=phase and with_signal=with_signal "
                "to verify ingress/egress ordering."
            )

        if v_planet_in is not None:
            print("\nPlanet velocity diagnostics:")
            print(f"  v_planet[0]  = {v_planet_in[0]: .3f} km/s")
            print(f"  v_planet[-1] = {v_planet_in[-1]: .3f} km/s")
            print(f"  v_planet min = {np.nanmin(v_planet_in): .3f} km/s")
            print(f"  v_planet max = {np.nanmax(v_planet_in): .3f} km/s")

        print("=" * 72 + "\n")

    # ------------------------------------------------------------
    # Compute CCFs for each interval
    # ------------------------------------------------------------
    results = {}

    for name, idx in intervals.items():

        idx = np.asarray(idx, dtype=int)

        ccf_interval = np.sum(ccf_values_shift[:, idx, :], axis=1)

        ccf_interval_sig, max_sig, interval_max_kp_idx, max_v_wind, _ = get_max_CCF_peak(
            inp_dat,
            ccf_interval,
            v_rest,
            kp_range,
            b=None,
            stats=None,
            sysrem_opt=sysrem_opt,
            CCF_Noise=False
        )

        max_sig = _as_scalar(max_sig)
        interval_max_kp_idx = _as_int_scalar(interval_max_kp_idx)
        max_v_wind = _as_scalar(max_v_wind)

        ccf_1d_global_kp = ccf_interval_sig[:, max_kp_idx]
        ccf_1d_free_kp = ccf_interval_sig[:, interval_max_kp_idx]

        global_peak_info = _peak_diagnostics(
            ccf_1d_global_kp,
            v_rest,
            xlim_use=error_xlim,
            subpixel=subpixel_peak
        )

        free_peak_info = _peak_diagnostics(
            ccf_1d_free_kp,
            v_rest,
            xlim_use=error_xlim,
            subpixel=subpixel_peak
        )

        snr_drop_errors = None
        if do_snr_drop_errors:
            snr_drop_errors = _snr_drop_error(
                ccf_1d_global_kp,
                v_rest,
                global_peak_info,
                xlim_use=error_xlim
            )

        bootstrap_errors = None
        if do_bootstrap_errors:
            bootstrap_errors = _bootstrap_interval_errors(
                idx=idx,
                fixed_kp_idx=max_kp_idx
            )

        interval_diag = {
            "index_first": int(idx[0]),
            "index_last": int(idx[-1]),
            "index_min": int(np.min(idx)),
            "index_max": int(np.max(idx)),
            "n_spectra": int(len(idx)),
            "global_kp_peak_snr": global_peak_info["peak_snr"],
            "global_kp_peak_vrest": global_peak_info["peak_vrest"],
            "global_kp_peak_vrest_grid": global_peak_info["peak_vrest_grid"],
            "global_kp_min_snr": global_peak_info["min_snr"],
            "global_kp_min_vrest": global_peak_info["min_vrest"],
            "free_kp_peak_snr": free_peak_info["peak_snr"],
            "free_kp_peak_vrest": free_peak_info["peak_vrest"],
            "free_kp_peak_vrest_grid": free_peak_info["peak_vrest_grid"],
            "free_kp_min_snr": free_peak_info["min_snr"],
            "free_kp_min_vrest": free_peak_info["min_vrest"]
        }

        if snr_drop_errors is not None:
            interval_diag["snr_drop_errors"] = snr_drop_errors
            interval_diag["global_kp_peak_vrest_err_minus"] = snr_drop_errors["err_minus"]
            interval_diag["global_kp_peak_vrest_err_plus"] = snr_drop_errors["err_plus"]
            interval_diag["global_kp_peak_vrest_err"] = snr_drop_errors["err_symmetric"]

        if bootstrap_errors is not None:
            interval_diag["bootstrap_errors"] = bootstrap_errors
            interval_diag["bootstrap_global_kp_peak_vrest_err"] = bootstrap_errors["vrest_std"]
            interval_diag["bootstrap_global_kp_peak_snr_err"] = bootstrap_errors["snr_std"]
            interval_diag["bootstrap_global_kp_peak_vrest_p16"] = bootstrap_errors["vrest_p16"]
            interval_diag["bootstrap_global_kp_peak_vrest_p50"] = bootstrap_errors["vrest_p50"]
            interval_diag["bootstrap_global_kp_peak_vrest_p84"] = bootstrap_errors["vrest_p84"]
            interval_diag["bootstrap_global_kp_peak_snr_p16"] = bootstrap_errors["snr_p16"]
            interval_diag["bootstrap_global_kp_peak_snr_p50"] = bootstrap_errors["snr_p50"]
            interval_diag["bootstrap_global_kp_peak_snr_p84"] = bootstrap_errors["snr_p84"]

        if phase_in is not None:
            ph = phase_in[idx]
            interval_diag["phase_first"] = float(ph[0])
            interval_diag["phase_last"] = float(ph[-1])
            interval_diag["phase_min"] = float(np.nanmin(ph))
            interval_diag["phase_max"] = float(np.nanmax(ph))
            interval_diag["phase_median"] = float(np.nanmedian(ph))

        if v_planet_in is not None:
            vp = v_planet_in[idx]
            interval_diag["v_planet_first"] = float(vp[0])
            interval_diag["v_planet_last"] = float(vp[-1])
            interval_diag["v_planet_min"] = float(np.nanmin(vp))
            interval_diag["v_planet_max"] = float(np.nanmax(vp))
            interval_diag["v_planet_median"] = float(np.nanmedian(vp))

        results[name] = {
            "indices": idx,
            "ccf_tot": ccf_interval,
            "ccf_sig": ccf_interval_sig,
            "ccf_1d_global_kp": ccf_1d_global_kp,
            "ccf_1d_free_kp": ccf_1d_free_kp,
            "global_kp_idx": int(max_kp_idx),
            "global_kp": float(kp_range[max_kp_idx]),
            "interval_max_kp_idx": int(interval_max_kp_idx),
            "interval_max_kp": float(kp_range[interval_max_kp_idx]),
            "max_sig": float(max_sig),
            "max_v_wind": float(max_v_wind),
            "snr_drop_errors": snr_drop_errors,
            "bootstrap_errors": bootstrap_errors,
            "diagnostics": interval_diag
        }

        if diagnostic:
            print(f"{name:12s}:")
            print(f"  Nspec                  = {len(idx)}")
            print(f"  index range             = {idx[0]} -> {idx[-1]} "
                  f"(min={np.min(idx)}, max={np.max(idx)})")

            if phase_in is not None:
                print(f"  phase first/last        = {interval_diag['phase_first']:.8f} "
                      f"-> {interval_diag['phase_last']:.8f}")
                print(f"  phase min/max/median    = {interval_diag['phase_min']:.8f}, "
                      f"{interval_diag['phase_max']:.8f}, "
                      f"{interval_diag['phase_median']:.8f}")

            if v_planet_in is not None:
                print(f"  v_planet first/last     = {interval_diag['v_planet_first']:.3f} "
                      f"-> {interval_diag['v_planet_last']:.3f} km/s")
                print(f"  v_planet min/max/median = {interval_diag['v_planet_min']:.3f}, "
                      f"{interval_diag['v_planet_max']:.3f}, "
                      f"{interval_diag['v_planet_median']:.3f} km/s")

            print(f"  free max S/N            = {max_sig:.3f}")
            print(f"  free max Kp             = {kp_range[interval_max_kp_idx]:.3f} km/s")
            print(f"  free max Vrest          = {max_v_wind:.3f} km/s")
            print(f"  global-Kp positive peak = {global_peak_info['peak_snr']:.3f} "
                  f"at Vrest = {global_peak_info['peak_vrest']:.3f} km/s")
            print(f"  global-Kp grid peak     = {global_peak_info['peak_vrest_grid']:.3f} km/s")
            print(f"  global-Kp most negative = {global_peak_info['min_snr']:.3f} "
                  f"at Vrest = {global_peak_info['min_vrest']:.3f} km/s")

            if snr_drop_errors is not None:
                print(
                    f"  S/N-drop error          = "
                    f"-{snr_drop_errors['err_minus']:.3f} / "
                    f"+{snr_drop_errors['err_plus']:.3f} km/s "
                    f"(floor = {snr_drop_errors['velocity_floor']:.3f} km/s)"
                )
                print(
                    f"  S/N-drop interval       = "
                    f"[{snr_drop_errors['v_left']:.3f}, "
                    f"{snr_drop_errors['v_right']:.3f}] km/s "
                    f"at CCF = peak - {snr_drop:.2f}"
                )

            if bootstrap_errors is not None:
                print(
                    f"  bootstrap Vrest error   = ±{bootstrap_errors['vrest_std']:.3f} km/s "
                    f"[16,50,84] = "
                    f"[{bootstrap_errors['vrest_p16']:.3f}, "
                    f"{bootstrap_errors['vrest_p50']:.3f}, "
                    f"{bootstrap_errors['vrest_p84']:.3f}] km/s"
                )
                print(
                    f"  bootstrap S/N error     = ±{bootstrap_errors['snr_std']:.3f} "
                    f"[16,50,84] = "
                    f"[{bootstrap_errors['snr_p16']:.3f}, "
                    f"{bootstrap_errors['snr_p50']:.3f}, "
                    f"{bootstrap_errors['snr_p84']:.3f}]"
                )
                print(
                    f"  bootstrap samples used  = "
                    f"{bootstrap_errors['n_bootstrap_used']} / "
                    f"{bootstrap_errors['n_bootstrap_requested']}"
                )

            print("")

    print(
        f"Full-transit reference: "
        f"S/N = {full_max_sig:.3f}, "
        f"Kp = {kp_range[full_max_kp_idx]:.2f}, "
        f"Vrest = {full_max_v_wind:.2f} km/s"
    )

    # ------------------------------------------------------------
    # Ingress/egress comparison
    # ------------------------------------------------------------
    if diagnostic and "Ingress" in results and "Egress" in results:
        ving = results["Ingress"]["diagnostics"]["global_kp_peak_vrest"]
        vegr = results["Egress"]["diagnostics"]["global_kp_peak_vrest"]

        print("\nIngress/Egress global-Kp peak comparison:")
        print(f"  Ingress peak Vrest = {ving:.3f} km/s")
        print(f"  Egress  peak Vrest = {vegr:.3f} km/s")

        if vegr < ving:
            print("  Egress is more blueshifted than ingress in this plot.")
        elif ving < vegr:
            print("  Ingress is more blueshifted than egress in this plot.")
            print(
                "  If your expected geometry predicts a more blueshifted egress, "
                "check phase ordering, the sign convention in get_shifted_ccf_matrix, "
                "and whether sort_by_phase=True changes the result."
            )
        else:
            print("  Ingress and egress peak at the same Vrest.")

        delta_v = ving - vegr
        delta_info = {
            "delta_v_ingress_minus_egress": float(delta_v),
            "ingress_vrest": float(ving),
            "egress_vrest": float(vegr)
        }

        if do_snr_drop_errors:
            eing = results["Ingress"]["diagnostics"].get("global_kp_peak_vrest_err", np.nan)
            eegr = results["Egress"]["diagnostics"].get("global_kp_peak_vrest_err", np.nan)

            if np.isfinite(eing) and np.isfinite(eegr):
                delta_v_err = np.sqrt(eing**2 + eegr**2)

                print(
                    f"  Delta V = V_ingress - V_egress = "
                    f"{delta_v:.3f} ± {delta_v_err:.3f} km/s "
                    f"(S/N-drop errors)"
                )

                delta_info.update({
                    "delta_v_error_snr_drop": float(delta_v_err),
                    "ingress_vrest_error_snr_drop": float(eing),
                    "egress_vrest_error_snr_drop": float(eegr)
                })

        if do_bootstrap_errors:
            b_ing = results["Ingress"]["bootstrap_errors"]
            b_egr = results["Egress"]["bootstrap_errors"]

            if b_ing is not None and b_egr is not None:
                eing = b_ing["vrest_std"]
                eegr = b_egr["vrest_std"]

                if np.isfinite(eing) and np.isfinite(eegr):
                    delta_v_err = np.sqrt(eing**2 + eegr**2)

                    print(
                        f"  Delta V = V_ingress - V_egress = "
                        f"{delta_v:.3f} ± {delta_v_err:.3f} km/s "
                        f"(bootstrap errors)"
                    )

                    delta_info.update({
                        "delta_v_error_bootstrap": float(delta_v_err),
                        "ingress_vrest_error_bootstrap": float(eing),
                        "egress_vrest_error_bootstrap": float(eegr)
                    })

        results["Ingress_Egress_delta"] = delta_info

        print("")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    if plot_halves:
        plot_order = ["Ingress", "First half", "Second half", "Egress"]
    else:
        plot_order = ["Ingress", "Egress"]

    for name in plot_order:
        res = results[name]

        v_peak = res["diagnostics"]["global_kp_peak_vrest"]

        ax.plot(
            v_rest,
            res["ccf_1d_global_kp"],
            lw=4,
            ls="-",
            color=colors_dict[name],
            label=rf"{name} ($V_{{\rm peak}}={v_peak:.1f}$ km s$^{{-1}}$)"
        )

    ax.axhline(0, color="k", lw=1.7, ls="--", alpha=0.4)
    ax.axvline(0, color="k", lw=1.7, ls="--", alpha=0.4)

    ax.set_xlim(xlim)
    ax.set_xlabel(r"$V_{\rm rest}$ [km s$^{-1}$]", fontsize=20)
    ax.set_ylabel("CCF S/N", fontsize=20)

    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=16, frameon=True)

    plt.tight_layout()

    if save_plot:
        if output_file is None:
            output_file = f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    results["diagnostics"] = diagnostics
    results["figure"] = fig

    return results


def plot_time_resolved_1D_CCFs_withHRLRS(inp_dat,
    ccf_values_shift,
    v_rest,
    kp_range,
    max_kp_idx=None,
    ingress_n=None,
    egress_n=None,
    xlim=(-100, 100),
    output_file=None,
    show_plot=True,
    save_plot=True,
    sysrem_opt=False,
    colors=None,
    plot_halves=True,
    phase=None,
    with_signal=None,
    v_planet=None,
    sort_by_phase=False,
    diagnostic=True,

    # ------------------------------------------------------------------
    # Velocity-error options
    # ------------------------------------------------------------------
    velocity_error_method="snr_drop",
    snr_drop=1.0,
    velocity_error_floor="ccf_step",
    error_xlim=None,

    # ------------------------------------------------------------------
    # Bootstrap options, preserved from the old version
    # ------------------------------------------------------------------
    estimate_errors=False,
    n_bootstrap=500,
    bootstrap_seed=123,
    velocity_error_mode="global_kp",
    subpixel_peak=True,
    compute_lsm_metric=True,
    lsm_reference=("Ingress", "Egress"),
    lsm_error_source="auto",
    lsm_min_detection_snr=5.0,
    lsm_shape_window_kms=25.0
):
    """
    Plot time-resolved 1D CCFs for ingress, first half, second half, and egress.

    Parameters
    ----------
    inp_dat : dict
        Input dictionary.

    ccf_values_shift : ndarray
        Shifted CCF cube with shape:
            (n_vrest, n_spectra_with_signal, n_kp)

    v_rest : ndarray
        Velocity grid corresponding to axis 0 of ccf_values_shift.

    kp_range : ndarray
        Kp grid corresponding to axis 2 of ccf_values_shift.

    max_kp_idx : int, optional
        Global Kp index to use. If None, it is estimated from the full-transit CCF.

    ingress_n, egress_n : int, optional
        Number of in-transit spectra assigned to ingress/egress.

    xlim : tuple
        Velocity range shown in the plot.

    output_file : str, optional
        Path to save figure.

    show_plot, save_plot : bool
        Show/save figure.

    sysrem_opt : bool
        Passed to get_max_CCF_peak.

    colors : dict or list, optional
        Colors for plotted intervals.

    plot_halves : bool
        If False, only ingress and egress are shown, although all intervals
        are still computed.

    phase, with_signal, v_planet : ndarray, optional
        Optional diagnostics arrays.

    sort_by_phase : bool
        If True, define ingress/egress from sorted phase.

    diagnostic : bool
        If True, print detailed diagnostics.

    velocity_error_method : {"none", "snr_drop", "bootstrap", "both"}
        Method used to estimate velocity uncertainties.

        "snr_drop":
            Uses the interval where the CCF drops by `snr_drop` from the
            peak value.

        "bootstrap":
            Uses the old bootstrap-over-spectra method.

        "both":
            Computes both and stores both.

        "none":
            No velocity uncertainties are computed.

    snr_drop : float
        S/N drop used for the S/N-drop uncertainty interval.
        The default is 1.0.

    velocity_error_floor : None, float, "ccf_step", or "resolution"
        Lower limit imposed on one-sided velocity errors.

        None:
            No floor.

        float:
            Use this value in km/s.

        "ccf_step":
            Use the median spacing of v_rest.

        "resolution":
            Use c/R, where R is read from one of:
            inp_dat["Resolving_power"], inp_dat["R"], inp_dat["resolution"].
            If none is found, no resolution floor is applied.

    error_xlim : tuple, optional
        Velocity range used to search for peaks and errors.
        If None, uses xlim.

    estimate_errors : bool
        Backwards-compatible bootstrap switch. If True and
        velocity_error_method="none", the function switches to bootstrap.
        Otherwise, velocity_error_method has priority.

    n_bootstrap : int
        Number of bootstrap realizations.

    bootstrap_seed : int
        Random seed for bootstrap.

    velocity_error_mode : {"global_kp", "free_kp"}
        Bootstrap mode only.

    subpixel_peak : bool
        If True, refine peak velocities by local quadratic interpolation.

    HRS-LSM-LIKE METRIC
    -------------------
    compute_lsm_metric : bool
        If True, compute a compact high-resolution limb-sensitivity metric
        between two intervals, by default Ingress and Egress. The metric is
        stored in results["HRS_LSM"]. Nothing in the plotting is changed.

    lsm_reference : tuple of str
        Names of the two intervals to compare. Default is ("Ingress", "Egress").

    lsm_error_source : {"auto", "bootstrap", "snr_drop"}
        Source of the velocity uncertainty used in the metric. "auto" uses
        bootstrap errors when available and otherwise falls back to S/N-drop
        errors.

    lsm_min_detection_snr : float
        Reference single-limb S/N used to convert velocity separation into a
        detection-weighted metric. A value of 5 means that two 5-sigma limb
        detections receive unit detection weight.

    lsm_shape_window_kms : float
        Half-width around each interval CCF peak used to compare the normalised
        CCF shapes after recentering both peaks.

    Returns
    -------
    results : dict
        Dictionary containing CCFs, peak values, error estimates, diagnostics,
        and the figure.
        
    Example of use:
    
    HD18:
    
    results = exosims.plot_time_resolved_1D_CCFs_withHRLRS(
        inp_dat=inp_dat,
        ccf_values_shift=ccf_values_shift,
        v_rest=v_rest,
        kp_range=kp_range,
        max_kp_idx=155 + 321,
        ingress_n=22,
        egress_n=22,
        xlim=(-100, 100),
        error_xlim=(-30, 30),
        output_file=f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf",
        show_plot=True,
        save_plot=True,
        sysrem_opt=inp_dat["Opt_PCA_its_ord_by_ord"],
        colors={
            "Ingress": "tab:cyan",
            "First half": "tab:cyan",
            "Second half": "tab:orange",
            "Egress": "tomato"
        },
        plot_halves=False,
        phase=phase,
        with_signal=with_signal,
        v_planet=v_planet,
        sort_by_phase=False,
        diagnostic=True,
        velocity_error_method="snr_drop",
        snr_drop=1.0,
        velocity_error_floor="ccf_step",
        subpixel_peak=True
    )
    
    W76b:
        
    results = exosims.plot_time_resolved_1D_CCFs_withHRLRS(
        inp_dat=inp_dat,
        ccf_values_shift=ccf_values_shift,
        v_rest=v_rest,
        kp_range=kp_range,
        max_kp_idx=198 + 321,
        ingress_n=22,
        egress_n=22,
        xlim=(-100, 100),
        error_xlim=(-30, 30),
        output_file=f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf",
        show_plot=True,
        save_plot=True,
        sysrem_opt=inp_dat["Opt_PCA_its_ord_by_ord"],
        colors={
            "Ingress": "tab:cyan",
            "First half": "tab:cyan",
            "Second half": "tab:orange",
            "Egress": "tomato"
        },
        plot_halves=False,
        phase=phase,
        with_signal=with_signal,
        v_planet=v_planet,
        sort_by_phase=False,
        diagnostic=True,
        velocity_error_method="snr_drop",
        snr_drop=1.0,
        velocity_error_floor="ccf_step",
        subpixel_peak=True
    )
    
    W127b:
        
    results = exosims.plot_time_resolved_1D_CCFs_withHRLRS(
        inp_dat=inp_dat,
        ccf_values_shift=ccf_values_shift,
        v_rest=v_rest,
        kp_range=kp_range,
        max_kp_idx=129 + 321,
        ingress_n=22,
        egress_n=22,
        xlim=(-100, 100),
        error_xlim=(-30, 30),
        output_file=f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf",
        show_plot=True,
        save_plot=True,
        sysrem_opt=inp_dat["Opt_PCA_its_ord_by_ord"],
        colors={
            "Ingress": "tab:cyan",
            "First half": "tab:cyan",
            "Second half": "tab:orange",
            "Egress": "tomato"
        },
        plot_halves=False,
        phase=phase,
        with_signal=with_signal,
        v_planet=v_planet,
        sort_by_phase=False,
        diagnostic=True,
        velocity_error_method="snr_drop",
        snr_drop=1.0,
        velocity_error_floor="ccf_step",
        subpixel_peak=True
    )
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------
    ccf_values_shift = np.asarray(ccf_values_shift)
    v_rest = np.asarray(v_rest)
    kp_range = np.asarray(kp_range)

    if ccf_values_shift.ndim != 3:
        raise ValueError(
            "ccf_values_shift must have shape "
            "(n_vrest, n_spectra_with_signal, n_kp)"
        )

    n_v, n_spec, n_kp = ccf_values_shift.shape

    if len(v_rest) != n_v:
        raise ValueError(
            f"len(v_rest) = {len(v_rest)} but ccf_values_shift has "
            f"{n_v} velocity points."
        )

    if len(kp_range) != n_kp:
        raise ValueError(
            f"len(kp_range) = {len(kp_range)} but ccf_values_shift has "
            f"{n_kp} Kp values."
        )

    if ingress_n is None:
        ingress_n = 41

    if egress_n is None:
        egress_n = 41

    ingress_n = int(ingress_n)
    egress_n = int(egress_n)

    if ingress_n <= 0:
        raise ValueError("ingress_n must be positive")

    if egress_n <= 0:
        raise ValueError("egress_n must be positive")

    if ingress_n + egress_n > n_spec:
        raise ValueError(
            f"ingress_n + egress_n = {ingress_n + egress_n}, "
            f"but only {n_spec} spectra are available."
        )

    if error_xlim is None:
        error_xlim = xlim

    allowed_error_methods = ["none", "snr_drop", "bootstrap", "both"]
    if velocity_error_method not in allowed_error_methods:
        raise ValueError(
            f"velocity_error_method must be one of {allowed_error_methods}"
        )

    if velocity_error_mode not in ["global_kp", "free_kp"]:
        raise ValueError(
            "velocity_error_mode must be either 'global_kp' or 'free_kp'"
        )

    # Backwards compatibility:
    # If someone still passes estimate_errors=True but asks for no method,
    # interpret that as the old bootstrap behaviour.
    if estimate_errors and velocity_error_method == "none":
        velocity_error_method = "bootstrap"

    do_snr_drop_errors = velocity_error_method in ["snr_drop", "both"]
    do_bootstrap_errors = velocity_error_method in ["bootstrap", "both"]

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------
    def _as_scalar(x, dtype=float):
        x = np.asarray(x)
        if x.size == 0:
            return np.nan
        return dtype(np.ravel(x)[0])

    def _as_int_scalar(x):
        return int(_as_scalar(x, dtype=float))

    def _select_in_transit_array(arr, name):
        """
        Return array with length n_spec.

        Accepts:
            - already in-transit array, length n_spec
            - full array + with_signal mask/indices
        """
        if arr is None:
            return None

        arr = np.asarray(arr)

        if arr.shape[0] == n_spec:
            return arr

        if with_signal is None:
            if diagnostic:
                print(
                    f"[diagnostic warning] {name} has length {arr.shape[0]}, "
                    f"but n_spec={n_spec} and with_signal was not provided. "
                    f"Skipping {name} diagnostics."
                )
            return None

        ws = np.asarray(with_signal)

        try:
            selected = arr[ws]
        except Exception as e:
            if diagnostic:
                print(
                    f"[diagnostic warning] Could not apply with_signal to {name}: {e}"
                )
            return None

        if selected.shape[0] != n_spec:
            if diagnostic:
                print(
                    f"[diagnostic warning] {name}[with_signal] has length "
                    f"{selected.shape[0]}, but n_spec={n_spec}. "
                    f"Skipping {name} diagnostics."
                )
            return None

        return selected

    def _get_velocity_floor():
        """
        Return velocity floor in km/s.
        """
        if velocity_error_floor is None:
            return 0.0

        if isinstance(velocity_error_floor, (int, float)):
            return float(velocity_error_floor)

        if isinstance(velocity_error_floor, str):
            mode = velocity_error_floor.lower()

            if mode == "ccf_step":
                dv = np.nanmedian(np.abs(np.diff(v_rest)))
                if np.isfinite(dv):
                    return float(dv)
                return 0.0

            if mode == "resolution":
                R = None
                for key in ["Resolving_power", "R", "resolution", "Resolution"]:
                    if key in inp_dat:
                        try:
                            R = float(inp_dat[key])
                            break
                        except Exception:
                            pass

                if R is not None and np.isfinite(R) and R > 0:
                    c_kms = 299792.458
                    return float(c_kms / R)

                if diagnostic:
                    print(
                        "[diagnostic warning] velocity_error_floor='resolution' "
                        "was requested, but no valid resolving power was found in inp_dat. "
                        "No resolution floor applied."
                    )
                return 0.0

        raise ValueError(
            "velocity_error_floor must be None, a number, 'ccf_step', or 'resolution'"
        )

    velocity_floor_value = _get_velocity_floor()

    def _peak_diagnostics(y, x, xlim_use=None, subpixel=True):
        """
        Peak information for a 1D CCF.

        Uses the maximum positive peak inside xlim_use.
        If subpixel=True, the peak velocity is refined with a local
        quadratic interpolation around the grid maximum.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if xlim_use is None:
            mask = np.ones_like(x, dtype=bool)
        else:
            mask = (x >= xlim_use[0]) & (x <= xlim_use[1])

        if not np.any(mask):
            return {
                "peak_snr": np.nan,
                "peak_vrest": np.nan,
                "peak_vrest_grid": np.nan,
                "peak_index_local": None,
                "peak_index_global": None,
                "min_snr": np.nan,
                "min_vrest": np.nan
            }

        idx_global = np.where(mask)[0]
        xx = x[mask]
        yy = y[mask]

        good = np.isfinite(xx) & np.isfinite(yy)

        if np.sum(good) < 3:
            return {
                "peak_snr": np.nan,
                "peak_vrest": np.nan,
                "peak_vrest_grid": np.nan,
                "peak_index_local": None,
                "peak_index_global": None,
                "min_snr": np.nan,
                "min_vrest": np.nan
            }

        idx_global = idx_global[good]
        xx = xx[good]
        yy = yy[good]

        imax = int(np.nanargmax(yy))
        imin = int(np.nanargmin(yy))

        peak_snr = float(yy[imax])
        peak_vrest_grid = float(xx[imax])
        peak_vrest = peak_vrest_grid

        if subpixel and imax > 0 and imax < len(xx) - 1:
            x3 = xx[imax - 1:imax + 2]
            y3 = yy[imax - 1:imax + 2]

            try:
                coeff = np.polyfit(x3, y3, 2)
                a, b, c = coeff

                if np.isfinite(a) and a < 0:
                    xv = -b / (2.0 * a)
                    yv = a * xv**2 + b * xv + c

                    if np.min(x3) <= xv <= np.max(x3):
                        peak_vrest = float(xv)
                        peak_snr = float(yv)

            except Exception:
                pass

        return {
            "peak_snr": peak_snr,
            "peak_vrest": peak_vrest,
            "peak_vrest_grid": peak_vrest_grid,
            "peak_index_local": imax,
            "peak_index_global": int(idx_global[imax]),
            "min_snr": float(yy[imin]),
            "min_vrest": float(xx[imin])
        }

    def _interpolate_crossing(x1, y1, x2, y2, y_target):
        """
        Linear interpolation for x where y crosses y_target.
        """
        if not np.isfinite(x1 + y1 + x2 + y2 + y_target):
            return np.nan

        if y2 == y1:
            return np.nan

        frac = (y_target - y1) / (y2 - y1)
        return float(x1 + frac * (x2 - x1))

    def _snr_drop_error(y, x, peak_info, xlim_use=None):
        """
        Estimate asymmetric velocity errors from the interval where the CCF
        drops by `snr_drop` from the peak value.

        The search is done on the grid CCF. The peak itself may be subpixel,
        but crossings are found by linear interpolation on the sampled CCF.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if xlim_use is None:
            mask = np.ones_like(x, dtype=bool)
        else:
            mask = (x >= xlim_use[0]) & (x <= xlim_use[1])

        if not np.any(mask):
            return {
                "method": "snr_drop",
                "snr_drop": float(snr_drop),
                "velocity_floor": float(velocity_floor_value),
                "v_left": np.nan,
                "v_right": np.nan,
                "err_minus_raw": np.nan,
                "err_plus_raw": np.nan,
                "err_minus": np.nan,
                "err_plus": np.nan,
                "err_symmetric": np.nan,
                "target_snr": np.nan
            }

        xx = x[mask]
        yy = y[mask]

        good = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[good]
        yy = yy[good]

        if len(xx) < 3:
            return {
                "method": "snr_drop",
                "snr_drop": float(snr_drop),
                "velocity_floor": float(velocity_floor_value),
                "v_left": np.nan,
                "v_right": np.nan,
                "err_minus_raw": np.nan,
                "err_plus_raw": np.nan,
                "err_minus": np.nan,
                "err_plus": np.nan,
                "err_symmetric": np.nan,
                "target_snr": np.nan
            }

        imax = int(np.nanargmax(yy))
        peak_snr = float(peak_info["peak_snr"])
        peak_v = float(peak_info["peak_vrest"])

        target = peak_snr - float(snr_drop)

        v_left = np.nan
        v_right = np.nan

        # Search left from the peak grid index
        for j in range(imax - 1, -1, -1):
            if yy[j] <= target:
                v_left = _interpolate_crossing(
                    xx[j], yy[j],
                    xx[j + 1], yy[j + 1],
                    target
                )
                break

        # Search right from the peak grid index
        for j in range(imax + 1, len(xx)):
            if yy[j] <= target:
                v_right = _interpolate_crossing(
                    xx[j - 1], yy[j - 1],
                    xx[j], yy[j],
                    target
                )
                break

        err_minus_raw = np.nan
        err_plus_raw = np.nan

        if np.isfinite(v_left):
            err_minus_raw = peak_v - v_left

        if np.isfinite(v_right):
            err_plus_raw = v_right - peak_v

        err_minus = err_minus_raw
        err_plus = err_plus_raw

        if np.isfinite(err_minus):
            err_minus = max(float(err_minus), float(velocity_floor_value))

        if np.isfinite(err_plus):
            err_plus = max(float(err_plus), float(velocity_floor_value))

        if np.isfinite(err_minus) and np.isfinite(err_plus):
            err_symmetric = 0.5 * (err_minus + err_plus)
        elif np.isfinite(err_minus):
            err_symmetric = err_minus
        elif np.isfinite(err_plus):
            err_symmetric = err_plus
        else:
            err_symmetric = np.nan

        return {
            "method": "snr_drop",
            "snr_drop": float(snr_drop),
            "velocity_floor": float(velocity_floor_value),
            "v_left": float(v_left) if np.isfinite(v_left) else np.nan,
            "v_right": float(v_right) if np.isfinite(v_right) else np.nan,
            "err_minus_raw": float(err_minus_raw) if np.isfinite(err_minus_raw) else np.nan,
            "err_plus_raw": float(err_plus_raw) if np.isfinite(err_plus_raw) else np.nan,
            "err_minus": float(err_minus) if np.isfinite(err_minus) else np.nan,
            "err_plus": float(err_plus) if np.isfinite(err_plus) else np.nan,
            "err_symmetric": float(err_symmetric) if np.isfinite(err_symmetric) else np.nan,
            "target_snr": float(target)
        }

    def _bootstrap_interval_errors(idx, fixed_kp_idx):
        """
        Bootstrap over spectra inside one interval.

        With velocity_error_mode='global_kp', this estimates errors on the
        same fixed-Kp CCF that is plotted.
        With velocity_error_mode='free_kp', this allows the Kp to vary in
        each bootstrap realization.
        """
        if (not do_bootstrap_errors) or n_bootstrap is None or n_bootstrap <= 1:
            return None

        idx = np.asarray(idx, dtype=int)
        n_this = len(idx)

        if n_this < 3:
            return None

        rng = np.random.default_rng(bootstrap_seed)

        boot_peak_v = []
        boot_peak_v_grid = []
        boot_peak_snr = []
        boot_kp = []
        boot_kp_idx = []

        for _ in range(int(n_bootstrap)):

            boot_idx = rng.choice(idx, size=n_this, replace=True)
            ccf_boot = np.sum(ccf_values_shift[:, boot_idx, :], axis=1)

            try:
                ccf_boot_sig, boot_max_sig, boot_max_kp_idx, boot_max_v_wind, _ = get_max_CCF_peak(
                    inp_dat,
                    ccf_boot,
                    v_rest,
                    kp_range,
                    b=None,
                    stats=None,
                    sysrem_opt=sysrem_opt,
                    CCF_Noise=False
                )

                boot_max_kp_idx = _as_int_scalar(boot_max_kp_idx)

                if velocity_error_mode == "global_kp":
                    use_kp_idx = fixed_kp_idx
                else:
                    use_kp_idx = boot_max_kp_idx

                if use_kp_idx < 0 or use_kp_idx >= n_kp:
                    continue

                y_boot = ccf_boot_sig[:, use_kp_idx]

                pinfo = _peak_diagnostics(
                    y_boot,
                    v_rest,
                    xlim_use=error_xlim,
                    subpixel=subpixel_peak
                )

                if np.isfinite(pinfo["peak_vrest"]):
                    boot_peak_v.append(pinfo["peak_vrest"])
                    boot_peak_v_grid.append(pinfo["peak_vrest_grid"])
                    boot_peak_snr.append(pinfo["peak_snr"])
                    boot_kp.append(kp_range[use_kp_idx])
                    boot_kp_idx.append(use_kp_idx)

            except Exception:
                continue

        boot_peak_v = np.asarray(boot_peak_v, dtype=float)
        boot_peak_v_grid = np.asarray(boot_peak_v_grid, dtype=float)
        boot_peak_snr = np.asarray(boot_peak_snr, dtype=float)
        boot_kp = np.asarray(boot_kp, dtype=float)
        boot_kp_idx = np.asarray(boot_kp_idx, dtype=int)

        out = {
            "method": "bootstrap",
            "n_bootstrap_requested": int(n_bootstrap),
            "n_bootstrap_used": int(boot_peak_v.size),
            "velocity_error_mode": velocity_error_mode,
            "error_xlim": error_xlim,
            "subpixel_peak": bool(subpixel_peak),
            "vrest_std": np.nan,
            "vrest_p16": np.nan,
            "vrest_p50": np.nan,
            "vrest_p84": np.nan,
            "vrest_grid_std": np.nan,
            "snr_std": np.nan,
            "snr_p16": np.nan,
            "snr_p50": np.nan,
            "snr_p84": np.nan,
            "kp_std": np.nan,
            "kp_p16": np.nan,
            "kp_p50": np.nan,
            "kp_p84": np.nan,
            "boot_peak_vrest": boot_peak_v,
            "boot_peak_vrest_grid": boot_peak_v_grid,
            "boot_peak_snr": boot_peak_snr,
            "boot_kp": boot_kp,
            "boot_kp_idx": boot_kp_idx
        }

        if boot_peak_v.size < max(20, int(0.1 * int(n_bootstrap))):
            return out

        out.update({
            "vrest_std": float(np.nanstd(boot_peak_v, ddof=1)),
            "vrest_p16": float(np.nanpercentile(boot_peak_v, 16)),
            "vrest_p50": float(np.nanpercentile(boot_peak_v, 50)),
            "vrest_p84": float(np.nanpercentile(boot_peak_v, 84)),
            "vrest_grid_std": float(np.nanstd(boot_peak_v_grid, ddof=1)),
            "snr_std": float(np.nanstd(boot_peak_snr, ddof=1)),
            "snr_p16": float(np.nanpercentile(boot_peak_snr, 16)),
            "snr_p50": float(np.nanpercentile(boot_peak_snr, 50)),
            "snr_p84": float(np.nanpercentile(boot_peak_snr, 84)),
            "kp_std": float(np.nanstd(boot_kp, ddof=1)),
            "kp_p16": float(np.nanpercentile(boot_kp, 16)),
            "kp_p50": float(np.nanpercentile(boot_kp, 50)),
            "kp_p84": float(np.nanpercentile(boot_kp, 84))
        })

        return out

    def _get_metric_velocity_error(res):
        """Return the velocity uncertainty used by the HRS-LSM metric."""
        diag = res.get("diagnostics", {})
        boot = res.get("bootstrap_errors", None)

        if lsm_error_source not in ("auto", "bootstrap", "snr_drop"):
            raise ValueError("lsm_error_source must be 'auto', 'bootstrap', or 'snr_drop'")

        if lsm_error_source in ("auto", "bootstrap"):
            if boot is not None:
                err = boot.get("vrest_std", np.nan)
                if np.isfinite(err) and err > 0:
                    return float(err), "bootstrap"
            if lsm_error_source == "bootstrap":
                return np.nan, "bootstrap"

        if lsm_error_source in ("auto", "snr_drop"):
            err = diag.get("global_kp_peak_vrest_err", np.nan)
            if np.isfinite(err) and err > 0:
                return float(err), "snr_drop"
            if lsm_error_source == "snr_drop":
                return np.nan, "snr_drop"

        return np.nan, "none"

    def _normalised_recentred_ccf(res, half_window):
        """Return a peak-centred, peak-normalised CCF segment for shape tests."""
        ccf = np.asarray(res["ccf_1d_global_kp"], dtype=float)
        peak_v = res["diagnostics"].get("global_kp_peak_vrest", np.nan)
        peak_snr = res["diagnostics"].get("global_kp_peak_snr", np.nan)

        if not np.isfinite(peak_v) or not np.isfinite(peak_snr) or peak_snr == 0:
            return None, None

        x = np.asarray(v_rest, dtype=float) - float(peak_v)
        mask = np.isfinite(x) & np.isfinite(ccf) & (np.abs(x) <= float(half_window))

        if np.count_nonzero(mask) < 5:
            return None, None

        y = ccf[mask].astype(float)
        y = y - np.nanmedian(y)

        norm = np.nanmax(np.abs(y))
        if not np.isfinite(norm) or norm <= 0:
            return None, None

        return x[mask], y / norm

    def _compute_hrs_lsm_metric(res_a, res_b, name_a, name_b):
        """
        Compute an LSM-like metric for high-resolution CCFs.

        The low-resolution LSM idea is translated here into the HRS observable:
        a phase-dependent change in the CCF peak position and/or shape. The
        first term quantifies the velocity separation of two limb-dominated CCFs
        in units of the measured peak-position uncertainty. The second term is a
        detection-weighted version of the same quantity, penalising cases where
        only one limb is significantly detected. The third term compares the
        peak-centred CCF morphology, useful for double-peaked or jet-like cases.
        """
        diag_a = res_a.get("diagnostics", {})
        diag_b = res_b.get("diagnostics", {})

        va = diag_a.get("global_kp_peak_vrest", np.nan)
        vb = diag_b.get("global_kp_peak_vrest", np.nan)
        sa = diag_a.get("global_kp_peak_snr", np.nan)
        sb = diag_b.get("global_kp_peak_snr", np.nan)

        ea, ea_source = _get_metric_velocity_error(res_a)
        eb, eb_source = _get_metric_velocity_error(res_b)

        delta_v = np.nan
        delta_v_err = np.nan
        velocity_lsm_signed = np.nan
        velocity_lsm_abs = np.nan

        if np.isfinite(va) and np.isfinite(vb):
            delta_v = float(va - vb)

        if np.isfinite(ea) and np.isfinite(eb):
            delta_v_err = float(np.sqrt(ea**2 + eb**2))

        if np.isfinite(delta_v) and np.isfinite(delta_v_err) and delta_v_err > 0:
            velocity_lsm_signed = float(delta_v / delta_v_err)
            velocity_lsm_abs = float(abs(velocity_lsm_signed))

        min_peak_snr = np.nanmin([abs(sa), abs(sb)])
        mean_peak_snr = np.nanmean([abs(sa), abs(sb)])

        if np.isfinite(min_peak_snr) and float(lsm_min_detection_snr) > 0:
            detection_weight = float(min_peak_snr / float(lsm_min_detection_snr))
        else:
            detection_weight = np.nan

        if np.isfinite(velocity_lsm_abs) and np.isfinite(detection_weight):
            velocity_lsm_detection_weighted = float(velocity_lsm_abs * detection_weight)
        else:
            velocity_lsm_detection_weighted = np.nan

        x_a, y_a = _normalised_recentred_ccf(res_a, lsm_shape_window_kms)
        x_b, y_b = _normalised_recentred_ccf(res_b, lsm_shape_window_kms)

        shape_l2 = np.nan
        shape_corr = np.nan
        shape_lsm_detection_weighted = np.nan

        if x_a is not None and x_b is not None:
            dx = np.nanmedian(np.diff(np.sort(np.unique(v_rest))))
            if not np.isfinite(dx) or dx <= 0:
                dx = 0.5

            x_common = np.arange(
                -float(lsm_shape_window_kms),
                float(lsm_shape_window_kms) + 0.5 * dx,
                dx
            )

            ya_common = np.interp(x_common, x_a, y_a, left=np.nan, right=np.nan)
            yb_common = np.interp(x_common, x_b, y_b, left=np.nan, right=np.nan)
            ok = np.isfinite(ya_common) & np.isfinite(yb_common)

            if np.count_nonzero(ok) >= 5:
                diff = ya_common[ok] - yb_common[ok]
                shape_l2 = float(np.sqrt(np.nanmean(diff**2)))

                if np.nanstd(ya_common[ok]) > 0 and np.nanstd(yb_common[ok]) > 0:
                    shape_corr = float(np.corrcoef(ya_common[ok], yb_common[ok])[0, 1])

                if np.isfinite(detection_weight):
                    shape_lsm_detection_weighted = float(shape_l2 * detection_weight)

        if np.isfinite(velocity_lsm_detection_weighted) and np.isfinite(shape_lsm_detection_weighted):
            combined_lsm = float(np.sqrt(
                velocity_lsm_detection_weighted**2 + shape_lsm_detection_weighted**2
            ))
        elif np.isfinite(velocity_lsm_detection_weighted):
            combined_lsm = float(velocity_lsm_detection_weighted)
        else:
            combined_lsm = np.nan

        interpretation = "unclassified"
        if np.isfinite(velocity_lsm_abs):
            if velocity_lsm_abs >= 5:
                interpretation = "strong limb-to-limb velocity asymmetry"
            elif velocity_lsm_abs >= 3:
                interpretation = "moderate limb-to-limb velocity asymmetry"
            elif velocity_lsm_abs >= 1:
                interpretation = "weak/marginal limb-to-limb velocity asymmetry"
            else:
                interpretation = "no significant velocity asymmetry"

        return {
            "definition": (
                "HRS_LSM_velocity = (V_peak_A - V_peak_B) / "
                "sqrt(sigma_A^2 + sigma_B^2), measured from global-Kp 1D CCFs. "
                "The detection-weighted version multiplies |HRS_LSM_velocity| "
                "by min(SNR_A, SNR_B) / lsm_min_detection_snr. "
                "HRS_LSM_shape compares the peak-centred, normalised CCF profiles."
            ),
            "interval_A": str(name_a),
            "interval_B": str(name_b),
            "vrest_A": float(va) if np.isfinite(va) else np.nan,
            "vrest_B": float(vb) if np.isfinite(vb) else np.nan,
            "peak_snr_A": float(sa) if np.isfinite(sa) else np.nan,
            "peak_snr_B": float(sb) if np.isfinite(sb) else np.nan,
            "vrest_error_A": float(ea) if np.isfinite(ea) else np.nan,
            "vrest_error_B": float(eb) if np.isfinite(eb) else np.nan,
            "vrest_error_source_A": ea_source,
            "vrest_error_source_B": eb_source,
            "delta_v_A_minus_B": float(delta_v) if np.isfinite(delta_v) else np.nan,
            "delta_v_error": float(delta_v_err) if np.isfinite(delta_v_err) else np.nan,
            "HRS_LSM_velocity_signed": float(velocity_lsm_signed) if np.isfinite(velocity_lsm_signed) else np.nan,
            "HRS_LSM_velocity_abs": float(velocity_lsm_abs) if np.isfinite(velocity_lsm_abs) else np.nan,
            "min_peak_snr": float(min_peak_snr) if np.isfinite(min_peak_snr) else np.nan,
            "mean_peak_snr": float(mean_peak_snr) if np.isfinite(mean_peak_snr) else np.nan,
            "lsm_min_detection_snr": float(lsm_min_detection_snr),
            "detection_weight": float(detection_weight) if np.isfinite(detection_weight) else np.nan,
            "HRS_LSM_velocity_detection_weighted": (
                float(velocity_lsm_detection_weighted)
                if np.isfinite(velocity_lsm_detection_weighted) else np.nan
            ),
            "HRS_LSM_shape_l2": float(shape_l2) if np.isfinite(shape_l2) else np.nan,
            "HRS_LSM_shape_corr": float(shape_corr) if np.isfinite(shape_corr) else np.nan,
            "HRS_LSM_shape_detection_weighted": (
                float(shape_lsm_detection_weighted)
                if np.isfinite(shape_lsm_detection_weighted) else np.nan
            ),
            "HRS_LSM_combined": float(combined_lsm) if np.isfinite(combined_lsm) else np.nan,
            "lsm_shape_window_kms": float(lsm_shape_window_kms),
            "interpretation": interpretation
        }

    # ------------------------------------------------------------
    # Prepare optional diagnostic arrays
    # ------------------------------------------------------------
    phase_in = _select_in_transit_array(phase, "phase")
    v_planet_in = _select_in_transit_array(v_planet, "v_planet")

    # ------------------------------------------------------------
    # Determine global Kp index from the full transit if needed
    # ------------------------------------------------------------
    full_ccf_tot = np.sum(ccf_values_shift, axis=1)

    full_ccf_sig, full_max_sig, full_max_kp_idx, full_max_v_wind, _ = get_max_CCF_peak(
        inp_dat,
        full_ccf_tot,
        v_rest,
        kp_range,
        b=None,
        stats=None,
        sysrem_opt=sysrem_opt,
        CCF_Noise=False
    )

    full_max_sig = _as_scalar(full_max_sig)
    full_max_kp_idx = _as_int_scalar(full_max_kp_idx)
    full_max_v_wind = _as_scalar(full_max_v_wind)

    if max_kp_idx is None:
        max_kp_idx = full_max_kp_idx
    else:
        max_kp_idx = int(max_kp_idx)

    if max_kp_idx < 0 or max_kp_idx >= n_kp:
        raise ValueError(
            f"max_kp_idx={max_kp_idx} is outside allowed range 0--{n_kp - 1}"
        )

    # ------------------------------------------------------------
    # Define time intervals
    # ------------------------------------------------------------
    mid = n_spec // 2

    if sort_by_phase and phase_in is not None:
        order = np.argsort(phase_in)

        intervals = {
            "Ingress": order[:ingress_n],
            "First half": order[:mid],
            "Second half": order[mid:],
            "Egress": order[-egress_n:]
        }

        interval_definition = "phase-sorted"

    else:
        intervals = {
            "Ingress": np.arange(0, ingress_n),
            "First half": np.arange(0, mid),
            "Second half": np.arange(mid, n_spec),
            "Egress": np.arange(n_spec - egress_n, n_spec)
        }

        interval_definition = "array-order"

        if sort_by_phase and phase_in is None and diagnostic:
            print(
                "[diagnostic warning] sort_by_phase=True was requested, "
                "but no usable phase array was provided. Falling back to array-order intervals."
            )

    # ------------------------------------------------------------
    # Default colors
    # ------------------------------------------------------------
    default_colors = {
        "Ingress": "tab:blue",
        "First half": "tab:cyan",
        "Second half": "tab:orange",
        "Egress": "tab:red"
    }

    if colors is None:
        colors_dict = default_colors

    elif isinstance(colors, dict):
        colors_dict = default_colors.copy()
        colors_dict.update(colors)

    else:
        color_list = list(colors)
        interval_names = list(intervals.keys())

        if len(color_list) < len(interval_names):
            raise ValueError(
                "If colors is a list/tuple, it must contain at least "
                f"{len(interval_names)} colors."
            )

        colors_dict = {
            name: color_list[i]
            for i, name in enumerate(interval_names)
        }

    # ------------------------------------------------------------
    # Global diagnostics
    # ------------------------------------------------------------
    diagnostics = {
        "ccf_values_shift_shape": ccf_values_shift.shape,
        "n_vrest": int(n_v),
        "n_spectra_with_signal": int(n_spec),
        "n_kp": int(n_kp),
        "interval_definition": interval_definition,
        "sort_by_phase": bool(sort_by_phase),
        "full_max_sig": float(full_max_sig),
        "full_max_kp_idx": int(full_max_kp_idx),
        "full_max_kp": float(kp_range[full_max_kp_idx]),
        "full_max_v_wind": float(full_max_v_wind),
        "used_global_kp_idx": int(max_kp_idx),
        "used_global_kp": float(kp_range[max_kp_idx]),
        "velocity_error_method": velocity_error_method,
        "snr_drop": float(snr_drop),
        "velocity_error_floor": velocity_error_floor,
        "velocity_error_floor_value": float(velocity_floor_value),
        "n_bootstrap": int(n_bootstrap) if n_bootstrap is not None else None,
        "bootstrap_seed": int(bootstrap_seed),
        "error_xlim": error_xlim,
        "velocity_error_mode": velocity_error_mode,
        "subpixel_peak": bool(subpixel_peak)
    }

    if phase_in is not None:
        dphase = np.diff(phase_in)
        diagnostics["phase_first"] = float(phase_in[0])
        diagnostics["phase_last"] = float(phase_in[-1])
        diagnostics["phase_min"] = float(np.nanmin(phase_in))
        diagnostics["phase_max"] = float(np.nanmax(phase_in))
        diagnostics["phase_monotonic_increasing"] = bool(np.all(dphase >= 0))
        diagnostics["phase_monotonic_decreasing"] = bool(np.all(dphase <= 0))
        diagnostics["phase_n_negative_steps"] = int(np.sum(dphase < 0))
        diagnostics["phase_n_positive_steps"] = int(np.sum(dphase > 0))

    if v_planet_in is not None:
        diagnostics["v_planet_first"] = float(v_planet_in[0])
        diagnostics["v_planet_last"] = float(v_planet_in[-1])
        diagnostics["v_planet_min"] = float(np.nanmin(v_planet_in))
        diagnostics["v_planet_max"] = float(np.nanmax(v_planet_in))

    if diagnostic:
        print("\n" + "=" * 72)
        print("TIME-RESOLVED CCF DIAGNOSTICS")
        print("=" * 72)
        print(f"ccf_values_shift shape = {ccf_values_shift.shape}")
        print("Assumed axes: (Vrest, in-transit spectrum index, Kp)")
        print(f"len(v_rest)  = {len(v_rest)}")
        print(f"len(kp_range)= {len(kp_range)}")
        print(f"Interval definition = {interval_definition}")
        print(f"Full-transit best Kp index = {full_max_kp_idx}")
        print(f"Full-transit best Kp       = {kp_range[full_max_kp_idx]:.3f} km/s")
        print(f"Full-transit best Vrest    = {full_max_v_wind:.3f} km/s")
        print(f"Full-transit max S/N       = {full_max_sig:.3f}")
        print(f"Using global Kp index      = {max_kp_idx}")
        print(f"Using global Kp            = {kp_range[max_kp_idx]:.3f} km/s")
        print(f"Velocity error method      = {velocity_error_method}")
        print(f"S/N-drop value             = {snr_drop:.3f}")
        print(f"Velocity error floor       = {velocity_error_floor}")
        print(f"Velocity floor value       = {velocity_floor_value:.3f} km/s")
        print(f"Error peak xlim            = {error_xlim}")
        print(f"Subpixel peak refinement   = {subpixel_peak}")

        if do_bootstrap_errors:
            print(f"Bootstrap realizations     = {n_bootstrap}")
            print(f"Bootstrap mode             = {velocity_error_mode}")

        if phase_in is not None:
            print("\nPhase diagnostics:")
            print(f"  phase[0]  = {phase_in[0]: .8f}")
            print(f"  phase[-1] = {phase_in[-1]: .8f}")
            print(f"  phase min = {np.nanmin(phase_in): .8f}")
            print(f"  phase max = {np.nanmax(phase_in): .8f}")
            print(f"  monotonic increasing = {diagnostics['phase_monotonic_increasing']}")
            print(f"  monotonic decreasing = {diagnostics['phase_monotonic_decreasing']}")
            print(f"  negative phase steps = {diagnostics['phase_n_negative_steps']}")
            print(f"  positive phase steps = {diagnostics['phase_n_positive_steps']}")

            if not diagnostics["phase_monotonic_increasing"]:
                print(
                    "  WARNING: phase is not monotonically increasing. "
                    "Array-order ingress/egress labels may be wrong."
                )

        else:
            print(
                "\nPhase diagnostics unavailable. Pass phase=phase and with_signal=with_signal "
                "to verify ingress/egress ordering."
            )

        if v_planet_in is not None:
            print("\nPlanet velocity diagnostics:")
            print(f"  v_planet[0]  = {v_planet_in[0]: .3f} km/s")
            print(f"  v_planet[-1] = {v_planet_in[-1]: .3f} km/s")
            print(f"  v_planet min = {np.nanmin(v_planet_in): .3f} km/s")
            print(f"  v_planet max = {np.nanmax(v_planet_in): .3f} km/s")

        print("=" * 72 + "\n")

    # ------------------------------------------------------------
    # Compute CCFs for each interval
    # ------------------------------------------------------------
    results = {}

    for name, idx in intervals.items():

        idx = np.asarray(idx, dtype=int)

        ccf_interval = np.sum(ccf_values_shift[:, idx, :], axis=1)

        ccf_interval_sig, max_sig, interval_max_kp_idx, max_v_wind, _ = get_max_CCF_peak(
            inp_dat,
            ccf_interval,
            v_rest,
            kp_range,
            b=None,
            stats=None,
            sysrem_opt=sysrem_opt,
            CCF_Noise=False
        )

        max_sig = _as_scalar(max_sig)
        interval_max_kp_idx = _as_int_scalar(interval_max_kp_idx)
        max_v_wind = _as_scalar(max_v_wind)

        ccf_1d_global_kp = ccf_interval_sig[:, max_kp_idx]
        ccf_1d_free_kp = ccf_interval_sig[:, interval_max_kp_idx]

        global_peak_info = _peak_diagnostics(
            ccf_1d_global_kp,
            v_rest,
            xlim_use=error_xlim,
            subpixel=subpixel_peak
        )

        free_peak_info = _peak_diagnostics(
            ccf_1d_free_kp,
            v_rest,
            xlim_use=error_xlim,
            subpixel=subpixel_peak
        )

        snr_drop_errors = None
        if do_snr_drop_errors:
            snr_drop_errors = _snr_drop_error(
                ccf_1d_global_kp,
                v_rest,
                global_peak_info,
                xlim_use=error_xlim
            )

        bootstrap_errors = None
        if do_bootstrap_errors:
            bootstrap_errors = _bootstrap_interval_errors(
                idx=idx,
                fixed_kp_idx=max_kp_idx
            )

        interval_diag = {
            "index_first": int(idx[0]),
            "index_last": int(idx[-1]),
            "index_min": int(np.min(idx)),
            "index_max": int(np.max(idx)),
            "n_spectra": int(len(idx)),
            "global_kp_peak_snr": global_peak_info["peak_snr"],
            "global_kp_peak_vrest": global_peak_info["peak_vrest"],
            "global_kp_peak_vrest_grid": global_peak_info["peak_vrest_grid"],
            "global_kp_min_snr": global_peak_info["min_snr"],
            "global_kp_min_vrest": global_peak_info["min_vrest"],
            "free_kp_peak_snr": free_peak_info["peak_snr"],
            "free_kp_peak_vrest": free_peak_info["peak_vrest"],
            "free_kp_peak_vrest_grid": free_peak_info["peak_vrest_grid"],
            "free_kp_min_snr": free_peak_info["min_snr"],
            "free_kp_min_vrest": free_peak_info["min_vrest"]
        }

        if snr_drop_errors is not None:
            interval_diag["snr_drop_errors"] = snr_drop_errors
            interval_diag["global_kp_peak_vrest_err_minus"] = snr_drop_errors["err_minus"]
            interval_diag["global_kp_peak_vrest_err_plus"] = snr_drop_errors["err_plus"]
            interval_diag["global_kp_peak_vrest_err"] = snr_drop_errors["err_symmetric"]

        if bootstrap_errors is not None:
            interval_diag["bootstrap_errors"] = bootstrap_errors
            interval_diag["bootstrap_global_kp_peak_vrest_err"] = bootstrap_errors["vrest_std"]
            interval_diag["bootstrap_global_kp_peak_snr_err"] = bootstrap_errors["snr_std"]
            interval_diag["bootstrap_global_kp_peak_vrest_p16"] = bootstrap_errors["vrest_p16"]
            interval_diag["bootstrap_global_kp_peak_vrest_p50"] = bootstrap_errors["vrest_p50"]
            interval_diag["bootstrap_global_kp_peak_vrest_p84"] = bootstrap_errors["vrest_p84"]
            interval_diag["bootstrap_global_kp_peak_snr_p16"] = bootstrap_errors["snr_p16"]
            interval_diag["bootstrap_global_kp_peak_snr_p50"] = bootstrap_errors["snr_p50"]
            interval_diag["bootstrap_global_kp_peak_snr_p84"] = bootstrap_errors["snr_p84"]

        if phase_in is not None:
            ph = phase_in[idx]
            interval_diag["phase_first"] = float(ph[0])
            interval_diag["phase_last"] = float(ph[-1])
            interval_diag["phase_min"] = float(np.nanmin(ph))
            interval_diag["phase_max"] = float(np.nanmax(ph))
            interval_diag["phase_median"] = float(np.nanmedian(ph))

        if v_planet_in is not None:
            vp = v_planet_in[idx]
            interval_diag["v_planet_first"] = float(vp[0])
            interval_diag["v_planet_last"] = float(vp[-1])
            interval_diag["v_planet_min"] = float(np.nanmin(vp))
            interval_diag["v_planet_max"] = float(np.nanmax(vp))
            interval_diag["v_planet_median"] = float(np.nanmedian(vp))

        results[name] = {
            "indices": idx,
            "ccf_tot": ccf_interval,
            "ccf_sig": ccf_interval_sig,
            "ccf_1d_global_kp": ccf_1d_global_kp,
            "ccf_1d_free_kp": ccf_1d_free_kp,
            "global_kp_idx": int(max_kp_idx),
            "global_kp": float(kp_range[max_kp_idx]),
            "interval_max_kp_idx": int(interval_max_kp_idx),
            "interval_max_kp": float(kp_range[interval_max_kp_idx]),
            "max_sig": float(max_sig),
            "max_v_wind": float(max_v_wind),
            "snr_drop_errors": snr_drop_errors,
            "bootstrap_errors": bootstrap_errors,
            "diagnostics": interval_diag
        }

        if diagnostic:
            print(f"{name:12s}:")
            print(f"  Nspec                  = {len(idx)}")
            print(f"  index range             = {idx[0]} -> {idx[-1]} "
                  f"(min={np.min(idx)}, max={np.max(idx)})")

            if phase_in is not None:
                print(f"  phase first/last        = {interval_diag['phase_first']:.8f} "
                      f"-> {interval_diag['phase_last']:.8f}")
                print(f"  phase min/max/median    = {interval_diag['phase_min']:.8f}, "
                      f"{interval_diag['phase_max']:.8f}, "
                      f"{interval_diag['phase_median']:.8f}")

            if v_planet_in is not None:
                print(f"  v_planet first/last     = {interval_diag['v_planet_first']:.3f} "
                      f"-> {interval_diag['v_planet_last']:.3f} km/s")
                print(f"  v_planet min/max/median = {interval_diag['v_planet_min']:.3f}, "
                      f"{interval_diag['v_planet_max']:.3f}, "
                      f"{interval_diag['v_planet_median']:.3f} km/s")

            print(f"  free max S/N            = {max_sig:.3f}")
            print(f"  free max Kp             = {kp_range[interval_max_kp_idx]:.3f} km/s")
            print(f"  free max Vrest          = {max_v_wind:.3f} km/s")
            print(f"  global-Kp positive peak = {global_peak_info['peak_snr']:.3f} "
                  f"at Vrest = {global_peak_info['peak_vrest']:.3f} km/s")
            print(f"  global-Kp grid peak     = {global_peak_info['peak_vrest_grid']:.3f} km/s")
            print(f"  global-Kp most negative = {global_peak_info['min_snr']:.3f} "
                  f"at Vrest = {global_peak_info['min_vrest']:.3f} km/s")

            if snr_drop_errors is not None:
                print(
                    f"  S/N-drop error          = "
                    f"-{snr_drop_errors['err_minus']:.3f} / "
                    f"+{snr_drop_errors['err_plus']:.3f} km/s "
                    f"(floor = {snr_drop_errors['velocity_floor']:.3f} km/s)"
                )
                print(
                    f"  S/N-drop interval       = "
                    f"[{snr_drop_errors['v_left']:.3f}, "
                    f"{snr_drop_errors['v_right']:.3f}] km/s "
                    f"at CCF = peak - {snr_drop:.2f}"
                )

            if bootstrap_errors is not None:
                print(
                    f"  bootstrap Vrest error   = ±{bootstrap_errors['vrest_std']:.3f} km/s "
                    f"[16,50,84] = "
                    f"[{bootstrap_errors['vrest_p16']:.3f}, "
                    f"{bootstrap_errors['vrest_p50']:.3f}, "
                    f"{bootstrap_errors['vrest_p84']:.3f}] km/s"
                )
                print(
                    f"  bootstrap S/N error     = ±{bootstrap_errors['snr_std']:.3f} "
                    f"[16,50,84] = "
                    f"[{bootstrap_errors['snr_p16']:.3f}, "
                    f"{bootstrap_errors['snr_p50']:.3f}, "
                    f"{bootstrap_errors['snr_p84']:.3f}]"
                )
                print(
                    f"  bootstrap samples used  = "
                    f"{bootstrap_errors['n_bootstrap_used']} / "
                    f"{bootstrap_errors['n_bootstrap_requested']}"
                )

            print("")

    print(
        f"Full-transit reference: "
        f"S/N = {full_max_sig:.3f}, "
        f"Kp = {kp_range[full_max_kp_idx]:.2f}, "
        f"Vrest = {full_max_v_wind:.2f} km/s"
    )

    # ------------------------------------------------------------
    # Ingress/egress comparison
    # ------------------------------------------------------------
    if diagnostic and "Ingress" in results and "Egress" in results:
        ving = results["Ingress"]["diagnostics"]["global_kp_peak_vrest"]
        vegr = results["Egress"]["diagnostics"]["global_kp_peak_vrest"]

        print("\nIngress/Egress global-Kp peak comparison:")
        print(f"  Ingress peak Vrest = {ving:.3f} km/s")
        print(f"  Egress  peak Vrest = {vegr:.3f} km/s")

        if vegr < ving:
            print("  Egress is more blueshifted than ingress in this plot.")
        elif ving < vegr:
            print("  Ingress is more blueshifted than egress in this plot.")
            print(
                "  If your expected geometry predicts a more blueshifted egress, "
                "check phase ordering, the sign convention in get_shifted_ccf_matrix, "
                "and whether sort_by_phase=True changes the result."
            )
        else:
            print("  Ingress and egress peak at the same Vrest.")

        delta_v = ving - vegr
        delta_info = {
            "delta_v_ingress_minus_egress": float(delta_v),
            "ingress_vrest": float(ving),
            "egress_vrest": float(vegr)
        }

        if do_snr_drop_errors:
            eing = results["Ingress"]["diagnostics"].get("global_kp_peak_vrest_err", np.nan)
            eegr = results["Egress"]["diagnostics"].get("global_kp_peak_vrest_err", np.nan)

            if np.isfinite(eing) and np.isfinite(eegr):
                delta_v_err = np.sqrt(eing**2 + eegr**2)

                print(
                    f"  Delta V = V_ingress - V_egress = "
                    f"{delta_v:.3f} ± {delta_v_err:.3f} km/s "
                    f"(S/N-drop errors)"
                )

                delta_info.update({
                    "delta_v_error_snr_drop": float(delta_v_err),
                    "ingress_vrest_error_snr_drop": float(eing),
                    "egress_vrest_error_snr_drop": float(eegr)
                })

        if do_bootstrap_errors:
            b_ing = results["Ingress"]["bootstrap_errors"]
            b_egr = results["Egress"]["bootstrap_errors"]

            if b_ing is not None and b_egr is not None:
                eing = b_ing["vrest_std"]
                eegr = b_egr["vrest_std"]

                if np.isfinite(eing) and np.isfinite(eegr):
                    delta_v_err = np.sqrt(eing**2 + eegr**2)

                    print(
                        f"  Delta V = V_ingress - V_egress = "
                        f"{delta_v:.3f} ± {delta_v_err:.3f} km/s "
                        f"(bootstrap errors)"
                    )

                    delta_info.update({
                        "delta_v_error_bootstrap": float(delta_v_err),
                        "ingress_vrest_error_bootstrap": float(eing),
                        "egress_vrest_error_bootstrap": float(eegr)
                    })

        results["Ingress_Egress_delta"] = delta_info

        print("")

    # ------------------------------------------------------------
    # HRS-LSM-like metric
    # ------------------------------------------------------------
    if compute_lsm_metric:
        if not isinstance(lsm_reference, (tuple, list)) or len(lsm_reference) != 2:
            raise ValueError("lsm_reference must be a tuple/list with two interval names")

        lsm_a, lsm_b = str(lsm_reference[0]), str(lsm_reference[1])

        if lsm_a in results and lsm_b in results:
            hrs_lsm = _compute_hrs_lsm_metric(
                results[lsm_a],
                results[lsm_b],
                lsm_a,
                lsm_b
            )

            results["HRS_LSM"] = hrs_lsm

            if diagnostic:
                print("HRS-LSM-like metric:")
                print(f"  intervals = {lsm_a} - {lsm_b}")
                print(
                    f"  Delta V = {hrs_lsm['delta_v_A_minus_B']:.3f} ± "
                    f"{hrs_lsm['delta_v_error']:.3f} km/s"
                )
                print(
                    f"  HRS_LSM_velocity = "
                    f"{hrs_lsm['HRS_LSM_velocity_signed']:.3f} "
                    f"(|.| = {hrs_lsm['HRS_LSM_velocity_abs']:.3f})"
                )
                print(
                    f"  HRS_LSM_velocity_detection_weighted = "
                    f"{hrs_lsm['HRS_LSM_velocity_detection_weighted']:.3f}"
                )
                print(
                    f"  HRS_LSM_shape_l2 = {hrs_lsm['HRS_LSM_shape_l2']:.3f}, "
                    f"HRS_LSM_shape_corr = {hrs_lsm['HRS_LSM_shape_corr']:.3f}"
                )
                print(f"  HRS_LSM_combined = {hrs_lsm['HRS_LSM_combined']:.3f}")
                print(f"  interpretation = {hrs_lsm['interpretation']}")
                print("")
        else:
            if diagnostic:
                print(
                    "HRS-LSM-like metric not computed because at least one "
                    f"requested interval is missing: {lsm_reference}"
                )
                print("")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    if plot_halves:
        plot_order = ["Ingress", "First half", "Second half", "Egress"]
    else:
        plot_order = ["Ingress", "Egress"]

    for name in plot_order:
        res = results[name]

        v_peak = res["diagnostics"]["global_kp_peak_vrest"]

        ax.plot(
            v_rest,
            res["ccf_1d_global_kp"],
            lw=4,
            ls="-",
            color=colors_dict[name],
            label=rf"{name} ($V_{{\rm peak}}={v_peak:.1f}$ km s$^{{-1}}$)"
        )

    ax.axhline(0, color="k", lw=1.7, ls="--", alpha=0.4)
    ax.axvline(0, color="k", lw=1.7, ls="--", alpha=0.4)

    ax.set_xlim(xlim)
    ax.set_xlabel(r"$V_{\rm rest}$ [km s$^{-1}$]", fontsize=20)
    ax.set_ylabel("CCF S/N", fontsize=20)

    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=16, frameon=True)

    plt.tight_layout()

    if save_plot:
        if output_file is None:
            output_file = f"{inp_dat['plots_dir']}time_resolved_1D_CCFs.pdf"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    results["diagnostics"] = diagnostics
    results["figure"] = fig

    return results




       
"""           
# BLOCK TO POTENTIALLY SIMULATE NYQUIST-SAMPLED INSTRUMENTS AVOIDING
# THE CARMENES ORIGINAL GRID AS A STARTING POINT, HENCE AVOIDING
# PROBLEMS WITH SAMPLING WHEN INCREASING OR DECREASING THE RESOLUTION R     

def make_log_wave_grid(lambda_min, lambda_max, R, oversample=2):
dlog_lambda = 1. / (R * oversample)
log_lambda = np.arange(np.log(lambda_min), np.log(lambda_max) + dlog_lambda, dlog_lambda)
return np.exp(log_lambda)


def simulate_instrument(wave_hi,     # 1D array at R~1e6
flux_hi,     # matching flux
R_target,
oversample=2 # Nyquist sampled
):
# 1) build new grid
wave_new = make_log_wave_grid(wave_hi[0], wave_hi[-1], R_target, oversample)
# 2) smooth high-res flux
flux_smooth = convolve(wave_hi, flux_hi, R_target)
# 3) interpolate to new grid
flux_inst   = np.interp(wave_new, wave_hi, flux_smooth)

# Example:
#wave_hi = np.linspace(5000, 6000, 200000)        # R~1e6 sampling
#flux_hi = my_hires_model(wave_hi)               # your function
#wave_50k, flux_50k = simulate_instrument(wave_hi, flux_hi, R_target=50_000)
#wave_100k, flux_100k = simulate_instrument(wave_hi, flux_hi, R_target=100_000)


return wave_new, flux_inst
"""

"""
# FROM BLAIN ET AL. 2024
import astropy.units as u
from astropy.coordinates import (EarthLocation, 
                             SkyCoord)
from astropy.time import Time

site_name = "CAHA"  # Calar Alto astropy site name
ra = 300.1821223 * u.deg  # (degree) for HD 189733
dec = 22.7097759 * u.deg  # (degree) for HD 189733

times_utc = ...  # placeholder to load MJD_UTC times

observer_location = EarthLocation.of_site(site_name)

target_coordinates = SkyCoord(
ra=ra,
dec=dec
)

times_utc = Time(times_utc, format="jd", scale="utc")
times_tdb = (
    times_utc.tdb
    + times_utc.light_travel_time(
        target_coordinates,
        location=observer_location
    )
)
"""

"""
inp_dat['n_nights']=200
scaling_factors = np.asarray([0.2, 0.5, 1., 2., 3., 4., 5.])

simulation_name = "BL19_with_signal_n_nights_200_stdnoisex"
filepath = str(
    __import__('pathlib').Path(inp_dat['home_dir']).parent
    / simulation_name / 'matrices'
)
       
       
mean_SNR = np.zeros((len(scaling_factors)), float)
mean_SNR_error = np.zeros((len(scaling_factors)), float)
mean_SNR_gauss = np.zeros((len(scaling_factors)), float)
mean_SNR_error_gauss = np.zeros((len(scaling_factors)), float)
pearson_coeff_SNR = np.zeros((len(scaling_factors)), float)
pearson_coeff_SNR_error = np.zeros((len(scaling_factors)), float)
corr_coeff_noise = np.zeros((inp_dat["n_orders"], len(scaling_factors)), float)
corr_coeff_noise_error = np.zeros((inp_dat["n_orders"], len(scaling_factors)), float)
corr_coeff_data_TC = np.zeros((inp_dat["n_orders"], len(scaling_factors)), float)
corr_coeff_data_TC_error = np.zeros((inp_dat["n_orders"], len(scaling_factors)), float)
corr_coeff_data_NTC = np.zeros((inp_dat["n_orders"], len(scaling_factors)), float)
corr_coeff_data_NTC_error = np.zeros((inp_dat["n_orders"], len(scaling_factors)), float)

for sf_idx, sf in enumerate(scaling_factors):
print(f"Running {sf_idx+1}/{len(scaling_factors)} (scaling_factor = {sf})")
filename_flag = exosims.format_number(sf)
mat_res = np.zeros(
    (inp_dat['n_orders'], inp_dat['n_nights'], 
     n_spectra, n_pixels), float
    )
for h in range(inp_dat['n_orders']):
    filename = f"{filepath}{filename_flag}/mat_res_order_{h}_{simulation_name}{filename_flag}" 
    mat_res[h, :] = np.load(f"{filename}.npz")['a']

mat_noise = np.zeros(
    (inp_dat['n_orders'], inp_dat['n_nights'], 
     n_spectra, n_pixels), float
    )
for h in range(inp_dat['n_orders']):
    filename = f"{filepath}{filename_flag}/mat_noise_order_{h}_{simulation_name}{filename_flag}" 
    mat_noise[h, :] = np.load(f"{filename}.npz")['a']

gauss_noise = np.zeros(
    (inp_dat['n_orders'], inp_dat['n_nights'], 
     n_spectra, n_pixels), float
    )
for h in range(inp_dat['n_orders']):
    filename = f"{filepath}{filename_flag}/gauss_noise_order_{h}_{simulation_name}{filename_flag}" 
    gauss_noise[h, :] = np.load(f"{filename}.npz")['a']

mat_cc = np.zeros(
    (inp_dat['n_orders'],
     n_spectra, n_pixels), float
    )
for h in range(inp_dat['n_orders']):
    filename = f"{filepath}{filename_flag}/mat_cc_order_{h}_{simulation_name}{filename_flag}" 
    mat_cc[h, :] = np.load(f"{filename}.npz")['a']
    
filename = f"{filepath}{filename_flag}/stats_{simulation_name}{filename_flag}" 
stats = np.load(f"{filename}.npz")['a']
mean_SNR[sf_idx] = np.mean(stats[:, 0])
mean_SNR_error[sf_idx] = np.std(stats[:, 0])
mean_SNR_gauss[sf_idx], mean_SNR_error_gauss[sf_idx] = norm.fit(stats[1:,0])
night_min = np.where(stats[:,0] == stats[:,0].min())[0][0]
night_max = np.where(stats[1:,0] == stats[1:,0].max())[0][0]+1

filename = f"{filepath}{filename_flag}/stats_noise_{simulation_name}{filename_flag}" 
stats_noise = np.load(f"{filename}.npz")['a']


for h in range(inp_dat["n_orders"]):
    # These are correlations per order, so we might need to run all orders!
    corr_coeff_noise[h,sf_idx], corr_coeff_noise_error[h,sf_idx] = exosims.get_corr_coeff(
        inp_dat, with_signal, gauss_noise, mat_cc, range(inp_dat['n_nights']-1), 
        h, stats,"", night_max, night_min, phase, 
        plotname = "", CC_2D=True,show_plot = False, save_plot = False
        )

    corr_coeff_data_TC[h,sf_idx], corr_coeff_data_TC_error[h,sf_idx] = exosims.get_corr_coeff(
        inp_dat, with_signal, mat_res, mat_cc, range(inp_dat['n_nights']-1), 
        h, stats,"", night_max, night_min, phase,
        plotname = "", CC_2D=True,show_plot = False, save_plot = False
        )

    corr_coeff_data_NTC[h,sf_idx], corr_coeff_data_NTC_error[h,sf_idx] = exosims.get_corr_coeff(
        inp_dat, with_signal, mat_noise, mat_cc, range(inp_dat['n_nights']-1), 
        h, stats,"", night_max, night_min, phase,
        plotname = "", CC_2D=True,show_plot = False, save_plot = False
        )


#corr_coeff_dataNTC, corr_coeff_dataNTC_uncertainty = exosims.compare_correlations(
#    inp_dat, corr_coeff_noise, corr_coeff_data_NTC, stats[1:, 0], 
#    xlabel="Corr. Coeff. Noise", ylabel="Corr. Coeff. Data After TC", 
#    colorbar_title="S/N CCF Data After TC", plot_lims=[-0.012,0.014]
#    )
#corr_coeff_dataTC, corr_coeff_dataTC_uncertainty = exosims.compare_correlations(
#    inp_dat, corr_coeff_noise, corr_coeff_data_TC, stats[1:, 0], 
#    xlabel="Corr. Coeff. Noise", ylabel="Corr. Coeff. Data After TC", 
#    colorbar_title="S/N CCF Data After TC", plot_lims=[-0.012,0.014]
#    )
#corr_coeff_noise_dataTC, corr_coeff_noise_dataTC_uncertainty = exosims.compare_correlations(
#    inp_dat, corr_coeff_noise, corr_coeff_data_after_TC, stats[1:, 0], 
#    xlabel="Corr. Coeff. Noise", ylabel="Corr. Coeff. Data After TC", 
#    colorbar_title="S/N CCF Data After TC", plot_lims=[-0.012,0.014]
#    ) 

# This is the Pearson corr. of the S/N of the MSS of simulated data with
# the S/N of the CCF obtained from noise spectral matrix vs. template
# at the MSS Kp-Vrest
pearson_coeff_SNR[sf_idx], pearson_coeff_SNR_error[sf_idx] = exosims.compare_correlations(
    inp_dat, stats_noise[1:,0], stats[1:,0],
    plotname = "",
    xlabel="S/N Noise", ylabel="S/N Data After TC", 
    show_plot= False, save_plot = False,
    plot_lims=[-5,7]
    )


# Create a figure with two panels
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(8, 6))

# Adjust the spacing between the two panels
plt.subplots_adjust(hspace=0)

# Plot on the first panel
ax1.errorbar(scaling_factors, pearson_coeff_SNR, yerr=pearson_coeff_SNR_error, 
         fmt='o', color='k', markersize=8, capsize=9, zorder=2,
         label = "S/N Data_TC vs. S/N Noise")
ax1.set_ylabel("Corr.", fontsize=15, multialignment='center')
ax1.axvline(x=1, color='firebrick', alpha=0.6, zorder=1)
ax1.grid(True, alpha=0.6, zorder=0)  # Add a grid behind everything
ax1.legend(loc='upper right')

# Plot on the second panel
scatter1 = ax2.errorbar(scaling_factors, np.mean(corr_coeff_data_NTC, axis=0), 
         yerr=np.mean(corr_coeff_data_NTC_error, axis=0), fmt='o', 
         color='k', linewidth=1, markersize=8, capsize=9, zorder=2,
         label="Data Before TC") 
scatter2 = ax2.errorbar(scaling_factors, np.mean(corr_coeff_data_TC, axis=0), 
         yerr=np.mean(corr_coeff_data_TC_error, axis=0), fmt='o', 
         color='goldenrod', linewidth=1, markersize=8, capsize=9, zorder=2,
         label="Data After TC") 
scatter3 = ax2.errorbar(scaling_factors, np.mean(corr_coeff_noise, axis=0), 
         yerr=np.mean(corr_coeff_noise_error, axis=0), fmt='o', 
         color='dodgerblue', linewidth=1, markersize=8, capsize=9, zorder=2,
         label="Noise spectral matrix") 

ax2.set_ylabel("Corr.", fontsize=15, multialignment='center')
ax2.axvline(x=1, color='firebrick', label='Real uncertainties', alpha=0.6, zorder=1)
ax2.grid(True, alpha=0.6, zorder=0)  # Add a grid behind everything
ax2.legend(handles=[scatter1, scatter2, scatter3], loc='upper right')

# Plot on the second panel
scatter4 = ax3.errorbar(scaling_factors, mean_SNR, yerr=mean_SNR_error, 
                    fmt='o', color='k', markersize=8, capsize=9,
                    zorder=2)
scatter5 = ax3.errorbar(scaling_factors, mean_SNR_gauss, 
                    yerr=mean_SNR_error_gauss,  fmt='o', 
                    color='k', markersize=8, capsize=9, 
                    zorder=2)
ax3.set_xlabel("Noise scaling factor", fontsize = 15)
ax3.set_xlabel("Noise scaling factor", fontsize = 15)
ax3.set_ylabel("<S/N>", fontsize = 15, multialignment='center')
ax3.axvline(x=1, color='firebrick', label='Nominal case', alpha=0.6, zorder=1)
ax3.grid(True, alpha=0.6, zorder=0)  # Add a grid behind everything
# Define the tick locations you want
ytick_locations = [0, 5, 10, 15, 20, 25, 30]
# Set the yticks for ax3
ax3.set_yticks(ytick_locations)

plt.tick_params(axis='both', width=1.5, direction='in', labelsize=15)

# Show the plot
plt.show()
"""



"""
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(4, 3))  # Adjust the figsize as needed

# Define your equation
equation = r'$F_{scaled} (\lambda) = 1 + \left(\frac{F_P (\lambda)}{F_{s} (\lambda)}\right) \left(\frac{R_P}{R_{s}}\right)^2$'

# Hide axes
ax.axis('off')

# Add the equation as text
ax.text(0.5, 0.5, equation, ha='center', va='center', fontsize=20)

# Save the figure as a PNG with transparent background
plt.savefig('dayside_flux_equation.png', transparent=True)
plt.show()

import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(4, 3))  # Adjust the figsize as needed

# Define your equation
equation = r'$R_{scaled} (\lambda) = 1 - \left(\frac{R_P (\lambda)}{R_{s} (\lambda)}\right)^2$'

# Hide axes
ax.axis('off')

# Add the equation as text
ax.text(0.5, 0.5, equation, ha='center', va='center', fontsize=20)

# Save the figure as a PNG with transparent background
plt.savefig('transit_radius_equation.png', transparent=True)
plt.show()
"""

