"""Overlay corner plot for multiple EXoPLORE retrieval runs.

Generates a corner plot showing 1D marginal posteriors (diagonal) and 2D
joint posteriors (off-diagonal) for up to four EXoPLORE retrieval runs
overlaid in different colours.  Intended for pipeline bias tests and
retrieval comparisons (Tutorial 7 of the EXoPLORE documentation).

Usage
-----
python scripts/plot_corner_overlay.py \\
    --output-root /path/to/EXoPLORE_clean_run/HD189733b/CARMENES_NIR/transit \\
    --runs  BL19_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1 \\
            BLASP24_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1 \\
            Gibson22_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1 \\
    --labels "Brogi & Line (2019)" "Blain et al. (2024)" "Gibson et al. (2022)" \\
    --truths -3.0 149.4 1170.0 0.0 \\
    --param-names "log10(X_H2O)" "Kp" "T_eq" "v_wind" \\
    --output docs/figures/tutorial_retrieval_bias_corner.png

Arguments
---------
--output-root   Directory containing the run subdirectories.
--runs          One or more run directory names (basename only).
--labels        Display labels for the legend (one per run).
--truths        Truth values for vertical/horizontal dashed lines.
                Must match the number of shared parameters.
--param-names   LaTeX or plain parameter names for axis labels.
--output        Output PNG path.
--colors        Hex colours for each run (optional; defaults provided).
--n-bins        Number of histogram/2D-density bins (default 50).
--smooth-sigma  Gaussian smoothing sigma for 2D densities (default 1.2).
--pad           Fractional padding on each axis range (default 0.18).
--dpi           Output DPI (default 180).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter


_DEFAULT_COLORS = ['#2166ac', '#d6604d', '#4dac26', '#8856a7']
_DEFAULT_PARAM_NAMES = [
    r'$\log_{10}(X_{\rm H_2O})$',
    r'$K_{\rm P}$ (km s$^{-1}$)',
    r'$T_{\rm eq}$ (K)',
    r'$v_{\rm wind}$ (km s$^{-1}$)',
    r'$\beta$',
]


def _load(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    import os, glob
    basename = os.path.basename(run_dir)
    mat_dir  = os.path.join(run_dir, 'matrices')
    samp = np.load(os.path.join(mat_dir, f'retrieval_night_0_dat_{basename}.npz'))['a']
    wts  = np.load(os.path.join(mat_dir, f'retrieval_night_0_weights_{basename}.npz'))['a']
    return samp, wts / wts.sum()


def _wpct(s: np.ndarray, w: np.ndarray, p: float) -> float:
    idx = np.argsort(s); cs = np.cumsum(w[idx]); cs /= cs[-1]
    return float(np.interp(p / 100, cs, s[idx]))


def _get_range(s_list, w_list, col: int, pad: float = 0.18):
    lo = min(_wpct(s[:, col], w, 0.5)  for s, w in zip(s_list, w_list))
    hi = max(_wpct(s[:, col], w, 99.5) for s, w in zip(s_list, w_list))
    m = (hi - lo) * pad
    return (lo - m, hi + m)


def _hist1d(ax, s, w, col, color, rng, n_bins):
    bins = np.linspace(rng[0], rng[1], n_bins + 1)
    h, e = np.histogram(s[:, col], bins=bins, weights=w)
    h = h / h.max()
    cx = 0.5 * (e[:-1] + e[1:])
    ax.fill_between(cx, h, step='mid', alpha=0.40, color=color)
    ax.step(cx, h, where='mid', color=color, lw=1.5)


def _contour2d(ax, s, w, cx, cy, color, rng_x, rng_y, n_bins, sigma):
    H, xe, ye = np.histogram2d(s[:, cx], s[:, cy], bins=n_bins, weights=w,
                                range=[rng_x, rng_y])
    H = gaussian_filter(H / H.sum(), sigma=sigma)
    Hf = H.flatten()
    idx = np.argsort(Hf)[::-1]
    cs  = np.cumsum(Hf[idx]); cs /= cs[-1]
    t68 = Hf[idx[np.searchsorted(cs, 0.68)]]
    t95 = Hf[idx[np.searchsorted(cs, 0.95)]]
    xc  = 0.5 * (xe[:-1] + xe[1:])
    yc  = 0.5 * (ye[:-1] + ye[1:])
    ax.contourf(xc, yc, H.T, levels=[t95, H.max()], colors=[color], alpha=0.15)
    ax.contourf(xc, yc, H.T, levels=[t68, H.max()], colors=[color], alpha=0.20)
    ax.contour(xc, yc, H.T,  levels=[t95, t68],     colors=[color, color],
               linewidths=[0.9, 1.6])
    ax.set_xlim(rng_x); ax.set_ylim(rng_y)


def make_corner(
    run_dirs:    list[str],
    labels:      list[str],
    truths:      list[float],
    param_names: list[str],
    output:      str,
    colors:      list[str] | None = None,
    n_bins:      int   = 50,
    smooth_sigma:float = 1.2,
    pad:         float = 0.18,
    dpi:         int   = 180,
) -> None:
    """Generate and save the overlay corner plot."""

    if colors is None:
        colors = _DEFAULT_COLORS[:len(run_dirs)]

    # Load all runs
    samples, weights, n_params_per_run = [], [], []
    for rd in run_dirs:
        s, w = _load(rd)
        samples.append(s)
        weights.append(w)
        n_params_per_run.append(s.shape[1])

    n_params = max(n_params_per_run)
    if len(param_names) < n_params:
        param_names = list(param_names) + _DEFAULT_PARAM_NAMES[len(param_names):n_params]

    # Axis ranges: shared params from all runs, extra params from runs that have them
    ranges = []
    for col in range(n_params):
        s_col = [s for s, n in zip(samples, n_params_per_run) if n > col]
        w_col = [w for w, n in zip(weights, n_params_per_run) if n > col]
        ranges.append(_get_range(s_col, w_col, col, pad))

    fig, axes = plt.subplots(n_params, n_params, figsize=(11, 11))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            ax.set_facecolor('white')
            for sp in ax.spines.values(): sp.set_linewidth(0.6)

            if col > row:
                ax.set_visible(False)
                continue

            if row == col:
                for s, w, color, npar in zip(samples, weights, colors, n_params_per_run):
                    if col < npar:
                        _hist1d(ax, s, w, col, color, ranges[col], n_bins)
                if col < len(truths):
                    ax.axvline(truths[col], color='black', lw=1.4, ls='--')
                ax.set_xlim(ranges[col]); ax.set_ylim(0, 1.15); ax.set_yticks([])
            else:
                for s, w, color, npar in zip(samples, weights, colors, n_params_per_run):
                    if col < npar and row < npar:
                        _contour2d(ax, s, w, col, row, color,
                                   ranges[col], ranges[row], n_bins, smooth_sigma)
                if col < len(truths):
                    ax.axvline(truths[col], color='black', lw=0.9, ls='--', alpha=0.5)
                if row < len(truths):
                    ax.axhline(truths[row], color='black', lw=0.9, ls='--', alpha=0.5)

            max_npar = max(n_params_per_run)

            # x-axis labels and ticks
            if row == n_params - 1:
                ax.set_xlabel(param_names[col], fontsize=13, labelpad=5)
                if col == max_npar - 1:
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
            else:
                ax.set_xticklabels([])

            # y-axis: only show on leftmost column; suppress ticks everywhere else
            if col == 0 and row > 0:
                ax.set_ylabel(param_names[row], fontsize=13, labelpad=5)
                if row == max_npar - 1:
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', left=False)

            ax.tick_params(labelsize=9)

    # Legend in top-right slot
    ax_leg = axes[0, n_params - 1]
    ax_leg.set_visible(True); ax_leg.set_facecolor('white'); ax_leg.axis('off')
    legend_elements = [
        Line2D([0], [0], color=c, lw=2.5, label=l)
        for c, l in zip(colors, labels)
    ]
    legend_elements.append(Line2D([0], [0], color='black', lw=1.6, ls='--', label='Truth'))
    ax_leg.legend(handles=legend_elements, loc='center', fontsize=11, frameon=False)

    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output}')


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Overlay corner plot for EXoPLORE retrieval runs.')
    p.add_argument('--output-root', required=True,
                   help='Directory containing the run subdirectories.')
    p.add_argument('--runs', nargs='+', required=True,
                   help='Run directory basenames.')
    p.add_argument('--labels', nargs='+', required=True,
                   help='Legend labels (one per run).')
    p.add_argument('--truths', nargs='+', type=float, default=[],
                   help='Truth values for dashed lines.')
    p.add_argument('--param-names', nargs='+', default=[],
                   help='Parameter axis labels (LaTeX ok).')
    p.add_argument('--output', required=True,
                   help='Output PNG path.')
    p.add_argument('--colors', nargs='+', default=None,
                   help='Hex colours (one per run).')
    p.add_argument('--n-bins', type=int, default=50)
    p.add_argument('--smooth-sigma', type=float, default=1.2)
    p.add_argument('--pad', type=float, default=0.18)
    p.add_argument('--dpi', type=int, default=180)
    return p.parse_args()


if __name__ == '__main__':
    import os
    args = _parse()
    run_dirs = [os.path.join(args.output_root, r) for r in args.runs]
    make_corner(
        run_dirs=run_dirs,
        labels=args.labels,
        truths=args.truths,
        param_names=args.param_names,
        output=args.output,
        colors=args.colors,
        n_bins=args.n_bins,
        smooth_sigma=args.smooth_sigma,
        pad=args.pad,
        dpi=args.dpi,
    )
