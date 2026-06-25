#!/usr/bin/env bash
# Sync workspace → git repo and commit v0.35
set -e

SRC="/Users/alexsl/Documents/Claude/Projects/EXoPLORE Repository/"
DST="/Users/alexsl/Documents/Simulador/EXoPLORE_github/"

rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.venv' --exclude='*.egg-info' --exclude='legacy/' \
      "$SRC" "$DST"

cd "$DST"
rm -rf src/exoplore/legacy/

git add -u

git commit -m 'v0.35: function signature audit — remove all unused/phantom parameters

Extensive audit of all 26 clean-package source files.  Every function
signature now contains only parameters that are (a) actually used in
the function body and (b) fully described in the docstring.  No
scientific logic changed; no legacy code touched.

pipelines/prepare.py:
  preparing_pipeline — remove tell_mask_threshold_BLASP24 (BLASP24 branch
    hardcodes 1e-16, never reads the passed value) and sysrem_division
    (never referenced anywhere in the body); add full descriptions to
    remaining optional parameters.
  pipeline_fixedTellurics call-site bug FIXED: was called with 5 positional
    args but signature requires 7; now passes mask and an empty mask_snr.

pipelines/masking.py:
  merge_masks — remove n_pixels (never used; body does pure set union).
  mask_low_snr_columns — remove data (never used; only snr_map matters).
  mask_telluric_columns — remove n_pixels (only needed to forward to
    merge_masks, which no longer requires it).
  mask_telluric_columns_with_window — same; n_pixels now inferred from
    template shape for the safety-window expansion loop.
  All internal merge_masks call sites updated to 2-argument form.

pipelines/blasp24.py:
  pipeline_BLASP24_tellcorr — add full NumPy-style Parameters section
    (was previously bare: "Returns" only).

ccf/statistics.py:
  Welch_ttest_map — remove plotting, v_rest_plot, kp_plot (never read
    in body); update all internal call sites.
  statistical_study — remove auto_lims (never passed anywhere), save_plot
    (no file I/O in body), ccf_SSIM (never read); update docstring.
  get_corr_coeff — remove night_max, night_min, phase (never read in body).

analysis/stats.py:
  plot_stats docstring — remove phantom Args entries mark_positives,
    true_kp, true_vrest (described in docstring but absent from signature).

observation/airmass.py:
  get_airmass — remove path (never used in body).

io/stellar.py:
  LoadPhoenix — remove n_pixels (never used in body).

analysis/diagnostics.py:
  diff_res_model — remove night_ref, night_max, night_min (body hardcodes
    matrix_res[1:,:,:,:] unconditionally; none of the three params are
    ever read); update docstring to document the hardcoded exclusion.'

git log --oneline -5
