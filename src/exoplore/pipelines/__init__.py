"""
exoplore.pipelines
==================

Data preparation and cleaning pipelines for high-resolution
spectroscopic time-series.

Each pipeline takes a spectral matrix (n_spectra × n_pixels) and
returns a cleaned matrix ready for cross-correlation analysis.

Available modules
-----------------
masking
    Telluric, SNR, and column-scatter masking.
sysrem
    SYSREM systematic noise removal (Tamuz et al. 2005).
bl19
    Brogi & Line (2019) normalisation + telluric correction.
blain24
    Blain, Sánchez-López & Mollière (2024) polynomial pipeline.
tellurics
    Telluric transmittance loading and fixed-telluric pipeline.
prepare
    Main pipeline orchestrator (``preparing_pipeline``) and injection
    utilities.
"""

from exoplore.pipelines.masking import (
    # clean API
    mask_telluric_columns,
    mask_telluric_columns_with_window,
    mask_low_snr_columns,
    mask_noisy_columns,
    merge_masks,
    good_pixel_indices,
    # public API
    _merge_masks,
    mask_tellurics,
    mask_tellurics_window,
    mask_columns,
    Correct_NaN,
    Remove_Outliers,
    Robust_Outlier_Removal,
)
from exoplore.pipelines.sysrem import (
    # clean API
    sysrem_iteration,
    apply_sysrem,
    # public API
    sysrem,
    SYSREM_filtering_projector,
    SYSREM_filtering_projector_singleorder,
    filter_model_singleorder,
    get_SYSREM_its_ordbyord,
)
from exoplore.pipelines.bl19 import (
    # clean API
    bl19_normalise,
    bl19_telluric_correct,
    run_bl19_pipeline,
    # public API
    pipeline_BL19_norm,
    pipeline_BL19_tellcorr,
)
from exoplore.pipelines.blain24 import (
    # clean API
    blain24_normalise,
    blain24_telluric_correct,
    run_blain24_pipeline,
    # public API
    remove_telluric_lines_fit,
    remove_throughput_fit,
    # v0.25
    compute_k_sigma,
)
from exoplore.pipelines.tellurics import (
    Load_Telluric_Transmittances,
    pipeline_fixedTellurics,
)
from exoplore.pipelines.prepare import (
    preparing_pipeline,
    injection,
    init_pipeline_outputs,
    remove_throughput_fit_og,
    # v0.25
    remove_telluric_lines_fit_og,
)

__all__ = [
    # ---- clean masking ----
    "mask_telluric_columns",
    "mask_telluric_columns_with_window",
    "mask_low_snr_columns",
    "mask_noisy_columns",
    "merge_masks",
    "good_pixel_indices",
    # ---- masking ----
    "_merge_masks",
    "mask_tellurics",
    "mask_tellurics_window",
    "mask_columns",
    "Correct_NaN",
    "Remove_Outliers",
    "Robust_Outlier_Removal",
    # ---- clean sysrem ----
    "sysrem_iteration",
    "apply_sysrem",
    # ---- sysrem ----
    "sysrem",
    "SYSREM_filtering_projector",
    "SYSREM_filtering_projector_singleorder",
    "filter_model_singleorder",
    "get_SYSREM_its_ordbyord",
    # ---- clean BL19 ----
    "bl19_normalise",
    "bl19_telluric_correct",
    "run_bl19_pipeline",
    # ---- BL19 ----
    "pipeline_BL19_norm",
    "pipeline_BL19_tellcorr",
    # ---- clean Blain24 ----
    "blain24_normalise",
    "blain24_telluric_correct",
    "run_blain24_pipeline",
    # ---- Blain24 ----
    "remove_telluric_lines_fit",
    "remove_throughput_fit",
    "compute_k_sigma",
    # ---- tellurics ----
    "Load_Telluric_Transmittances",
    "pipeline_fixedTellurics",
    # ---- prepare / injection ----
    "preparing_pipeline",
    "injection",
    "init_pipeline_outputs",
    "remove_throughput_fit_og",
    "remove_telluric_lines_fit_og",
]
