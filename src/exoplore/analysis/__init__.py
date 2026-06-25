"""
exoplore.analysis
==================

Post-processing and analysis tools for high-resolution exoplanet atmosphere
simulations, grouped into thematic submodules.

Submodules
----------
stats            Statistical analysis of CCF detections.
diagnostics      Diagnostic and model-comparison plots.
retrieval_plots  Post-processing plots for MultiNest retrievals.
multi_night      Multi-night / multi-instrument combination plots.
ccf_time         Time-resolved 1D CCF analysis.
utils            Miscellaneous helpers.
"""

# ---- stats ---------------------------------------------------------------
from exoplore.analysis.stats import (
    plot_stats,
    statistical_study,
    get_SYSREM_its_ordbyord,
    get_CCvalues_dist,
    compare_empirical_SN,
    plot_std_errors,
    compare_KpVr_dist,
    get_corr_coeff,
    bootstrap_corrcoeffs,
    compare_correlations,
    find_nights_with_extrema,
    perform_correlations_with_noise,
    bayes_factor_to_sigma,
)

# ---- diagnostics ---------------------------------------------------------
from exoplore.analysis.diagnostics import (
    plot_absolute_differences,
    diff_res_model,
    plot_difference,
    autocorrelation_examples,
    plot_params_vs_order,
    plot_detectability_maps_FeH_CO,
)

# ---- retrieval_plots -----------------------------------------------------
from exoplore.analysis.retrieval_plots import (
    plot_live_posterior,
    plot_live_posterior2,
    compare_retrieval_corners,
    compare_multinest_evidence,
    compare_CtoO_corners_from_multispecies,
    compare_CtoO_corners_from_multispecies_flexible,
    compare_CtoO_corners_from_CO_H2O,
    compute_CtoO_text,
)

# ---- multi_night ---------------------------------------------------------
from exoplore.analysis.multi_night import (
    combine_nights_and_plot_3params,
    combine_nights_and_make_ZPc_and_beta_panels,
    combine_nights_and_plot_3params_2ins,
)

# ---- ccf_time ------------------------------------------------------------
from exoplore.analysis.ccf_time import (
    plot_time_resolved_1D_CCFs,
    plot_time_resolved_1D_CCFs_withHRLRS,
)

# ---- utils ---------------------------------------------------------------
from exoplore.analysis.utils import (
    remove_all_elements,
)

__all__ = [
    # stats
    "plot_stats", "statistical_study", "get_SYSREM_its_ordbyord",
    "get_CCvalues_dist", "compare_empirical_SN", "plot_std_errors",
    "compare_KpVr_dist", "get_corr_coeff", "bootstrap_corrcoeffs",
    "compare_correlations", "find_nights_with_extrema",
    "perform_correlations_with_noise",
    # diagnostics
    "plot_absolute_differences", "diff_res_model", "plot_difference",
    "autocorrelation_examples", "plot_params_vs_order",
    "plot_detectability_maps_FeH_CO",
    # retrieval_plots
    "plot_live_posterior", "plot_live_posterior2",
    "compare_retrieval_corners", "compare_multinest_evidence",
    "compare_CtoO_corners_from_multispecies",
    "compare_CtoO_corners_from_multispecies_flexible",
    "compare_CtoO_corners_from_CO_H2O", "compute_CtoO_text",
    # multi_night
    "combine_nights_and_plot_3params",
    "combine_nights_and_make_ZPc_and_beta_panels",
    "combine_nights_and_plot_3params_2ins",
    # ccf_time
    "plot_time_resolved_1D_CCFs", "plot_time_resolved_1D_CCFs_withHRLRS",
    # utils
    "remove_all_elements",
]
