"""Smoke tests for exoplore.analysis, import and signature checks only.

No analysis functions render pixels or run retrievals in this suite.
These tests verify:

1. All public symbols are importable from the top-level package.
2. Each function has the expected parameter names (via inspect).
"""

import inspect
import pytest

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Import checks, stats
# ---------------------------------------------------------------------------

def test_stats_imports():
    from exoplore.analysis import (
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
    )
    for fn in [plot_stats, statistical_study, get_SYSREM_its_ordbyord,
               get_CCvalues_dist, compare_empirical_SN, plot_std_errors,
               compare_KpVr_dist, get_corr_coeff, bootstrap_corrcoeffs,
               compare_correlations, find_nights_with_extrema,
               perform_correlations_with_noise]:
        assert callable(fn)


# ---------------------------------------------------------------------------
# Import checks, diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_imports():
    from exoplore.analysis import (
        plot_absolute_differences,
        diff_res_model,
        plot_difference,
        autocorrelation_examples,
        plot_params_vs_order,
        plot_detectability_maps_FeH_CO,
    )
    for fn in [plot_absolute_differences, diff_res_model, plot_difference,
               autocorrelation_examples, plot_params_vs_order,
               plot_detectability_maps_FeH_CO]:
        assert callable(fn)


# ---------------------------------------------------------------------------
# Import checks, retrieval_plots
# ---------------------------------------------------------------------------

def test_retrieval_plots_imports():
    from exoplore.analysis import (
        plot_live_posterior,
        plot_live_posterior2,
        compare_retrieval_corners,
        compare_multinest_evidence,
        compare_CtoO_corners_from_multispecies,
        compare_CtoO_corners_from_multispecies_flexible,
        compare_CtoO_corners_from_CO_H2O,
        compute_CtoO_text,
    )
    for fn in [plot_live_posterior, plot_live_posterior2,
               compare_retrieval_corners, compare_multinest_evidence,
               compare_CtoO_corners_from_multispecies,
               compare_CtoO_corners_from_multispecies_flexible,
               compare_CtoO_corners_from_CO_H2O, compute_CtoO_text]:
        assert callable(fn)


# ---------------------------------------------------------------------------
# Import checks, multi_night
# ---------------------------------------------------------------------------

def test_multi_night_imports():
    from exoplore.analysis import (
        combine_nights_and_plot_3params,
        combine_nights_and_make_ZPc_and_beta_panels,
        combine_nights_and_plot_3params_2ins,
    )
    for fn in [combine_nights_and_plot_3params,
               combine_nights_and_make_ZPc_and_beta_panels,
               combine_nights_and_plot_3params_2ins]:
        assert callable(fn)


# ---------------------------------------------------------------------------
# Import checks, ccf_time
# ---------------------------------------------------------------------------

def test_ccf_time_imports():
    from exoplore.analysis import (
        plot_time_resolved_1D_CCFs,
        plot_time_resolved_1D_CCFs_withHRLRS,
    )
    assert callable(plot_time_resolved_1D_CCFs)
    assert callable(plot_time_resolved_1D_CCFs_withHRLRS)


# ---------------------------------------------------------------------------
# Import checks, utils
# ---------------------------------------------------------------------------

def test_utils_imports():
    from exoplore.analysis import remove_all_elements
    assert callable(remove_all_elements)


# ---------------------------------------------------------------------------
# Signature checks for key functions
# ---------------------------------------------------------------------------

class TestSignatures:
    def test_plot_stats_params(self):
        from exoplore.analysis import plot_stats
        params = list(inspect.signature(plot_stats).parameters)
        assert "stats" in params
        assert "kp_lim_inf" in params
        assert "kp_lim_sup" in params

    def test_statistical_study_params(self):
        from exoplore.analysis import statistical_study
        params = list(inspect.signature(statistical_study).parameters)
        assert "inp_dat" in params

    def test_get_CCvalues_dist_params(self):
        from exoplore.analysis import get_CCvalues_dist
        params = list(inspect.signature(get_CCvalues_dist).parameters)
        assert "inp_dat" in params
        assert "ccf_matrix" in params
        assert "v_ccf" in params
        assert "v_rest" in params

    def test_compare_KpVr_dist_params(self):
        from exoplore.analysis import compare_KpVr_dist
        params = list(inspect.signature(compare_KpVr_dist).parameters)
        assert "inp_dat" in params
        assert "v_rest" in params
        assert "ccf_matrix" in params

    def test_plot_time_resolved_1D_CCFs_params(self):
        from exoplore.analysis import plot_time_resolved_1D_CCFs
        params = list(inspect.signature(plot_time_resolved_1D_CCFs).parameters)
        assert "inp_dat" in params
        assert "ccf_values_shift" in params
        assert "v_rest" in params
        assert "kp_range" in params

    def test_plot_time_resolved_withHRLRS_params(self):
        from exoplore.analysis import plot_time_resolved_1D_CCFs_withHRLRS
        params = list(inspect.signature(plot_time_resolved_1D_CCFs_withHRLRS).parameters)
        assert "inp_dat" in params
        assert "ccf_values_shift" in params
        assert "v_rest" in params
        assert "kp_range" in params

    def test_compare_retrieval_corners_params(self):
        from exoplore.analysis import compare_retrieval_corners
        params = list(inspect.signature(compare_retrieval_corners).parameters)
        assert "base_dir" in params
        assert "prefixes" in params
        assert "param_names" in params

    def test_combine_nights_3params_params(self):
        from exoplore.analysis import combine_nights_and_plot_3params
        params = list(inspect.signature(combine_nights_and_plot_3params).parameters)
        assert "base_dir_template" in params
        assert "retrieval_name" in params

    def test_plot_detectability_maps_params(self):
        from exoplore.analysis import plot_detectability_maps_FeH_CO
        params = list(inspect.signature(plot_detectability_maps_FeH_CO).parameters)
        assert "data_dirs" in params
        assert "labels" in params

    def test_perform_correlations_params(self):
        """perform_correlations_with_noise has the expected parameters."""
        from exoplore.analysis import perform_correlations_with_noise
        params = list(inspect.signature(perform_correlations_with_noise).parameters)
        assert "inp_dat" in params
        assert "stats" in params
        assert "stats_noise" in params
