"""exoplore.ccf, cross-correlation function analysis."""

from exoplore.ccf.kernels import (
    compute_inverse_variance_weighted_ccf,
    ccf_numba_par_weighted,
    # additional kernels
    ccf_numba,
    ccf_numba_par,
    ccf_numba_par_weighted_ordbord_opt,
    ccf_literature,
)
from exoplore.ccf.compute import (
    compute_ccf_timeseries,
    compute_kp_vsys_map,
    build_velocity_grid,
    # public API
    get_shifted_ccf_matrix,
    get_max_CCF_peak,
    call_ccf_numba,
    call_ccf_numba_par,
    call_ccf_numba_par_weighted,
    call_ccf_literature,
    call_ccf_numba_par_weighted_ordbord_opt,
    quick_CCF,
)
from exoplore.ccf.statistics import (
    Welch_ttest_map,
    Combine_Nights,
    statistical_study,
    get_corr_coeff,
)

__all__ = [
    # kernels (clean API)
    "compute_inverse_variance_weighted_ccf",
    # kernels
    "ccf_numba_par_weighted",
    "ccf_numba",
    "ccf_numba_par",
    "ccf_numba_par_weighted_ordbord_opt",
    "ccf_literature",
    # compute (clean API)
    "compute_ccf_timeseries",
    "compute_kp_vsys_map",
    "build_velocity_grid",
    # compute
    "get_shifted_ccf_matrix",
    "get_max_CCF_peak",
    "call_ccf_numba",
    "call_ccf_numba_par",
    "call_ccf_numba_par_weighted",
    "call_ccf_literature",
    "call_ccf_numba_par_weighted_ordbord_opt",
    "quick_CCF",
    # statistics
    "Welch_ttest_map",
    "Combine_Nights",
    "statistical_study",
    "get_corr_coeff",
]
