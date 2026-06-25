"""
exoplore.plotting
=================

Diagnostic and publication-quality figures for high-resolution
exoplanet atmosphere simulations.

Modules
-------
kpvsys
    Kp-Vsys detection maps, 1-D CCF slices, and combined detection panels.
matrices
    Spectral time-series matrices and CCF time-series heatmaps.
"""

from exoplore.plotting.kpvsys import (
    plot_kp_vsys_map,
    plot_1d_ccf,
    # public API
    plot_Kp_Vrest,
    plot_1D_CCF,
    plot_combined_KpVrest_1DCCF,
)
from exoplore.plotting.matrices import (
    plot_spectral_matrix,
    plot_ccf_timeseries,
    # public API
    CCF_matrix_ERF,
    plot_ccf_matrices_per_night,
    plot_mat_with_collapse,
    plot_steps,
    plot_matrix_difference,
)

__all__ = [
    # kpvsys (clean API)
    "plot_kp_vsys_map",
    "plot_1d_ccf",
    # kpvsys
    "plot_Kp_Vrest",
    "plot_1D_CCF",
    "plot_combined_KpVrest_1DCCF",
    # matrices (clean API)
    "plot_spectral_matrix",
    "plot_ccf_timeseries",
    # matrices
    "CCF_matrix_ERF",
    "plot_ccf_matrices_per_night",
    "plot_mat_with_collapse",
    "plot_steps",
    "plot_matrix_difference",
]
