"""exoplore.io, file I/O and output path management."""

from exoplore.io.paths import (
    build_output_tree,
    simulation_output_dir,
    matrices_dir,
    plots_dir,
    correlations_dir,
)
from exoplore.io.stellar import LoadPhoenix, get_stellar_matrix, spec_to_mat_fraction
from exoplore.io.utils import (
    save_compressed,
    create_directory,
    format_number,
    weighted_quantile,
    check_consistent_wavelengths,
    convert_masked_arrays,
    find_nearest,
    convert_vega_to_ab,
    bootstrap_corrcoeffs,
    Utils_permute_nights_indices,
)

__all__ = [
    "build_output_tree",
    "simulation_output_dir",
    "matrices_dir",
    "plots_dir",
    "correlations_dir",
    # stellar loaders
    "LoadPhoenix",
    "get_stellar_matrix",
    "spec_to_mat_fraction",
    # utilities
    "save_compressed",
    "create_directory",
    "format_number",
    "weighted_quantile",
    "check_consistent_wavelengths",
    "convert_masked_arrays",
    "find_nearest",
    "convert_vega_to_ab",
    "bootstrap_corrcoeffs",
    "Utils_permute_nights_indices",
]
