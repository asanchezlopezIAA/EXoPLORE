"""exoplore.io, file I/O and output path management."""

# Reading these imports
# These public names make file/path helpers convenient to import. Look in
# the implementing module to distinguish reading, writing, name construction
# and numerical helpers retained under IO. A path returned by a helper does
# not by itself establish that the file already exists.


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
