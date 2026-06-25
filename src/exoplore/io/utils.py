"""
exoplore.io.utils
=================

General I/O utility helpers.

Functions
---------
save_compressed
    Save multiple numpy arrays as compressed .npz files.
create_directory
    Create a directory (with ``exist_ok=True``).
format_number
    Format a float for use in simulation run-name strings.
"""

from __future__ import annotations

import os

import numpy as np


def save_compressed(filename_base: str, sim_name: str, data_dict: dict) -> None:
    """Save multiple arrays as compressed .npz files.

    For each key-value pair in *data_dict*, writes a file::

        {filename_base}/{key}_{sim_name}.npz

    The array is stored under the key ``'a'`` inside the .npz archive,
    matching the convention used by :func:`numpy.savez_compressed`.

    Parameters
    ----------
    filename_base : str
        Directory where the files will be written.
    sim_name : str
        Simulation run name appended to each filename.
    data_dict : dict
        Mapping of ``{label: array}`` pairs to save.
    """
    for key, data in data_dict.items():
        filename = f"{filename_base}/{key}_{sim_name}"
        np.savez_compressed(filename, a=data)


def create_directory(directory_path: str, cluster: bool = False) -> None:
    """Create a directory if it does not already exist.

    Parameters
    ----------
    directory_path : str
        Path to the directory to create.
    cluster : bool
        Unused flag retained for backward compatibility.
    """
    os.makedirs(directory_path, exist_ok=True)
    return


def format_number(x, decimals: int = 1):
    """Format a float (or array of floats) for use in simulation run names.

    Integers are returned as ``str(int(x))``.
    Non-integer floats are formatted as ``"{integer_part}p{decimal_part:02d}"``
    where *decimal_part* is two digits.  For example:

    * ``1.0``  → ``"1"``
    * ``1.2``  → ``"1p20"``
    * ``0.5``  → ``"0p50"``
    * ``1.35`` → ``"1p35"``

    Parameters
    ----------
    x : float or numpy.ndarray
        Value(s) to format.
    decimals : int
        Currently unused; retained for API parity.

    Returns
    -------
    str or list of str
        Formatted string (scalar input) or list of strings (array input).
    """
    if not isinstance(x, np.ndarray):
        if int(x) == x:
            formatted_x = str(int(x))
        else:
            integer_part = int(x)
            decimal_part = int(round((x - integer_part), 1) * 100)
            decimal_str = f"{decimal_part:02d}"
            formatted_x = f"{integer_part}p{decimal_str}"
    else:
        formatted_x = list()
        for n in range(len(x)):
            if int(x[n]) == x[n]:
                formatted_x.append(str(int(x[n])))
            else:
                integer_part = int(x[n])
                decimal_part = int(abs(x[n] - integer_part) * 100)
                decimal_str = f"{decimal_part:02d}"
                formatted_x.append(f"{integer_part}p{decimal_str}")
    return formatted_x


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles,
) -> np.ndarray:
    """Compute weighted quantiles of an array.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights corresponding to each value.
    quantiles : array-like
        Quantile(s) to compute, in [0, 1].

    Returns
    -------
    numpy.ndarray
        Weighted quantile values.  Returns NaN entries if ``values`` is
        empty or all finite-weight pairs are exhausted.
    """
    values = np.asarray(values).astype(float)
    weights = np.asarray(weights).astype(float)
    if values.size == 0:
        return np.array([np.nan for _ in quantiles])
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.array([np.nan for _ in quantiles])
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumw = np.cumsum(weights)
    if cumw[-1] == 0:
        return np.percentile(values, 100.0 * np.asarray(quantiles))
    cumw = cumw / cumw[-1]
    return np.interp(quantiles, cumw, values)


def check_consistent_wavelengths(wave: np.ndarray) -> bool:
    """Check whether wavelengths are consistent across all spectra.

    Parameters
    ----------
    wave : numpy.ndarray, shape (n_spectra, n_pixels)
        Wavelength array for each spectrum.

    Returns
    -------
    bool
        True if all spectra share the same wavelength grid.
    """
    ref_wave = wave[0, :]
    for i in range(1, wave.shape[0]):
        current_wave = wave[i, :]
        if not np.array_equal(ref_wave, current_wave):
            return False
    return True


def convert_masked_arrays(
    arr1: np.ma.MaskedArray,
    arr2: np.ma.MaskedArray,
):
    """Convert two masked arrays to regular NumPy arrays.

    Parameters
    ----------
    arr1, arr2 : numpy.ma.MaskedArray
        Input masked arrays.

    Returns
    -------
    arr1_data : numpy.ndarray
    arr2_data : numpy.ndarray
    arr1_masked_indices : numpy.ndarray
        Indices of masked values in *arr1*.
    arr2_masked_indices : numpy.ndarray
        Indices of masked values in *arr2*.
    """
    arr1_data = arr1.data
    arr2_data = arr2.data
    arr1_masked_indices = np.where(arr1.mask)[0]
    arr2_masked_indices = np.where(arr2.mask)[0]
    return arr1_data, arr2_data, arr1_masked_indices, arr2_masked_indices


# ---------------------------------------------------------------------------
# v0.23 additions
# ---------------------------------------------------------------------------

def find_nearest(array, value):
    """
    Find the nearest value to *value* in a 1-D numpy array.

    Parameters
    ----------
    array : array_like
        1-D array of values to search.
    value : float
        Target value.

    Returns
    -------
    nearest : scalar
        Element of *array* closest to *value*.

    Examples
    --------
    >>> find_nearest(np.array([1, 2, 3, 4, 5]), 2.7)
    3
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def convert_vega_to_ab(magnitude, band):
    """
    Convert a Vega-system magnitude to AB magnitude.

    Parameters
    ----------
    magnitude : float
        Magnitude in the Vega system.
    band : str
        Photometric band.  Valid values: ``'U'``, ``'B'``, ``'V'``,
        ``'R'``, ``'I'``, ``'J'``, ``'H'``, ``'K'``.

    Returns
    -------
    ab_magnitude : float
        Magnitude in the AB system.

    Raises
    ------
    ValueError
        If *band* is not in the supported set.
    """
    # Offsets from Vega to AB magnitudes
    vega_to_ab_offsets = {
        'U': 0.79,
        'B': -0.09,
        'V': 0.02,
        'R': 0.21,
        'I': 0.45,
        'J': 0.91,
        'H': 1.39,
        'K': 1.85
    }

    # Check if the band is valid
    if band not in vega_to_ab_offsets:
        raise ValueError(f"Invalid band '{band}'. Valid bands are: {', '.join(vega_to_ab_offsets.keys())}")

    # Calculate AB magnitude
    ab_magnitude = magnitude + vega_to_ab_offsets[band]

    return ab_magnitude


def bootstrap_corrcoeffs(X, Y, samples=1000):
    """
    Estimate the bootstrap standard error of the Pearson correlation coefficient.

    Parameters
    ----------
    X, Y : array_like
        Input data arrays (same length).
    samples : int, optional
        Number of bootstrap resamples.  Default 1000.

    Returns
    -------
    std_err : float
        Standard deviation of the bootstrap distribution of Pearson r.
    """
    # Number of bootstrap samples
    num_samples = samples

    # Store the calculated correlation coefficients
    bootstrap_corrcoeffs_list = []

    # Perform bootstrapping
    for _ in range(num_samples):
        # Resample with replacement
        resampled_x = np.random.choice(X, size=len(X), replace=True)
        resampled_y = np.random.choice(Y, size=len(Y), replace=True)

        # Calculate Pearson correlation coefficient for the resampled data
        correlation = np.corrcoef(resampled_x, resampled_y)[0, 1]

        bootstrap_corrcoeffs_list.append(correlation)

    # Calculate the standard error of the correlation coefficients
    return np.std(bootstrap_corrcoeffs_list)


def Utils_permute_nights_indices(array):
    """
    Permute the nights axis of a 4-D CCF array.

    Parameters
    ----------
    array : numpy.ndarray
        4-D array of shape ``(ccf_iterations, n_spectra, n_orders, n_nights)``.

    Returns
    -------
    permuted_array : numpy.ndarray
        Same shape as *array*, with the ``n_nights`` axis independently
        permuted for each ``(i, j, k)`` slice.
    """
    ccf_iterations, n_spectra, n_orders, n_nights = array.shape
    permuted_array = np.empty_like(array)

    for i in range(ccf_iterations):
        for j in range(n_spectra):
            for k in range(n_orders):
                permuted_array[i, j, k, :] = np.random.permutation(
                    array[i, j, k, :]
                    )
    return permuted_array
