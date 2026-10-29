"""Repeat stellar-reference preparation to propagate fitted continuum noise.

prepare_transmission() supplies the reduction function used for every draw.
Repeating that function propagates the common out-of-transit reference and
continuum fitting into correlations between final wavelength pixels.
"""

import numpy as np
from scipy.sparse import csr_matrix


def sample_preparation_covariance(flux, error, expectation, draws, seed, reduce):
    """Draw native-pixel noise and repeat the supplied preparation function.

    Each draw gets its own continuum estimates, exposure weights and shared
    master-out. Telluric and stellar correction models remain fixed. Supplying
    a noiseless expectation is useful for injection tests. Otherwise the
    measured flux supplies the mean of this parametric bootstrap.
    """
    # Use a separate recorded seed for uncertainty resampling.
    rng = np.random.default_rng(seed)
    # Choose the simulation expectation when supplied, or the measured spectrum.
    mean = flux if expectation is None else np.asarray(expectation, dtype=float)
    # Keep the same usable native pixels throughout the noise draws.
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    if mean.shape != flux.shape or np.any(valid & ~np.isfinite(mean)):
        raise ValueError("noise_expectation must match flux and be finite on valid pixels")
    # Collect the final spectrum and optional light curve from each reduction.
    spectra, curves = [], []
    # Each iteration represents one Gaussian noise realization of the input arrays.
    for _ in range(draws):
        # Draw pixel noise with the supplied standard deviations and preserve masks.
        realization = np.where(valid, mean + rng.normal(size=flux.shape) * error, np.nan)
        # Repeat alignment, continuum normalization, master construction and coaddition.
        result = reduce(realization)
        # Store the planet-frame mean spectrum from this draw.
        spectra.append(result.planet_coadd)
        # Store the corresponding exposure-by-exposure band averages when requested.
        if result.planet_lightcurve is not None:
            curves.append(result.planet_lightcurve)
    # Arrange draws as rows and wavelength channels as columns.
    spectra = np.asarray(spectra)
    # Fixed coverage across draws gives one common covariance for valid pixels.
    # Missing edge channels retain zero entries, matching the analytical route.
    # Select channels measured successfully in every draw.
    covered = np.isfinite(spectra).all(axis=0)
    # Remember their locations in the complete output spectrum.
    indices = np.flatnonzero(covered)
    # Subtract the across-draw mean before estimating covariance.
    centered = spectra[:, covered] - spectra[:, covered].mean(axis=0)
    # Allocate the full wavelength-by-wavelength covariance matrix.
    covariance = np.zeros((spectra.shape[1], spectra.shape[1]))
    # The sample covariance uses draws - 1 in the denominator.
    covariance[np.ix_(indices, indices)] = centered.T @ centered / (draws - 1)
    # Leave the light-curve covariance empty when the reduction supplied no curve.
    lightcurve_covariance = None
    # Apply the same sample-covariance calculation to the light curves.
    # Here columns are exposures, so off-diagonal entries describe correlations in time.
    if curves:
        curves = np.asarray(curves)
        covered = np.isfinite(curves).all(axis=0)
        centered = curves[:, covered] - curves[:, covered].mean(axis=0)
        lightcurve_covariance = np.full((curves.shape[1], curves.shape[1]), np.nan)
        indices = np.flatnonzero(covered)
        lightcurve_covariance[np.ix_(indices, indices)] = centered.T @ centered / (draws - 1)
    # Store the spectral covariance in the sparse format expected by preparation.
    return csr_matrix(covariance), lightcurve_covariance
