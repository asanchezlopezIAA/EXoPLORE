"""CARMENES transmission preparation following Pallé et al. (2020).

Normalize before stellar alignment, divide by the arithmetic-mean reference,
then take the arithmetic mean of planet-aligned T2--T3 spectra. The synthetic
adapter excludes OH emission; this is not a complete CARACAL/sky reduction.
Reference: https://doi.org/10.1051/0004-6361/202037719
"""
from __future__ import annotations
import numpy as np
from exoplore.pipelines.czesla2024 import exposure_indices
from exoplore.pipelines.helium_math import stellar_wavelength, planet_wavelength, window_mask, resample


def gap_resample(wave, flux, variance, destination):
    """Interpolate within detector sections, never across their physical gap."""
    y, v = resample(wave, flux, variance, destination)
    widths = np.diff(wave)
    gaps = np.flatnonzero(widths > 5*np.median(widths))
    for i in gaps:
        missing = (destination > wave[i]) & (destination < wave[i+1])
        y[missing] = np.nan
        v[missing] = np.nan
    return y, v


def arithmetic_mean(flux, variance):
    """Require complete frame coverage for an unweighted mean spectrum."""
    complete = np.isfinite(flux).all(axis=0) & np.isfinite(variance).all(axis=0)
    value = np.where(complete, np.mean(flux, axis=0), np.nan)
    error = np.where(complete, np.sum(variance, axis=0)/len(flux)**2, np.nan)
    return value, error


def prepare_carmenes_transmission(waves_nm, flux, error, berv_kms, planet_rv_kms, config):
    """Prepare a corrected time series and propagate diagonal pixel weights.

    Include finite-reference variance in the ratio's marginal variance for
    in-transit frames. Reference noise is shared between frames; consequently
    the final conditional coadd variance excludes its cross-frame covariance.
    Continuum-estimation covariance is also excluded. No significance follows
    from these diagnostic weights alone.
    """
    waves, data, errors = map(lambda a: np.asarray(a, float), (waves_nm, flux, error))
    if waves.ndim != 2 or data.shape != waves.shape or errors.shape != waves.shape:
        raise ValueError('Expected matching exposure-by-pixel wavelength, flux and error')
    n = len(data)
    refs = np.asarray(config._reference_indices, dtype=int)
    inside = np.asarray(config._full_transit_indices, dtype=int)
    if not len(refs) or not len(inside):
        raise ValueError('Calculate exposure membership from observing times before preparing spectra')
    stellar = np.asarray([stellar_wavelength(w, b, config.gamma_kms) for w, b in zip(waves, berv_kms)])
    grid = np.median(stellar, axis=0)
    rows, variances = [], []
    for w, sw, row, err in zip(waves, stellar, data, errors):
        bands = window_mask(w, config.continuum_stellar_windows_nm)
        good = bands & np.isfinite(row) & np.isfinite(err) & (err > 0)
        if good.sum() < config.minimum_continuum_pixels:
            raise ValueError('Insufficient continuum coverage')
        continuum = float(np.mean(row[good]))
        if continuum <= 0 or not np.isfinite(continuum):
            raise ValueError('Invalid continuum normalization')
        y, v = gap_resample(sw, row/continuum, (err/continuum)**2, grid)
        rows.append(y); variances.append(v)
    rows, variances = np.asarray(rows), np.asarray(variances)
    master, master_variance = arithmetic_mean(rows[refs], variances[refs])
    ratio = rows/master
    # Conditional reference-fixed errors retained separately for coaddition.
    ratio_variance = variances/master**2
    marginal = ratio_variance + rows**2*master_variance/master**4
    marginal[refs] -= 2*rows[refs]*variances[refs]/(len(refs)*master**3)
    planet, pvar = [], []
    for row, var, rv in zip(ratio, ratio_variance, planet_rv_kms):
        y, v = gap_resample(planet_wavelength(grid, rv), row, var, grid)
        planet.append(y); pvar.append(v)
    planet, pvar = np.asarray(planet), np.asarray(pvar)
    coadd, variance = arithmetic_mean(planet[inside], pvar[inside])
    def curve(array, window):
        return np.mean(array[:, window_mask(grid, [window])], axis=1)
    return dict(stellar_wave_nm=grid, stellar_transmission=ratio,
                stellar_marginal_variance=marginal, planet_wave_nm=grid,
                planet_transmission=planet, planet_coadd=coadd,
                conditional_coadd_variance=variance, master=master,
                master_variance=master_variance, normalized_stellar_flux=rows,
                normalized_stellar_variance=variances,
                planet_lightcurve=curve(planet, config.planet_lightcurve_window_nm),
                stellar_lightcurve=curve(ratio, config.stellar_lightcurve_window_nm))
