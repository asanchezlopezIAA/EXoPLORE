"""Analytical noise propagation for an unweighted helium transmission coadd.

Propagate the one supplied pixel-variance array, without drawing additional
observations. Continuum and molecfit fit parameters remain fixed. Shared
reference noise and interpolation correlations are retained in linear operators.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from exoplore.pipelines.helium_math import stellar_wavelength, planet_wavelength, window_mask


def interpolation_operator(wave, destination, valid):
    """Sparse linear interpolator with finite-pixel and detector-gap masking."""
    index=np.clip(np.searchsorted(wave,destination,side='right')-1,0,len(wave)-2)
    widths=np.diff(wave)
    t=(destination-wave[index])/widths[index]
    good=(t>=0)&(t<=1)&valid[index]&valid[index+1]&(widths[index]<=5*np.median(widths))
    rows=np.flatnonzero(good);left=index[good]
    return csr_matrix((np.r_[1-t[good],t[good]],(np.r_[rows,rows],np.r_[left,left+1])),shape=(len(destination),len(wave)))


def carmenes_coadd_covariance(waves_nm, flux, error, berv_kms, rv_kms, config, result):
    """Return spectral covariance conditional on fitted continua and correction.

    This differentiates the arithmetic-reference division and both wavelength
    interpolations. Out-of-transit noise enters once, shared by every transit
    spectrum; it is not divided by the number of transit exposures again.
    The continuum means are fixed at their fitted values in this calculation.
    Invalid coadd pixels must remain masked when using the covariance.
    """
    refs=np.asarray(config.reference_running_numbers)-1
    inside=np.asarray(config.in_transit_running_numbers)-1
    grid=result['planet_wave_nm'];master=result['master']
    inv=np.divide(1.,master,out=np.zeros_like(master),where=np.isfinite(master)&(master>0))
    operators=[];normalized_variance=[]
    for w,y,e,b in zip(waves_nm,flux,error,berv_kms):
        good=np.isfinite(y)&np.isfinite(e)&(e>0)
        bands=window_mask(w,config.continuum_stellar_windows_nm)&good
        continuum=np.mean(y[bands])
        operators.append(interpolation_operator(stellar_wavelength(w,b,config.gamma_kms),grid,good))
        normalized_variance.append(np.where(good,(e/continuum)**2,0.))
    covariance=csr_matrix((len(grid),len(grid)),dtype=float)
    reference_operator=csr_matrix(covariance.shape,dtype=float)
    for i in inside:
        f=result['normalized_stellar_flux'][i]
        valid=np.isfinite(result['stellar_transmission'][i])
        shift=interpolation_operator(planet_wavelength(grid,rv_kms[i]),grid,valid)
        numerator=shift@diags(inv)@operators[i]/len(inside)
        covariance+=numerator@diags(normalized_variance[i])@numerator.T
        derivative=np.where(np.isfinite(f),-f*inv**2,0.)
        reference_operator+=shift@diags(derivative)/len(inside)
    for j in refs:
        operator=reference_operator@operators[j]/len(refs)
        covariance+=operator@diags(normalized_variance[j])@operator.T
    return covariance


def carmenes_lightcurve_covariance(waves_nm, flux, error, berv_kms,
                                  rv_kms, config, result):
    """Propagate native noise into the planetary-band light curve.

    Returns dimensionless transmission covariance between exposures. Both
    interpolations and the shared arithmetic reference are differentiated,
    including numerator/reference cancellation for reference exposures.
    Continuum estimates and telluric fits are held fixed. A missing band pixel
    invalidates that exposure, matching the preparation's arithmetic mean.
    """
    refs = np.asarray(config.reference_running_numbers) - 1
    grid = result['planet_wave_nm']
    band = window_mask(grid, [config.planet_lightcurve_window_nm])
    if not band.any():
        raise ValueError('Light-curve band contains no pixels')
    weights = band.astype(float) / band.sum()
    master = result['master']
    inv = np.divide(1., master, out=np.zeros_like(master),
                    where=np.isfinite(master) & (master > 0))
    operators, variances = [], []
    for w, y, e, b in zip(waves_nm, flux, error, berv_kms):
        good = np.isfinite(y) & np.isfinite(e) & (e > 0)
        continuum = np.mean(y[window_mask(w, config.continuum_stellar_windows_nm) & good])
        operators.append(interpolation_operator(
            stellar_wavelength(w, b, config.gamma_kms), grid, good))
        variances.append(np.where(good, (e / continuum)**2, 0.))
    n = len(operators)
    numerator = []
    reference = []
    valid_rows = np.isfinite(result['planet_transmission'][:, band]).all(axis=1)
    for i, rv in enumerate(rv_kms):
        valid = np.isfinite(result['stellar_transmission'][i])
        shift = interpolation_operator(planet_wavelength(grid, rv), grid, valid)
        projection = csr_matrix(weights[None]) @ shift
        numerator.append((projection @ diags(inv) @ operators[i]).toarray()[0])
        f = result['normalized_stellar_flux'][i]
        derivative = np.where(np.isfinite(f), -f * inv**2, 0.)
        reference.append(projection @ diags(derivative) / len(refs))
    reference = vstack(reference)
    covariance = np.zeros((n, n))
    for j, variance in enumerate(variances):
        jacobian = (reference @ operators[j]).toarray() if j in refs else np.zeros((n, len(variance)))
        jacobian[j] += numerator[j]
        weighted = jacobian * np.sqrt(variance)[None]
        covariance += weighted @ weighted.T
    covariance[~valid_rows, :] = np.nan
    covariance[:, ~valid_rows] = np.nan
    return covariance
