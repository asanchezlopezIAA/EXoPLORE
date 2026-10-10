"""Analytical noise propagation for direct helium transmission spectra.

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
    refs=np.asarray(config._reference_indices, dtype=int)
    inside=np.asarray(config._full_transit_indices, dtype=int)
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
    refs = np.asarray(config._reference_indices, dtype=int)
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


def weighted_crires_uncertainties(waves_nm, flux, error, berv_kms, rv_kms,
                                  config, result):
    """Propagate extraction errors through the weighted CRIRES preparation.

    Differentiate the reference division and both interpolations, retaining
    shared-reference covariance. Continuum coefficients, weights and telluric
    fits are held fixed. This is linear error propagation, not resampling or
    the parameter-posterior calculation of Czesla et al. (2024).
    """
    grid = result['stellar_wave_nm']
    refs = np.asarray(config._reference_indices, dtype=int)
    inside = np.asarray(config._full_transit_indices, dtype=int)
    n, pixels = np.shape(flux)
    bands = window_mask(grid, config.continuum_stellar_windows_nm)
    x = grid - grid[bands].mean()
    master = result['master']
    inv = np.divide(1., master, out=np.zeros_like(master),
                    where=np.isfinite(master) & (master > 0))
    f = result['normalized_stellar_flux']
    variance = result['normalized_stellar_variance']
    weights = np.divide(1., variance[refs], out=np.zeros_like(variance[refs]),
                        where=np.isfinite(variance[refs]) & (variance[refs] > 0))
    denominator = weights.sum(axis=0)
    coefficients = np.divide(weights, denominator, out=np.zeros_like(weights),
                             where=denominator > 0)
    native_operators, native_variance, shifts, pvariance = [], [], [], []
    for i, (w, y, e, b, rv) in enumerate(zip(waves_nm, flux, error, berv_kms, rv_kms)):
        valid = np.isfinite(y) & np.isfinite(e) & (e > 0)
        align = interpolation_operator(stellar_wavelength(w, b, config.gamma_kms), grid, valid)
        row = align @ np.where(valid, y, 0.)
        var = align.multiply(align) @ np.where(valid, e**2, 0.)
        good = bands & np.isfinite(variance[i]) & (var > 0)
        fitted = np.polyfit(x[good], row[good], config.continuum_polynomial_degree,
                           w=1/np.sqrt(var[good]))
        continuum = np.polyval(fitted, x)
        native_operators.append(diags(1/continuum) @ align)
        native_variance.append(np.where(valid, e**2, 0.))
        shift = interpolation_operator(planet_wavelength(grid, rv), grid,
                                       np.isfinite(result['stellar_transmission'][i]))
        shifts.append(shift)
        pvariance.append(shift.multiply(shift) @ np.nan_to_num(variance[i]*inv**2))
    pvariance = np.asarray(pvariance)
    valid = np.isfinite(result['planet_transmission'][inside]) & (pvariance[inside] > 0)
    weight = np.divide(1., pvariance[inside], out=np.zeros_like(pvariance[inside]), where=valid)
    total = weight.sum(axis=0)
    beta = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    reference = csr_matrix((pixels, pixels))
    for k, i in enumerate(inside):
        reference += diags(beta[k]) @ shifts[i] @ diags(np.nan_to_num(-f[i]*inv**2))
    covariance = csr_matrix((pixels, pixels))
    curve_reference, curve_numerator = {}, {}
    for frame in ('planet', 'stellar'):
        selected = window_mask(grid, [getattr(config, frame+'_lightcurve_window_nm')])
        if not selected.any():
            raise ValueError('Light-curve band contains no pixels')
        band = csr_matrix((selected.astype(float)/selected.sum())[None, :])
        projections = [band @ shifts[i] if frame == 'planet' else band for i in range(n)]
        curve_reference[frame] = vstack([projection @ diags(np.nan_to_num(-f[i]*inv**2))
                                        for i, projection in enumerate(projections)])
        curve_numerator[frame] = [projection @ diags(inv) @ native_operators[i]
                                  for i, projection in enumerate(projections)]
    curve_covariance = {frame: np.zeros((n, n)) for frame in curve_reference}
    for j, native in enumerate(native_operators):
        jacobian = csr_matrix((pixels, pixels))
        if j in refs:
            coefficient = diags(coefficients[np.flatnonzero(refs == j)[0]])
            jacobian += reference @ coefficient @ native
        if j in inside:
            k = np.flatnonzero(inside == j)[0]
            jacobian += diags(beta[k]) @ shifts[j] @ diags(inv) @ native
        covariance += jacobian @ diags(native_variance[j]) @ jacobian.T
        for frame in curve_reference:
            derivative = (curve_reference[frame] @ coefficient @ native).toarray() if j in refs else np.zeros((n, pixels))
            derivative[j] += curve_numerator[frame][j].toarray()[0]
            curve_covariance[frame] += (derivative*native_variance[j]) @ derivative.T
    output = {'planet_coadd_error': np.where(np.isfinite(result['planet_coadd']),
                                             np.sqrt(np.maximum(covariance.diagonal(), 0)), np.nan)}
    for frame, cov in curve_covariance.items():
        good = np.isfinite(result[frame+'_lightcurve'])
        output[frame+'_lightcurve_error'] = np.where(good, np.sqrt(np.maximum(np.diag(cov), 0)), np.nan)
        output[frame+'_lightcurve_covariance'] = np.where(good[:, None] & good[None, :], cov, np.nan)
    selected = np.flatnonzero(window_mask(grid, [config.equivalent_width_window_nm]))
    integration = np.zeros(pixels)
    if len(selected) > 1 and np.isfinite(result['planet_coadd'][selected]).all():
        steps = np.diff(grid[selected])
        integration[selected[:-1]] += steps/2
        integration[selected[1:]] += steps/2
        ew_error = np.sqrt(max(float(integration @ (covariance @ integration)), 0))*1e4
    else:
        ew_error = np.nan
    output['equivalent_width_error_mA'] = np.asarray(ew_error)
    return output
