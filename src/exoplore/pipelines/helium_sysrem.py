"""Optional Earth-frame SYSREM alternative for synthetic helium observations.

Normalize native spectra, divide by an Earth-frame out-of-transit reference,
then subtract weighted SYSREM components from the resulting residual matrix.
The output is differential transmission, not a fitted telluric transmittance.
Wavelengths are vacuum nm. No wavelength refinement is performed.
"""
from __future__ import annotations
import numpy as np
from exoplore.pipelines.helium_math import window_mask, resample
from exoplore.pipelines.sysrem import apply_sysrem


def correct_helium_sysrem(waves_nm, flux, error, science, n_iterations):
    """Return differential flux, conditional errors and removed components.

    Align onto a median Earth-frame grid before fitting. Only columns valid
    in every exposure enter SYSREM. Native OH masks are applied first.
    Errors hold the Earth-frame reference and fitted SYSREM components fixed.
    These errors exclude the temporal covariance and signal attenuation caused
    by estimating SYSREM components and must not establish detection claims.
    """
    if type(n_iterations) is not int or n_iterations < 1:
        raise ValueError('Supply a positive integer helium SYSREM iteration count')
    waves, data, errors = map(lambda x: np.asarray(x, float), (waves_nm, flux, error))
    if waves.ndim != 2 or data.shape != waves.shape or errors.shape != waves.shape:
        raise ValueError('Helium SYSREM requires matching exposure-by-pixel arrays')
    refs=np.asarray(science._reference_indices,dtype=int)
    if not len(refs):raise ValueError('Compute out-of-transit membership before SYSREM')
    grid=np.median(waves,axis=0)
    rows=[];variances=[]
    for wave,y,e in zip(waves,data,errors):
        valid=np.isfinite(y)&np.isfinite(e)&(e>0)
        if science.oh_mode=='mask':valid &= ~window_mask(wave,science.oh_topocentric_windows_nm)
        band=valid&window_mask(wave,science.continuum_stellar_windows_nm)
        if band.sum()<science.minimum_continuum_pixels:
            raise ValueError('Insufficient native continuum pixels for helium SYSREM')
        continuum=np.mean(y[band])
        if not np.isfinite(continuum) or continuum<=0:raise ValueError('Invalid SYSREM continuum')
        row,var=resample(wave,np.where(valid,y/continuum,np.nan),
                         np.where(valid,(e/continuum)**2,np.nan),grid)
        rows.append(row);variances.append(var)
    rows=np.asarray(rows);variances=np.asarray(variances)
    reference=np.mean(rows[refs],axis=0)
    good=np.flatnonzero(np.isfinite(rows).all(axis=0)&np.isfinite(variances).all(axis=0)
                       &(variances>0).all(axis=0)&np.isfinite(reference)&(reference>0))
    if len(good)<=n_iterations or n_iterations>=len(rows):
        raise ValueError('Too many SYSREM components for the valid spectral matrix')
    residual=rows[:,good]/reference[good]-1
    sigma=np.sqrt(variances[:,good])/reference[good]
    # Existing SYSREM initializes from one column. An exactly zero residual
    # contains no component to fit, and must not produce a 0/0 iteration.
    cleaned=residual.copy();components=[]
    for _ in range(n_iterations):
        if not np.any(cleaned):break
        order=np.arange(len(good))
        seed_column=len(good)//3
        if not np.any(cleaned[:,seed_column]):
            strongest=int(np.argmax(np.sum((cleaned/sigma)**2,axis=0)))
            order[seed_column],order[strongest]=order[strongest],order[seed_column]
        filtered,removed_component=apply_sysrem(cleaned[:,order],sigma[:,order],1)
        inverse=np.argsort(order)
        cleaned=filtered[:,inverse]
        components.append(removed_component[0][:,inverse])
    if not np.isfinite(cleaned).all():raise ValueError('Helium SYSREM returned nonfinite residuals')
    corrected=np.full_like(rows,np.nan);uncertainty=np.full_like(rows,np.nan)
    corrected[:,good]=1+cleaned;uncertainty[:,good]=sigma
    removed=np.zeros((len(components),*rows.shape))
    for i,component in enumerate(components):removed[i,:,good]=component.T
    return dict(wave_nm=np.broadcast_to(grid,rows.shape).copy(),flux=corrected,
                error=uncertainty,components=removed,earth_reference=reference,
                good_pixels=good,residual_before_sysrem=np.where(np.isfinite(rows),rows/reference-1,np.nan))
