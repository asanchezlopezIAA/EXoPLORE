"""Palle2020 preparation of real CARMENES He I transit observations.

Implement the OH-masked alternative of Pallé et al. (2020). Source CARACAL
spectra are preserved; fibre B locates emission and is never science flux.
The existing arithmetic-reference CARMENES transmission implementation is
reused. No SYSREM or uncertainty resampling is applied.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import numpy as np
from astropy.io import fits

from exoplore.config.carmenes_helium import CarmenesHeliumConfig
from exoplore.config.helium_oh import validate_oh_treatment, uses_sky_fibre
from exoplore.instruments.carmenes_palle2020 import read_night, sky_windows, fit_exposure
from exoplore.pipelines.carmenes_helium import prepare_carmenes_transmission
from exoplore.pipelines.helium_math import window_mask


def validate_run_config(config) -> None:
    """Validate an observed, single-night Palle2020 choice before I/O."""
    if config.instrument.name!='CARMENES_NIR' or not config.observation.use_real_data:
        raise ValueError('Observed Palle2020 requires CARMENES_NIR and use_real_data=true')
    if config.observation.n_nights!=1 or config.observation.event_type!='transit':
        raise ValueError('Prepare each observed transit night separately')
    if config.pipeline.sysrem_iterations!=0 or config.retrieval.enabled:
        raise ValueError('Palle2020 uses direct transmission without SYSREM or molecular retrieval')
    if not config.pipeline.palle2020 or not config.planet.parameter_file:
        raise ValueError('Supply pipeline.palle2020 settings and the planet parameter file')
    validate_oh_treatment(config)
    if len(config.instrument.order_indices)!=1:
        raise ValueError('Select one helium-containing NIR order')
    if not 0<=config.instrument.order_indices[0]<28:
        raise ValueError('CARMENES NIR order index must lie between 0 and 27')


def selection(science, bjd, duration):
    """Select transit spectra by midpoint/half-exposure overlap with T1--T4.

    Pallé et al. count an exposure in transit when at least half is between
    T1 and T4. The final spectrum uses midpoint membership between T2 and T3.
    """
    from dataclasses import replace
    from exoplore.observation.timing import orbital_phase
    phase=orbital_phase(bjd,science.t0_bjd_tdb,science.period_days)
    t1,t2,t3,t4=science.optical_contact_phases
    half=duration/(2*science.period_days*86400)
    fraction=np.maximum(0,np.minimum(phase+half,t4)-np.maximum(phase-half,t1))/(2*half)
    inside=np.flatnonzero(fraction>=.5)
    outside=np.flatnonzero(fraction<.5)
    full=np.flatnonzero((phase>=t2)&(phase<=t3))
    resolved=replace(science)
    object.__setattr__(resolved,'_reference_indices',tuple(outside.tolist()))
    object.__setattr__(resolved,'_full_transit_indices',tuple(full.tolist()))
    if not len(outside) or not len(full):
        raise ValueError('Night needs out-of-transit and T2--T3 exposures')
    return resolved,phase,inside,outside,full


def _write_inputs(directory: Path,night: dict) -> None:
    """Create EXoPLORE per-order input matrices, retaining native masks/grids."""
    directory.mkdir()
    arrays=dict(julian_date=night['bjd_tdb'],observations_berv=night['berv_kms'],
                airmass=np.array([h['AIRMASS'] for h in night['headers']]),sig=night['error'],
                wave_perframe=night['wave_nm']/1000,
                wave=np.median(night['wave_nm'],axis=0)/1000,
                valid_pixels=np.isfinite(night['flux'])&np.isfinite(night['error'])&(night['error']>0))
    arrays['snr']=np.divide(night['flux'],night['error'],out=np.full_like(night['flux'],np.nan),
                            where=np.isfinite(night['error'])&(night['error']>0))
    for name,data in arrays.items():
        fits.writeto(directory/f'{name}_0.fits',data.astype(np.uint8) if data.dtype==bool else data)
    for order in range(night['flux'].shape[1]):
        fits.writeto(directory/f'observations_night_0_order_{order}.fits',night['flux'][:,order])


def preparing_pipeline_adapter(inp_dat,data,noise,wave,mask,masks,correct_uncertainties,retrieval):
    """Prepare an already corrected, native-masked order through the dispatcher."""
    treatment=inp_dat.get('oh_treatment','mask')
    if retrieval or inp_dat.get('telluric_corrected') is not True:
        raise ValueError('Palle2020 array preparation requires telluric-corrected spectra')
    if treatment not in ('ignore','mask','fiber_b_matched_extraction','fiber_b_calibrated','fiber_a_oot_profile'):
        raise ValueError('Unknown Palle2020 OH treatment')
    if treatment=='mask' and inp_dat.get('oh_mask_applied') is not True:
        raise ValueError('Palle2020 mask mode requires the native OH mask before interpolation')
    if uses_sky_fibre(treatment) and inp_dat.get('oh_treatment_applied')!=treatment:
        raise ValueError('Apply the selected fibre-B subtraction upstream of array preparation')
    science=inp_dat['palle2020_config']
    if isinstance(science,dict):
        science=CarmenesHeliumConfig(**science)
    science=replace(science,oh_mode='mask' if treatment=='mask' else 'baseline')
    bjd=np.asarray(inp_dat['julian_date'])
    duration=np.broadcast_to(inp_dat['exposure_time_seconds'],bjd.shape)
    science,_,_,_,_=selection(science,bjd,duration)
    flux=np.array(data,float,copy=True);error=np.array(noise,float,copy=True)
    flux[:,np.asarray(mask,int)]=np.nan;error[:,np.asarray(mask,int)]=np.nan
    wavelengths=np.broadcast_to(wave,flux.shape)
    result=prepare_carmenes_transmission(wavelengths,flux,error,np.asarray(inp_dat['berv_kms']),
                                         np.asarray(inp_dat['planet_rv_kms']),science)
    prepared=result['stellar_transmission'].copy()
    propagated=np.sqrt(np.maximum(result['stellar_marginal_variance'],0))
    bad=np.flatnonzero(np.any(~np.isfinite(prepared)|~np.isfinite(propagated)|(propagated<=0),axis=0))
    useful=np.setdiff1d(np.arange(prepared.shape[1]),bad)
    prepared[:,bad]=1.;propagated[:,bad]=1.
    if not masks:
        return (prepared,propagated) if correct_uncertainties else prepared
    return prepared,propagated,useful,bad,0,bad.copy(),useful.copy(),None,None


def run_palle2020(config, *, pilot: bool=False, pilot_label: str='pilot') -> Path:
    """Read paired A/B spectra, mask OH, correct H2O and prepare transmission."""
    validate_run_config(config)
    choices=config.pipeline.palle2020
    treatment=validate_oh_treatment(config)
    science=replace(config.pipeline.carmenes_helium,
                    oh_mode='mask' if treatment=='mask' else 'baseline')
    night=read_night(Path(choices['input_directory']))
    order=config.instrument.order_indices[0]
    centres=np.asarray(science.helium_vacuum_lines_nm)
    native=night['wave_nm'][:,order]
    if not np.all((native[:,0,None]<centres)&(native[:,-1,None]>centres)):
        raise ValueError('The selected order does not contain the vacuum helium triplet')
    windows=(sky_windows(night,order,choices['oh_catalogue_nm'],choices['oh_mask_halfwidth_nm'])
             if treatment=='mask' else [])
    science,phase,inside,outside,full=selection(science,night['bjd_tdb'],night['duration_seconds'])
    rv=science.kp_kms*np.sin(2*np.pi*phase)
    label=(pilot_label if pilot else choices.get('output_name','night' if treatment=='mask' else f'night_{treatment}'))
    if not label or label in ('.','..') or Path(label).name!=label:
        raise ValueError('Output label must be a single directory name')
    output=Path(config.paths.output_root)/config.planet.name/'Palle2020'/label
    output.mkdir(parents=True,exist_ok=False)
    if not pilot:
        _write_inputs(output/'reference_night',night)
    with (output/'input_provenance.json').open('x') as stream:
        json.dump(dict(sources=night['sources'],configuration=choices,oh_windows_nm=windows,
                       science_fibre='A',sky_fibre='B',sky_subtraction=uses_sky_fibre(treatment),
                       oh_treatment=treatment,
                       exposure_membership='Calculated from BJD and optical contacts',
                       in_transit=inside.tolist(),out_of_transit=outside.tolist(),full_transit=full.tolist()),
                  stream,indent=2)
    raw=night['flux'][:,order].copy();error=night['error'][:,order].copy()
    sky_result=transfer=None
    if uses_sky_fibre(treatment):
        from exoplore.instruments.carmenes_oh import (
            OHFitConfig, fit_fibre_b, transfer_to_fibre_a, subtract_matched_fibre_b,
        )
        settings=OHFitConfig(**choices['oh_fit']['profile'])
        sky_result=fit_fibre_b(night,order,settings)
        transfer=(subtract_matched_fibre_b(night,order,sky_result,choices['oh_fit'].get('fiber_b_scale'),settings,reference_indices=outside)
                  if treatment=='fiber_b_matched_extraction' else
                  transfer_to_fibre_a(night,order,sky_result,outside,settings))
        raw=transfer['corrected_flux'].copy();error=transfer['conditional_error'].copy()
        arrays={k:v for k,v in sky_result.items() if isinstance(v,np.ndarray)}
        arrays.update({k:v for k,v in transfer.items() if isinstance(v,np.ndarray)})
        if treatment != 'fiber_a_oot_profile':
            with (output/'oh_model.npz').open('xb') as stream:
                np.savez_compressed(stream,**arrays)
            with (output/'oh_report.json').open('x') as stream:
                json.dump(dict(oh_treatment=treatment,settings=asdict(settings),
                    sky={k:v for k,v in sky_result.items() if not isinstance(v,np.ndarray)},
                    transfer={k:v for k,v in transfer.items() if not isinstance(v,np.ndarray)}),stream,indent=2)
    masked=window_mask(native,windows) if treatment=='mask' else np.zeros_like(native,dtype=bool)
    valid=np.isfinite(raw)&np.isfinite(error)&(error>0)&~masked
    raw[~valid]=np.nan;error[~valid]=np.nan
    corrections=[None]*len(raw)
    reused=choices.get('validated_first_exposure_path') if not pilot else None
    if reused:
        if treatment!='mask':
            raise ValueError('The existing masked pilot cannot be reused for a different OH treatment; clear validated_first_exposure_path and fit the new science input')
        prior=Path(reused).resolve()
        provenance=json.loads((prior/'input_provenance.json').read_text())
        report_path=prior/'molecfit/exposure_01/report.json'
        report=json.loads(report_path.read_text())
        if (not report['accepted'] or provenance['sources']!=night['sources'] or
            provenance['configuration']['transmission']!=choices['transmission'] or
            provenance['oh_windows_nm']!=windows):
            raise ValueError('Pilot correction does not match the current data and preparation')
        lsf=report['grid_and_lsf']
        for field in ('gaussian_relative_fwhm','lorentzian_relative_fwhm'):
            if lsf.get(field)!=choices['molecfit'].get(field):
                raise ValueError('Pilot and night LSF coefficients differ')
        expected=provenance['configuration']['molecfit']['windows_nm']
        if expected!=choices['molecfit']['windows_nm']:
            raise ValueError('Pilot and night fitting intervals differ')
        path=prior/'molecfit/exposure_01/correction.npz'
        with np.load(path) as saved:
            if not np.array_equal(saved['native_wave_nm'],native[0]):
                raise ValueError('Pilot native wavelength differs')
            trans=saved['transmittance'].copy()
            corrections[0]=dict(wave_nm=saved['refined_wave_nm'].copy(),
                flux=raw[0]/trans,error=error[0]/trans,transmittance=trans)
        with (output/'reused_first_exposure.json').open('x') as stream:
            json.dump(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           report=str(report_path),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest()),stream,indent=2)
    def fit_one(i):
        correction=fit_exposure((output/'molecfit'/f'exposure_{i+1:02d}').resolve(),
            native[i],raw[i],error[i],night['headers'][i],choices['molecfit'],science,
            float(rv[i]),float(night['berv_kms'][i]),float(night['utc_mjd'][i]))
        return i,correction
    selected=[0] if pilot else [i for i,c in enumerate(corrections) if c is None]
    workers=1 if pilot else choices['molecfit_workers']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fit_one,i) for i in selected]
        for future in as_completed(futures):
            i,correction=future.result()
            corrections[i]=correction
            print(f'Palle2020: molecfit {i+1}/{len(raw)} accepted',flush=True)
    if pilot:
        return output
    from exoplore.plotting.helium_molecfit import plot_molecfit_diagnostics
    plot_molecfit_diagnostics(output,output/'molecfit_diagnostics.png','CARMENES_NIR')
    waves=np.array([c['wave_nm'] for c in corrections])
    flux=np.array([c['flux'] for c in corrections]);errors=np.array([c['error'] for c in corrections])
    if treatment == 'fiber_a_oot_profile':
        from exoplore.instruments.carmenes_oh_oot import OOTProfileConfig, fit_oot_profile
        oot_settings=OOTProfileConfig(**choices['oh_fit']['oot_profile'])
        transmission=np.array([c['transmittance'] for c in corrections])
        preliminary_transfer=transfer
        transfer=fit_oot_profile(native,waves,night['flux'][:,order],night['error'][:,order],
            transmission,night['berv_kms'],outside,science,sky_result,
            preliminary_transfer,settings,oot_settings)
        arrays={k:v for k,v in sky_result.items() if isinstance(v,np.ndarray)}
        arrays.update({k:v for k,v in transfer.items() if isinstance(v,np.ndarray)})
        # The initial correction was used only for telluric fitting. Verify
        # that the final OH profile leaves its accepted anchors unchanged.
        anchors=window_mask(native,choices['molecfit']['windows_nm'])
        change=np.divide(abs(transfer['model_a']-preliminary_transfer['model_a']),
            night['error'][:,order],out=np.full_like(native,np.inf),
            where=np.isfinite(night['error'][:,order])&(night['error'][:,order]>0))
        maximum_change=float(np.max(change[anchors])) if anchors.any() else float('inf')
        threshold=choices['oh_fit']['maximum_anchor_change_sigma']
        if maximum_change>threshold:
            transfer['accepted']=False
            transfer['rejection_reasons'].append('Final OH profile changes the telluric fitting anchors; refitting is required')
        with (output/'oh_model.npz').open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        with (output/'oh_report.json').open('x') as stream:
            json.dump(dict(oh_treatment=treatment,settings=asdict(settings),
                oot_profile_settings=asdict(oot_settings),maximum_anchor_change_sigma=maximum_change,
                anchor_change_limit_sigma=threshold,
                sky={k:v for k,v in sky_result.items() if not isinstance(v,np.ndarray)},
                transfer={k:v for k,v in transfer.items() if not isinstance(v,np.ndarray)}),stream,indent=2)
        if not transfer['accepted']:
            raise ValueError('OOT OH profile rejected: '+'; '.join(transfer['rejection_reasons']))
        raw=transfer['corrected_flux'].copy();error=transfer['conditional_error'].copy()
        flux=raw/transmission;errors=error/transmission
        valid &= np.isfinite(transmission)&(transmission>=science.telluric_min_transmission)
    # Keep the native OH mask through the wavelength refinement.
    flux[~valid]=np.nan;errors[~valid]=np.nan
    with (output/'observations.npz').open('xb') as stream:
        np.savez_compressed(stream,native_wave_nm=native,raw_flux=night['flux'][:,order],
            raw_error=night['error'][:,order],wave_nm=waves,corrected_flux=flux,
            corrected_error=errors,oh_mask=masked,oh_treated_flux=raw,
            oh_treated_error=error,
            telluric_transmission=np.array([c['transmittance'] for c in corrections]))
    if uses_sky_fibre(treatment):
        from exoplore.plotting.carmenes_oh import plot_oh_correction
        plot_oh_correction(output,output/'oh_correction.png',science,int(np.argmin(abs(phase)))+1)
    result=prepare_carmenes_transmission(waves,flux,errors,night['berv_kms'],rv,science)
    from exoplore.pipelines.helium_noise_propagation import carmenes_coadd_covariance, carmenes_lightcurve_covariance
    covariance=carmenes_coadd_covariance(waves,flux,errors,night['berv_kms'],rv,science,result)
    result['planet_coadd_error']=np.where(np.isfinite(result['planet_coadd']),
                                        np.sqrt(np.maximum(covariance.diagonal(),0)),np.nan)
    curve=carmenes_lightcurve_covariance(waves,flux,errors,night['berv_kms'],rv,science,result)
    result['planet_lightcurve_error']=np.sqrt(np.maximum(np.diag(curve),0))
    result.update(phase=phase,bjd_tdb=night['bjd_tdb'],berv_kms=night['berv_kms'],planet_rv_kms=rv,
                  with_signal=inside,without_signal=outside,full_transit_indices=full)
    with (output/'transmission.npz').open('xb') as stream:
        np.savez_compressed(stream,**result)
    from scipy.sparse import save_npz
    save_npz(output/'planet_coadd_covariance.npz',covariance)
    from exoplore.plotting.helium import plot_helium_summary
    plot_helium_summary(output/'transmission_diagnostic.png',result,phase,rv,night['berv_kms'],science)
    # Reuse the standard EXoPLORE stages renderer, with all panels displayed
    # on the native Earth-frame grid. The reference division stays stellar.
    from exoplore.plotting.steps import plot_pipeline_steps
    from exoplore.pipelines.carmenes_helium import gap_resample
    from exoplore.pipelines.helium_math import stellar_wavelength
    display_wave=native[0]
    display=np.array([gap_resample(result['stellar_wave_nm'],row-1,np.ones_like(row),
        stellar_wavelength(display_wave,b,science.gamma_kms))[0]
        for row,b in zip(result['stellar_transmission'],night['berv_kms'])])
    corrected_display=np.array([gap_resample(w,row,e**2,display_wave)[0]
                               for w,row,e in zip(waves,flux,errors)])
    stage={'mask':'Telluric corrected; OH masked','ignore':'Telluric corrected; OH retained',
           'fiber_b_matched_extraction':'Telluric corrected; fibre-B OH subtracted',
           'fiber_b_calibrated':'Telluric corrected; calibrated fibre-B OH subtracted',
           'fiber_a_oot_profile':'Telluric corrected; OOT fibre-A OH profile subtracted'}[treatment]
    plot_pipeline_steps(f'{config.planet.name}_Palle2020',str(output)+'/',display_wave/1000,
        phase,inside,outside,np.flatnonzero(np.isfinite(display).any(axis=0)),
        night['flux'][:,order],night['flux'][:,order],display,
        order_label=science.order_segment,xlim_1d=tuple(np.array(science.plot_stellar_window_nm)/1000),
        sysrem_stages={stage:corrected_display,
                       'Divided by master out-of-transit (Earth-frame display)':display},
        sysrem_iters=[stage,'Divided by master out-of-transit (Earth-frame display)'],
        use_real_data=True,save_plot=True,show_plot=False)
    with (output/'summary.json').open('x') as stream:
        json.dump(dict(pipeline='Palle2020',reference='https://doi.org/10.1051/0004-6361/202037719',
            oh_treatment=treatment,oh_mode=science.oh_mode,oh_windows_nm=windows,n_exposures=len(raw),
            n_in_transit=len(inside),n_reference=len(outside),n_full_transit=len(full),
            uncertainty='Analytical extraction-noise covariance; continuum/telluric fit parameters held fixed',
            uncertainty_resamples=0,sky_subtraction=uses_sky_fibre(treatment),
            sky_uncertainty=None if transfer is None else transfer['uncertainty_scope'],
            transmission_configuration=asdict(science)),stream,indent=2)
    return output
