"""Molecfit adapter for synthetic CARMENES detector sections.

Use ordinary wavelength/flux/error tables and explicit CAHA metadata; do not
fabricate CRIRES nods or detector products. Fit each native detector section
separately, using only the configured helium-order telluric anchors.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Event
import shutil
import time
import numpy as np
from astropy.io import fits
from exoplore.instruments.crires_czesla2024 import HeliumMolecfitConfig, protected_windows, _recipe
from exoplore.pipelines.helium_math import C_KMS


@dataclass(frozen=True)
class CarmenesMolecfitConfig(HeliumMolecfitConfig):
    """Calibrated synthetic LSF; fitting its width is an explicit choice."""
    fit_gaussian_width: bool
    wavelength_degree: int = 1
    calibrated_resolving_power: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.wavelength_degree not in (0,1):
            raise ValueError("Choose an offset-only or offset-and-stretch wavelength correction")
        if self.calibrated_resolving_power is not None and self.calibrated_resolving_power <= 0:
            raise ValueError("Calibrated resolving power must be positive")


def fit_section(directory, wave, flux, error, settings, science, mjd, berv, rv, airmass, duration):
    """Run ESO recipes for one contiguous detector section, with acceptance gates."""
    directory.mkdir(exist_ok=False)
    executable = shutil.which(settings.esorex)
    if executable is None: raise FileNotFoundError(settings.esorex)
    windows = [w for w in protected_windows(settings.windows_nm, np.asarray(science.helium_vacuum_lines_nm), rv, berv, science.gamma_kms, settings.helium_guard_kms) if wave[0] <= w[0] and w[1] <= wave[-1]]
    if not windows: raise ValueError(f'No guarded telluric anchors for detector section {wave[0]}--{wave[-1]} nm')
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    scale = float(np.median(flux[valid]))
    header = fits.Header({'INSTRUME':'CARMENES', 'SYNTHET':True, 'ORIGIN':'EXOPLORE_SIMULATION', 'MJD-OBS':float(mjd), 'UTC':float((mjd%1)*86400), 'EXPTIME':float(duration), 'ALTITUDE':float(np.degrees(np.arcsin(1/airmass))), 'AIRMASS':airmass, 'RHUM':15., 'PRESSURE':780., 'AMBTEMP':15., 'MIRTEMP':15., 'SITELON':science.observatory_lon_lat_height[0], 'SITELAT':science.observatory_lon_lat_height[1], 'SITEELEV':science.observatory_lon_lat_height[2]})
    table = fits.BinTableHDU.from_columns([fits.Column(name='lambda',format='D',unit='um',array=wave/1000),fits.Column(name='flux',format='D',array=np.where(valid,flux/scale,0)),fits.Column(name='dflux',format='D',array=np.where(valid,error/scale,1)),fits.Column(name='mask',format='J',array=valid.astype(int))],header=header,name='SPECTRUM')
    fits.HDUList([fits.PrimaryHDU(header=header),table]).writeto(directory/'science.fits',overwrite=False)
    molecules = fits.BinTableHDU.from_columns([fits.Column(name='LIST_MOLEC',format='10A',array=settings.molecules),fits.Column(name='FIT_MOLEC',format='J',array=settings.fit_molecules),fits.Column(name='REL_COL',format='D',array=settings.relative_columns)])
    fits.HDUList([fits.PrimaryHDU(),molecules]).writeto(directory/'molecules.fits',overwrite=False)
    n = len(windows)
    fwhm = settings.gaussian_initial_fwhm_pixels
    if settings.calibrated_resolving_power is not None:
        # A pixel integration is approximated by a box of variance 1/12;
        # express the known resolving power at this detector's central pixel.
        center = len(wave)//2
        sampling = wave[center]/settings.calibrated_resolving_power/np.gradient(wave)[center]
        fwhm = float(np.sqrt(sampling**2+2.354820045**2/12))
    includes = fits.BinTableHDU.from_columns([fits.Column(name='LOWER_LIMIT',format='D',array=np.asarray(windows)[:,0]/1000),fits.Column(name='UPPER_LIMIT',format='D',array=np.asarray(windows)[:,1]/1000),fits.Column(name='MAPPED_TO_CHIP',format='J',array=np.ones(n,int)),fits.Column(name='WLC_FIT_FLAG',format='J',array=np.ones(n,int)),fits.Column(name='CONT_FIT_FLAG',format='J',array=np.ones(n,int))])
    fits.HDUList([fits.PrimaryHDU(),includes]).writeto(directory/'wave_include.fits',overwrite=False)
    with (directory/'model.sof').open('x') as f:
        for file,tag in [('science.fits','SCIENCE'),('molecules.fits','MOLECULES'),('wave_include.fits','WAVE_INCLUDE')]: f.write(f'{directory/file} {tag}\n')
    command = [executable,f'--output-dir={directory}','molecfit_model','--COLUMN_LAMBDA=lambda','--COLUMN_FLUX=flux','--COLUMN_DFLUX=dflux','--COLUMN_MASK=mask','--WLG_TO_MICRON=1','--WAVELENGTH_FRAME=VAC','--FIT_WLC=1',f'--WLC_N={settings.wavelength_degree}','--WLC_CONST=0','--FIT_CONTINUUM='+','.join(['1']*n),'--CONTINUUM_N='+','.join(['0']*n),'--MAP_REGIONS_TO_CHIP='+','.join(['1']*n),'--LIST_MOLEC='+','.join(settings.molecules),'--FIT_MOLEC='+','.join(map(str,settings.fit_molecules)),'--REL_COL='+','.join(map(str,settings.relative_columns)),'--FIT_RES_BOX=FALSE','--RES_BOX=0','--FIT_RES_LORENTZ=FALSE','--RES_LORENTZ=0','--FIT_RES_GAUSS='+str(settings.fit_gaussian_width).upper(),f'--RES_GAUSS={fwhm}','--VARKERN=TRUE',f'--KERNFAC={settings.kernel_factor}',f'--FTOL={settings.ftol}',f'--XTOL={settings.xtol}']
    for key,value in [('TELESCOPE_ANGLE','ALTITUDE'),('RELATIVE_HUMIDITY','RHUM'),('PRESSURE','PRESSURE'),('TEMPERATURE','AMBTEMP'),('MIRROR_TEMPERATURE','MIRTEMP'),('ELEVATION','SITEELEV'),('LONGITUDE','SITELON'),('LATITUDE','SITELAT')]: command.append(f'--{key}_KEYWORD={value}')
    # Molecfit requires a slit-width metadata parameter even when its boxcar
    # kernel is disabled. Use the fibre aperture, without claiming a slit.
    command.extend(['--SLIT_WIDTH_KEYWORD=NONE','--SLIT_WIDTH_VALUE=1.5'])
    command.append(str(directory/'model.sof'))
    report = dict(calibrated_fwhm_pixels=fwhm,calibrated_resolving_power=settings.calibrated_resolving_power,wavelength_degree=settings.wavelength_degree,accepted=False,command=command,windows_nm=windows,science_sha256=hashlib.sha256((directory/'science.fits').read_bytes()).hexdigest(),input_origin='exoplore_simulated_CARMENES_1d',sky_emission='absent',frame='topocentric vacuum nm')
    start = time.monotonic()
    try:
        _recipe(command,directory,'model',settings.timeout_seconds)
        params = {str(row['parameter']):float(row['value']) for row in fits.getdata(directory/'BEST_FIT_PARAMETERS.fits',1) if np.isfinite(row['value'])}
        model = fits.getdata(directory/'BEST_FIT_MODEL.fits',1)
        # BEST_FIT_MODEL can contain corrupt edge mlambda outside mrange.
        # Recover the declared low-order correction only from fitted pixels;
        # the accepted flux divisor and wavelength grid still come from calctrans.
        anchored = (model['mrange'] > 0) & np.isfinite(model['mlambda'])
        if np.count_nonzero(anchored) < 10:
            raise ValueError('Insufficient calibrated pixels in fitted intervals')
        correction = np.polynomial.Polynomial.fit(model['lambda'][anchored],
                    model['mlambda'][anchored]-model['lambda'][anchored],settings.wavelength_degree)
        reference = float(np.mean(science.helium_vacuum_lines_nm)/1000)
        if not wave[0]/1000 <= reference <= wave[-1]/1000:
            reference = float(np.mean(wave)/1000)
        shift = float(C_KMS*correction(reference)/reference)
        report.update(parameters=params,wavelength_shift_kms=shift)
        if params['status'] not in (1,2,3,4) or abs(shift)>science.max_wavelength_shift_kms or not settings.gaussian_fwhm_range_pixels[0]<=params['gaussfwhm']<=settings.gaussian_fwhm_range_pixels[1] or params['reduced_chi2']>settings.max_reduced_chi2:
            raise ValueError('Molecfit acceptance gates failed; no substitute correction')
        calcdir = directory/'calctrans';calcdir.mkdir()
        with (calcdir/'calctrans.sof').open('x') as f:
            for file,tag in [('science.fits','SCIENCE'),('ATM_PARAMETERS.fits','ATM_PARAMETERS'),('MODEL_MOLECULES.fits','MODEL_MOLECULES'),('BEST_FIT_PARAMETERS.fits','BEST_FIT_PARAMETERS')]:f.write(f'{directory/file} {tag}\n')
        calc = [executable,f'--output-dir={calcdir}','molecfit_calctrans','--MAPPING_ATMOSPHERIC=0,1','--MAPPING_CONVOLVE=0,1','--HDR_EXP=EXPTIME',str(calcdir/'calctrans.sof')]
        report['calctrans_command']=calc;_recipe(calc,calcdir,'calctrans',settings.timeout_seconds)
        full = fits.getdata(calcdir/'TELLURIC_DATA.fits',1)
        refined, trans = full['mlambda']*1000,full['mtrans']
        if len(full)!=len(wave) or not np.isfinite(refined).all() or np.any(np.diff(refined)<=0) or not np.isfinite(trans).all() or np.any(trans<=0):raise ValueError('Invalid full-section molecfit correction')
        with (directory/'correction.npz').open('xb') as f:np.savez_compressed(f,native_wave_nm=wave,refined_wave_nm=refined,flux=flux,error=error,transmittance=trans,valid=valid)
        report['accepted']=True
        return dict(wave_nm=refined,flux=flux/trans,error=error/trans,transmittance=trans,shift=shift)
    except Exception as exc:
        report['failure']=str(exc);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        with (directory/'report.json').open('x') as f:json.dump(report,f,indent=2)


def fit_carmenes_night(output, arrays, raw, error, settings, science, utc_mjd, duration):
    """Fit the complete selected order, independently on both detector sections."""
    fit_settings = CarmenesMolecfitConfig(**json.loads(Path(settings.molecfit_config_path).read_text()))
    root = (output/'molecfit').resolve();root.mkdir(exist_ok=False)
    stop = Event()
    def one(index):
        if stop.is_set():
            raise RuntimeError('Pending fit cancelled after another calibration failure')
        directory=root/f'exposure_{index+1:03d}';directory.mkdir()
        w=arrays['wave_nm'][index];gap=np.flatnonzero(np.diff(w)>5*np.median(np.diff(w)))
        sections=np.split(np.arange(len(w)),gap+1)
        results=[]
        for j,pixels in enumerate(sections):
            if stop.is_set():
                raise RuntimeError('Pending detector fit cancelled after another calibration failure')
            try:
                result=fit_section(directory/f'detector_{j+1}',w[pixels],raw[index,pixels],error[index,pixels],fit_settings,science,utc_mjd[index],settings.berv_kms[index],science.kp_kms*np.sin(2*np.pi*settings.phase_midpoints[index]),settings.airmass[index],duration)
            except Exception:
                stop.set()
                raise
            results.append(result)
        print(f'  CARMENES molecfit: exposure {index+1}/{len(raw)}, shifts {[round(r["shift"],3) for r in results]} km/s',flush=True)
        return {key:np.concatenate([r[key] for r in results]) for key in ('wave_nm','flux','error','transmittance')} | {'shift':np.asarray([r['shift'] for r in results])}
    # The first exposure is part of this night, and gates the remaining fits.
    results=[one(0)]
    with ThreadPoolExecutor(max_workers=settings.molecfit_workers) as executor:
        results.extend(executor.map(one,range(1,len(raw))))
    return {key:np.asarray([r[key] for r in results]) for key in results[0]}
