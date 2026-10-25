"""Native CARACAL ingestion and real-data molecfit for Palle2020.

Fibre A supplies the observed science spectrum. Fibre B locates OH emission
for the masked alternative of Pallé et al. (2020), doi:10.1051/0004-6361/202037719.
All source files are read-only. All correction products must be new.
"""
from __future__ import annotations

# Role in the calculation
# Read extracted CARACAL exposures for CARMENES helium analysis and fit their telluric
# corrections. Preserve native pixel orientation and match fibres by exposure before
# constructing transmission spectra.

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from exoplore.instruments.crires_czesla2024 import _recipe, protected_windows
from exoplore.pipelines.helium_math import C_KMS


def native_array(array):
    """Accept both recorded CARACAL orientations. Return orders by pixels."""
    value = np.asarray(array, float)
    if value.shape == (28, 4080):
        return value.copy()
    if value.shape == (4080, 28):
        return value.T.copy()
    raise ValueError(f'Unexpected CARACAL NIR dimensions: {value.shape}')


def read_night(directory: Path) -> dict:
    """Read simultaneous A/B exposures without interpolating or filling pixels."""
    pairs = []
    for a in sorted(directory.glob('*sci-*nir_A.fits')):
        b = a.with_name(a.name[:-6]+'B.fits')
        if not b.is_file():
            raise FileNotFoundError(b)
        with fits.open(a, mode='readonly') as science, fits.open(b, mode='readonly') as sky:
            header = science[0].header.copy()
            if header.get('CAHA INS ICS FIB-MODE') == 'FP,FP':
                continue
            if header['DATE-OBS'] != sky[0].header['DATE-OBS']:
                raise ValueError('Science and sky exposure timestamps differ')
            pairs.append(dict(header=header,
                sky_header=sky[0].header.copy(),
                flux=native_array(science[1].data), error=native_array(science[3].data),
                wave_nm=native_array(science[4].data)/10,
                sky_flux=native_array(sky[1].data), sky_error=native_array(sky[3].data),
                sky_wave_nm=native_array(sky[4].data)/10,
                sources=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (a,b)]))
    if not pairs:
        raise ValueError('No CARACAL NIR fibre A science exposures')
    pairs.sort(key=lambda p:p['header']['CARACAL BJD'])
    night = {key:np.asarray([p[key] for p in pairs]) for key in
             ('flux','error','wave_nm','sky_flux','sky_error','sky_wave_nm')}
    night.update(headers=[p['header'] for p in pairs], sky_headers=[p['sky_header'] for p in pairs],
                 sources=[s for p in pairs for s in p['sources']],
                 bjd_tdb=np.array([p['header']['CARACAL BJD']+2400000 for p in pairs]),
                 utc_mjd=np.array([p['header']['CARACAL JD']-.5 for p in pairs]),
                 berv_kms=np.array([p['header']['CARACAL BERV'] for p in pairs]),
                 duration_seconds=np.array([p['header']['EXPTIME'] for p in pairs]))
    return night


def sky_windows(night: dict, order: int, catalogue_nm: list[float], halfwidth_nm: float) -> list[list[float]]:
    """Locate resolved sky lines on a median-filtered master B spectrum.

    The three-pixel filter suppresses isolated detector spikes for locating
    emission only. It never changes the fibre A spectrum. Weak sky lines retain
    their catalogue centres. Mask width is an explicit local choice.
    """
    wave = np.median(night['sky_wave_nm'][:,order], axis=0)
    master = np.nanmedian(night['sky_flux'][:,order], axis=0)
    smooth = median_filter(master, size=3)
    centres=[]
    for centre in catalogue_nm:
        indices=np.flatnonzero(abs(wave-centre)<.012)
        if len(indices)<3:
            raise ValueError('Sky-line catalogue lies outside the selected order')
        peak=indices[np.argmax(smooth[indices])]
        # Follow a resolved peak only. do not shift a weak line to noise.
        baseline=float(np.median(smooth[indices]))
        scatter=max(float(1.4826*np.median(abs(smooth[indices]-baseline))),1e-12)
        measured=float(wave[peak]) if smooth[peak]-baseline>5*scatter else float(centre)
        centres.append(measured)
    return [[c-halfwidth_nm,c+halfwidth_nm] for c in centres]


def fit_exposure(directory: Path, wave, flux, error, header, settings: dict,
                 science, rv: float, berv: float, utc_mjd: float) -> dict:
    """Fit H2O using explicit fixed or fitted Voigt component widths.

    Nagel2023 mode uses a uniform wavelength grid only for model fitting.
    Component FWHMs scale with wavelength. Their relative coefficients remain
    fixed. Evaluate the correction on the native wavelength grid.
    """
    directory.mkdir(parents=True, exist_ok=False)
    native_wave=np.array(wave,float,copy=True)
    native_flux=np.array(flux,float,copy=True)
    native_error=np.array(error,float,copy=True)
    settings=dict(settings)
    grid_provenance=dict(mode='native',native_pixels=len(wave))
    if settings.get('lsf_mode')=='nagel2023_nir':
        from exoplore.pipelines.carmenes_helium import gap_resample
        widths=np.diff(wave)
        count=int(np.ceil((wave[-1]-wave[0])/np.median(widths)))
        fit_wave=np.linspace(wave[0],wave[-1],count+1)
        valid=np.isfinite(flux)&np.isfinite(error)&(error>0)
        source_flux=np.where(valid,flux,np.nan)
        source_variance=np.where(valid,error**2,np.nan)
        flux,variance=gap_resample(wave,source_flux,source_variance,fit_wave)
        error=np.sqrt(variance)
        wave=fit_wave
        step=wave[1]-wave[0]
        central=(wave[0]+wave[-1])/2
        gaussian=settings['gaussian_relative_fwhm']
        lorentz=settings['lorentzian_relative_fwhm']
        if not 0<gaussian<1e-3 or not 0<lorentz<1e-3:
            raise ValueError('Invalid relative instrumental widths')
        settings.update(gaussian_fwhm_pixels=gaussian*central/step,
                        lorentzian_fwhm_pixels=lorentz*central/step,
                        fit_gaussian_width=False,fit_lorentzian_width=False)
        grid_provenance=dict(mode='uniform wavelength grid for telluric fitting only',
            native_pixels=len(native_wave),fit_pixels=len(wave),step_nm=float(step),
            reference_wavelength_nm=float(central),
            gaussian_relative_fwhm=gaussian,lorentzian_relative_fwhm=lorentz,
            source=settings['lsf_source'],wavelength_scaling='FWHM(lambda)=relative_FWHM*lambda',
            fitting_error='Squared interpolation weights; interpolation covariance is not used in molecfit',
            detector_gaps='Masked; no interpolation across gaps',
            native_observation='Unchanged; only the computed correction is interpolated back')
    valid=np.isfinite(flux)&np.isfinite(error)&(error>0)
    scale=float(np.median(flux[valid]))
    windows=protected_windows(settings['windows_nm'],np.asarray(science.helium_vacuum_lines_nm),
                             rv,berv,science.gamma_kms,settings['helium_guard_kms'])
    windows=[w for w in windows if wave[0]<=w[0] and w[1]<=wave[-1]]
    if not windows:
        raise ValueError('No helium-protected telluric fitting intervals')
    meta=fits.Header(dict(INSTRUME='CARMENES',SYNTHET=False,**{'MJD-OBS':utc_mjd},
        UTC=float((utc_mjd%1)*86400),
        EXPTIME=float(header['EXPTIME']), AIRMASS=float(header['AIRMASS']),
        ALTITUDE=float(np.degrees(np.arcsin(1/header['AIRMASS']))),
        RHUM=float(header['CAHA GEN AMBI RHUM']),PRESSURE=float(header['CAHA GEN AMBI PRESSURE']),
        AMBTEMP=float(header['CAHA GEN AMBI TEMPERATURE']),
        MIRTEMP=float(header['CAHA INS FRONTEND TEMP1']),
        SITELON=science.observatory_lon_lat_height[0],SITELAT=science.observatory_lon_lat_height[1],
        SITEELEV=science.observatory_lon_lat_height[2]))
    table=fits.BinTableHDU.from_columns([
        fits.Column(name='lambda',format='D',unit='um',array=wave/1000),
        fits.Column(name='flux',format='D',array=np.where(valid,flux/scale,0)),
        fits.Column(name='dflux',format='D',array=np.where(valid,error/scale,1)),
        fits.Column(name='mask',format='J',array=valid.astype(int))],name='SPECTRUM')
    fits.HDUList([fits.PrimaryHDU(header=meta),table]).writeto(directory/'science.fits')
    molecules=fits.BinTableHDU.from_columns([
        fits.Column(name='LIST_MOLEC',format='10A',array=['H2O']),
        fits.Column(name='FIT_MOLEC',format='J',array=[1]),
        fits.Column(name='REL_COL',format='D',array=[1.])])
    fits.HDUList([fits.PrimaryHDU(),molecules]).writeto(directory/'molecules.fits')
    intervals=np.asarray(windows)/1000
    include=fits.BinTableHDU.from_columns([
        fits.Column(name='LOWER_LIMIT',format='D',array=intervals[:,0]),
        fits.Column(name='UPPER_LIMIT',format='D',array=intervals[:,1]),
        *[fits.Column(name=n,format='J',array=np.ones(len(windows),int)) for n in
          ('MAPPED_TO_CHIP','WLC_FIT_FLAG','CONT_FIT_FLAG')]])
    fits.HDUList([fits.PrimaryHDU(),include]).writeto(directory/'wave_include.fits')
    with (directory/'model.sof').open('x') as stream:
        for name,tag in (('science','SCIENCE'),('molecules','MOLECULES'),('wave_include','WAVE_INCLUDE')):
            stream.write(f'{directory}/{name}.fits {tag}\n')
    count=len(windows)
    command=[settings['esorex'],f'--output-dir={directory}','molecfit_model',
        '--COLUMN_LAMBDA=lambda','--COLUMN_FLUX=flux','--COLUMN_DFLUX=dflux','--COLUMN_MASK=mask',
        '--WLG_TO_MICRON=1','--WAVELENGTH_FRAME=VAC','--FIT_WLC=1','--WLC_N=1','--WLC_CONST=0',
        '--FIT_CONTINUUM='+','.join(['1']*count),'--CONTINUUM_N='+','.join(['1']*count),
        '--MAP_REGIONS_TO_CHIP='+','.join(['1']*count),
        '--LIST_MOLEC=H2O','--FIT_MOLEC=1','--REL_COL=1',
        '--FIT_RES_BOX=FALSE','--RES_BOX=0',
        '--FIT_RES_GAUSS='+str(settings.get('fit_gaussian_width',False)).upper(),
        f"--RES_GAUSS={settings['gaussian_fwhm_pixels']}",
        '--FIT_RES_LORENTZ='+str(settings.get('fit_lorentzian_width',False)).upper(),
        f"--RES_LORENTZ={settings['lorentzian_fwhm_pixels']}",
        '--VARKERN=TRUE','--KERNFAC=6','--FTOL=0.0001','--XTOL=0.0001',
        '--SLIT_WIDTH_KEYWORD=NONE','--SLIT_WIDTH_VALUE=1.5']
    for name,key in (('TELESCOPE_ANGLE','ALTITUDE'),('RELATIVE_HUMIDITY','RHUM'),
                     ('PRESSURE','PRESSURE'),('TEMPERATURE','AMBTEMP'),('MIRROR_TEMPERATURE','MIRTEMP'),
                     ('ELEVATION','SITEELEV'),('LONGITUDE','SITELON'),('LATITUDE','SITELAT')):
        command.append(f'--{name}_KEYWORD={key}')
    command.append(str(directory/'model.sof'))
    report=dict(accepted=False,command=command,windows_nm=windows,
                grid_and_lsf=grid_provenance,
                gaussian_fwhm_pixels=settings['gaussian_fwhm_pixels'],
                lorentzian_fwhm_pixels=settings['lorentzian_fwhm_pixels'])
    try:
        _recipe(command,directory,'model',settings['timeout_seconds'])
        params={str(row['parameter']):float(row['value']) for row in
                fits.getdata(directory/'BEST_FIT_PARAMETERS.fits',1) if np.isfinite(row['value'])}
        report['parameters']=params
        gaussian,lorentz=params['gaussfwhm'],params['lorentzfwhm']
        if gaussian<=0 or lorentz<0:
            raise ValueError('Unphysical fitted instrumental widths')
        fwhm=.5346*lorentz+np.sqrt(.2166*lorentz**2+gaussian**2)
        centre=int(np.argmin(abs(wave-np.mean(science.helium_vacuum_lines_nm))))
        reference=grid_provenance.get('reference_wavelength_nm',(wave[0]+wave[-1])/2)
        report['effective_resolving_power']=float(reference/(fwhm*np.gradient(wave)[centre]))
        if params['status'] not in (1,2,3,4) or params['reduced_chi2']>settings['max_reduced_chi2']:
            raise ValueError('Telluric fit failed its convergence/chi-square gate')
        calc=directory/'calctrans';calc.mkdir()
        with (calc/'calctrans.sof').open('x') as stream:
            for name,tag in (('science','SCIENCE'),('ATM_PARAMETERS','ATM_PARAMETERS'),
                             ('MODEL_MOLECULES','MODEL_MOLECULES'),('BEST_FIT_PARAMETERS','BEST_FIT_PARAMETERS')):
                stream.write(f'{directory}/{name}.fits {tag}\n')
        cmd=[settings['esorex'],f'--output-dir={calc}','molecfit_calctrans',
             '--MAPPING_ATMOSPHERIC=0,1','--MAPPING_CONVOLVE=0,1','--HDR_EXP=EXPTIME',str(calc/'calctrans.sof')]
        report['calctrans_command']=cmd
        _recipe(cmd,calc,'calctrans',settings['timeout_seconds'])
        full=fits.getdata(calc/'TELLURIC_DATA.fits',1)
        refined,trans=full['mlambda']*1000,full['mtrans']
        if len(refined)!=len(wave) or not np.isfinite(refined).all() or np.any(np.diff(refined)<=0):
            raise ValueError('Invalid refined wavelength grid')
        line=float(np.mean(science.helium_vacuum_lines_nm))
        shift=float(C_KMS*np.interp(line,wave,refined-wave)/line)
        report['wavelength_shift_kms']=shift
        if abs(shift)>science.max_wavelength_shift_kms or not np.isfinite(trans).all() or np.any(trans<=0):
            raise ValueError('Wavelength or transmission acceptance gate failed')
        if grid_provenance['mode']!='native':
            refined=native_wave+np.interp(native_wave,wave,refined-wave)
            trans=np.interp(native_wave,wave,trans)
            wave,flux,error=native_wave,native_flux,native_error
            valid=np.isfinite(flux)&np.isfinite(error)&(error>0)
        report['accepted']=True
        with (directory/'correction.npz').open('xb') as stream:
            np.savez_compressed(stream,native_wave_nm=wave,refined_wave_nm=refined,
                                transmittance=trans,valid=valid)
        return dict(wave_nm=refined,flux=flux/trans,error=error/trans,transmittance=trans)
    except Exception as exc:
        report['failure']=str(exc)
        raise
    finally:
        with (directory/'report.json').open('x') as stream:
            json.dump(report,stream,indent=2)
