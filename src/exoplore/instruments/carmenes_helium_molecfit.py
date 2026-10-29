"""Fit terrestrial absorption in synthetic CARMENES or ANDES detector sections.

The sodium and CARMENES helium runners call fit_section() with observer-frame
vacuum wavelengths, flux and standard deviations. The same ESO model and
calctrans recipes fit molecular absorption and calculate a full-section
transmission. fit_carmenes_night() splits the helium order at detector gaps.
Each exposure/section has its own recipe files and acceptance report.
"""
from __future__ import annotations

# Role in the calculation
# Fit telluric transmission separately in the synthetic CARMENES detector sections. The
# configured acceptance checks determine which fitted corrections can be used.

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
    """Calibrated synthetic LSF. fitting its width is an explicit choice."""
    # Choose whether Molecfit fits the Gaussian line-spread width.
    fit_gaussian_width: bool
    # Zero fits a wavelength offset. One also permits a stretch.
    wavelength_degree: int = 1
    # Optional known resolution sets the initial Gaussian width on this detector grid.
    calibrated_resolving_power: float | None = None
    # Polynomial degree of the continuum fitted independently in each anchor interval.
    continuum_degree: int = 0
    # Enable or hold fixed the wavelength correction during the telluric fit.
    fit_wavelength: bool = True

    # Validate these additional choices after the common Molecfit settings.
    def __post_init__(self):
        super().__post_init__()
        if self.wavelength_degree not in (0,1):
            raise ValueError("Choose an offset-only or offset-and-stretch wavelength correction")
        if self.calibrated_resolving_power is not None and self.calibrated_resolving_power <= 0:
            raise ValueError("Calibrated resolving power must be positive")
        if self.continuum_degree not in (0, 1, 2):
            raise ValueError("Choose continuum_degree 0, 1 or 2")
        if type(self.fit_wavelength) is not bool:
            raise ValueError("fit_wavelength must be boolean")


def fit_section(directory, wave, flux, error, settings, science, mjd, berv, rv, airmass, duration,
                *, protected_lines_nm=None, guard_kms=None):
    """Run ESO recipes for one contiguous detector section, with acceptance gates."""
    # ESO recipes run inside this directory. Absolute paths keep their SOF
    # inputs valid when the caller supplies a relative output location.
    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=False)
    # Locate the external esorex program in the active command-line environment.
    executable = shutil.which(settings.esorex)
    if executable is None: raise FileNotFoundError(settings.esorex)
    # Use the supplied sodium rest wavelengths or the helium triplet from science.
    lines = np.asarray(science.helium_vacuum_lines_nm if protected_lines_nm is None
                       else protected_lines_nm)
    # Choose the velocity width used to protect the planetary absorption during fitting.
    guard = settings.helium_guard_kms if guard_kms is None else guard_kms
    # Shift the protected planetary lines to this exposure and keep only guarded
    # telluric intervals fully contained in the detector section.
    windows = [w for w in protected_windows(settings.windows_nm, lines, rv, berv,
        science.gamma_kms, guard) if wave[0] <= w[0] and w[1] <= wave[-1]]
    if not windows: raise ValueError(f'No guarded telluric anchors for detector section {wave[0]}--{wave[-1]} nm')
    # Construct the usable-pixel mask from flux and uncertainty values.
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    # Put flux and error on a common near-unity scale for the external recipe.
    scale = float(np.median(flux[valid]))
    # Build the synthetic exposure header used by Molecfit. Site, time and airmass
    # come from the experiment. Weather defaults here are RH 15%, pressure 780 hPa
    # and ambient/mirror temperature 15 degrees C.
    header = fits.Header({'INSTRUME':getattr(science, 'instrument_name', 'CARMENES'), 'SYNTHET':True, 'ORIGIN':'EXOPLORE_SIMULATION', 'MJD-OBS':float(mjd), 'UTC':float((mjd%1)*86400), 'EXPTIME':float(duration), 'ALTITUDE':float(np.degrees(np.arcsin(1/airmass))), 'AIRMASS':airmass, 'RHUM':15., 'PRESSURE':780., 'AMBTEMP':15., 'MIRTEMP':15., 'SITELON':science.observatory_lon_lat_height[0], 'SITELAT':science.observatory_lon_lat_height[1], 'SITEELEV':science.observatory_lon_lat_height[2]})
    # Store vacuum wavelengths in micrometres, scaled flux/errors and a valid-pixel mask.
    table = fits.BinTableHDU.from_columns([fits.Column(name='lambda',format='D',unit='um',array=wave/1000),fits.Column(name='flux',format='D',array=np.where(valid,flux/scale,0)),fits.Column(name='dflux',format='D',array=np.where(valid,error/scale,1)),fits.Column(name='mask',format='J',array=valid.astype(int))],header=header,name='SPECTRUM')
    # Write the science table once in this exposure's working directory.
    fits.HDUList([fits.PrimaryHDU(header=header),table]).writeto(directory/'science.fits',overwrite=False)
    # List the fitted molecules, fit flags and initial relative columns from the JSON.
    molecules = fits.BinTableHDU.from_columns([fits.Column(name='LIST_MOLEC',format='10A',array=settings.molecules),fits.Column(name='FIT_MOLEC',format='J',array=settings.fit_molecules),fits.Column(name='REL_COL',format='D',array=settings.relative_columns)])
    fits.HDUList([fits.PrimaryHDU(),molecules]).writeto(directory/'molecules.fits',overwrite=False)
    # Each retained anchor interval gets its own continuum fit in the model recipe.
    n = len(windows)
    # Start with the user's Gaussian FWHM in detector pixels.
    fwhm = settings.gaussian_initial_fwhm_pixels
    # When a calibrated resolution is supplied, derive the initial pixel width
    # from lambda / R and the local detector sampling.
    if settings.calibrated_resolving_power is not None:
        # A pixel integration is approximated by a box of variance 1/12.
        # express the known resolving power at this detector's central pixel.
        center = len(wave)//2
        # Calculate the resolution-element width in pixels at the section centre.
        sampling = wave[center]/settings.calibrated_resolving_power/np.gradient(wave)[center]
        # Combine Gaussian width with the pixel-integration box variance of 1/12 pixel squared.
        fwhm = float(np.sqrt(sampling**2+2.354820045**2/12))
    # Write wavelength anchors in micrometres with their chip and fitting flags.
    includes = fits.BinTableHDU.from_columns([fits.Column(name='LOWER_LIMIT',format='D',array=np.asarray(windows)[:,0]/1000),fits.Column(name='UPPER_LIMIT',format='D',array=np.asarray(windows)[:,1]/1000),fits.Column(name='MAPPED_TO_CHIP',format='J',array=np.ones(n,int)),fits.Column(name='WLC_FIT_FLAG',format='J',array=np.ones(n,int)),fits.Column(name='CONT_FIT_FLAG',format='J',array=np.ones(n,int))])
    fits.HDUList([fits.PrimaryHDU(),includes]).writeto(directory/'wave_include.fits',overwrite=False)
    # The SOF associates each input FITS file with the role expected by molecfit_model.
    with (directory/'model.sof').open('x') as f:
        for file,tag in [('science.fits','SCIENCE'),('molecules.fits','MOLECULES'),('wave_include.fits','WAVE_INCLUDE')]: f.write(f'{directory/file} {tag}\n')
    # Build the ESO recipe options from the declared columns, continuum and molecule
    # choices. The instrumental model uses the Gaussian width and the requested
    # wavelength dependence. The box and Lorentzian terms are fixed at zero.
    command = [executable,f'--output-dir={directory}','molecfit_model','--COLUMN_LAMBDA=lambda','--COLUMN_FLUX=flux','--COLUMN_DFLUX=dflux','--COLUMN_MASK=mask','--WLG_TO_MICRON=1','--WAVELENGTH_FRAME=VAC','--FIT_WLC='+str(int(settings.fit_wavelength)),f'--WLC_N={settings.wavelength_degree}','--WLC_CONST=0','--FIT_CONTINUUM='+','.join(['1']*n),'--CONTINUUM_N='+','.join([str(settings.continuum_degree)]*n),'--MAP_REGIONS_TO_CHIP='+','.join(['1']*n),'--LIST_MOLEC='+','.join(settings.molecules),'--FIT_MOLEC='+','.join(map(str,settings.fit_molecules)),'--REL_COL='+','.join(map(str,settings.relative_columns)),'--FIT_RES_BOX=FALSE','--RES_BOX=0','--FIT_RES_LORENTZ=FALSE','--RES_LORENTZ=0','--FIT_RES_GAUSS='+str(settings.fit_gaussian_width).upper(),f'--RES_GAUSS={fwhm}','--VARKERN=TRUE',f'--KERNFAC={settings.kernel_factor}',f'--FTOL={settings.ftol}',f'--XTOL={settings.xtol}']
    # Tell Molecfit which synthetic header keyword supplies each observing condition.
    for key,value in [('TELESCOPE_ANGLE','ALTITUDE'),('RELATIVE_HUMIDITY','RHUM'),('PRESSURE','PRESSURE'),('TEMPERATURE','AMBTEMP'),('MIRROR_TEMPERATURE','MIRTEMP'),('ELEVATION','SITEELEV'),('LONGITUDE','SITELON'),('LATITUDE','SITELAT')]: command.append(f'--{key}_KEYWORD={value}')
    # Supply the fibre-aperture value for Molecfit's required slit-width metadata.
    # The boxcar convolution term remains disabled in the recipe options.
    command.extend(['--SLIT_WIDTH_KEYWORD=NONE','--SLIT_WIDTH_VALUE=1.5'])
    command.append(str(directory/'model.sof'))
    # Record the exact command, accepted intervals and science-input hash for inspection.
    report = dict(calibrated_fwhm_pixels=fwhm,calibrated_resolving_power=settings.calibrated_resolving_power,wavelength_degree=settings.wavelength_degree,accepted=False,command=command,windows_nm=windows,science_sha256=hashlib.sha256((directory/'science.fits').read_bytes()).hexdigest(),input_origin='exoplore_simulated_'+getattr(science, 'instrument_name', 'CARMENES')+'_1d',sky_emission='absent',frame='topocentric vacuum nm')
    # Measure the external fit and transmission calculation runtime.
    start = time.monotonic()
    try:
        # Run the molecular fit with the configured timeout.
        _recipe(command,directory,'model',settings.timeout_seconds)
        # Read the optimizer status, fitted width, chi-squared and other fitted parameters.
        params = {str(row['parameter']):float(row['value']) for row in fits.getdata(directory/'BEST_FIT_PARAMETERS.fits',1) if np.isfinite(row['value'])}
        # Read the model wavelengths used to evaluate the fitted wavelength correction.
        model = fits.getdata(directory/'BEST_FIT_MODEL.fits',1)
        # Assess the wavelength correction using fitted pixels inside mrange,
        # where BEST_FIT_MODEL provides calibrated wavelengths. The full-section
        # correction and wavelength grid are subsequently read from calctrans.
        # Use finite wavelengths inside fitted anchor intervals to assess that correction.
        anchored = (model['mrange'] > 0) & np.isfinite(model['mlambda'])
        if np.count_nonzero(anchored) < 10:
            raise ValueError('Insufficient calibrated pixels in fitted intervals')
        # Fit the declared low-order wavelength displacement across the accepted anchors.
        correction = np.polynomial.Polynomial.fit(model['lambda'][anchored],
                    model['mlambda'][anchored]-model['lambda'][anchored],settings.wavelength_degree)
        # Evaluate its velocity displacement near the protected spectral feature.
        reference = float(np.mean(lines)/1000)
        # If that feature is outside this detector section, use the section mean wavelength.
        if not wave[0]/1000 <= reference <= wave[-1]/1000:
            reference = float(np.mean(wave)/1000)
        # Convert the wavelength displacement to km/s with the optical Doppler convention.
        shift = float(C_KMS*correction(reference)/reference)
        # Retain the fitted parameters and measured displacement in the exposure report.
        report.update(parameters=params,wavelength_shift_kms=shift)
        # Require an accepted optimizer status, wavelength displacement, Gaussian width
        # and reduced chi-squared before using this correction.
        if params['status'] not in (1,2,3,4) or abs(shift)>science.max_wavelength_shift_kms or not settings.gaussian_fwhm_range_pixels[0]<=params['gaussfwhm']<=settings.gaussian_fwhm_range_pixels[1] or params['reduced_chi2']>settings.max_reduced_chi2:
            raise ValueError('Molecfit acceptance gates failed; no substitute correction')
        # Keep the full-section transmission recipe products in their own subdirectory.
        calcdir = directory/'calctrans';calcdir.mkdir()
        # Pass the fitted atmosphere, molecules and parameters to molecfit_calctrans.
        with (calcdir/'calctrans.sof').open('x') as f:
            for file,tag in [('science.fits','SCIENCE'),('ATM_PARAMETERS.fits','ATM_PARAMETERS'),('MODEL_MOLECULES.fits','MODEL_MOLECULES'),('BEST_FIT_PARAMETERS.fits','BEST_FIT_PARAMETERS')]:f.write(f'{directory/file} {tag}\n')
        # Request transmission across the complete science section using the fitted atmosphere.
        calc = [executable,f'--output-dir={calcdir}','molecfit_calctrans','--MAPPING_ATMOSPHERIC=0,1','--MAPPING_CONVOLVE=0,1','--HDR_EXP=EXPTIME',str(calcdir/'calctrans.sof')]
        report['calctrans_command']=calc;_recipe(calc,calcdir,'calctrans',settings.timeout_seconds)
        # Read the full-section wavelength solution and transmission from calctrans.
        full = fits.getdata(calcdir/'TELLURIC_DATA.fits',1)
        # Convert the corrected wavelengths back to nm for the EXoPLORE preparation.
        refined, trans = full['mlambda']*1000,full['mtrans']
        # Require a complete, ordered wavelength grid and positive finite transmission.
        if len(full)!=len(wave) or not np.isfinite(refined).all() or np.any(np.diff(refined)<=0) or not np.isfinite(trans).all() or np.any(trans<=0):raise ValueError('Invalid full-section molecfit correction')
        # Retain the raw input and fitted correction together for later checks.
        with (directory/'correction.npz').open('xb') as f:np.savez_compressed(f,native_wave_nm=wave,refined_wave_nm=refined,flux=flux,error=error,transmittance=trans,valid=valid)
        # Mark this exposure accepted after every fit and product check has passed.
        report['accepted']=True
        # Divide the flux and its standard deviation by the same fitted transmission.
        return dict(wave_nm=refined,flux=flux/trans,error=error/trans,transmittance=trans,shift=shift)
    # Record any fitting or acceptance failure, then let the caller stop the reduction.
    except Exception as exc:
        report['failure']=str(exc);raise
    # Write the exposure report on both success and failure.
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        with (directory/'report.json').open('x') as f:json.dump(report,f,indent=2)


def fit_carmenes_night(output, arrays, raw, error, settings, science, utc_mjd, duration):
    """Fit the complete selected order, independently on both detector sections."""
    # Read the telluric-fit settings used for this CARMENES helium night.
    fit_settings = CarmenesMolecfitConfig(**json.loads(Path(settings.molecfit_config_path).read_text()))
    # Create a fresh set of external recipe directories under the run output.
    root = (output/'molecfit').resolve();root.mkdir(exist_ok=False)
    # Share a failure flag between workers so pending detector fits can stop.
    stop = Event()
    # Process every contiguous detector section in one exposure.
    def one(index):
        # Check for an earlier worker failure before starting another exposure.
        if stop.is_set():
            raise RuntimeError('Pending fit cancelled after another calibration failure')
        # Keep this exposure's detector sections together.
        directory=root/f'exposure_{index+1:03d}';directory.mkdir()
        # Identify detector gaps where wavelength steps exceed five times the median spacing.
        w=arrays['wave_nm'][index];gap=np.flatnonzero(np.diff(w)>5*np.median(np.diff(w)))
        # Split at those gaps before fitting each section independently.
        sections=np.split(np.arange(len(w)),gap+1)
        # Collect the corrected detector sections in their wavelength order.
        results=[]
        # Run the shared single-section fitter on each set of contiguous pixels.
        for j,pixels in enumerate(sections):
            if stop.is_set():
                raise RuntimeError('Pending detector fit cancelled after another calibration failure')
            try:
                # Pass the section arrays and this exposure's time, BERV, orbital velocity and airmass.
                result=fit_section(directory/f'detector_{j+1}',w[pixels],raw[index,pixels],error[index,pixels],fit_settings,science,utc_mjd[index],settings.berv_kms[index],science.kp_kms*np.sin(2*np.pi*settings.phase_midpoints[index]),settings.airmass[index],duration)
            except Exception:
                stop.set()
                raise
            results.append(result)
        print(f'  CARMENES molecfit: exposure {index+1}/{len(raw)}, shifts {[round(r["shift"],3) for r in results]} km/s',flush=True)
        # Join corrected wavelength, flux, error and transmission arrays across the sections.
        return {key:np.concatenate([r[key] for r in results]) for key in ('wave_nm','flux','error','transmittance')} | {'shift':np.asarray([r['shift'] for r in results])}
    # The first exposure is part of this night, and gates the remaining fits.
    results=[one(0)]
    # Run the remaining exposures using the configured concurrency limit.
    with ThreadPoolExecutor(max_workers=settings.molecfit_workers) as executor:
        results.extend(executor.map(one,range(1,len(raw))))
    # Return exposure-by-pixel arrays in the same order as the input time series.
    return {key:np.asarray([r[key] for r in results]) for key in results[0]}
