"""
exoplore.instruments.crires_czesla2024
===================================

Explicit, target-segment-only molecfit preparation for the Czesla et al. (2024)
direct helium workflow. ESO cr2res dark/flat/wavelength/nodding extraction is
upstream. Every output must be new; rejected fits never become unity models.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from astropy.io import fits

from exoplore.config.czesla2024 import Czesla2024Config
from exoplore.instruments.crires_molecfit import segments_of
from exoplore.pipelines.czesla2024 import exposure_metadata
from exoplore.pipelines.helium_math import C_KMS


@dataclass(frozen=True)
class HeliumMolecfitConfig:
    """Explicit target-order anchors, instrumental kernel and acceptance gates."""
    esorex: str
    windows_nm: list[list[float]]
    molecules: list[str]
    fit_molecules: list[int]
    relative_columns: list[float]
    helium_guard_kms: float
    gaussian_initial_fwhm_pixels: float
    gaussian_fwhm_range_pixels: list[float]
    max_reduced_chi2: float
    ftol: float
    xtol: float
    kernel_factor: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.windows_nm or any(len(w)!=2 or not np.isfinite(w).all() or w[0]>=w[1] for w in self.windows_nm):raise ValueError('Explicit increasing telluric anchors required')
        if not len(self.molecules)==len(self.fit_molecules)==len(self.relative_columns) or not self.molecules:raise ValueError('Molecule settings disagree')
        if not any(self.fit_molecules) or any(flag not in (0,1) for flag in self.fit_molecules):raise ValueError('At least one molecule must be fitted')
        if len(self.gaussian_fwhm_range_pixels)!=2 or not 0<self.gaussian_fwhm_range_pixels[0]<self.gaussian_fwhm_range_pixels[1]:raise ValueError('Invalid Gaussian width gate')
        if any(not np.isfinite(value) or value<=0 for value in [self.helium_guard_kms,self.gaussian_initial_fwhm_pixels,self.max_reduced_chi2,self.ftol,self.xtol,self.kernel_factor,self.timeout_seconds]):raise ValueError('Molecfit settings must be finite and positive')


def protected_windows(windows: list[list[float]],lines_nm: np.ndarray,
                      planet_rv_kms: float,berv_kms: float,gamma_kms: float,
                      guard_kms: float) -> list[list[float]]:
    """Remove whole anchors overlapping any predicted observer-frame He line."""
    centers=np.asarray(lines_nm)*(1+planet_rv_kms/C_KMS)*(1+gamma_kms/C_KMS)/(1+berv_kms/C_KMS)
    low=centers*(1-guard_kms/C_KMS);high=centers*(1+guard_kms/C_KMS)
    return [list(w) for w in windows if not np.any((w[0]<=high)&(w[1]>=low))]


def _recipe(command: list[str],directory: Path,name: str,timeout: float) -> None:
    """Run a bounded recipe only inside a new exposure directory."""
    with (directory/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=False)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {directory/(name+".log")}')


def prepare_crires_czesla2024(manifest: Path,raw_directory: Path,
                             science: Czesla2024Config,settings: HeliumMolecfitConfig,
                             output: Path) -> Path:
    """Fit each chronological extracted exposure on guarded helium-order anchors.

    This can take minutes per exposure. Run one exposure pilot before a whole
    night. All settings and recipe commands are recorded. The full calctrans
    output supplies refined vacuum wavelengths and pure absorption transmission.
    Source raw/extracted files are opened read-only.
    """
    executable=shutil.which(settings.esorex)
    if executable is None:raise FileNotFoundError(f'ESO executable not found: {settings.esorex}')
    rows=sorted((float(line.split(maxsplit=1)[0]),Path(line.split(maxsplit=1)[1])) for line in manifest.read_text().splitlines() if line.strip())
    if not rows:raise ValueError('Empty extraction manifest')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    with (output/'configuration.json').open('x') as stream:
        json.dump({'science':asdict(science),'molecfit':asdict(settings),'manifest':str(manifest.resolve()),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},stream,indent=2)
    reports=[]
    for number,(mjd,extracted) in enumerate(rows,1):
        start=time.monotonic();nod='A' if extracted.name.endswith('extractedA.fits') else 'B' if extracted.name.endswith('extractedB.fits') else None
        if nod is None:raise ValueError('Manifest must contain individual extracted A/B products')
        header=fits.getheader(extracted);matches=[]
        for key in header:
            if key.startswith('ESO PRO REC1 RAW') and key.endswith(' NAME'):
                candidate=raw_directory/header[key];rh=fits.getheader(candidate)
                if rh['ESO SEQ NODPOS']==nod and abs(rh['MJD-OBS']-mjd)<1e-7:matches.append(candidate)
        if len(matches)!=1:raise ValueError('Cannot resolve an individual raw exposure using manifest time and nod')
        raw=matches[0];raw_header=fits.getheader(raw);raw_hash=hashlib.sha256(raw_header.tostring().encode()).hexdigest()
        bjd,berv,_=exposure_metadata(raw,science);phase=((bjd-science.t0_bjd_tdb)/science.period_days+.5)%1-.5
        rv=science.kp_kms*np.sin(2*np.pi*phase)
        name,wave,flux,error={item[0]:item for item in segments_of(extracted)}[science.order_segment]
        if np.any(np.diff(wave)<=0):raise ValueError('Invalid native vacuum wavelength grid')
        windows=protected_windows(settings.windows_nm,np.asarray(science.helium_vacuum_lines_nm),rv,berv,science.gamma_kms,settings.helium_guard_kms)
        if not windows or any(lo<wave.min() or hi>wave.max() for lo,hi in windows):raise ValueError('No usable target-segment anchors')
        work=output/f'exposure_{number:02d}_{nod}';work.mkdir()
        valid=np.isfinite(flux)&np.isfinite(error)&(error>0)
        scale=float(np.nanmedian(flux[valid]));
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid extraction scale')
        table=fits.BinTableHDU.from_columns([
            fits.Column(name='lambda',format='D',array=wave/1000),
            fits.Column(name='flux',format='D',array=np.where(valid,flux/scale,0)),
            fits.Column(name='dflux',format='D',array=np.where(valid,error/scale,1)),
            fits.Column(name='mask',format='J',array=valid.astype(int)),
        ],header=raw_header.copy(),name=name)
        fits.HDUList([fits.PrimaryHDU(header=raw_header.copy()),table]).writeto(work/'science.fits',overwrite=False)
        molecules=fits.BinTableHDU.from_columns([
            fits.Column(name='LIST_MOLEC',format='10A',array=settings.molecules),
            fits.Column(name='FIT_MOLEC',format='J',array=settings.fit_molecules),
            fits.Column(name='REL_COL',format='D',array=settings.relative_columns)])
        fits.HDUList([fits.PrimaryHDU(),molecules]).writeto(work/'molecules.fits',overwrite=False)
        includes=fits.BinTableHDU.from_columns([
            fits.Column(name='LOWER_LIMIT',format='D',array=np.asarray(windows)[:,0]/1000),
            fits.Column(name='UPPER_LIMIT',format='D',array=np.asarray(windows)[:,1]/1000),
            fits.Column(name='MAPPED_TO_CHIP',format='J',array=np.ones(len(windows),int)),
            fits.Column(name='WLC_FIT_FLAG',format='J',array=np.ones(len(windows),int)),
            fits.Column(name='CONT_FIT_FLAG',format='J',array=np.ones(len(windows),int))])
        fits.HDUList([fits.PrimaryHDU(),includes]).writeto(work/'wave_include.fits',overwrite=False)
        with (work/'model.sof').open('x') as stream:
            for filename,tag in [('science.fits','SCIENCE'),('molecules.fits','MOLECULES'),('wave_include.fits','WAVE_INCLUDE')]:stream.write(f'{work/filename} {tag}\n')
        n=len(windows)
        command=[executable,f'--output-dir={work}','molecfit_model',
                 '--COLUMN_LAMBDA=lambda','--COLUMN_FLUX=flux','--COLUMN_DFLUX=dflux','--COLUMN_MASK=mask',
                 '--WLG_TO_MICRON=1','--WAVELENGTH_FRAME=VAC','--FIT_WLC=1','--WLC_N=1','--WLC_CONST=0',
                 '--FIT_CONTINUUM='+','.join(['1']*n),'--CONTINUUM_N='+','.join(['0']*n),'--MAP_REGIONS_TO_CHIP='+','.join(['1']*n),
                 '--LIST_MOLEC='+','.join(settings.molecules),'--FIT_MOLEC='+','.join(map(str,settings.fit_molecules)),
                 '--REL_COL='+','.join(map(str,settings.relative_columns)),
                 '--FIT_RES_BOX=FALSE','--RES_BOX=0','--FIT_RES_LORENTZ=FALSE','--RES_LORENTZ=0','--FIT_RES_GAUSS=TRUE',
                 f'--RES_GAUSS={settings.gaussian_initial_fwhm_pixels}',f'--KERNFAC={settings.kernel_factor}',
                 f'--FTOL={settings.ftol}',f'--XTOL={settings.xtol}',str(work/'model.sof')]
        report={'exposure':number,'nod':nod,'extracted':str(extracted.resolve()),'raw':str(raw.resolve()),
                'extracted_sha256':hashlib.sha256(extracted.read_bytes()).hexdigest(),'raw_header_sha256':raw_hash,
                'frame':'topocentric vacuum nm','order_segment':science.order_segment,'fit_windows_nm':windows,
                'command':command,'accepted':False}
        try:
            _recipe(command,work,'model',settings.timeout_seconds)
            parameters={str(row['parameter']):float(row['value']) for row in fits.getdata(work/'BEST_FIT_PARAMETERS.fits',1) if np.isfinite(row['value'])}
            model=fits.getdata(work/'BEST_FIT_MODEL.fits',1)
            shift=float(np.interp(np.mean(science.helium_vacuum_lines_nm)/1000,model['lambda'],C_KMS*(model['mlambda']/model['lambda']-1)))
            report.update(parameters=parameters,he_shift_kms=shift)
            if not (parameters['status'] in [1,2,3,4] and abs(shift)<=science.max_wavelength_shift_kms and
                    settings.gaussian_fwhm_range_pixels[0]<=parameters['gaussfwhm']<=settings.gaussian_fwhm_range_pixels[1] and
                    parameters['reduced_chi2']<=settings.max_reduced_chi2 and np.all(np.diff(model['mlambda'][model['mrange']>0])>0)):
                raise ValueError('Molecfit calibration gates failed; no fallback correction')
            calcdir=work/'calctrans';calcdir.mkdir()
            with (calcdir/'calctrans.sof').open('x') as stream:
                for filename,tag in [('science.fits','SCIENCE'),('ATM_PARAMETERS.fits','ATM_PARAMETERS'),('MODEL_MOLECULES.fits','MODEL_MOLECULES'),('BEST_FIT_PARAMETERS.fits','BEST_FIT_PARAMETERS')]:stream.write(f'{work/filename} {tag}\n')
            calc=[executable,f'--output-dir={calcdir}','molecfit_calctrans','--MAPPING_ATMOSPHERIC=0,1','--MAPPING_CONVOLVE=0,1','--HDR_EXP=EXPTIME',str(calcdir/'calctrans.sof')]
            report['calctrans_command']=calc;_recipe(calc,calcdir,'calctrans',settings.timeout_seconds)
            full=fits.getdata(calcdir/'TELLURIC_DATA.fits',1)
            if len(full)!=len(wave) or np.any(~np.isfinite(full['mlambda'])) or np.any(np.diff(full['mlambda'])<=0) or np.any(~np.isfinite(full['mtrans'])) or np.any(full['mtrans']<=0):raise ValueError('Full-grid correction validation failed')
            with (work/'correction.npz').open('xb') as stream:np.savez(stream,native_wave_nm=wave,refined_wave_nm=full['mlambda']*1000,flux=flux,error=error,transmittance=full['mtrans'],valid=valid)
            report.update(accepted=True,correction_sha256=hashlib.sha256((work/'correction.npz').read_bytes()).hexdigest())
        except Exception as exception:
            report['failure']=str(exception)
            raise
        finally:
            report['elapsed_seconds']=time.monotonic()-start
            with (work/'report.json').open('x') as stream:json.dump(report,stream,indent=2)
        reports.append(report)
        print(f'  czesla2024 molecfit: exposure {number}/{len(rows)}, shift {shift:+.3f} km/s',flush=True)
    with (output/'night_report.json').open('x') as stream:json.dump(reports,stream,indent=2)
    return output
