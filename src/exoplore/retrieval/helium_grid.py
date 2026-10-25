"""Sunbather grid fitting of prepared planetary-frame He I spectra.

Fit physical Parker T0 and mass loss with a profiled shared Doppler shift.
Saved EXoPLORE inputs retain their native masks, observing times, continuum
normalization and reference/coadd weighting. No noise is drawn and no
telluric fit is performed. Covariance is conditional on upstream corrections.
"""
from __future__ import annotations

# Role in the calculation
# Fit a prepared helium spectrum by evaluating atmospheric grid points with its
# sampling and covariance information. _night_prediction samples each model using the
# stored observing inputs.


from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import load_npz

from exoplore.config.helium_grid import HeliumGridConfig, axis_values
from exoplore.pipelines.helium_math import C_KMS, stellar_wavelength, planet_wavelength
from exoplore.pipelines.helium_noise_propagation import interpolation_operator


def _json(path, value):
    """Create a new provenance or result file. Never overwrite a product."""
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def _archive(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def _read_single_folder(folder: Path, choices: dict) -> dict:
    """Read prepared spectra and native inputs, preserving saved membership."""
    from exoplore.config.czesla2024 import Czesla2024Config
    from exoplore.config.carmenes_helium import CarmenesHeliumConfig
    from exoplore.pipelines.czesla2024 import read_corrected_night, resample
    # The combined transmission alone is insufficient for model comparison.
    # Recover its saved selections/configuration as well, so a proposed model
    # can be sampled/prepared with the same observing and coaddition context.
    path=folder/'transmission.npz'
    data=_archive(path)
    sources=[path]
    science=pipeline=None
    duration=choices.get('exposure_seconds')
    for name in ('summary.json','oh_report.json','run_config.json','input_provenance.json'):
        metadata_path=folder/name
        if not metadata_path.is_file():
            continue
        metadata=json.loads(metadata_path.read_text());sources.append(metadata_path)
        if name=='summary.json':
            science=metadata.get('transmission_configuration',metadata.get('configuration',science))
            pipeline=metadata.get('pipeline',pipeline)
            duration=metadata.get('simulation_configuration',{}).get('observation',{}).get('exposure_time_seconds',duration)
        elif name=='oh_report.json':
            science=metadata.get('science_configuration',science)
            if science is None and 'settings' in metadata:
                # Native runner writes its full science choices to summary.json.
                raise ValueError('The OH folder lacks its science configuration')
            pipeline=pipeline or 'Palle2020'
        elif name=='run_config.json':
            pipeline=metadata['pipeline']['name']
            science=metadata['pipeline'].get('czesla2024') or metadata['pipeline'].get('carmenes_helium') or science
            duration=metadata['observation']['exposure_time_seconds']
        if duration is None and 'sources' in metadata:
            from astropy.io import fits
            paths=[s['path'] for s in metadata['sources'] if s['path'].endswith('nir_A.fits')]
            if paths:
                # read_night stores sources in chronological A/B order.
                duration=np.array([fits.getheader(p)['EXPTIME'] for p in paths])
                sources.extend(Path(p) for p in paths)
    if pipeline not in ('Palle2020','carmenes_helium','czesla2024') or science is None:
        raise ValueError('The folder must contain EXoPLORE transmission results and their preparation configuration')
    if duration is None:
        raise ValueError('Saved exposure duration is unavailable; supply helium_grid.input.exposure_seconds explicitly')
    cls=Czesla2024Config if pipeline=='czesla2024' else CarmenesHeliumConfig
    science=dict(science)
    science.pop('reference_running_numbers',None);science.pop('in_transit_running_numbers',None)
    science=cls(**science)
    reference_indices=data.get('reference_indices')
    if reference_indices is None:
        reference_indices=data.get('without_signal')
    if reference_indices is None:
        raise ValueError('Saved EXoPLORE spectra must retain their automatically calculated reference membership')
    refs=np.asarray(reference_indices,int)
    inside=np.asarray(data['full_transit_indices'],int)
    object.__setattr__(science,'_reference_indices',tuple(refs.tolist()))
    object.__setattr__(science,'_full_transit_indices',tuple(inside.tolist()))
    observations=folder/'observations.npz'
    if observations.is_file():
        obs=_archive(observations);sources.append(observations)
        native_wave=obs['wave_nm'];native_flux=obs['corrected_flux'];native_error=obs['corrected_error']
    elif pipeline=='czesla2024':
        night=read_corrected_night(science)
        native_wave,native_flux,native_error=(night[k] for k in ('wave_nm','flux','error'))
    else:
        raise ValueError('EXoPLORE grid input needs saved native corrected spectra and uncertainties')
    grid=data['stellar_wave_nm']
    continuum=[]
    for w,f,e,b in zip(native_wave,native_flux,native_error,data['berv_kms']):
        good=np.isfinite(f)&np.isfinite(e)&(e>0)
        if pipeline!='czesla2024':
            # All configured windows contribute to the native continuum mean.
            band=np.zeros(len(w),bool)
            for low,high in science.continuum_stellar_windows_nm:
                band|=(w>=low)&(w<=high)
            value=float(np.mean(f[good&band]));continuum.append(np.full_like(w,value))
        else:
            sw=stellar_wavelength(w,b,science.gamma_kms)
            row,var=resample(sw,f,e**2,grid)
            band=np.zeros(len(grid),bool)
            for low,high in science.continuum_stellar_windows_nm:
                band|=(grid>=low)&(grid<=high)
            valid=band&np.isfinite(row)&np.isfinite(var)&(var>0)
            pivot=grid[band].mean()
            coefficients=np.polyfit((grid-pivot)[valid],row[valid],science.continuum_polynomial_degree,w=1/np.sqrt(var[valid]))
            continuum.append(np.polyval(coefficients,sw-pivot))
    continuum=np.asarray(continuum)
    if not np.isfinite(continuum).all() or np.any(continuum<=0):
        raise ValueError('Cannot reconstruct the saved continuum for model preparation')
    return dict(data=data,science=science,pipeline=pipeline,refs=refs,inside=inside,
                native_wave=native_wave,native_error=native_error,continuum=continuum,
                native_valid=np.isfinite(native_flux)&np.isfinite(native_error)&(native_error>0),
                duration=np.broadcast_to(duration,(len(data['phase']),)).copy(),sources=sources)


def read_grid_spectrum(settings: HeliumGridConfig) -> dict:
    """Read an EXoPLORE folder or explicitly mapped external three-column data."""
    source=settings.input;path=Path(source['path']).resolve();sources=[];nights=[]
    covariance=None
    if source['kind']=='exoplore':
        if (path/'combined_transmission.npz').is_file():
            data=_archive(path/'combined_transmission.npz')
            manifest_path=path/'night_manifest.json';manifest=json.loads(manifest_path.read_text())
            nights=[_read_single_folder(Path(p),source) for p in manifest['night_paths']]
            wave,flux,error=(data[k] for k in ('planet_wave_nm','combined_spectrum','combined_noise_error'))
            covariance_file=path/'combined_conditional_covariance.npz'
            sources.extend((path/'combined_transmission.npz',manifest_path))
        else:
            night=_read_single_folder(path,source);nights=[night];data=night['data']
            wave,flux=data['planet_wave_nm'],data['planet_coadd']
            key='planet_coadd_error' if 'planet_coadd_error' in data else 'coadd_error'
            if key not in data:
                raise ValueError('The saved spectrum has no reported coadd uncertainties')
            error=data[key];covariance_file=path/'planet_coadd_covariance.npz'
        if source['uncertainty']=='saved_covariance':
            if not covariance_file.is_file():
                raise ValueError('This folder has no saved spectral covariance. Choose diagonal explicitly for a marginal-error grid ranking, or provide external_covariance.')
            covariance=load_npz(covariance_file);sources.append(covariance_file)
        for night in nights:sources.extend(night['sources'])
    else:
        columns=source['columns']
        if source['kind']=='ascii':
            table=np.loadtxt(path,skiprows=source.get('skiprows',0),delimiter=source.get('delimiter'))
            if table.ndim!=2:
                raise ValueError('External spectrum must be a two-dimensional table')
            wave,flux,error=(table[:,columns[k]] for k in ('wavelength','value','uncertainty'))
        else:
            from astropy.io import fits
            if 'extension' not in source:
                raise ValueError('Supply the external FITS table extension explicitly')
            table=fits.getdata(path,source['extension'])
            wave,flux,error=(np.asarray(table[columns[k]],float) for k in ('wavelength','value','uncertainty'))
        wave=np.asarray(wave,float)*{'nm':1.,'um':1000.,'angstrom':.1}[source['wavelength_unit']]
        value_kind=source['value_kind'];factor=.01 if value_kind.endswith('percent') else 1.
        flux=np.asarray(flux,float);error=np.asarray(error,float)*factor
        flux=(flux if value_kind=='transmission' else 1+flux*.01 if value_kind=='transmission_excess_percent' else 1-flux*factor)
        sources.append(path)
    if source['uncertainty']=='external_covariance':
        covariance_path=Path(source['covariance_path'])
        covariance=load_npz(covariance_path) if covariance_path.suffix=='.npz' else np.loadtxt(covariance_path)
        factor=(.01 if source['kind']!='exoplore' and source['value_kind'].endswith('percent') else 1.)
        covariance=covariance*factor**2;sources.append(covariance_path)
    elif source['uncertainty']=='saved_covariance' and source['kind']!='exoplore':
        raise ValueError('saved_covariance requires an EXoPLORE folder; use external_covariance or diagonal for external spectra')
    wave,flux,error=map(lambda x:np.asarray(x,float),(wave,flux,error))
    if wave.ndim!=1 or flux.shape!=wave.shape or error.shape!=wave.shape or not np.isfinite(wave).all() or np.any(np.diff(wave)<=0):
        raise ValueError('Provide matching 1-D spectra with finite increasing wavelengths; invalid flux pixels remain masked')
    if np.any(np.isfinite(error)&(error<=0)):
        raise ValueError('Reported finite uncertainties must be positive')
    good=(wave>=settings.fit_window_nm[0])&(wave<=settings.fit_window_nm[1])&np.isfinite(flux)&np.isfinite(error)&(error>0)
    indices=np.flatnonzero(good)
    if len(indices)<=3:
        raise ValueError('Too few valid spectrum pixels for a three-parameter grid fit')
    if covariance is None:
        matrix=np.diag(error[indices]**2)
    else:
        if covariance.shape!=(len(wave),len(wave)):
            raise ValueError('Covariance dimensions must match the original spectrum before masking')
        selected=covariance[indices][:,indices] if hasattr(covariance,'tocsr') else covariance[np.ix_(indices,indices)]
        matrix=selected.toarray() if hasattr(selected,'toarray') else selected
        if not np.allclose(np.diag(matrix),error[indices]**2,rtol=1e-4,atol=0):
            raise ValueError('Saved covariance diagonal and reported uncertainties disagree')
    if not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T,rtol=1e-8,atol=1e-16):
        raise ValueError('Fit covariance must be finite and symmetric')
    eigenvalues,eigenvectors=np.linalg.eigh(matrix)
    cutoff=settings.covariance_eigenvalue_rtol*eigenvalues.max()
    if eigenvalues.max()<=0 or eigenvalues.min() < -cutoff:
        raise ValueError('Fit covariance must be positive semidefinite')
    keep=eigenvalues>cutoff
    rank=int(keep.sum())
    if rank<=3:
        raise ValueError('Too few independent covariance modes for the grid fit')
    whitener=(eigenvectors[:,keep]/np.sqrt(eigenvalues[keep])).T
    return dict(wave_nm=wave,transmission=flux,error=error,fit_indices=indices,
                covariance=matrix,whitener=whitener,rank=rank,nights=nights,
                sources=list(dict.fromkeys(Path(p).resolve() for p in sources)))


def validate_grid_config(config) -> None:
    """Validate the fit stage without launching an atmosphere or recipe."""
    settings=config.helium_grid
    if settings is None or not settings.enabled:
        raise ValueError('Enable helium_grid explicitly to run this stage')
    if config.observation.helium_transmission_spectroscopy is False or config.retrieval.enabled:
        raise ValueError('Select helium spectroscopy and disable the molecular retrieval when using helium_grid')
    if not config.planet.parameter_file:
        raise ValueError('Provide the planet parameter JSON for the physical grid')
    for path in (config.planet.parameter_file,settings.atmosphere['irradiation_spectrum_path'],settings.input['path']):
        if not Path(path).exists():raise FileNotFoundError(path)
    cloudy=Path(settings.atmosphere['sunbather']['cloudy_path'])/'source/cloudy.exe'
    if not cloudy.is_file():raise FileNotFoundError(cloudy)


def _pixel_edges(wave):
    """Return per-pixel linear wavelength edges without bridging detector gaps."""
    cuts=np.flatnonzero(np.diff(wave)>5*np.median(np.diff(wave)))+1
    low,high=np.empty_like(wave),np.empty_like(wave)
    for indices in np.split(np.arange(len(wave)),cuts):
        if len(indices)<2:raise ValueError('A detector section needs at least two pixels')
        w=wave[indices]
        edges=np.r_[w[0]-(w[1]-w[0])/2,(w[:-1]+w[1:])/2,w[-1]+(w[-1]-w[-2])/2]
        low[indices],high[indices]=edges[:-1],edges[1:]
    return low,high


def _model_rows(model, phases, durations, kp, period, resolution, quadrature, dense_wave):
    """Integrate phase-dependent spectra, orbital smearing and one planet LSF."""
    from exoplore.atmosphere.prt import convolve
    nodes,weights=np.polynomial.legendre.leggauss(quadrature);weights/=2
    rows=[]
    for phase,duration in zip(phases,durations):
        row=np.zeros_like(dense_wave);opaque_mean=0.
        for node,weight in zip(nodes,weights):
            p=float(phase+node*duration/(2*period*86400))
            helium,opaque=model.spectrum(p)
            rv=kp*np.sin(2*np.pi*p)
            row+=weight*opaque*np.interp(dense_wave/(1+rv/C_KMS),model.wave_nm,helium,left=1.,right=1.)
            opaque_mean+=weight*opaque
        rows.append(convolve(dense_wave/1000,row/opaque_mean,resolution))
    return np.asarray(rows)


def _night_prediction(night, dense_wave, rows, velocity):
    """Replay native sampling/preparation with frozen data-derived fit weights."""
    from scipy.integrate import cumulative_trapezoid
    from exoplore.pipelines.czesla2024 import prepare_transmission
    from exoplore.pipelines.carmenes_helium import prepare_carmenes_transmission
    data,science=night['data'],night['science']
    sampled=[]
    for wave,row,berv in zip(night['native_wave'],rows,data['berv_kms']):
        low,high=_pixel_edges(wave)
        low=stellar_wavelength(low,berv,science.gamma_kms)/(1+velocity/C_KMS)
        high=stellar_wavelength(high,berv,science.gamma_kms)/(1+velocity/C_KMS)
        if low.min()<dense_wave[0] or high.max()>dense_wave[-1]:
            raise ValueError('Model sampling grid does not cover shifted detector pixels')
        integral=cumulative_trapezoid(row,dense_wave,initial=0)
        sampled.append((np.interp(high,dense_wave,integral)-np.interp(low,dense_wave,integral))/(high-low))
    flux=np.asarray(sampled)*night['continuum']
    flux=np.where(night['native_valid'],flux,np.nan)
    prepare=prepare_transmission if night['pipeline']=='czesla2024' else prepare_carmenes_transmission
    result=prepare(night['native_wave'],flux,night['native_error'],data['berv_kms'],data['planet_rv_kms'],science)
    # The likelihood must use the measured reference/coadd weights, rather
    # than adjusting its weighting independently for each proposed atmosphere.
    normalized=result['normalized_stellar_flux'];refs,inside=night['refs'],night['inside']
    variance=data['normalized_stellar_variance'];grid=data['stellar_wave_nm']
    if night['pipeline']=='czesla2024':
        weight=np.divide(1.,variance[refs],out=np.zeros_like(variance[refs]),where=np.isfinite(variance[refs])&(variance[refs]>0))
        total=weight.sum(axis=0);coeff=np.divide(weight,total,out=np.zeros_like(weight),where=total>0)
        master=np.sum(coeff*np.nan_to_num(normalized[refs]),axis=0)
    else:
        master=np.mean(normalized[refs],axis=0)
    ratio=normalized/master
    shifted=[];pvariance=[]
    for i,rv in enumerate(data['planet_rv_kms']):
        valid=np.isfinite(data['stellar_transmission'][i])
        operator=interpolation_operator(planet_wavelength(grid,rv),grid,valid)
        value=operator@np.nan_to_num(ratio[i]);value[~np.isfinite(data['planet_transmission'][i])]=np.nan
        shifted.append(value)
        var=np.divide(variance[i],data['master']**2,out=np.zeros_like(grid),where=np.isfinite(variance[i])&np.isfinite(data['master'])&(data['master']>0))
        pvariance.append(operator.multiply(operator)@var)
    shifted,pvariance=np.asarray(shifted),np.asarray(pvariance)
    if night['pipeline']=='czesla2024':
        valid=np.isfinite(data['planet_transmission'][inside])&(pvariance[inside]>0)
        weight=np.divide(1.,pvariance[inside],out=np.zeros_like(pvariance[inside]),where=valid)
        total=weight.sum(axis=0);coeff=np.divide(weight,total,out=np.zeros_like(weight),where=total>0)
        coadd=np.sum(coeff*np.nan_to_num(shifted[inside]),axis=0);coadd[total<=0]=np.nan
    else:
        coadd=np.mean(shifted[inside],axis=0)
    return coadd


def _point(task):
    """Solve one atmosphere and profile a common velocity shift."""
    payload,i,j,temperature,logmass=task
    config=json.loads(Path(payload).read_text())
    settings=HeliumGridConfig(**config['grid'])
    root=Path(config['output'])/f'model_{i:02d}_{j:02d}'
    started=time.monotonic()
    result=dict(T0_K=temperature,log10_mass_loss_rate_g_s=logmass,path=str(root))
    root.mkdir(exist_ok=False)
    try:
        from exoplore.planets import load_planet
        from exoplore.atmosphere.helium_sunbather import solve_sunbather,SunbatherTransit
        planet=load_planet(config['planet'])
        measured=read_grid_spectrum(settings)
        for night in measured['nights']:
            science=night['science']
            if not np.isclose(science.period_days,planet.orbital_period_days,rtol=0,atol=1e-10) or not np.isclose(science.kp_kms,planet.kp_kms,rtol=0,atol=1e-7):
                raise ValueError('Grid planet orbit differs from the saved preparation orbit')
        for name,digest in config['source_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Grid input changed during the run: {name}')
        if planet.eccentricity!=0:
            raise ValueError('The current Sunbather transit geometry requires a circular orbit')
        atmosphere=dict(settings.atmosphere)
        atmosphere.update(temperature_K=temperature,mass_loss_rate_g_s=10**logmass,irradiation_mode='spectrum')
        atmosphere['sunbather']=dict(atmosphere['sunbather'],project_path=str(root/'atmosphere'))
        physical=SimpleNamespace(**atmosphere)
        profile=solve_sunbather(physical,planet)
        with (root/'outflow.npz').open('xb') as stream:
            np.savez_compressed(stream,**{k:v for k,v in profile.items() if k!='warnings'})
        model=SunbatherTransit(physical,planet,physical,profile)
        if measured['nights']:
            rows=[];grids=[]
            for night in measured['nights']:
                science=night['science'];data=night['data']
                native=night['native_wave'];berv=data['berv_kms']
                sw=np.array([stellar_wavelength(w,b,science.gamma_kms) for w,b in zip(native,berv)])
                pad=6/settings.resolving_power+max(abs(np.asarray(settings.velocity_bounds_kms)))/C_KMS
                lo,hi=np.log(sw.min())-pad,np.log(sw.max())+pad
                step=min(1/(settings.resolving_power*settings.instrumental_oversampling),np.median(np.diff(model.wave_nm))/np.mean(model.wave_nm))
                dense=np.exp(np.linspace(lo,hi,int(np.ceil((hi-lo)/step))+1))
                grids.append(dense)
                rows.append(_model_rows(model,data['phase'],night['duration'],science.kp_kms,science.period_days,
                                       settings.resolving_power,settings.exposure_quadrature_points,dense))
            def prediction(velocity):
                values=[]
                for night,dense,integrated in zip(measured['nights'],grids,rows):
                    value=_night_prediction(night,dense,integrated,velocity)
                    w=night['data']['planet_wave_nm']
                    operator=interpolation_operator(w,measured['wave_nm'],np.isfinite(value))
                    aligned=operator@np.nan_to_num(value)
                    aligned[np.asarray(operator.sum(axis=1)).ravel()==0]=np.nan
                    values.append(aligned)
                return np.mean(values,axis=0)
        else:
            sampling=settings.phase_sampling
            if sampling['mode']=='mid_transit':
                phases,durations,weights=np.array([0.]),np.array([0.]),np.array([1.])
            else:
                phases=np.asarray(sampling['phase_midpoints']);durations=np.asarray(sampling['exposure_seconds']);weights=np.asarray(sampling['weights'],float);weights/=weights.sum()
            # External coadds are already in the planetary frame. retain
            # within-exposure orbital smearing, but remove each midpoint RV.
            dense=np.exp(np.linspace(np.log(model.wave_nm[0]),np.log(model.wave_nm[-1]),settings.atmosphere['model_wavelength_points']))
            rows=_model_rows(model,phases,durations,planet.kp_kms,planet.orbital_period_days,settings.resolving_power,settings.exposure_quadrature_points,dense)
            coadd=np.sum(weights[:,None]*np.asarray([np.interp(dense*(1+planet.kp_kms*np.sin(2*np.pi*p)/C_KMS),dense,row,left=1.,right=1.) for p,row in zip(phases,rows)]),axis=0)
            def prediction(velocity):
                if settings.input['sampling']=='point':
                    query=measured['wave_nm']/(1+velocity/C_KMS)
                    return np.interp(query,dense,coadd,left=np.nan,right=np.nan)
                from scipy.integrate import cumulative_trapezoid
                low,high=_pixel_edges(measured['wave_nm'])
                low/=1+velocity/C_KMS;high/=1+velocity/C_KMS
                integral=cumulative_trapezoid(coadd,dense,initial=0)
                value=(np.interp(high,dense,integral)-np.interp(low,dense,integral))/(high-low)
                value[(low<dense[0])|(high>dense[-1])]=np.nan
                return value
        indices=measured['fit_indices'];whitener=measured['whitener'];target=measured['transmission'][indices]
        def objective(velocity):
            residual=prediction(velocity)[indices]-target
            if not np.isfinite(residual).all():return np.inf
            white=whitener@residual
            return float(white@white)
        velocities=np.linspace(*settings.velocity_bounds_kms,settings.velocity_grid_points)
        scores=np.asarray([objective(v) for v in velocities]);best=int(np.argmin(scores))
        velocity,chi2=float(velocities[best]),float(scores[best])
        if 0<best<len(velocities)-1:
            fit=minimize_scalar(objective,bounds=(velocities[best-1],velocities[best+1]),method='bounded')
            if fit.fun<chi2:velocity,chi2=float(fit.x),float(fit.fun)
        if not np.isfinite(chi2):raise ValueError('No velocity trial covers all fixed fit pixels')
        r,n,t=(profile[k] for k in ('radius_m','helium_triplet_m3','temperature_K'))
        norm=np.trapz(n,r)
        result.update(status='complete',Tmax_K=float(t.max()),THe_K=float(np.trapz(n*t,r)/norm) if norm>0 else None,
                      THe_definition='Radial metastable-He number-density weighting over the saved domain',
                      chi2=chi2,reduced_chi2=chi2/(measured['rank']-3),degrees_of_freedom=measured['rank']-3,
                      covariance_rank=measured['rank'],n_fit_pixels=len(indices),velocity_shift_kms=velocity,
                      velocity_at_boundary=bool(np.isclose(velocity,settings.velocity_bounds_kms).any()),solver_warnings=profile['warnings'])
        with (root/'fitted_spectrum.npz').open('xb') as stream:
            np.savez_compressed(stream,wave_nm=measured['wave_nm'],transmission=prediction(velocity),unshifted_transmission=prediction(0.))
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
    result['elapsed_seconds']=time.monotonic()-started
    _json(root/'result.json',result)
    return result


def run_helium_grid(config) -> Path:
    """Run only the explicitly enabled grid. retain all solver products."""
    validate_grid_config(config)
    from importlib.util import find_spec
    missing=[name for name in ('sunbather','p_winds') if find_spec(name) is None]
    if missing:
        raise ImportError('Activate an environment containing Sunbather and p-winds before launching the grid; missing: '+', '.join(missing))
    settings=config.helium_grid
    measured=read_grid_spectrum(settings)
    output=Path(config.paths.output_root)/config.planet.name/'helium_sunbather'/settings.output_name
    files=measured['sources']+[Path(config.planet.parameter_file).resolve(),Path(settings.atmosphere['irradiation_spectrum_path']).resolve()]
    payload=dict(grid=asdict(settings),planet=str(Path(config.planet.parameter_file).resolve()),output=str(output.resolve()),
                 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                 statistic='Chi-square of supplied spectrum/covariance; three fitted parameters: T0, log mass loss and shared shift',
                 scope='Conditional on upstream continuum, telluric and OH corrections; grid ranking, not detection significance or posterior confidence',
                 model_preparation='Native pixel integration, saved masks and normalization; frozen data reference/coadd weights; no stellar/telluric simulation',
                 THe_definition='Radial metastable-helium density weighting, not an inferred temperature-consistency constraint')
    provenance=output/'grid_config.json'
    if output.exists():
        if not settings.resume or not provenance.is_file():raise FileExistsError(output)
        previous=json.loads(provenance.read_text());current=json.loads(json.dumps(payload))
        previous['grid'].pop('resume',None);current['grid'].pop('resume',None)
        if previous!=current:raise ValueError('Resume configuration or source hashes differ; use a new output_name')
    else:
        output.mkdir(parents=True,exist_ok=False);_json(provenance,payload)
        with (output/'fit_input.npz').open('xb') as stream:
            np.savez_compressed(stream,wave_nm=measured['wave_nm'],transmission=measured['transmission'],
                                error=measured['error'],fit_indices=measured['fit_indices'],fit_covariance=measured['covariance'])
    temperatures=axis_values(settings.temperature_grid_K,'temperature_grid_K',True)
    masses=axis_values(settings.log10_mass_loss_grid_g_s,'log10_mass_loss_grid_g_s')
    results=[];tasks=[]
    for i,t in enumerate(temperatures):
        for j,m in enumerate(masses):
            path=output/f'model_{i:02d}_{j:02d}'/'result.json'
            if path.is_file():results.append(json.loads(path.read_text()))
            elif path.parent.exists():raise ValueError(f'Preserving interrupted model {path.parent}; choose a new output_name to repeat it')
            else:tasks.append((str(provenance.resolve()),i,j,t,m))
    print(f'  Sunbather grid: {len(temperatures)} × {len(masses)} = {len(temperatures)*len(masses)} models; {settings.workers} workers; {len(tasks)} new calculations',flush=True)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    if tasks:
        # Python 3.10 lacks max_tasks_per_child. Keep the declared package
        # compatibility while recycling solver workers when it is supported.
        import inspect
        pool_settings=dict(max_workers=settings.workers,mp_context=multiprocessing.get_context('spawn'))
        if 'max_tasks_per_child' in inspect.signature(ProcessPoolExecutor).parameters:
            pool_settings['max_tasks_per_child']=1
        with ProcessPoolExecutor(**pool_settings) as pool:
            futures=[pool.submit(_point,task) for task in tasks]
            for future in as_completed(futures):
                result=future.result();results.append(result)
                print(f'  Grid {len(results)}/{len(temperatures)*len(masses)}: T0={result["T0_K"]:g} K, log MLR={result["log10_mass_loss_rate_g_s"]:.3f}, {result["status"]}',flush=True)
    summary=output/'grid_results.json'
    if not summary.exists():_json(summary,dict(results=results,configuration=str(provenance),noise_draws=0))
    from exoplore.plotting.helium_grid import plot_helium_grid
    plot_helium_grid(output,settings,results)
    return output
