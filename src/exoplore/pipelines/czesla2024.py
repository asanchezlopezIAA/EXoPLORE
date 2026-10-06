"""
exoplore.pipelines.czesla2024
=============================

Direct He I transmission preparation following Czesla et al. (2024),
A&A 692, A230, https://doi.org/10.1051/0004-6361/202451003.
The published stellar-frame reference division and planetary-frame coaddition
are retained. Continuum settings and noise propagation are explicit local
choices. This module neither detrends with SYSREM nor declares a detection.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
from scipy.optimize import least_squares

from exoplore.config.czesla2024 import Czesla2024Config
from exoplore.pipelines.helium_math import (
    C_KMS, multiplet, planet_wavelength, resample, stellar_wavelength,
    weighted_mean, window_mask,
)


def exposure_indices(numbers: list[int], n_exposures: int) -> np.ndarray:
    """Convert explicit one-based running numbers; never assume a 40-frame night."""
    if not numbers or len(set(numbers))!=len(numbers) or any(type(i) is not int or i<1 or i>n_exposures for i in numbers):
        raise ValueError('Invalid chronological exposure selection')
    return np.asarray(numbers,dtype=int)-1


def exposure_metadata(raw: Path, config: Czesla2024Config) -> tuple[float,float,float]:
    """Compute BJD_TDB and additive BERV from the individual raw UTC midpoint."""
    header=fits.getheader(raw)
    duration=float(header[config.exposure_time_keyword])
    if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid exposure duration')
    location=EarthLocation.from_geodetic(*config.observatory_lon_lat_height[:2],height=config.observatory_lon_lat_height[2]*u.m)
    target=SkyCoord(config.target_ra_deg*u.deg,config.target_dec_deg*u.deg)
    time=Time(header['MJD-OBS']+duration/2/86400,format='mjd',scale='utc',location=location)
    # Deterministic offline timing; restore global Astropy settings afterward.
    with iers.conf.set_temp('auto_download',False),iers.conf.set_temp('auto_max_age',None):
        bjd=float((time.tdb+time.light_travel_time(target)).jd)
        berv=float(target.radial_velocity_correction(obstime=time).to_value(u.km/u.s))
    return bjd,berv,duration


def read_corrected_night(config: Czesla2024Config) -> dict:
    """Read validated correction caches, verifying their source files and gates.

    Corrections contain raw extracted flux/error, native and refined vacuum
    grids and dimensionless ``transmittance``. Never use ``mflux`` as a divisor.
    Reports must name the individual raw exposure and preserve file hashes.
    The exploratory nod-wide caches are intentionally not accepted.
    """
    root=Path(config.input_path);report_path=root/'night_report.json'
    reports=json.loads(report_path.read_text());paths=[report_path]
    if not reports:raise ValueError('Empty corrected night')
    waves=[];flux=[];error=[];bjd=[];berv=[];duration=[];inflation=[];shifts=[]
    for number,report in enumerate(reports,1):
        if report['exposure']!=number or not report['accepted']:
            raise ValueError('Every chronological correction must be accepted; no dropping or fallback')
        if report.get('frame')!='topocentric vacuum nm':raise ValueError('Correction frame is not verified vacuum topocentric')
        extracted=Path(report['extracted']);raw=Path(report['raw'])
        if hashlib.sha256(extracted.read_bytes()).hexdigest()!=report['extracted_sha256']:
            raise ValueError('Extracted input changed since calibration')
        if hashlib.sha256(fits.getheader(raw).tostring().encode()).hexdigest()!=report['raw_header_sha256']:
            raise ValueError('Individual raw header changed since calibration')
        path=root/f"exposure_{number:02d}_{report['nod']}"/'correction.npz';paths.extend([path,extracted])
        if 'correction_sha256' not in report or hashlib.sha256(path.read_bytes()).hexdigest()!=report['correction_sha256']:
            raise ValueError('Correction needs a matching provenance hash')
        with np.load(path,allow_pickle=False) as data:
            native=data['native_wave_nm'];wave=data['refined_wave_nm'];transmission=data['transmittance']
            from exoplore.instruments.crires_molecfit import segments_of
            extracted_segments={segment[0]:segment for segment in segments_of(extracted)}
            if config.order_segment not in extracted_segments or not np.array_equal(native,extracted_segments[config.order_segment][1]):
                raise ValueError('Correction does not belong to the configured physical order segment')
            if np.any(~np.isfinite(wave)) or np.any(np.diff(wave)<=0) or np.any(~np.isfinite(transmission)) or np.any(transmission<=0):
                raise ValueError('Invalid full-grid wavelength or telluric model')
            if native.shape!=wave.shape or not all(data[key].shape==wave.shape for key in ['flux','error','valid','transmittance']):
                raise ValueError('Correction array shapes disagree')
            if not native.min()<=np.mean(config.helium_vacuum_lines_nm)<=native.max():raise ValueError('Chosen segment does not cover helium')
            shift=float(np.interp(np.mean(config.helium_vacuum_lines_nm),native,C_KMS*(wave/native-1)))
            if abs(shift)>config.max_wavelength_shift_kms or abs(report['he_shift_kms'])>config.max_wavelength_shift_kms:
                raise ValueError('Rejected molecfit wavelength displacement: investigate calibration')
            good=data['valid'].astype(bool)&np.isfinite(data['flux'])&np.isfinite(data['error'])&(data['error']>0)&(transmission>=config.telluric_min_transmission)
            if config.oh_mode=='mask':good&=~window_mask(wave,config.oh_topocentric_windows_nm)
            waves.append(wave.copy());flux.append(np.where(good,data['flux']/transmission,np.nan))
            error.append(np.where(good,data['error']/transmission,np.nan));shifts.append(shift)
        midpoint,correction,seconds=exposure_metadata(raw,config)
        bjd.append(midpoint);berv.append(correction);duration.append(seconds)
        inflation.append(max(1.,float(report['parameters']['rms_rel_to_err'])) if config.inflate_by_telluric_rms else 1.)
    if np.any(np.diff(bjd)<=0):raise ValueError('Individual BJD midpoints must strictly increase')
    return {'wave_nm':np.asarray(waves),'flux':np.asarray(flux),'error':np.asarray(error),
            'bjd_tdb':np.asarray(bjd),'berv_kms':np.asarray(berv),'duration_seconds':np.asarray(duration),
            'noise_inflation':np.asarray(inflation),'wavelength_shift_kms':shifts,
            'nod':[r['nod'] for r in reports],'paths':paths,
            'raw_header_hashes':[{'path':r['raw'],'sha256':r['raw_header_sha256']} for r in reports]}


def prepare_transmission(waves_nm: np.ndarray, flux: np.ndarray, error: np.ndarray,
                         berv_kms: np.ndarray, planet_rv_kms: np.ndarray,
                         config: Czesla2024Config) -> dict[str,np.ndarray]:
    """Stellar-align, normalize, reference-divide and planet-align one segment.

    Inputs are already telluric-corrected native flux/error arrays. Invalid
    pixels remain NaN. Native OH masking is performed by the reader before
    interpolation; callers of this array API must supply the same native mask.
    Returned variances are conditional marginal weights, not full covariance.
    """
    waves=np.asarray(waves_nm,float);data=np.asarray(flux,float);errors=np.asarray(error,float)
    if waves.ndim!=2 or waves.shape!=data.shape or errors.shape!=data.shape:
        raise ValueError('Expected matching exposure-by-pixel arrays')
    n=len(data)
    if np.shape(berv_kms)!=(n,) or np.shape(planet_rv_kms)!=(n,):raise ValueError('Velocity array lengths disagree')
    reference=exposure_indices(config.reference_running_numbers,n);inside=exposure_indices(config.in_transit_running_numbers,n)
    stellar=np.asarray([stellar_wavelength(w,b,config.gamma_kms) for w,b in zip(waves,berv_kms)])
    grid=np.median(stellar,axis=0);bands=window_mask(grid,config.continuum_stellar_windows_nm)
    if not bands.any():raise ValueError('Continuum windows outside the segment')
    x=grid-grid[bands].mean();normalized=[];variance=[]
    for w,y,e in zip(stellar,data,errors):
        row,var=resample(w,y,e**2,grid)
        good=bands&np.isfinite(row)&np.isfinite(var)&(var>0)
        if good.sum()<config.minimum_continuum_pixels:raise ValueError('Insufficient valid continuum pixels')
        coefficients=np.polyfit(x[good],row[good],config.continuum_polynomial_degree,w=1/np.sqrt(var[good]))
        continuum=np.polyval(coefficients,x)
        if np.any(~np.isfinite(continuum)) or np.any(continuum<=0):raise ValueError('Invalid continuum')
        normalized.append(row/continuum);variance.append(var/continuum**2)
    normalized=np.asarray(normalized);variance=np.asarray(variance)
    master,master_variance=weighted_mean(normalized[reference],variance[reference])
    ratio=normalized/master;ratio_variance=variance/master**2
    planet=[];planet_variance=[]
    for row,var,rv in zip(ratio,ratio_variance,planet_rv_kms):
        yy,vv=resample(planet_wavelength(grid,rv),row,var,grid);planet.append(yy);planet_variance.append(vv)
    planet=np.asarray(planet);planet_variance=np.asarray(planet_variance)
    coadd,conditional=weighted_mean(planet[inside],planet_variance[inside])
    def curve(array: np.ndarray,window: list[float]) -> np.ndarray:
        selected=window_mask(grid,[window]);values=array[:,selected]
        # Fixed-band integrated curves require complete coverage; no masked-gap bridge.
        return np.asarray([np.mean(row) if row.size and np.isfinite(row).all() else np.nan for row in values])
    return {'stellar_wave_nm':grid,'stellar_transmission':ratio,'planet_wave_nm':grid,
            'planet_transmission':planet,'planet_coadd':coadd,'conditional_coadd_variance':conditional,
            'master':master,'master_variance':master_variance,'normalized_stellar_flux':normalized,
            'normalized_stellar_variance':variance,'planet_lightcurve':curve(planet,config.planet_lightcurve_window_nm),
            'stellar_lightcurve':curve(ratio,config.stellar_lightcurve_window_nm)}


def fit_helium_multiplet(wave_nm: np.ndarray,transmission: np.ndarray,error: np.ndarray,
                        lines_nm: np.ndarray,strengths: np.ndarray,filling_factor: float,
                        effective_resolution: float,initial: list[float],bounds: list[list[float]]) -> dict:
    """Fit the Czesla slab's log-column, shift and sigma with baseline fixed at one.

    Filling factor and effective (already smear-broadened) resolution must be
    supplied. Marginal error weights yield only a diagnostic point fit; no
    detection significance or posterior intervals are returned.
    """
    if effective_resolution<=0:raise ValueError('Resolution must be positive')
    good=np.isfinite(transmission)&np.isfinite(error)&(error>0)
    if good.sum()<20:raise ValueError('Insufficient multiplet coverage')
    system=C_KMS/effective_resolution/np.sqrt(8*np.log(2))
    def residual(par):return (multiplet(wave_nm[good],10**par[0],par[1],par[2],filling_factor,system,lines_nm,strengths)-transmission[good])/error[good]
    fit=least_squares(residual,initial,bounds=bounds,max_nfev=1000)
    return {'success':bool(fit.success),'log10_column_cm2':float(fit.x[0]),'velocity_shift_kms':float(fit.x[1]),
            'intrinsic_sigma_kms':float(fit.x[2]),'doppler_b_kms':float(np.sqrt(2)*fit.x[2]),
            'filling_factor':filling_factor,'effective_resolution':effective_resolution,
            'uncertainty_scope':'Diagnostic marginal-error point fit; no posterior or significance'}


def run_czesla2024(simulation_config) -> Path:
    """Run the explicitly selected observed-data recipe through EXoPLORE's runner.

    The output directory must be new. Raw reduction and validated molecfit
    corrections are upstream inputs, described in the RTD tutorial. Native
    noise draws rebuild normalization, reference, interpolation and coaddition;
    physical extraction, stellar and telluric-model systematics remain excluded.
    """
    config=simulation_config.pipeline.czesla2024
    if not isinstance(config,Czesla2024Config):raise ValueError('Explicit pipeline.czesla2024 configuration required')
    destination=Path(simulation_config.paths.output_root)/simulation_config.planet.name/'czesla2024'
    if destination.exists():raise FileExistsError(f'Preserving existing output: {destination}')
    night=read_corrected_night(config)
    phase=(night['bjd_tdb']-config.t0_bjd_tdb)/config.period_days;phase-=np.round(np.median(phase))
    rv=config.kp_kms*np.sin(2*np.pi*phase)
    # Validate every data selection and normalization before creating any output.
    observed=prepare_transmission(night['wave_nm'],night['flux'],night['error'],night['berv_kms'],rv,config)
    output=Path(simulation_config.paths.output_root)/simulation_config.planet.name/'czesla2024'
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(config.monte_carlo_seed);coadds=[];planet_curves=[];stellar_curves=[]
    draw_error=night['error']*night['noise_inflation'][:,None]
    for draw in range(config.monte_carlo_draws):
        result=prepare_transmission(night['wave_nm'],night['flux']+rng.normal(size=night['flux'].shape)*draw_error,
                                    night['error'],night['berv_kms'],rv,config)
        coadds.append(result['planet_coadd']);planet_curves.append(result['planet_lightcurve']);stellar_curves.append(result['stellar_lightcurve'])
        if (draw+1)%64==0:print(f'  czesla2024: native-noise draws {draw+1}/{config.monte_carlo_draws}',flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        observed['planet_coadd_error']=np.nanstd(coadds,axis=0,ddof=1)
        observed['planet_lightcurve_error']=np.nanstd(planet_curves,axis=0,ddof=1)
        observed['stellar_lightcurve_error']=np.nanstd(stellar_curves,axis=0,ddof=1)
    selected=window_mask(observed['planet_wave_nm'],[config.equivalent_width_window_nm])
    complete=selected.any() and np.isfinite(observed['planet_coadd'][selected]).all() and np.isfinite(np.asarray(coadds)[:,selected]).all()
    ew=float(np.trapz(1-observed['planet_coadd'][selected],observed['planet_wave_nm'][selected])*1e4) if complete else None
    ew_error=float(np.std(np.trapz(1-np.asarray(coadds)[:,selected],observed['planet_wave_nm'][selected],axis=1)*1e4,ddof=1)) if complete else None
    with (output/'transmission.npz').open('xb') as stream:
        np.savez(stream,**observed,bjd_tdb=night['bjd_tdb'],berv_kms=night['berv_kms'],planet_rv_kms=rv,phase=phase,
                 coadd_noise_samples=np.asarray(coadds),planet_lightcurve_noise_samples=np.asarray(planet_curves),stellar_lightcurve_noise_samples=np.asarray(stellar_curves))
    summary={'pipeline':'czesla2024','scientific_reference':'https://doi.org/10.1051/0004-6361/202451003',
             'status':'Direct transmission preparation; no automatic detection decision',
             'configuration':asdict(config),'simulation_configuration':simulation_config.to_dict(),
             'n_exposures':len(rv),'nod':night['nod'],'wavelength_shift_kms':night['wavelength_shift_kms'],
             'noise_inflation_factors':night['noise_inflation'].tolist(),'equivalent_width_mA':ew,'ew_noise_error_mA':ew_error,
             'ew_coverage':'complete' if complete else 'not reported: masked or missing pixels; no gap bridging',
             'oh_emission_model_applied':False,'uncertainty_scope':'Native noise propagated through continuum, shared reference and interpolation. No stellar/telluric/calibration/physical AB systematic model, significance or physical upper limit.',
             'weight_convention':'Native ESO conditional variances; RMS inflation affects noise draws only',
             'inputs':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in night['paths']],
             'raw_header_hashes':night['raw_header_hashes']}
    with (output/'summary.json').open('x') as stream:json.dump(summary,stream,indent=2)
    plot_transmission(observed,phase,rv,night['berv_kms'],config,output/'transmission_diagnostic.png')
    print(f'  czesla2024: direct transmission products written to {output}',flush=True)
    return output


def plot_transmission(result: dict,phase: np.ndarray,rv: np.ndarray,berv: np.ndarray,
                      config: Czesla2024Config,path: Path) -> None:
    """Write a new diagnostic map, planetary coadd and two fixed-band curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if path.exists():raise FileExistsError(path)
    wave=result['stellar_wave_nm'];hours=phase*config.period_days*24
    fig,axes=plt.subplots(4,1,figsize=(10,12),constrained_layout=True)
    image=axes[0].pcolormesh(wave,hours,result['stellar_transmission'],cmap='RdBu',shading='auto',vmin=.975,vmax=1.025)
    for line in config.helium_vacuum_lines_nm:
        axes[0].axvline(line,color='magenta',ls=':');axes[0].plot(line*(1+rv/C_KMS),hours,color='red',ls='--',lw=.7)
        axes[1].axvline(line,color='magenta',ls=':')
    for line in config.oh_topocentric_lines_nm:
        axes[0].plot(stellar_wavelength(line,berv,config.gamma_kms),hours,color='gold',ls=':',lw=.7)
    axes[0].set(xlim=config.plot_stellar_window_nm,xlabel='Stellar-frame vacuum wavelength (nm)',ylabel='Hours from transit',title='Czesla et al. (2024) direct preparation — diagnostic')
    fig.colorbar(image,ax=axes[0],label='Relative transmission')
    axes[1].plot(wave,result['planet_coadd'],'k-',lw=.7)
    axes[1].fill_between(wave,result['planet_coadd']-result['planet_coadd_error'],result['planet_coadd']+result['planet_coadd_error'],alpha=.2)
    axes[1].set(xlim=config.plot_stellar_window_nm,xlabel='Planet-frame vacuum wavelength (nm)',ylabel='Transmission')
    for ax,key,title in [(axes[2],'planet','Planet-frame fixed-band curve'),(axes[3],'stellar','Stellar-frame fixed-band curve')]:
        ax.errorbar(hours,result[key+'_lightcurve'],yerr=result[key+'_lightcurve_error'],fmt='o',ms=3)
        ax.set(xlabel='Hours from transit',ylabel='Transmission',title=title)
    for contact in config.optical_contact_phases:
        axes[0].axhline(contact*config.period_days*24,color='black',ls='--',lw=.5)
        for ax in axes[2:]:ax.axvline(contact*config.period_days*24,color='black',ls='--',lw=.5)
    fig.savefig(path,dpi=160);plt.close(fig)


def validate_run_config(config) -> None:
    """Validate the opt-in direct branch without loading spectra or optional pRT."""
    if config.instrument.name not in ('CRIRES+','CRIRES_PLUS'):
        raise ValueError('czesla2024 currently supports observed CRIRES+ datasets')
    if config.observation.event_type!='transit' or not config.observation.use_real_data:
        raise ValueError('czesla2024 requires a real observed transit time series')
    if config.observation.n_nights!=1:
        raise ValueError('Run czesla2024 separately for each night with its own reference selection')
    if config.observation.simulate_planet:
        raise ValueError('Set observation.simulate_planet=false for direct observed helium preparation')
    if config.pipeline.sysrem_iterations!=0 or config.pipeline.optimize_sysrem_order_by_order or config.pipeline.sysrem_robust_halt:
        raise ValueError('czesla2024 has no SYSREM; set iterations=0 and disable optimization')
    if config.retrieval.enabled:
        raise ValueError('The direct helium branch does not use the molecular atmospheric retrieval')
    if config.pipeline.czesla2024 is None:
        raise ValueError('Explicit pipeline.czesla2024 settings are required')


def preparing_pipeline_adapter(inp_dat: dict,data: np.ndarray,noise: np.ndarray,
                               wave: np.ndarray,mask: np.ndarray,masks: bool,
                               correct_uncertainties: bool,retrieval: bool):
    """Expose the direct array preparation through the existing dispatcher.

    Requires ``czesla2024_config``, ``berv_kms``, ``planet_rv_kms`` and an
    explicit ``telluric_corrected=True`` statement in ``inp_dat``. Per-frame
    native wavelengths and masked errors are required for an OH-mask run.
    Output noise is marginal only; use the high-level runner for native MC.
    """
    if retrieval:raise ValueError('czesla2024 dispatcher does not implement molecular retrieval filtering')
    if inp_dat.get('telluric_corrected') is not True:
        raise ValueError('czesla2024 requires separately validated telluric corrections')
    config=inp_dat['czesla2024_config']
    if isinstance(config,dict):config=Czesla2024Config(**config)
    waves=np.broadcast_to(wave,data.shape);flux=np.array(data,dtype=float,copy=True);error=np.array(noise,dtype=float,copy=True)
    flux[:,np.asarray(mask,dtype=int)]=np.nan;error[:,np.asarray(mask,dtype=int)]=np.nan
    if config.oh_mode=='mask':
        bad=window_mask(waves,config.oh_topocentric_windows_nm);flux[bad]=np.nan;error[bad]=np.nan
    result=prepare_transmission(waves,flux,error,np.asarray(inp_dat['berv_kms']),np.asarray(inp_dat['planet_rv_kms']),config)
    reference=exposure_indices(config.reference_running_numbers,len(flux))
    var=result['normalized_stellar_variance'];f=result['normalized_stellar_flux'];master=result['master']
    marginal=var/master**2+f**2*result['master_variance']/master**4
    # Numerator/reference dependence for OOT members, with fixed reference weights.
    weight=np.divide(1,var[reference],out=np.zeros_like(var[reference]),where=np.isfinite(var[reference])&(var[reference]>0))
    coefficient=weight/weight.sum(axis=0)
    marginal[reference]-=2*f[reference]*coefficient*var[reference]/master**3
    prepared=result['stellar_transmission'].copy();propagated=np.sqrt(np.maximum(marginal,0))
    bad=np.flatnonzero(np.any(~np.isfinite(prepared)|~np.isfinite(propagated)|(propagated<=0),axis=0))
    useful=np.setdiff1d(np.arange(prepared.shape[1]),bad)
    prepared[:,bad]=1.;propagated[:,bad]=1.
    if not masks:return (prepared,propagated) if correct_uncertainties else prepared
    return prepared,propagated,useful,bad,0,bad.copy(),useful.copy(),None,None
