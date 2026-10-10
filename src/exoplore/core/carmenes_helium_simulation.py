"""Opt-in p-winds/CARMENES synthetic transit route within EXoPLORE.

The regular simulator and CRIRES helium route retain their existing behavior.
This route consumes full-channel per-pixel ETC arrays before order selection.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from astropy.io import fits
from exoplore.atmosphere.helium_pwinds import contact_phases
from exoplore.atmosphere.helium_provider import solve_outflow, HeliumTransit
from exoplore.core.helium_etc import require_helium_orders
from exoplore.core.helium_simulation import observing_arrays, _write_npz, plot_simulation, NoHeliumTransit
from exoplore.core.helium_synthetic_observation import utc_midpoints
from exoplore.instruments import load_instrument_v2, get_WaveGrid_v2
from exoplore.instruments.carmenes_helium_molecfit import fit_carmenes_night
from exoplore.observation.noise import photon_noise
from exoplore.pipelines.carmenes_helium import prepare_carmenes_transmission
from exoplore.pipelines.czesla2024 import timed_exposure_selection
from exoplore.pipelines.helium_math import window_mask
from exoplore.planets import load_planet


def validate_carmenes_helium(config):
    """Reject conflicting selections before physical calculations or outputs."""
    if config.instrument.name != 'CARMENES_NIR' or config.pipeline.carmenes_helium is None:
        raise ValueError('carmenes_helium requires CARMENES_NIR and explicit preparation settings')
    h = config.atmosphere.helium
    if h is None or config.observation.use_real_data or config.observation.n_nights < 1 or config.observation.event_type != 'transit':
        raise ValueError('Select synthetic transits with explicit p-winds settings')
    if config.pipeline.sysrem_iterations != 0 or config.retrieval.enabled or config.atmosphere.limb_asymmetries:
        raise ValueError('The direct spherical-helium route uses no SYSREM or molecular retrieval')
    if h.observing_grid_mode != 'instrument' or h.telluric_correction != 'molecfit':
        raise ValueError('This CARMENES route requires its native ETC grid and molecfit correction')
    if config.observation.different_nights:
        raise ValueError('Different epochs/BERV/cadences are not connected to the helium multi-night route yet')
    if h.completed_night_paths and config.observation.n_nights == 1:
        raise ValueError('Completed-night reuse belongs to a multi-night helium run')
    if h.molecfit_reuse_path:
        raise ValueError('CARMENES fit reuse is not implemented; retain prior run directories')
    if not config.planet.parameter_file or config.observation.exposure_time_seconds <= 0:
        raise ValueError('Supply the planet file and a positive exposure duration')


def load_carmenes_order(config):
    """Use the existing instrument loader, then select from the full ETC arrays."""
    instrument = load_instrument_v2(config)
    wave, pixels, _, snr, *_ = get_WaveGrid_v2(config, instrument, instrument.n_orders_total)
    wave = np.asarray(wave, float).T*1000  # Existing CARMENES wave_star convention.
    snr = np.asarray(snr, float)
    science = config.pipeline.carmenes_helium
    selected = config.instrument.order_indices
    require_helium_orders(wave, selected, science.helium_vacuum_lines_nm)
    if len(selected) != 1:
        raise ValueError('Select one helium order for the current spherical simulation')
    if snr.shape != wave.shape:
        raise ValueError('Full-channel ETC S/N must match the native orders-by-pixels grid')
    with fits.open(instrument.snr_file) as hdul:
        if hdul[0].header.get('SNRTYPE') != 'PER SPECTRAL PIXEL':
            raise ValueError('Convert ETC resel S/N before supplying the CARMENES input')
        if hdul[0].header['DIT']*hdul[0].header['NDIT'] != config.observation.exposure_time_seconds:
            raise ValueError('ETC exposure time differs from simulation')
        if not np.allclose(config.atmosphere.helium.airmass, hdul[0].header['AIRMASS']):
            raise ValueError('This first fixed-condition run must use the ETC airmass')
    index = selected[0]
    if not np.isfinite(snr[index]).all() or np.any(snr[index] <= 0):
        raise ValueError('Helium order has unavailable ETC noise estimates')
    if pixels != config.atmosphere.helium.observing_pixels:
        raise ValueError('Helium observing_pixels differs from the native grid')
    print(f'  Full CARMENES grid: {wave.shape}; selected order {index}; native R={instrument.res:.0f}',flush=True)
    return dict(wave_nm=wave[index],snr=snr[index],source_files=[instrument.wave_file,instrument.snr_file])


def observing_carmenes_arrays(settings, science, planet, model, duration, wave):
    """Integrate actual detector pixels, with no artificial pixels across gaps."""
    gaps = np.flatnonzero(np.diff(wave)>5*np.median(np.diff(wave)))
    sections = np.split(np.arange(len(wave)), gaps+1)
    cache = {}
    class CachedTransit:
        wave_nm = model.wave_nm
        def spectrum(self, phase):
            if phase not in cache: cache[phase] = model.spectrum(phase)
            return cache[phase]
    parts = [observing_arrays(settings,science,planet,CachedTransit(),duration,wave[p]) for p in sections]
    return {key:(parts[0][key] if key=='opaque_continuum' else np.concatenate([part[key] for part in parts],axis=1)) for key in parts[0]}


def run_carmenes_helium(config, *, expectation_source: Path | None = None):
    """Run selected helium nights, preserving independent correction references.

    expectation_source reuses only noiseless physical arrays for a new night;
    that night gets one fresh observation-noise draw from its recorded seed.
    """
    validate_carmenes_helium(config)
    if config.observation.n_nights > 1:
        from exoplore.core.helium_multi_night import run_carmenes_multi_night
        return run_carmenes_multi_night(config)
    h, science = config.atmosphere.helium, config.pipeline.carmenes_helium
    order = load_carmenes_order(config)
    planet = load_planet(config.planet.parameter_file)
    if planet.eccentricity != 0: raise ValueError('This geometry requires a circular orbit')
    if not np.isclose(science.kp_kms,planet.kp_kms) or not np.isclose(science.period_days,planet.orbital_period_days):
        raise ValueError('Preparation orbit differs from the injected orbit')
    if not np.allclose(science.optical_contact_phases, contact_phases(planet),atol=1e-7,rtol=0):
        raise ValueError('Preparation contacts differ from the injected transit')
    phases = np.asarray(h.phase_midpoints)
    science,_,_,_=timed_exposure_selection(science,
        science.t0_bjd_tdb+phases*science.period_days,config.observation.exposure_time_seconds)
    half = config.observation.exposure_time_seconds/(2*science.period_days*86400)
    if np.any(np.diff(phases)<2*half):raise ValueError('Synthetic exposures overlap')
    contacts = np.asarray(science.optical_contact_phases)
    refs = np.asarray(science._reference_indices, dtype=int)
    inside = np.asarray(science._full_transit_indices, dtype=int)
    if np.any((phases[refs]+half>contacts[0])&(phases[refs]-half<contacts[3])):raise ValueError('Reference exposure overlaps optical transit')
    if np.any(phases[inside]-half<contacts[1]) or np.any(phases[inside]+half>contacts[2]):raise ValueError('Coadd exposure extends beyond T2--T3')
    output = Path(config.paths.output_root)/config.planet.name/('helium_sunbather' if h.backend == 'sunbather' else 'helium_pwinds')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'run_config.json').open('x') as f:json.dump(asdict(config),f,indent=2)
    if h.resume_simulation_path:
        source = Path(h.resume_simulation_path)
        original = json.loads((source/'run_config.json').read_text())
        current = asdict(config)
        for item in (original,current):
            if item['atmosphere']['helium'].get('sunbather') is None:
                item['atmosphere']['helium'].pop('sunbather', None)
            if item['atmosphere']['helium'].get('completed_night_paths') is None:
                item['atmosphere']['helium'].pop('completed_night_paths', None)
            item['atmosphere']['helium']['resume_simulation_path'] = ''
            # Correction settings do not change an already saved observation.
            item['atmosphere']['helium']['molecfit_config_path'] = ''
            item['paths']['output_root'] = ''
        if original != current:
            raise ValueError('Resume must preserve the atmosphere, cadence, ETC, seed and preparation configuration')
        with np.load(source/'simulated_observations.npz',allow_pickle=False) as saved:
            arrays = {k:saved[k].copy() for k in ('wave_nm','expected_flux','no_helium_flux','telluric_transmission','opaque_continuum')}
            raw,raw_error = saved['raw_flux'].copy(),saved['raw_error'].copy()
        with np.load(source/'outflow.npz',allow_pickle=False) as saved:
            profile = {k:saved[k].copy() for k in saved.files}
        from importlib.metadata import version
        metadata = source/'outflow_metadata.json'
        profile.update(json.loads(metadata.read_text()) if metadata.exists() else dict(p_winds_version=version('p-winds'),warnings=['Original solver warning list was not saved before the adapter failure']))
        expected,control,tell = (arrays[k] for k in ('expected_flux','no_helium_flux','telluric_transmission'))
        print('  Reusing the identical saved 45-exposure matrix and its original noise realization.',flush=True)
    elif expectation_source is not None:
        from exoplore.core.helium_multi_night import read_saved_expectation
        arrays, profile, raw_error = read_saved_expectation(expectation_source, config)
        expected, control, tell = (arrays[k] for k in ('expected_flux', 'no_helium_flux', 'telluric_transmission'))
        rng = np.random.default_rng(h.noise_seed)
        raw = expected + photon_noise(1/raw_error, rng=rng) if h.noise_mode == 'gaussian' else expected.copy()
        print(f'  New independent observation-noise realization, seed {h.noise_seed}; noiseless expectation reused.', flush=True)
    else:
        print(f'  Solving spherical H/He with {h.backend}...',flush=True)
        if config.observation.simulate_planet:
            profile = solve_outflow(h,planet)
            model = HeliumTransit(h,planet,science,profile)
            if h.backend == 'sunbather':
                provider_spectra = [model.spectrum(float(phase)) for phase in h.phase_midpoints]
                _write_npz(output/'planet_model.npz',dict(syn_wave_nm=model.wave_nm,
                    syn_spec=np.asarray([item[0] for item in provider_spectra]),
                    opaque_continuum=np.asarray([item[1] for item in provider_spectra]),
                    phase=np.asarray(h.phase_midpoints),frame=np.asarray('planetary_rest_vacuum_nm')))
        else:
            profile = dict(p_winds_version=None,warnings=[],injected_helium=np.array(False))
            model = NoHeliumTransit(h,planet)
        arrays = observing_carmenes_arrays(h,science,planet,model,config.observation.exposure_time_seconds,order['wave_nm'])
        expected,control,tell = (arrays[k] for k in ('expected_flux','no_helium_flux','telluric_transmission'))
        # ETC includes its star/telluric/throughput and detector noise. Preserve
        # its wavelength dependence; do not apply a second blaze/telluric factor.
        # Here normalized-flux uncertainty is specified directly by the ETC S/N.
        baseline = control/arrays['opaque_continuum'][:,None]
        raw_error = baseline/order['snr'][None,:]
        rng = np.random.default_rng(h.noise_seed)
        raw = expected+photon_noise(1/raw_error,rng=rng) if h.noise_mode=='gaussian' else expected.copy()
    berv = np.asarray(h.berv_kms);rv=science.kp_kms*np.sin(2*np.pi*phases)
    bjd = science.t0_bjd_tdb+phases*science.period_days
    _write_npz(output/'simulated_observations.npz',dict(**arrays,raw_flux=raw,raw_error=raw_error,phase=phases,berv_kms=berv,planet_rv_kms=rv,bjd_tdb=bjd))
    _write_npz(output/'outflow.npz',{k:v for k,v in profile.items() if k not in ('warnings','p_winds_version')})
    with (output/'outflow_metadata.json').open('x') as f:json.dump(dict(p_winds_version=profile['p_winds_version'],warnings=profile['warnings']),f,indent=2)
    times = utc_midpoints(bjd,science)
    print('  Fitting the complete helium order with molecfit, independently per detector...',flush=True)
    night = fit_carmenes_night(output,arrays,raw,raw_error,h,science,times.mjd,config.observation.exposure_time_seconds)
    good = night['transmittance'] >= science.telluric_min_transmission
    if science.oh_mode=='mask':good &= ~window_mask(arrays['wave_nm'],science.oh_topocentric_windows_nm)
    corrected,error = np.where(good,night['flux'],np.nan),np.where(good,night['error'],np.nan)
    recovered = prepare_carmenes_transmission(night['wave_nm'],corrected,error,berv,rv,science)
    true_good = tell>=science.telluric_min_transmission
    truth = prepare_carmenes_transmission(arrays['wave_nm'],np.where(true_good,expected/tell,np.nan),np.where(true_good,raw_error/tell,np.nan),berv,rv,science)
    coadd_error = np.sqrt(recovered['conditional_coadd_variance'])
    from exoplore.pipelines.helium_noise_propagation import carmenes_lightcurve_covariance
    curve_covariance = carmenes_lightcurve_covariance(
        night['wave_nm'], corrected, error, berv, rv, science, recovered)
    curve_error = np.sqrt(np.diag(curve_covariance))
    _write_npz(output/'observations.npz',dict(**arrays,raw_flux=raw,raw_error=raw_error,corrected_flux=corrected,corrected_error=error,phase=phases,berv_kms=berv,planet_rv_kms=rv,bjd_tdb=bjd))
    _write_npz(output/'transmission.npz',dict(**recovered,coadd_error=coadd_error,lightcurve_error=curve_error,lightcurve_covariance=curve_covariance,truth_planet_wave_nm=truth['planet_wave_nm'],truth_planet_coadd=truth['planet_coadd'],truth_planet_lightcurve=truth['planet_lightcurve']))
    _write_npz(output/'molecfit_calibration.npz',dict(wavelength_shift_kms=night['shift'],refined_wave_nm=night['wave_nm']))
    def ew(result):
        band=window_mask(result['planet_wave_nm'],[science.equivalent_width_window_nm])
        return float(np.trapz(1-result['planet_coadd'][band],result['planet_wave_nm'][band])*1e4)
    sources=order['source_files']+[str(config.planet.parameter_file),h.irradiation_spectrum_path,h.stellar_template_path,h.telluric_template_path,h.molecfit_config_path,str(Path(config.paths.inputs_dir)/'input_provenance.json')]
    if h.resume_simulation_path:
        sources.extend([str(Path(h.resume_simulation_path)/name) for name in ('run_config.json','simulated_observations.npz','outflow.npz')])
    if expectation_source is not None:
        sources.extend([str(expectation_source/name) for name in ('run_config.json','simulated_observations.npz','outflow.npz')])
    summary=dict(reused_noiseless_expectation_path=str(expectation_source) if expectation_source else None, noise_seed=h.noise_seed, reused_simulation_path=h.resume_simulation_path or None,pipeline='carmenes_helium',scientific_reference='https://doi.org/10.1051/0004-6361/202037719',p_winds_version=profile['p_winds_version'],solver_warnings=profile['warnings'],source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},normalization='Native mean over 1081.5962--1082.7624 nm before stellar alignment',reference='Arithmetic mean of optical out-of-transit spectra',coaddition='Arithmetic mean of exposures entirely within T2--T3',noise_realizations=1 if h.noise_mode=='gaussian' else 0,uncertainty_resamples=0,uncertainty_scope='Spectral coadd errors are reference-fixed marginal errors. Light-curve errors retain native interpolation and shared-reference covariance, conditional on fitted continua and molecfit parameters; not a detection significance',noise_convention='Full ETC resel curve converted using local native sampling; normalized flux error = baseline_flux / ETC_pixel_SNR; no second blaze, telluric or detector-noise factor',oh_emission='Not simulated; no OH subtraction claimed',optical_gate='None; atmospheric overlap outside optical contacts retained',irradiation='GJ674 coronal/transition-region SED times 2.2/a_AU^2; no photospheric UV',stellar_assumptions='Static PHOENIX 3700K/logg5.0/FeH0, uniform disc, no stellar helium variability/RM/CLV/rotation',truth_equivalent_width_mA=ew(truth) if np.isfinite(ew(truth)) else None,recovered_equivalent_width_mA=ew(recovered) if np.isfinite(ew(recovered)) else None)
    summary['physical_backend'] = h.backend
    summary['irradiation'] = h.irradiation_source
    if h.backend == 'sunbather':
        summary['sunbather_version'] = profile.get('sunbather_version')
        summary['temperature_convention'] = 'temperature_K is Parker T0; Cloudy T(r) used for populations and broadening'
        summary['physical_provider_provenance'] = str(Path(h.sunbather['project_path'])/'provider_provenance.json')
    with (output/'provenance.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    plot_simulation(output,arrays,recovered,truth,phases,rv,coadd_error,curve_error,science)
    print(f'  Completed one CARMENES night: {output}\n  Prepared injected EW={ew(truth):.3f} mA; recovered EW={ew(recovered):.3f} mA',flush=True)
    return output
