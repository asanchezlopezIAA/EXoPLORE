"""Observed helium simulations with an optional p-winds atmosphere.

This opt-in branch generates one order, integrates finite exposures, convolves
flux with a Gaussian instrumental profile, samples detector pixels, and runs
the existing direct transmission preparation. Synthetic truth is kept apart
from real ESO calibration products. Files are written only to a new directory.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from exoplore.atmosphere.helium_pwinds import contact_phases, geometry
from exoplore.atmosphere.helium_provider import HeliumTransit, solve_outflow
from exoplore.pipelines.czesla2024 import exposure_indices, prepare_transmission, timed_exposure_selection
from exoplore.pipelines.helium_math import stellar_wavelength, window_mask
from exoplore.planets import load_planet
from exoplore.instruments import load_instrument_v2, get_WaveGrid_v2
from exoplore.observation.noise import photon_noise
from exoplore.core.helium_etc import require_helium_orders

C_KMS = 299792.458


def validate_helium_run(config) -> None:
    """Validate the optional simulation without importing p-winds or loading data."""
    if config.pipeline.name == "carmenes_helium":
        from exoplore.core.carmenes_helium_simulation import validate_carmenes_helium
        validate_carmenes_helium(config)
        return
    from exoplore.core.spectroscopy_mode import spectroscopy_route
    if spectroscopy_route(config) != 'helium_simulation':
        raise ValueError('The observation controls do not select a helium simulation')
    o, p, h = config.observation, config.pipeline, config.atmosphere.helium
    if h is None or p.name != "czesla2024" or p.czesla2024 is None:
        raise ValueError("Helium simulations require explicit helium and czesla2024 settings")
    if h.completed_night_paths:
        raise ValueError("Completed-night combination is currently connected only to carmenes_helium")
    if o.use_real_data or o.event_type != "transit" or o.n_nights != 1:
        raise ValueError("Select one synthetic transit with use_real_data=false")
    if p.sysrem_iterations != 0 or config.retrieval.enabled or config.atmosphere.limb_asymmetries:
        raise ValueError("The spherical direct-helium simulation uses zero SYSREM and no molecular retrieval")
    if config.instrument.name != "CRIRES+" or not config.planet.parameter_file:
        raise ValueError("Supply CRIRES+ and an explicit planet parameter file")
    if not np.isfinite(o.exposure_time_seconds) or o.exposure_time_seconds <= 0:
        raise ValueError("Exposure duration must be positive")
    science,_,_,_=timed_exposure_selection(p.czesla2024,
        p.czesla2024.t0_bjd_tdb+np.asarray(h.phase_midpoints)*p.czesla2024.period_days,
        o.exposure_time_seconds)
    if h.telluric_correction == "molecfit" and science.monte_carlo_draws != 0:
        raise ValueError("This synthetic molecfit run uses one noise realization and zero uncertainty resamples")
    if h.uncertainty_mode == 'conditional' and science.monte_carlo_draws != 0:
        raise ValueError('Set monte_carlo_draws=0 for a simulation without noise resampling')
    if h.uncertainty_mode == 'monte_carlo' and science.monte_carlo_draws < 100:
        raise ValueError('Monte Carlo propagation requires at least 100 draws')
    refs = np.asarray(science._reference_indices, dtype=int)
    inside = np.asarray(science._full_transit_indices, dtype=int)
    if np.intersect1d(refs, inside).size:
        raise ValueError("Reference and in-transit exposure selections overlap")


def detector_sample(flux: np.ndarray, log_step: float, resolving_power: float, oversampling: int) -> np.ndarray:
    """Convolve flux on a logarithmic grid and average equal log-width subpixels."""
    sigma = 1/(resolving_power*2*np.sqrt(2*np.log(2))*log_step)
    broadened = gaussian_filter1d(flux, sigma, mode="nearest")
    return broadened.reshape(-1, oversampling).mean(axis=1)


def _template(path: str, key: str, query: np.ndarray) -> np.ndarray:
    """Read a vacuum-nm template, refusing extrapolation and malformed inputs."""
    with np.load(path, allow_pickle=False) as archive:
        wave, values = archive["wave_nm"], archive[key]
    if wave.ndim != 1 or values.shape != wave.shape or len(wave) < 2:
        raise ValueError("Template must contain matching one-dimensional wavelength/value arrays")
    if not np.isfinite(wave).all() or not np.isfinite(values).all() or np.any(np.diff(wave) <= 0):
        raise ValueError("Invalid template wavelength grid or values")
    if query.min() < wave[0] or query.max() > wave[-1]:
        raise ValueError("Template does not cover the simulated order")
    if np.any(values < 0) or (key == "transmission" and np.any(values > 1)):
        raise ValueError("Template values are outside their physical range")
    return np.interp(query, wave, values)


def observing_arrays(settings, science, planet, model, exposure_seconds: float,
                     native_wave_nm: np.ndarray | None = None) -> dict:
    """Generate noiseless detector fluxes with and without helium.

    Atmospheric overlap is evaluated at each quadrature sample, including
    before first and after fourth optical contact, before exposure integration.
    BERV is held at the supplied exposure midpoint; the planet velocity varies
    within the exposure. A known telluric model is an ideal simulation input,
    not a molecfit result. Sky emission is absent in this first experiment.
    """
    h = settings
    n_fine = h.observing_pixels*h.instrumental_oversampling
    if native_wave_nm is None:
        edges = np.linspace(np.log(h.observing_wavelength_range_nm[0]),
                            np.log(h.observing_wavelength_range_nm[1]), n_fine+1)
    else:
        centres = np.log(native_wave_nm)
        native_edges = np.r_[centres[0]-(centres[1]-centres[0])/2,
                             (centres[:-1]+centres[1:])/2,
                             centres[-1]+(centres[-1]-centres[-2])/2]
        # Padding avoids convolution-edge artefacts in the native end pixels.
        pad = 6/h.instrumental_resolving_power
        step = np.min(np.diff(centres))/h.instrumental_oversampling
        edges = np.linspace(native_edges[0]-pad, native_edges[-1]+pad,
                            int(np.ceil((native_edges[-1]-native_edges[0]+2*pad)/step))+1)
    fine = np.exp((edges[1:]+edges[:-1])/2)
    wave = (native_wave_nm if native_wave_nm is not None else
            np.exp((edges[::h.instrumental_oversampling][1:]+edges[::h.instrumental_oversampling][:-1])/2))
    def sample(row):
        if native_wave_nm is None:
            return detector_sample(row, edges[1]-edges[0], h.instrumental_resolving_power, h.instrumental_oversampling)
        from scipy.integrate import cumulative_trapezoid
        sigma = 1/(h.instrumental_resolving_power*2*np.sqrt(2*np.log(2))*(edges[1]-edges[0]))
        blurred = gaussian_filter1d(row, sigma, mode='nearest')
        integral = cumulative_trapezoid(blurred, fine, initial=0)
        pixel_edges = np.exp(native_edges)
        return np.diff(np.interp(pixel_edges, fine, integral))/np.diff(pixel_edges)
    nodes, weights = np.polynomial.legendre.leggauss(h.exposure_quadrature_points)
    offsets = nodes*exposure_seconds/(2*science.period_days*86400)
    rows, controls, transmissions, opaque_means = [], [], [], []
    for frame, (phase, berv) in enumerate(zip(h.phase_midpoints, h.berv_kms)):
        starwave = stellar_wavelength(fine, berv, science.gamma_kms)
        star = np.ones_like(fine) if h.stellar_mode == "flat" else _template(h.stellar_template_path, "flux", starwave)
        tell = np.ones_like(fine) if h.telluric_mode == "none" else _template(h.telluric_template_path, "transmission", fine)
        if h.airmass is not None:
            tell = tell**h.airmass[frame]
        total, control, opaque_mean = np.zeros_like(fine), np.zeros_like(fine), 0.0
        for offset, weight in zip(offsets, weights/2):
            node_phase = phase+offset
            helium, opaque = model.spectrum(node_phase)
            rv = science.kp_kms*np.sin(2*np.pi*node_phase)
            planetwave = starwave/(1+rv/C_KMS)
            absorption = np.interp(planetwave, model.wave_nm, helium, left=1, right=1)
            total += weight*star*opaque*absorption*tell
            control += weight*star*opaque*tell
            opaque_mean += weight*opaque
        rows.append(sample(total)); controls.append(sample(control)); transmissions.append(sample(tell))
        opaque_means.append(opaque_mean)
        if (frame+1) % 20 == 0:
            print(f'  Integrated {frame+1}/{len(h.phase_midpoints)} exposures', flush=True)
    return dict(wave_nm=np.tile(wave, (len(rows), 1)), expected_flux=np.asarray(rows),
                no_helium_flux=np.asarray(controls), telluric_transmission=np.asarray(transmissions),
                opaque_continuum=np.asarray(opaque_means))


def load_observing_order(config) -> dict:
    """Use EXoPLORE's CRIRES+ loader and select one native ETC segment.

    ETC S/N is already per spectral pixel. TARGET and VARFIXED preserve the
    electron-count noise budget; readout and sky noise are not added twice.
    """
    from astropy.io import fits
    instrument = load_instrument_v2(config)
    wave_star, n_pixels, _, snr, *_ = get_WaveGrid_v2(config, instrument, instrument.n_orders_total)
    selected = config.instrument.order_indices
    require_helium_orders(np.asarray(wave_star, float)*1000, selected,
                          config.pipeline.czesla2024.helium_vacuum_lines_nm)
    if len(selected) != 1 or not 0 <= selected[0] < len(wave_star):
        raise ValueError('Select exactly one valid instrument order segment')
    index = selected[0]
    with fits.open(instrument.wave_file) as archive:
        if archive[0].header.get('BUNIT') != 'um' or archive[0].header.get('WFRAME') != 'TOPOCENTRIC VACUUM':
            raise ValueError('ETC FITS must explicitly declare topocentric vacuum micrometres')
        identity = archive['SEGMENTS'].data[index]
        if str(identity['DETECTOR']).strip() != 'detector1' or int(identity['ORDER']) != 52:
            raise ValueError('This observing configuration requires detector 1, physical order 52')
    with fits.open(instrument.snr_file) as archive:
        if archive[0].header.get('SNRTYPE') != 'PER SPECTRAL PIXEL':
            raise ValueError('Specify the native ETC S/N convention')
        if archive[0].header['DIT']*archive[0].header['NDIT'] != config.observation.exposure_time_seconds:
            raise ValueError('ETC integration time differs from the simulated exposure')
        counts = np.asarray(archive['TARGET'].data[index], float)
        fixed = np.asarray(archive['VARFIXED'].data[index], float)
    wave = np.asarray(wave_star[index], float)*1000
    snr = np.asarray(snr[index], float)
    if len(wave) != config.atmosphere.helium.observing_pixels or np.any(np.diff(wave) <= 0):
        raise ValueError('Native pixel count or wavelength order disagrees with helium settings')
    if not np.isfinite(snr).all() or np.any(snr <= 0):
        raise ValueError('ETC segment contains invalid S/N')
    print(f'  EXoPLORE instrument grid: {len(wave_star)} segments; selected {index}, {n_pixels} pixels', flush=True)
    return dict(wave_nm=wave, snr=snr, target=counts, variance_fixed=fixed,
                source_files=[instrument.wave_file, instrument.snr_file])


def _write_npz(path: Path, arrays: dict) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


class NoHeliumTransit:
    """Opaque transit control without importing or executing p-winds."""

    p_winds_air_lines_m: list[float] = []

    def __init__(self, settings, planet):
        self.settings, self.planet = settings, planet
        self.wave_nm = np.linspace(*settings.model_wavelength_range_nm, settings.model_wavelength_points)

    def spectrum(self, phase: float) -> tuple[np.ndarray, float]:
        """Integrate the occulted stellar intensity in concentric annuli."""
        from scipy.integrate import quad
        x, y, p = geometry(phase, self.planet)
        d = float(np.hypot(x, y))
        if d >= 1+p:
            return np.ones_like(self.wave_nm), 1.
        coefficients = self.settings.limb_darkening_coefficients
        u1 = coefficients[0] if coefficients else 0.
        u2 = coefficients[1] if len(coefficients) == 2 else 0.
        def blocked(radius):
            mu = np.sqrt(max(0., 1-radius*radius))
            intensity = 1-u1*(1-mu)-u2*(1-mu)**2
            if d == 0:
                angle = np.pi if radius < p else 0.
            elif radius == 0:
                return 0.
            else:
                angle = np.arccos(np.clip((radius*radius+d*d-p*p)/(2*radius*d), -1, 1))
            return 2*radius*angle*intensity
        points = [r for r in (abs(d-p), d+p) if 0 < r < 1]
        absorbed = quad(blocked, 0, 1, points=points, epsabs=1e-10)[0]
        return np.ones_like(self.wave_nm), 1-absorbed/(np.pi*(1-u1/3-u2/6))


def run_helium_simulation(config) -> Path:
    """Automatically calculate p-winds helium, simulate exposures and recover it."""
    validate_helium_run(config)
    if config.pipeline.name == "carmenes_helium":
        from exoplore.core.carmenes_helium_simulation import run_carmenes_helium
        return run_carmenes_helium(config)
    h, science = config.atmosphere.helium, config.pipeline.czesla2024
    science,_,_,_=timed_exposure_selection(science,
        science.t0_bjd_tdb+np.asarray(h.phase_midpoints)*science.period_days,
        config.observation.exposure_time_seconds)
    output = Path(config.paths.output_root)/config.planet.name/('helium_sunbather' if config.atmosphere.helium.backend == 'sunbather' else 'helium_pwinds')
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new output_root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    with (output/"run_config.json").open("x") as stream:
        json.dump(asdict(config), stream, indent=2, allow_nan=False)
    planet = load_planet(config.planet.parameter_file)
    if planet.eccentricity != 0:
        raise ValueError("The first p-winds observing adapter supports circular orbits")
    if not np.isclose(science.period_days, planet.orbital_period_days, rtol=1e-8, atol=0):
        raise ValueError("Planet and analysis periods disagree")
    if not np.isclose(science.kp_kms, planet.kp_kms, rtol=1e-6, atol=0):
        raise ValueError("Planet and analysis velocity amplitudes disagree")
    contacts = contact_phases(planet)
    if not np.allclose(science.optical_contact_phases, contacts, atol=1e-7, rtol=0):
        raise ValueError("Configured optical contacts disagree with the simulated geometry")
    phases = np.asarray(h.phase_midpoints)
    half = config.observation.exposure_time_seconds/(2*science.period_days*86400)
    if np.any(np.diff(phases) < 2*half):
        raise ValueError("Synthetic exposures overlap in time")
    refs = np.asarray(science._reference_indices, dtype=int)
    inside = np.asarray(science._full_transit_indices, dtype=int)
    if np.any((phases[refs]+half > contacts[0]) & (phases[refs]-half < contacts[3])):
        raise ValueError("A reference exposure overlaps the optical transit")
    if np.any(phases[inside]-half < contacts[1]) or np.any(phases[inside]+half > contacts[2]):
        raise ValueError("Selected coadd exposures must lie completely between contacts 2 and 3")
    instrument_order = load_observing_order(config) if h.observing_grid_mode == 'instrument' else None
    if h.resume_simulation_path:
        from importlib.metadata import version
        from types import SimpleNamespace
        from p_winds import lines
        source = Path(h.resume_simulation_path)
        original = json.loads((source/'run_config.json').read_text())
        current = asdict(config)
        old_h, new_h = original['atmosphere']['helium'].copy(), current['atmosphere']['helium'].copy()
        old_h.pop('resume_simulation_path', None); new_h.pop('resume_simulation_path', None)
        for fields in (old_h, new_h):
            if fields.get('sunbather') is None:
                fields.pop('sunbather', None)
            if fields.get('completed_night_paths') is None:
                fields.pop('completed_night_paths', None)
        original['pipeline'].setdefault('carmenes_helium', None)
        # Refitting changes only the recovery recipe, never the saved observation.
        old_h.pop('molecfit_config_path', None); new_h.pop('molecfit_config_path', None)
        for key in ('molecfit_workers', 'molecfit_reuse_path'):
            old_h.pop(key, None); new_h.pop(key, None)
        if old_h != new_h or original['pipeline'] != current['pipeline'] or original['planet'] != current['planet'] or original['instrument'] != current['instrument'] or original['observation'] != current['observation']:
            raise ValueError('Resume must use the exact original atmosphere, exposures, instrument and preparation')
        with np.load(source/'simulated_observations.npz', allow_pickle=False) as saved:
            arrays = {k:saved[k].copy() for k in ('wave_nm', 'expected_flux', 'no_helium_flux', 'telluric_transmission', 'opaque_continuum')}
            raw, raw_error = saved['raw_flux'].copy(), saved['raw_error'].copy()
        with np.load(source/'outflow.npz', allow_pickle=False) as saved:
            profile = {k:saved[k].copy() for k in saved.files}
        profile.update(p_winds_version=version('p-winds'), warnings=None)
        model = SimpleNamespace(p_winds_air_lines_m=list(lines.he_3_properties()[:3]))
        expected, control, tell = arrays['expected_flux'], arrays['no_helium_flux'], arrays['telluric_transmission']
        print('  Reusing the saved complete matrix and its original noise; no new physical model or noise draw.', flush=True)
    else:
        if config.observation.simulate_planet:
            print(f"  Solving spherical hydrogen and helium with {h.backend}...", flush=True)
            profile = solve_outflow(h, planet)
            model = HeliumTransit(h, planet, science, profile)
        else:
            print("  Generating the control without injected helium; p-winds is not called.", flush=True)
            profile = dict(p_winds_version=None, warnings=[], injected_helium=np.array(False))
            model = NoHeliumTransit(h, planet)
        print("  Integrating exposures and applying the instrumental profile...", flush=True)
        arrays = observing_arrays(h, science, planet, model, config.observation.exposure_time_seconds,
                                  None if instrument_order is None else instrument_order['wave_nm'])
        expected, control, tell = arrays["expected_flux"], arrays["no_helium_flux"], arrays["telluric_transmission"]
        if instrument_order is None:
            raw_error = np.full_like(expected, 1/h.snr_per_pixel)
        else:
            # Scale only target shot noise with the transit flux. The ETC already
            # includes sky, detector readout and dark variance in extracted units.
            counts = instrument_order['target'][None, :]
            # Map the ETC target counts to the injected out-of-transit spectrum.
            # Thus its fractional baseline uncertainty is exactly 1/(ETC S/N),
            # including the ETC's existing stellar and atmospheric attenuation.
            baseline = control/arrays['opaque_continuum'][:, None]
            gain = counts/baseline
            raw_error = np.sqrt(gain*expected+instrument_order['variance_fixed'][None, :])/gain
        rng = np.random.default_rng(h.noise_seed)
        raw = expected+photon_noise(1/raw_error, rng=rng) if h.noise_mode == "gaussian" else expected.copy()
    divisor = tell if h.telluric_correction == "known_model" else np.ones_like(tell)
    valid = tell >= science.telluric_min_transmission
    if science.oh_mode == "mask":
        valid &= ~window_mask(arrays["wave_nm"], science.oh_topocentric_windows_nm)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.where(valid, raw_error/divisor, np.nan)
        corrected = np.where(valid, raw/divisor, np.nan)
        corrected_expected = np.where(valid, expected/divisor, np.nan)
        corrected_control = np.where(valid, control/divisor, np.nan)
    rv = science.kp_kms*np.sin(2*np.pi*phases)
    berv = np.asarray(h.berv_kms)
    def prepare(data):
        return prepare_transmission(arrays["wave_nm"], data, error, berv, rv, science)
    _write_npz(output/'simulated_observations.npz', dict(**arrays, raw_flux=raw,
               raw_error=raw_error, phase=phases, berv_kms=berv, planet_rv_kms=rv,
               bjd_tdb=science.t0_bjd_tdb+phases*science.period_days))
    _write_npz(output/'outflow.npz', {k:v for k,v in profile.items() if k not in ('warnings', 'p_winds_version')})
    if h.telluric_correction == 'molecfit':
        from exoplore.core.helium_synthetic_observation import fit_synthetic_night
        print('  Fitting the entire synthetic matrix with molecfit...', flush=True)
        night = fit_synthetic_night(output, arrays, raw, raw_error, h, science,
                                    config.observation.exposure_time_seconds)
        corrected, error = night['flux'], night['error']
        recovered = prepare_transmission(night['wave_nm'], corrected, error, berv, rv, science)
        _write_npz(output/'molecfit_calibration.npz', dict(wavelength_shift_kms=night['wavelength_shift_kms'],
                   refined_wave_nm=night['wave_nm']))
        # Injected transmission is used only to display the known noiseless truth.
        truth_error = np.where(valid, raw_error/tell, np.nan)
        truth = prepare_transmission(arrays['wave_nm'], np.where(valid, expected/tell, np.nan),
                                     truth_error, berv, rv, science)
        null = prepare_transmission(arrays['wave_nm'], np.where(valid, control/tell, np.nan),
                                    truth_error, berv, rv, science)
    else:
        recovered, truth, null = prepare(corrected), prepare(corrected_expected), prepare(corrected_control)
    draws = []
    curves = []
    mc_rng = np.random.default_rng(science.monte_carlo_seed)
    if science.monte_carlo_draws:
        print(f"  Propagating native noise through {science.monte_carlo_draws} preparations...", flush=True)
    else:
        print('  One noise realisation; no uncertainty resampling.', flush=True)
    for _ in range(science.monte_carlo_draws):
        trial = prepare(corrected+mc_rng.normal(size=corrected.shape)*error)
        draws.append(trial["planet_coadd"]); curves.append(trial["planet_lightcurve"])
    draws = np.asarray(draws) if draws else np.empty((0, expected.shape[1]))
    curves = np.asarray(curves) if curves else np.empty((0, len(phases)))
    # Incomplete edge coverage remains NaN; standard deviations exclude only absent coverage.
    coadd_error = np.full(draws.shape[1], np.nan)
    good = np.isfinite(draws).all(axis=0)
    if len(draws):
        coadd_error[good] = draws[:, good].std(axis=0, ddof=1)
    else:
        # These conditional weights hold the fitted continuum and noisy
        # reference fixed. They are not full inference uncertainties.
        coadd_error = np.sqrt(recovered['conditional_coadd_variance'])
    curve_error = np.full(curves.shape[1], np.nan)
    good_curve = np.isfinite(curves).all(axis=0)
    if len(curves):
        curve_error[good_curve] = curves[:, good_curve].std(axis=0, ddof=1)
    selected = window_mask(recovered["planet_wave_nm"], [science.equivalent_width_window_nm])
    ew_draws = np.trapz(1-draws[:, selected], recovered["planet_wave_nm"][selected], axis=1)*1e4
    truth_selected = window_mask(truth["planet_wave_nm"], [science.equivalent_width_window_nm])
    truth_ew = float(np.trapz(1-truth["planet_coadd"][truth_selected], truth["planet_wave_nm"][truth_selected])*1e4)
    recovered_ew = float(np.trapz(1-recovered["planet_coadd"][selected], recovered["planet_wave_nm"][selected])*1e4)
    _write_npz(output/"observations.npz", dict(**arrays, raw_flux=raw, raw_error=raw_error,
               corrected_flux=corrected, corrected_error=error, phase=phases, berv_kms=berv,
               planet_rv_kms=rv, bjd_tdb=science.t0_bjd_tdb+phases*science.period_days))
    _write_npz(output/"transmission.npz", dict(**recovered, coadd_error=coadd_error,
               lightcurve_error=curve_error, truth_planet_wave_nm=truth["planet_wave_nm"],
               truth_planet_coadd=truth["planet_coadd"],
               truth_planet_lightcurve=truth["planet_lightcurve"], null_planet_coadd=null["planet_coadd"],
               monte_carlo_coadds=draws, equivalent_width_draws_mA=ew_draws))
    sources = [config.planet.parameter_file, h.stellar_template_path, h.telluric_template_path,
               h.molecfit_config_path, config.paths.phoenix_wave_file, config.paths.phoenix_flux_file,
               config.tellurics.reference_telluric_file]
    if h.resume_simulation_path:
        sources.extend([str(Path(h.resume_simulation_path)/name) for name in ('simulated_observations.npz', 'outflow.npz', 'run_config.json')])
    if config.observation.simulate_planet:
        sources.append(h.irradiation_spectrum_path)
    if instrument_order is not None:
        sources.extend(instrument_order['source_files'])
    provenance = dict(config=asdict(config), planet=asdict(planet), p_winds_version=profile["p_winds_version"],
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources if p},
        reused_simulation_path=h.resume_simulation_path or None,
        solver_warnings=profile["warnings"], p_winds_default_air_lines_m=getattr(model, 'p_winds_air_lines_m', None),
        adopted_vacuum_lines_nm=science.helium_vacuum_lines_nm,
        injected_helium=config.observation.simulate_planet,
        physical_model=("Steady spherical isothermal Parker wind with tidal gravity; circular orbit"
                        if config.observation.simulate_planet else "Opaque transit without helium injection"),
        optical_gate="None; foreground atmospheric overlap retained outside optical contacts, within the supplied radial profile",
        irradiation_location="At planet; no automatic flux rescaling",
        telluric_correction=h.telluric_correction, sky_emission="Absent",
        uncertainty_scope=("Native independent Gaussian noise, normalization, shared reference and resampling; excludes atmospheric, stellar and telluric systematics"
                           if len(draws) else "Conditional marginal pixel weights only, holding reference and fitted continuum fixed; interpolation covariance and shared-reference uncertainty excluded. No EW uncertainty or significance inferred."),
        noise_convention="ETC per-pixel target counts and existing sky/read/dark budget mapped to the injected baseline; one Gaussian draw via EXoPLORE photon_noise",
        stellar_assumptions="Static PHOENIX photosphere shifted by gamma and BERV; no chromospheric helium, RM/CLV, or rotational broadening",
        airmass=h.airmass,
        noise_realizations=1 if h.noise_mode == 'gaussian' else 0,
        uncertainty_resamples=len(draws),
        observing_grid_mode=h.observing_grid_mode,
        solver_convergence=("p-winds relaxation tolerance and iteration cap; validate independently using refined grids and tolerances"
                            if config.observation.simulate_planet else "Not run: helium injection disabled"),
        truth_equivalent_width_mA=truth_ew if np.isfinite(truth_ew) else None,
        recovered_equivalent_width_mA=recovered_ew if np.isfinite(recovered_ew) else None,
        equivalent_width_error_mA=float(np.std(ew_draws, ddof=1)) if len(ew_draws) and np.isfinite(ew_draws).all() else None,
        equivalent_width_coverage="Complete fixed window required; masked gaps yield no EW estimate")
    if h.backend == 'sunbather':
        provenance.update(physical_backend='sunbather', physical_model='Parker wind plus Cloudy nonisothermal temperature and atomic populations; resolved wind radiative transfer',
            irradiation_location='Native Sunbather outer boundary a-rmax',
            solver_convergence='Sunbather heating/cooling and temperature convergence; not a guarantee of T0 consistency',
            physical_provider_provenance=str(Path(h.sunbather['project_path'])/'provider_provenance.json'))
    with (output/"provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
    plot_simulation(output, arrays, recovered, truth, phases, rv, coadd_error, curve_error, science)
    print(f"  Saved helium simulation: {output}\n  Injected/prepared EW: {truth_ew:.2f} mA; recovered: {recovered_ew:.2f} mA", flush=True)
    return output


def plot_simulation(output, arrays, recovered, truth, phases, rv, coadd_error, curve_error, science) -> None:
    """Save the moving signal, recovered spectrum and fixed-band transit curve."""
    if any((output/f'helium_recovery.{extension}').exists() for extension in ('png', 'pdf')):
        raise FileExistsError('Choose a new directory for the helium figures')
    import matplotlib
    matplotlib.use("Agg")
    significance = None
    choices = json.loads((output/'run_config.json').read_text())['atmosphere']['helium'].get('allart2023_significance')
    if choices is not None:
        from exoplore.pipelines.helium_significance import Allart2023SignificanceConfig, allart2023_significance
        significance = allart2023_significance(recovered['planet_wave_nm'], recovered['planet_coadd'],
                                              Allart2023SignificanceConfig(**choices))
        with (output/'allart2023_significance.json').open('x') as stream:
            json.dump(significance, stream, indent=2, allow_nan=False)
    with matplotlib.rc_context({'font.size': 11, 'axes.titlesize': 13,
                                'axes.labelsize': 11, 'legend.fontsize': 9,
                                'xtick.labelsize': 10, 'ytick.labelsize': 10}):
        _plot_simulation(output, arrays, recovered, truth, phases, rv, coadd_error, curve_error, science,
                         significance=significance)
    from exoplore.plotting.helium import write_helium_run_plots
    write_helium_run_plots(output, arrays, recovered, truth, phases, rv, science)
    if significance is not None:
        from exoplore.pipelines.helium_significance import plot_allan_noise
        plot_allan_noise(output/'plots', [significance], ['Single night'])


def _plot_simulation(output, arrays, recovered, truth, phases, rv, coadd_error, curve_error, science, *, significance=None) -> None:
    """Draw with local style settings, independently of the molecular simulator."""
    from exoplore.plotting.helium import plot_helium_summary
    display=dict(recovered,planet_coadd_error=coadd_error,planet_lightcurve_error=curve_error)
    berv=np.asarray(json.loads((output/'run_config.json').read_text())['atmosphere']['helium']['berv_kms'])
    for suffix in ('png','pdf'):
        plot_helium_summary(output/f'helium_recovery.{suffix}',display,phases,rv,berv,
                            science,significance=significance,synthetic=True)
