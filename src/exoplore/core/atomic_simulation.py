"""Generate and reduce a CARMENES or ANDES sodium observation.

ExoploreSimulator.run() selects this file when atomic_transmission.enabled
is true. The sequence is pRT atmosphere -> spatial stellar transit ->
terrestrial absorption -> instrument and pixels -> noise -> Molecfit ->
Seidel22 -> sodium spectra and light curves.

Spectral matrices have one exposure per row and wavelength per column.
Internal wavelength grids use micrometres. Preparation uses vacuum nm.
Velocities use km/s, radii use cm and exposure times use seconds.
"""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import numpy as np
from astropy.io import fits
from exoplore.planets import load_planet
from exoplore.instruments import load_instrument
from exoplore.observation import get_event
from exoplore.observation.carmenes_noise import configured_carmenes_etc
from exoplore.observation.sodium_noise import sodium_photon_uncertainty
from exoplore.observation.randomness import throughput_child_seed
from exoplore.observation.noise import add_throughput
from exoplore.observation.spectral_sampling import sample_native_pixels
from exoplore.pipelines.seidel22 import prepare_transmission


def validate_atomic_run(config):
    """Check the supported sodium experiment before creating outputs."""
    # Read the direct-sodium observing and reduction choices.
    a = config.atomic_transmission
    a.__post_init__()
    # Atmospheric temperatures, abundances and pressure limits come from this block.
    model = config.atmosphere.planet_model
    # Check the supported instrument and one-order observing setup.
    if config.instrument.name not in ('CARMENES_VIS', 'ANDES_UBV') or len(config.instrument.order_indices or []) != 1:
        raise ValueError('Direct sodium requires one CARMENES_VIS or ANDES_UBV order')
    if config.observation.use_real_data or config.observation.event_type != 'transit' or config.observation.n_nights != 1:
        raise ValueError('Direct sodium currently requires one synthetic transit')
    if config.observation.different_nights or config.observation.use_accurate_berv:
        raise ValueError('Supply the explicit fixed BERV for this direct sodium experiment')
    # Check that preparation and atmospheric assumptions match this runner.
    if config.pipeline.name != 'Seidel22' or config.retrieval.enabled or config.atmosphere.limb_asymmetries:
        raise ValueError('Direct sodium requires Seidel22 and a homogeneous forward atmosphere')
    if not model.isothermal or model.use_easychem or set(model.species) - {'H2', 'He', 'Na', 'Na__LorCut', 'Na__NewAllard'}:
        raise ValueError('Direct sodium currently supports an explicit isothermal H2/He/Na atmosphere')
    if len(model.species) != len(model.mass_fractions) or not np.isclose(sum(model.mass_fractions), 1):
        raise ValueError('Supply one mass fraction per atmospheric species, summing to one')
    # The spatial calculation uses pRT rays and the physical transit geometry.
    if config.observation.external_planet_model or config.observation.scale_injection != 1 or not config.observation.signal_uses_light_curve:
        raise ValueError('Direct sodium uses the pRT atmosphere with physical transit geometry and scale_injection=1')
    if ((not config.observation.simulate_planet and not (config.stellar_surface and config.stellar_surface.active))
            or config.noise.snr_correction != 0):
        raise ValueError('Use snr_correction=0. Stellar-only controls require an active stellar surface')
    if not {'H2', 'He'}.issubset(model.species) or not any(name.startswith('Na') for name in model.species):
        raise ValueError('Supply H2, He and a sodium opacity species')
    if config.observation.noise_scaling_factor != 1 or model.wind_velocity_kms != 0 or model.cloud_pressure_bar is not None:
        raise ValueError('This direct sodium route requires unit noise scaling, zero bulk wind and a cloud-free model')
    # This runner scales one reference transmission by the exposure airmass.
    if not config.tellurics.include_tellurics or config.tellurics.use_full_skycalc:
        raise ValueError('Supply one reference telluric spectrum and include_tellurics=true')
    # Check the supported stellar injection and correction choices together.
    if config.rm_clv.simulate and config.rm_clv.coupling != 'spatial':
        raise ValueError('Direct sodium currently supports spatial RM+CLV or a disabled stellar effect')
    if config.rm_clv.correct and (not config.rm_clv.simulate or config.rm_clv.correction_method != 'matched'):
        raise ValueError('Direct sodium correction currently uses the matched opaque-body model')
    # Each instrument supplies its own ETC anchor and exposure-time information.
    if config.instrument.name == 'CARMENES_VIS':
        configured_carmenes_etc(config)
        if config.noise.carmenes_etc_file is None:
            raise ValueError('Direct sodium requires a CARMENES ETC bundle')
    else:
        if config.noise.carmenes_etc_file is not None:
            raise ValueError('ANDES uses its own instrument ETC table')
        if (a.etc_reference_exposure_seconds is None
                or not np.isclose(a.etc_reference_exposure_seconds,
                                  config.observation.exposure_time_seconds,
                                  rtol=0, atol=1e-6)):
            raise ValueError('Record an ANDES ETC reference exposure matching the simulation')
        if (config.noise.noise_choice != 'SNR' or config.noise.fixed_snr is not None
                or config.noise.use_mean_snr):
            raise ValueError('Direct ANDES sodium uses the instrument ETC S/N')
    # Require recorded seeds for both stochastic components in this experiment.
    if config.noise.noise_seed is None or config.observation.throughput_seed is None:
        raise ValueError('Record explicit Gaussian and throughput seeds for the direct sodium experiment')
    # Check the external inputs before starting the expensive atmosphere calculation.
    for name in [a.observing_sequence_file, config.rm_clv.model_file,
                 config.tellurics.reference_telluric_file, config.planet.parameter_file]:
        if not Path(name).is_file():
            raise FileNotFoundError(name)
    from exoplore.instruments.carmenes_helium_molecfit import CarmenesMolecfitConfig
    # Validate the telluric-fitting settings with the shared Molecfit configuration.
    CarmenesMolecfitConfig(**a.molecfit)


def _factorized_transit(chords, planet, phase, planet_velocity, duration, samples, fine):
    """Apply a chromatic BATMAN transit to a fixed disk-integrated star."""
    import batman
    # Sample evenly spaced midpoints within each integration, expressed in orbital phase.
    offsets = ((np.arange(samples) + .5) / samples - .5) * duration / 86400 / planet.orbital_period_days
    # Keep one phase for every exposure and temporal subsample.
    subphase = phase[:, None] + offsets
    # Build the orbital geometry and limb darkening used for the RM+CLV-disabled case.
    params = batman.TransitParams()
    # Phase zero sets mid-transit at time zero for these light-curve evaluations.
    params.t0 = 0.
    params.per = planet.orbital_period_days
    # Convert the semimajor axis from km to cm, then express it in stellar radii.
    params.a = planet.a * 1e5 / planet.R_star
    # BATMAN takes inclination in degrees and the catalog orbital parameters.
    params.inc = planet.inclination_deg
    params.ecc = planet.eccentricity
    params.w = planet.argument_of_periastron_deg
    # Use the catalog quadratic limb-darkening coefficients.
    params.u = planet.limb_darkening_coeffs
    params.limb_dark = 'quadratic'
    # Start with the catalog radius ratio. Each wavelength replaces this below.
    params.rp = planet.R_pl / planet.R_star
    # Flatten the exposure/subsample times for BATMAN.
    transit = batman.TransitModel(params, (subphase * params.per).ravel())
    # Store one chromatic light curve for every atmospheric wavelength.
    curves = np.empty((subphase.size, len(chords.wavelength_um)))
    # The effective pRT radius gives the opaque-equivalent transit area at each wavelength.
    for j, radius in enumerate(chords.effective_radius_cm()):
        # Convert that effective radius into the radius ratio required by BATMAN.
        params.rp = radius / planet.R_star
        # Evaluate the transit at every temporal subsample for this wavelength.
        curves[:, j] = transit.light_curve(params)
    # Prepare one observer-frame transit spectrum per exposure.
    observed = np.zeros((len(phase), len(fine)))
    # Average the chromatic transit over the samples within each exposure.
    for n in range(len(phase)):
        for k in range(samples):
            # Update the orbital velocity from the exposure midpoint to this subsample.
            velocity = planet_velocity[n] + planet.kp_kms * (np.sin(2*np.pi*subphase[n, k]) - np.sin(2*np.pi*phase[n]))
            # Convert observed wavelengths to the planet frame for interpolation.
            rest = fine / (1 + velocity / 299792.458)
            # Require pRT coverage of all Doppler-shifted wavelengths.
            if rest.min() < chords.wavelength_um.min() or rest.max() > chords.wavelength_um.max():
                raise ValueError('Extend model_wavelength_limits_um to cover planetary Doppler shifts')
            # Interpolate this transit spectrum and add its equal time weight.
            observed[n] += np.interp(rest, chords.wavelength_um, curves[n*samples+k]) / samples
    return observed


def calculate_atomic_atmosphere(config, planet):
    """Calculate the direct-sodium rays reused when only the stellar surface changes."""
    from petitRADTRANS.radtrans import Radtrans
    from petitRADTRANS import physical_constants as cst
    from exoplore.atmosphere.transmission import calculate_chord_transmission
    a = config.atomic_transmission
    # Use the same atmospheric model for the spatial rays and the synthetic observation.
    region = config.atmosphere.planet_model
    # Sample pressure uniformly in log space between the configured bar limits.
    pressure = np.logspace(np.log10(region.pressure_min_bar), np.log10(region.pressure_max_bar), region.pressure_grid_size)
    # H2 and He supply Rayleigh scattering and collision-induced absorption.
    # The remaining species supply line opacity.
    species = [name for name in region.species if name not in ('H2', 'He')]
    # Load the line-by-line opacities over the padded pRT wavelength interval.
    rt = Radtrans(pressures=pressure, line_species=species, rayleigh_species=['H2', 'He'],
                  gas_continuum_contributors=['H2--H2', 'H2--He'],
                  wavelength_boundaries=a.model_wavelength_limits_um,
                  line_opacity_mode='lbl', path_input_data=config.paths.prt_input_data)
    # Assign the configured mass fraction to every pressure level in this homogeneous model.
    fractions = {name: np.full_like(pressure, value) for name, value in zip(region.species, region.mass_fractions)}
    # Calculate ray transmission versus wavelength and atmospheric radius.
    # The gravity uses the catalog mass and radius in CGS units.
    return calculate_chord_transmission(rt, temperatures=np.full_like(pressure, region.isothermal_temperature_K),
        mass_fractions=fractions, mean_molar_masses=np.full_like(pressure, region.mean_molecular_weight),
        reference_gravity=cst.G*planet.M_pl/planet.R_pl**2,
        reference_pressure=region.reference_pressure_bar, planet_radius=planet.R_pl)


def build_atomic_observation(config, output, *, planetary_rays=None):
    """Compute stellar/planetary transfer, terrestrial absorption and native noise."""
    from exoplore.stellar.coupled import SpatialTransitModel
    # Read the JSON and the separate planet file before constructing the observation.
    a = config.atomic_transmission
    planet = load_planet(config.planet.parameter_file)
    if planet.eccentricity != 0:
        raise ValueError('Direct sodium currently uses circular orbital velocities')
    # Resolve the instrument wavelength file and resolving power.
    instrument = load_instrument(config)
    # Use the physical order index selected by the JSON.
    order = config.instrument.order_indices[0]
    if config.instrument.name == 'ANDES_UBV':
        from exoplore.instruments.wavegrid import make_wave_grid_ANDES_modes
        with fits.open(instrument.wave_file) as h:
            # Convert the stored ETC order centre from nm to micrometres.
            center_um = float(h['WAVE_MIDS'].data[order])/1000
            # Read the reference S/N per resolution element for this order.
            snr_resel = float(h['SNR_MID'].data[order])
            # Match the mode=1 grid used by the current ANDES runner.
            full_wave = make_wave_grid_ANDES_modes(
                [h['WAVE_STARTS'].data[order]], [h['WAVE_MIDS'].data[order]],
                [h['WAVE_ENDS'].data[order]], R=instrument.res,
                px_per_resel=config.instrument.pixels_per_resolution_element,
                scale=1e-3, mode=1)[0]
    else:
        # CARMENES supplies the native wavelength grid directly in its FITS file.
        full_wave = np.asarray(fits.getdata(instrument.wave_file)[order], float)
    # Keep the detector pixels within the requested observing interval.
    indices = np.flatnonzero((full_wave >= a.wavelength_interval_um[0]) & (full_wave <= a.wavelength_interval_um[1]))
    # This cropped native grid will be used for the saved observation.
    wave = full_wave[indices]
    if wave.size < 3:
        raise ValueError('Observing wavelength interval contains too few native pixels')
    # Read the supplied exposure midpoints and matching terrestrial airmasses.
    with np.load(a.observing_sequence_file, allow_pickle=False) as sequence:
        jd, airmass = sequence['bjd'], sequence['airmass']
    if jd.ndim != 1 or jd.shape != airmass.shape or not np.isfinite(jd).all() or np.any(np.diff(jd) <= 0):
        raise ValueError('Supply finite increasing BJD midpoints and matching airmasses')
    # Use the optical transit geometry to select in-transit and reference exposures.
    _, inside, outside, midpoint = get_event(config.observation, planet, jd)
    if min(len(inside), len(outside)) < 2:
        raise ValueError('At least two in-transit and two reference exposures are required')
    # Phase zero is the chosen transit midpoint.
    phase = (jd - midpoint) / planet.orbital_period_days
    # The direct-sodium experiment uses the explicit fixed BERV from the JSON.
    berv = config.observation.berv_kms
    # The star has systemic motion, reflex motion and the selected BERV correction.
    stellar_velocity = planet.systemic_velocity_kms - planet.stellar_rv_semiamplitude_kms*np.sin(2*np.pi*phase) - berv
    # The planet has the opposite orbital sign. Both velocities are in the observer frame.
    planet_velocity = planet.systemic_velocity_kms + planet.kp_kms*np.sin(2*np.pi*phase) - berv
    chords = None
    if config.observation.simulate_planet:
        if planetary_rays is None:
            chords = calculate_atomic_atmosphere(config, planet)
        else:
            from exoplore.atmosphere.transmission import ChordTransmission
            if not isinstance(planetary_rays, ChordTransmission):
                raise TypeError('planetary_rays must be a ChordTransmission calculated for this atmosphere')
            chords = planetary_rays
    # Use the configured integration time when averaging the transit geometry.
    duration = config.observation.exposure_time_seconds
    # Load the local stellar intensities and prepare rotation and surface sampling.
    stellar = SpatialTransitModel(config.rm_clv, planet, instrument.res, surface_config=config.stellar_surface)
    # Integrate the same pRT atmospheric rays against the local stellar light.
    # Keep intrinsic spectra so telluric absorption can be applied before the instrument.
    result = stellar.evaluate(wave, phase, stellar_velocity, duration,
        chords=chords if config.rm_clv.simulate and config.observation.simulate_planet else None,
        planet_velocity_kms=planet_velocity, return_intrinsic=True, times_bjd=jd)
    if stellar.field is not None:
        (output/'stellar_surface_provenance.json').write_text(
            json.dumps(stellar.field.provenance(), indent=2))
    # Access the exposure-averaged flux components on their fine wavelength grid.
    intrinsic = result.intrinsic
    # For the disabled RM+CLV control, use the stationary disk-integrated star
    # and multiply it by the wavelength-dependent BATMAN transit.
    if not config.rm_clv.simulate:
        transit = _factorized_transit(chords, planet, phase, planet_velocity, duration,
                                      config.rm_clv.exposure_subsamples, intrinsic.wavelength_um)
        intrinsic.flux[:] = intrinsic.stellar_flux * transit
    # Save the intrinsic star, opaque-body and atmospheric components for inspection.
    np.savez_compressed(output/'intrinsic_spectra.npz', wavelength_um=intrinsic.wavelength_um,
        flux=intrinsic.flux, opaque_flux=intrinsic.opaque_flux, stellar_flux=intrinsic.stellar_flux,
        continuum=intrinsic.continuum, phase=phase, native_wave_um=wave,
        stellar_velocity_kms=stellar_velocity, planet_velocity_kms=planet_velocity)
    # Use the fine intrinsic grid for terrestrial transmission and convolution.
    fine = intrinsic.wavelength_um
    # Calculate native pixel boundaries halfway between neighbouring log wavelengths.
    centers = np.log(wave)
    # Extend the first and last boundaries by half a local pixel spacing.
    edges = np.r_[centers[0]-(centers[1]-centers[0])/2, (centers[:-1]+centers[1:])/2,
                  centers[-1]+(centers[-1]-centers[-2])/2]
    # Record the fine-grid log-wavelength spacing for the instrumental Gaussian.
    step = float(np.log(fine[1]/fine[0]))
    # Apply the same instrumental profile and pixel integration to every flux component.
    def sample(row):
        return sample_native_pixels(row, fine, edges, step, instrument.res)
    # Apply terrestrial absorption before instrumental convolution, using the
    # same reference transmission for photon response and synthetic spectra.
    with fits.open(config.tellurics.reference_telluric_file) as h:
        # The SkyCalc table stores wavelengths in nm. Convert them to the internal micrometres.
        tw, transmission = h[1].data['lam']/1000, h[1].data['trans']
    # Check coverage before interpolating the terrestrial transmission.
    if fine.min() < tw.min() or fine.max() > tw.max():
        raise ValueError('Reference telluric model does not cover the observing interval')
    # Place the reference terrestrial transmission on the fine observer-frame grid.
    reference = np.interp(fine, tw, transmission)
    # Generate one throughput factor per exposure using this order/night child stream.
    if config.observation.add_throughput_variations:
        # Use physical order and night zero to derive the component seed.
        seed = throughput_child_seed(config.observation.throughput_seed, order, 0)
        # The existing throughput routine supplies the configured jitter with its usual 0.02 scale.
        throughput = add_throughput(np.ones((len(jd), 1)), jitter_frac=.02, seed=seed)[:, 0]
    else:
        # With throughput variation disabled, every exposure retains unit throughput.
        throughput = np.ones(len(jd))
    # Reuse the quiet correction already integrated alongside the forward model.
    correction_intrinsic = result.correction_intrinsic
    # Collect the observed expectation and comparison components using the same sampling.
    arrays = {key: [] for key in ['expected_flux', 'bare_expected', 'no_telluric', 'bare_model', 'matched_bare_model', 'telluric_sampled']}
    # Build each exposure at its own airmass and throughput.
    for n, x in enumerate(airmass):
        # Scale optical depth with airmass through T(X) = T_ref ** (X / X_ref).
        trans = reference**(x/config.tellurics.reference_airmass)
        # Sample the out-of-transit stellar continuum for a common relative-flux scale.
        continuum = sample(intrinsic.continuum[n])
        # Combine atmosphere, star and tellurics before instrumental sampling.
        # Then apply the exposure throughput and express the result relative to the continuum.
        arrays['expected_flux'].append(sample(intrinsic.flux[n]*trans)*throughput[n]/continuum)
        # Save the corresponding opaque-body observation as a stellar comparison.
        arrays['bare_expected'].append(sample(intrinsic.opaque_flux[n]*trans)*throughput[n]/continuum)
        # Keep the noiseless planetary observation for the prepared injected-model comparison.
        arrays['no_telluric'].append(sample(intrinsic.flux[n])*throughput[n]/continuum)
        # Keep the atmosphere-free stellar spectrum used for the matched correction.
        arrays['matched_bare_model'].append(sample(intrinsic.opaque_flux[n])/continuum)
        if result.correction_flux is None:
            arrays['bare_model'].append(sample(intrinsic.opaque_flux[n])/continuum)
        else:
            # Sample the separately declared correction on its own continuum scale.
            correction_flux = np.interp(fine, correction_intrinsic.wavelength_um,
                                        correction_intrinsic.opaque_flux[n])
            correction_continuum = np.interp(fine, correction_intrinsic.wavelength_um,
                                             correction_intrinsic.continuum[n])
            arrays['bare_model'].append(sample(correction_flux)/sample(correction_continuum))
        # Sample the terrestrial transmission for the photon-response calculation.
        arrays['telluric_sampled'].append(sample(trans))
    # Convert exposure lists into matrices with shape (exposure, pixel).
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    # Choose the instrument-specific ETC conversion on the same expected-flux scale.
    if config.instrument.name == 'ANDES_UBV':
        from exoplore.observation.andes_sodium_noise import andes_sodium_photon_uncertainty
        noise = andes_sodium_photon_uncertainty(
            full_wave, indices, arrays['expected_flux'], arrays['telluric_sampled'],
            snr_resel, config.instrument.pixels_per_resolution_element, center_um)
    else:
        noise = sodium_photon_uncertainty(configured_carmenes_etc(config), order, full_wave,
            arrays['expected_flux'], arrays['telluric_sampled'], airmass, duration, instrument.res, indices)
    # Draw additive Gaussian noise independently of the throughput random stream.
    gaussian = np.random.default_rng(config.noise.noise_seed).normal(0., noise.std_noise)
    # A noiseless control retains the uncertainty arrays for weighting and sets the draw to zero.
    if config.observation.noiseless or config.observation.first_night_noiseless:
        gaussian[:] = 0
    # Save flux, errors, component draws and coordinates together for later reduction.
    arrays.update(raw_flux=arrays['expected_flux']+gaussian, raw_error=noise.std_noise,
        std_noise=noise.std_noise, gauss_noise=gaussian, throughput=throughput,
        wave_nm=np.tile(wave*1000, (len(jd), 1)), phase=phase, bjd=jd, airmass=airmass,
        snr_per_pixel=noise.local_snr_pixel, continuum_snr_pixel=noise.continuum_snr_pixel,
        photon_response=noise.photon_response)
    # Write the synthetic observation before beginning any external telluric fits.
    np.savez_compressed(output/'observations.npz', **arrays)
    # Pass the observer-frame velocities to the reduction alongside the saved arrays.
    velocities = SimpleNamespace(stellar_velocity_kms=stellar_velocity,
                                 planet_velocity_kms=planet_velocity)
    return arrays, velocities, planet, instrument.res


def run_atomic_simulation(config):
    """Run a fresh sodium observation, Molecfit fits and both line diagnostics."""
    # Check supported choices and input files before allocating the output directory.
    validate_atomic_run(config)
    # Resolve the run directory and require a fresh location to preserve existing products.
    output = Path(config.paths.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Save the actual choices used for this execution.
    config.to_json(output/'resolved.json')
    # Measure elapsed time with a clock that is independent of wall-clock adjustments.
    started = time.monotonic()
    # Build the noisy observation. Here intrinsic carries the returned velocity pair
    # used by reduction. The intrinsic flux components are stored in their own NPZ.
    data, intrinsic, planet, resolution = build_atomic_observation(config, output)
    # Record the forward-model runtime separately from the telluric reduction.
    physical_seconds = time.monotonic() - started
    print(f'Physical observation complete ({len(data["phase"])} exposures, {physical_seconds:.1f} s)', flush=True)
    # Continue with Molecfit and the two direct-transmission measurements.
    reduce_atomic_observation(config, output, data, intrinsic, planet, resolution, started, physical_seconds)


def fit_atomic_tellurics(config, output, data, planet, resolution):
    """Fit each exposure once and return the correction for paired stellar analyses."""
    from exoplore.core.helium_synthetic_observation import utc_midpoints
    from exoplore.instruments.carmenes_helium_molecfit import fit_section, CarmenesMolecfitConfig
    a = config.atomic_transmission
    # Supply site, target coordinates and velocity convention to the shared telluric wrapper.
    science = SimpleNamespace(gamma_kms=planet.systemic_velocity_kms,
        observatory_lon_lat_height=a.observatory_lon_lat_height,
        target_ra_deg=planet.ra_deg, target_dec_deg=planet.dec_deg, max_wavelength_shift_kms=2.,
        instrument_name='ANDES' if config.instrument.name == 'ANDES_UBV' else 'CARMENES')
    # Convert BJD_TDB midpoints to UTC times for Molecfit atmospheric calculations.
    times = utc_midpoints(data['bjd'], science)
    # Use the same telluric fitter for sodium and helium, with this JSON setting its inputs.
    settings = CarmenesMolecfitConfig(**a.molecfit)
    # Require the telluric fit and synthetic observation to use the same instrument resolution.
    if settings.calibrated_resolving_power != resolution:
        raise ValueError('Molecfit calibrated resolution differs from the instrument')
    # Keep each exposure fit in a separate subdirectory under this run.
    root = output/'molecfit'
    root.mkdir()
    # Fit one observer-frame exposure, protecting the predicted planetary sodium lines.
    def fit_one(n):
        # Pass its flux, uncertainty, UTC time, orbital velocity, airmass and exposure duration.
        # helium_guard_kms is the shared wrapper field used for the sodium protection width.
        result = fit_section(root/f'exposure_{n+1:03d}', data['wave_nm'][n], data['raw_flux'][n],
            data['raw_error'][n], settings, science, float(times.mjd[n]), config.observation.berv_kms,
            float(planet.kp_kms*np.sin(2*np.pi*data['phase'][n])), float(data['airmass'][n]),
            config.observation.exposure_time_seconds, protected_lines_nm=a.line_wavelengths_nm,
            guard_kms=settings.helium_guard_kms)
        print(f'Molecfit accepted exposure {n+1}/{len(times)}', flush=True)
        return n, result
    # A real first-exposure fit gates the full set of external recipe calls.
    first = fit_one(0)
    # Retain exposure indices because concurrent fits can finish in a different order.
    fits_by_index = {first[0]: first[1]}
    # Run independent external recipe processes up to the requested worker limit.
    with ThreadPoolExecutor(max_workers=a.molecfit_workers) as pool:
        # Queue the remaining exposures after the first fit has passed its quality checks.
        jobs = [pool.submit(fit_one, n) for n in range(1, len(times))]
        # Collect successful fits as they finish.
        for job in as_completed(jobs):
            try:
                n, result = job.result()
            except Exception:
                # Stop queued fits when an exposure fails its acceptance checks.
                # The currently running recipes can finish writing diagnostics.
                for pending in jobs:
                    pending.cancel()
                raise
            fits_by_index[n] = result
            # Write the accepted exposure numbers so a long run can be monitored.
            (output/'fit_progress.json').write_text(json.dumps({'accepted': sorted(n+1 for n in fits_by_index)}))
    # Restore chronological order before constructing the time series.
    fitted = [fits_by_index[n] for n in range(len(times))]
    # Read the telluric-corrected flux, its propagated errors and fitted transmission.
    waves = np.array([r['wave_nm'] for r in fitted])
    flux = np.array([r['flux'] for r in fitted])
    error = np.array([r['error'] for r in fitted])
    T = np.array([r['transmittance'] for r in fitted])
    corrections = dict(wave_nm=waves, flux=flux, error=error, transmittance=T,
                       input_sha256=np.array(telluric_input_hash(config, data)))
    np.savez_compressed(output/'corrected_observations.npz', **corrections)
    return corrections


def telluric_input_hash(config, data):
    """Identify the observation and fitting choices used by a saved correction."""
    digest = hashlib.sha256()
    for name in ('raw_flux', 'raw_error', 'wave_nm', 'bjd', 'airmass'):
        value = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    settings = dict(molecfit=config.atomic_transmission.molecfit,
        protected_lines=config.atomic_transmission.line_wavelengths_nm,
        site=config.atomic_transmission.observatory_lon_lat_height,
        berv=config.observation.berv_kms, exposure=config.observation.exposure_time_seconds,
        planet_sha256=hashlib.sha256(Path(config.planet.parameter_file).read_bytes()).hexdigest())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def reduce_atomic_observation(config, output, data, intrinsic, planet, resolution, started, physical_seconds, *, corrections=None):
    """Fit terrestrial absorption and prepare each configured sodium line."""
    from exoplore.atmosphere.helium_pwinds import contact_phases
    from exoplore.plotting.helium import plot_helium_summary
    a = config.atomic_transmission
    if corrections is None:
        corrections = fit_atomic_tellurics(config, output, data, planet, resolution)
    if str(corrections['input_sha256']) != telluric_input_hash(config, data):
        raise ValueError('Telluric corrections belong to a different observation or fitting setup')
    waves, flux, error, T = (corrections[k] for k in ('wave_nm', 'flux', 'error', 'transmittance'))
    # Exclude deep terrestrial absorption and samples with invalid uncertainties.
    good = (T > config.tellurics.mask_threshold) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    # Represent those excluded samples as NaN so preparation preserves their masks.
    flux, error = np.where(good, flux, np.nan), np.where(good, error, np.nan)
    # Select the same optical in-transit and out-of-transit exposures used in injection.
    _, inside, outside, _ = get_event(config.observation, planet, data['bjd'])
    # Planet-frame alignment uses the planetary velocity relative to the star.
    relative_velocity = intrinsic.planet_velocity_kms - intrinsic.stellar_velocity_kms
    # The atmosphere-free opaque-body spectrum supplies the stellar correction.
    # Select the saved opaque-body spectrum when stellar correction is enabled.
    stellar_model = data['bare_model'] if config.rm_clv.correct else None
    # Share velocities, reference membership and preparation choices between both lines.
    # Telluric fitting has completed. Sky emission is absent from this synthetic observation.
    common = dict(stellar_rv_kms=intrinsic.stellar_velocity_kms,
        planet_rv_stellar_kms=relative_velocity, reference_indices=outside, transit_indices=inside,
        settings=config.pipeline.seidel22, telluric_corrected=True, sky_subtracted=True,
        stellar_model_flux=stellar_model)
    # Begin a compact result record with exposure counts and forward-model runtime.
    report = dict(exposures=len(data['bjd']),in_transit=len(inside),reference_exposures=len(outside),
        physical_seconds=physical_seconds,results={})
    # Analyse the two rest wavelengths in their supplied D2, D1 order.
    for k, line in enumerate(a.line_wavelengths_nm):
        # Use these stable labels for each line's numerical products and figure.
        label = ['Na_D2', 'Na_D1'][k]
        # Convert the requested planet-frame velocity half-width to wavelength limits.
        bounds = (line*(1-a.band_half_width_kms/299792.458), line*(1+a.band_half_width_kms/299792.458))
        # Prepare the measured spectra and propagate their final covariance.
        # For noise resampling, divide the noiseless expectation by the fitted transmission
        # to express the draws on the same telluric-corrected scale as the data.
        result = prepare_transmission(waves, flux, error, calculate_covariance=True,
            lightcurve_window_nm=bounds, noise_resamples=config.pipeline.seidel22.noise_resamples,
            noise_resample_seed=config.pipeline.seidel22.noise_resample_seed,
            noise_expectation=data['expected_flux']/T if config.pipeline.seidel22.noise_resamples else None,
            **common)
        # Process the noiseless, telluric-free observation with the same estimator
        # to show the expected recovered atmospheric profile.
        truth = prepare_transmission(data['wave_nm'], data['no_telluric'], data['raw_error'],
                                     lightcurve_window_nm=bounds, **common)
        # Locate the wavelength channels used for the reported band depth.
        band = (result.planet_wave_nm >= bounds[0]) & (result.planet_wave_nm <= bounds[1])
        # Use an arithmetic mean over those channels for this band statistic.
        weights = band.astype(float)/band.sum()
        # Report absorption as a positive percentage deficit from unit transmission.
        depth = 100*(1-np.mean(result.planet_coadd[band]))
        # Contract the full spectral covariance with the band weights, then convert to percent.
        sigma = 100*np.sqrt(weights@result.coadd_covariance@weights)
        # Keep the measured depth, its uncertainty and prepared-model depth together.
        report['results'][label] = dict(depth_percent=float(depth),error_percent=float(sigma),
            prepared_truth_percent=float(100*(1-np.mean(truth.planet_coadd[band]))),covariance_method=result.covariance_method)
        # Package the common Tutorial 12 plotting arrays for this sodium line.
        products = dict(stellar_wave_nm=result.stellar_wave_nm,stellar_transmission=result.stellar_transmission,
            planet_wave_nm=result.planet_wave_nm,planet_coadd=result.planet_coadd,
            coadd_error=np.sqrt(result.coadd_covariance.diagonal()),planet_lightcurve=result.planet_lightcurve,
            lightcurve_error=np.sqrt(np.maximum(np.diag(result.lightcurve_covariance),0)),
            truth_planet_coadd=truth.planet_coadd,truth_planet_lightcurve=truth.planet_lightcurve)
        # Save spectra, light curves and their covariance for later inspection.
        np.savez_compressed(output/(label+'_transmission.npz'), **products, phase=data['phase'],
            lightcurve_covariance=result.lightcurve_covariance, band_covariance=result.coadd_covariance[band][:,band].toarray())
        # Describe the displayed wavelength interval, line identity and optical contacts.
        plot = SimpleNamespace(period_days=planet.orbital_period_days,plot_stellar_window_nm=[line-.16,line+.16],
            planet_lightcurve_window_nm=bounds,optical_contact_phases=contact_phases(planet),line_label=label,
            line_wavelengths_nm=[line],line_transition_label=label)
        # Use the shared three-panel renderer for the stellar-frame map, planet-frame
        # spectrum and band light curve. The supplied line metadata labels it as sodium.
        plot_helium_summary(output/(label+'_transmission.png'),products,data['phase'],relative_velocity,np.zeros(len(data['bjd'])),plot,synthetic=True)
        print(f'{label}: {depth:.4f} +/- {sigma:.4f} percent',flush=True)
    # Finish the run record with elapsed time, random seeds and cadence checks.
    report['seconds'] = time.monotonic()-started
    report['gaussian_seed'] = config.noise.noise_seed
    report['throughput_seed'] = config.observation.throughput_seed
    report['cadence_separations_below_exposure'] = int(np.sum(np.diff(data['bjd'])*86400 < config.observation.exposure_time_seconds))
    # Read the installed package version and source revision for provenance.
    import exoplore
    import subprocess
    report['exoplore_version'] = exoplore.__version__
    report['git_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    # Record which instrument file or ETC bundle anchored the observation noise.
    noise_source = (load_instrument(config).wave_file if config.instrument.name == 'ANDES_UBV'
                    else config.noise.carmenes_etc_file)
    # Hash each scientific input so future runs can check that their files agree.
    report['input_sha256'] = {str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [
        a.observing_sequence_file,config.planet.parameter_file,config.rm_clv.model_file,
        config.tellurics.reference_telluric_file,noise_source]}
    # Write the human-readable measurements and provenance next to the numerical products.
    (output/'final_summary.json').write_text(json.dumps(report,indent=2))
