"""Stellar-reference transmission spectroscopy following Seidel et al. (2022).

Reference: MNRAS 513, L15, https://doi.org/10.1093/mnrasl/slac027.
The inputs are extracted, sky-subtracted and telluric-corrected spectra.
Wavelengths are vacuum nm. Positive radial velocities mean recession.


The direct sodium runner calls prepare_transmission() after Molecfit. The
standard CCF runner calls preparing_pipeline_adapter() and receives spectra
back on its observer-frame grid. Shared helium helpers provide frame shifts,
interpolation across valid pixels and weighted averages.

Follow prepare_transmission() in this order: align with the star, normalize
the continuum, build the out-of-transit reference, correct the stellar transit,
align with the planet, and average the selected exposures.
"""

from dataclasses import dataclass, replace
import numpy as np
from scipy.sparse import csr_matrix

from exoplore.config.seidel22 import Seidel22Config
from exoplore.pipelines.helium_math import (
    C_KMS, window_mask, weighted_mean, stellar_wavelength, planet_wavelength, gap_resample,
)


@dataclass(frozen=True)
class TransmissionResult:
    """Intermediate spectra and the planet-aligned mean for inspection.

    Marginal variances include the shared reference's diagonal variance.
    Conditional variances hold that reference and the continuum fits fixed.
    Interpolation also creates pixel covariance. The optional final coadd and
    light-curve covariance can include continuum refitting through noise
    resampling. covariance_method records which propagation was used.
    Intermediate diagonal arrays retain their conditional definitions.
    """

    # Common stellar-frame wavelength grid in vacuum nm.
    stellar_wave_nm: np.ndarray
    # One corrected, reference-divided stellar-frame spectrum per exposure.
    stellar_transmission: np.ndarray
    # Diagonal ratio variances including the shared reference contribution.
    stellar_variance: np.ndarray
    # Diagonal ratio variances with the fitted reference treated as fixed.
    stellar_conditional_variance: np.ndarray
    # Continuum-normalized out-of-transit stellar reference and its variance.
    master: np.ndarray
    master_variance: np.ndarray
    # Continuum S/N used as the exposure weight in the reference and coadd.
    exposure_snr: np.ndarray
    # Common planet-frame wavelength grid and the aligned exposure spectra.
    planet_wave_nm: np.ndarray
    planet_transmission: np.ndarray
    # S/N-weighted mean of the selected in-transit spectra.
    planet_coadd: np.ndarray
    # Diagonal coadd variance with reference and continuum held fixed.
    conditional_coadd_variance: np.ndarray
    # Number of finite in-transit samples contributing to each coadd channel.
    contributing_exposures: np.ndarray
    # Optional sparse covariance retains shared-reference and interpolation
    # terms. Rows with missing coadd coverage have zero covariance entries
    # and must be excluded using the finite planet_coadd mask.
    coadd_covariance: csr_matrix | None = None
    # Prepared opaque-body stellar ratio used to divide the individual transmissions.
    stellar_correction: np.ndarray | None = None
    # One arithmetic band-average transmission per exposure, in the planet frame.
    planet_lightcurve: np.ndarray | None = None
    lightcurve_covariance: np.ndarray | None = None
    # Record which uncertainty calculation produced the final covariance.
    covariance_method: str = "fixed_continuum"
    noise_resamples: int = 0
    noise_resample_seed: int | None = None


def _indices(values, n, label):
    """Keep reference and transit membership explicit and unambiguous."""
    # Use explicit exposure membership supplied by the caller.
    result = np.asarray(values)
    # Check that each exposure index is an integer, unique and within the input rows.
    if (result.ndim != 1 or result.size == 0 or result.dtype.kind not in 'iu'
            or np.any(result < 0) or np.any(result >= n)
            or len(np.unique(result)) != len(result)):
        raise ValueError(f"{label} must contain distinct valid exposure indices")
    return result


def _normalize_stellar_rows(waves, data, errors, star_rv, grid, settings,
                            calculate_covariance):
    """Use identical alignment and continuum operations for data and stellar models."""
    # Remove the supplied observed stellar shift from every exposure wavelength.
    # BERV is already included in star_rv, so this helper receives zero separately.
    stellar_wave = stellar_wavelength(waves, 0, star_rv[:, None])
    # Collect the aligned flux, variance and continuum S/N for each row.
    normalized, variance, snr = [], [], []
    # Keep the linear interpolation and native variances for covariance propagation.
    stellar_operators, native_variances = [], []
    # Normalize each exposure on the same stellar-rest-frame wavelength grid.
    for w, row, sigma in zip(stellar_wave, data, errors):
        # Exclude missing flux or invalid errors before any interpolation.
        valid = np.isfinite(row) & np.isfinite(sigma) & (sigma > 0)
        # Align first so the same stellar-frame channels define the continuum
        # in every exposure. Selecting anchors on moving native samples can
        # introduce level jumps when a pixel crosses a window boundary.
        aligned, aligned_variance = gap_resample(
            w, np.where(valid, row, np.nan),
            np.where(valid, sigma**2, np.nan), grid)
        # Select finite continuum pixels inside the user's stellar-frame windows.
        anchors = (window_mask(grid, settings.continuum_windows_nm)
                   & np.isfinite(aligned) & np.isfinite(aligned_variance)
                   & (aligned_variance > 0))
        # Stop when the surviving continuum coverage is too small.
        if anchors.sum() < settings.minimum_continuum_pixels:
            raise ValueError("Seidel22 continuum windows have insufficient valid pixels")
        # Use the median continuum flux as this exposure's normalization level.
        level = np.median(aligned[anchors])
        # Measure the median continuum S/N for the exposure weighting.
        signal_to_noise = np.median(aligned[anchors] / np.sqrt(aligned_variance[anchors]))
        if not np.isfinite(level) or level <= 0 or signal_to_noise <= 0:
            raise ValueError("Seidel22 requires a positive continuum and continuum S/N")
        # Apply the same continuum level to aligned flux and variance. The
        # optional covariance calculation holds this fitted level fixed.
        y, v = aligned / level, aligned_variance / level**2
        # Retain the normalized row, its variance and its measured S/N.
        normalized.append(y)
        variance.append(v)
        snr.append(signal_to_noise)
        # Record the native-to-stellar interpolation as a matrix. This lets the
        # uncertainty calculation track shared contributions between neighbouring pixels.
        # Propagate native noise through both interpolations and the shared reference.
        if calculate_covariance:
            from exoplore.pipelines.helium_noise_propagation import interpolation_operator
            stellar_operators.append(interpolation_operator(w, grid, valid))
            native_variances.append(np.where(valid, (sigma / level)**2, 0.))
    # Stack the row lists into arrays for reference construction.
    normalized, variance, snr = map(np.asarray, (normalized, variance, snr))
    return normalized, variance, snr, stellar_operators, native_variances


def prepare_transmission(wave_nm, flux, error, *, stellar_rv_kms,
                         planet_rv_stellar_kms, reference_indices,
                         transit_indices, settings: Seidel22Config,
                         telluric_corrected: bool, sky_subtracted: bool,
                         calculate_covariance: bool = False,
                         stellar_model_flux: np.ndarray | None = None,
                         lightcurve_window_nm: tuple[float, float] | None = None,
                         noise_resamples: int = 0,
                         noise_resample_seed: int | None = None,
                         noise_expectation: np.ndarray | None = None):
    """Align spectra, divide by the master-out and combine in the planet frame.

    ``stellar_rv_kms`` is the full observed stellar shift, including systemic,
    reflex and Earth velocities in the input's convention. The planet velocity
    is relative to the star. Supply zero stellar velocities for spectra already
    in the stellar frame. Input correction flags must describe the actual data.
    stellar_model_flux supplies an atmosphere-free observer-frame stellar model.
    It undergoes the same alignment and continuum normalization. Its master-out
    uses the data's reference weights. The individual residuals are divided by
    this fixed model before planetary alignment. Model uncertainty is omitted.
    Set calculate_covariance=True for sparse propagation of native pixel noise
    through the common reference and both interpolations. Continuum estimates
    and S/N weights remain fixed at their measured values in that calculation.
    With noise_resamples >= 2, repeat preparation on independent native-pixel
    Gaussian draws to include continuum and weight variability in the final
    coadd and light-curve covariance. This replaces the analytical covariance.
    The point estimates stay those of the supplied data. noise_expectation
    optionally supplies the noiseless mean for simulation validation. With
    no expectation, the measured flux supplies a parametric-bootstrap mean.
    Telluric and stellar models remain fixed in both uncertainty methods.
    """
    # Check the continuum windows and uncertainty settings before using the data.
    settings.__post_init__()
    # Validate the optional covariance and resampling requests together.
    if type(calculate_covariance) is not bool:
        raise ValueError("calculate_covariance must be boolean")
    if type(noise_resamples) is not int or noise_resamples < 0 or noise_resamples == 1:
        raise ValueError("noise_resamples must be zero or an integer of at least two")
    if noise_resamples and not calculate_covariance:
        raise ValueError("noise_resamples requires calculate_covariance=True")
    if noise_expectation is not None and not noise_resamples:
        raise ValueError("noise_expectation requires noise_resamples >= 2")
    # Require the caller to confirm the terrestrial corrections already applied.
    if telluric_corrected is not True or sky_subtracted is not True:
        raise ValueError("Seidel22 requires telluric-corrected, sky-subtracted spectra")
    # Fluxes and standard deviations must have shape (exposure, pixel).
    data = np.asarray(flux, dtype=float)
    errors = np.asarray(error, dtype=float)
    if data.ndim != 2 or data.shape != errors.shape or min(data.shape) < 2:
        raise ValueError("Flux and error must be matching exposure-by-pixel matrices")
    # Allow one common wavelength vector or a separate vector for each exposure.
    waves = np.broadcast_to(np.asarray(wave_nm, dtype=float), data.shape)
    if (np.any(~np.isfinite(waves)) or np.any(waves <= 0)
            or np.any(np.diff(waves, axis=1) <= 0)):
        raise ValueError("Wavelengths must be positive, finite and increasing in each exposure")
    # Select the reference and in-transit rows explicitly.
    n = len(data)
    refs = _indices(reference_indices, n, 'reference_indices')
    inside = _indices(transit_indices, n, 'transit_indices')
    # Keep the out-of-transit reference separate from the exposures being coadded.
    if np.intersect1d(refs, inside).size:
        raise ValueError("Reference and transit exposures must be disjoint")
    # Allow either a constant velocity or one velocity per exposure, in km/s.
    star_rv = np.broadcast_to(np.asarray(stellar_rv_kms, float), (n,))
    planet_rv = np.broadcast_to(np.asarray(planet_rv_stellar_kms, float), (n,))
    if (np.any(~np.isfinite(star_rv)) or np.any(~np.isfinite(planet_rv))
            or np.any(abs(star_rv) >= C_KMS) or np.any(abs(planet_rv) >= C_KMS)):
        raise ValueError("Radial velocities must be finite and subluminal")

    # Removing the observed stellar shift aligns the stellar absorption lines.
    stellar_wave = stellar_wavelength(waves, 0, star_rv[:, None])
    # Use the median stellar-frame wavelengths as the common interpolation grid.
    grid = np.median(stellar_wave, axis=0)
    # Align and continuum-normalize the observed spectra on that grid.
    normalized, variance, snr, stellar_operators, native_variances = _normalize_stellar_rows(
        waves, data, errors, star_rv, grid, settings, calculate_covariance)
    # Average the out-of-transit spectra with weights proportional to continuum S/N.
    master, master_var = weighted_mean(normalized[refs], variance[refs],
                                      exposure_weights=snr[refs])
    # Division requires a finite, positive reference flux at each channel.
    valid_master = np.isfinite(master) & (master > 0)
    # Mark unusable reference wavelengths so they remain excluded downstream.
    master = np.where(valid_master, master, np.nan)
    # Divide each normalized exposure by the common out-of-transit spectrum.
    ratio = normalized / master
    # Propagate each exposure variance through division by a fixed reference.
    conditional = variance / master**2
    # Include the reference variance through the derivative of normalized/master.
    marginal = conditional + normalized**2 * master_var / master**4
    # Reference exposures also appear in their own denominator. Include that
    # covariance in their individual ratio errors with the same S/N weights.
    # The reference's per-channel weights use its finite flux and variance samples.
    ref_good = np.isfinite(normalized[refs]) & np.isfinite(variance[refs])
    # Assign each valid reference sample its exposure S/N weight.
    weight = np.where(ref_good, snr[refs, None], 0.0)
    # Sum those weights separately at each wavelength.
    total = weight.sum(axis=0)
    # Normalize the reference weights to sum to one wherever coverage exists.
    weight = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    # For reference exposures, subtract the covariance between the numerator
    # and the master that contains the same exposure.
    marginal[refs] -= 2 * normalized[refs] * weight * variance[refs] / master**3
    # Clip small negative propagated variances at zero.
    marginal = np.maximum(marginal, 0)
    # Record the stellar correction only when a model has been supplied.
    correction = None
    # Prepare the atmosphere-free stellar transit using the same data masks,
    # continuum intervals, reference exposure membership and S/N weights.
    if stellar_model_flux is not None:
        # Require one model flux value for every observation pixel.
        model = np.asarray(stellar_model_flux, dtype=float)
        if model.shape != data.shape:
            raise ValueError("Stellar model must match the exposure-by-pixel flux shape")
        # Use the observation's usable pixels to define model coverage.
        valid = np.isfinite(data) & np.isfinite(errors) & (errors > 0)
        if np.any(valid & (~np.isfinite(model) | (model <= 0))):
            raise ValueError("Stellar model must be finite and positive at valid data pixels")
        # Apply that coverage to the model before alignment and normalization.
        model = np.where(valid, model, np.nan)
        # Normalize the model with the same stellar-frame preparation as the data.
        model_rows, model_variance, _, _, _ = _normalize_stellar_rows(
            waves, model, errors, star_rv, grid, settings, False)
        # The same reference exposures and weights define both masters.
        model_master, _ = weighted_mean(model_rows[refs], model_variance[refs],
                                        exposure_weights=snr[refs])
        # The model's exposure/reference ratio describes its RM+CLV distortion.
        correction = model_rows/model_master
        # Keep finite positive correction factors and mask the remaining channels.
        correction = np.where(np.isfinite(correction) & (correction > 0),
                              correction, np.nan)
        # Divide the data transmission by the prepared stellar distortion.
        ratio = ratio/correction
        # Scale both variance definitions by the square of that fixed divisor.
        conditional = conditional/correction**2
        marginal = marginal/correction**2
    # A stellar-core mask follows the stellar frame through planet alignment.
    # Identify the stellar-frame core intervals requested by the user.
    core = window_mask(grid, settings.stellar_mask_windows_nm)
    # Apply the same core mask to the transmission and both variance arrays.
    ratio[:, core] = np.nan
    conditional[:, core] = np.nan
    marginal[:, core] = np.nan
    # Collect the planet-aligned spectra and their conditional variances.
    planet, planet_variance = [], []
    # Remove the planet's velocity relative to the star in each exposure.
    for row, var, rv in zip(ratio, conditional, planet_rv):
        # Interpolate the shifted spectrum onto the common planet-frame grid,
        # carrying its variance and excluded wavelength gaps through the same operation.
        y, v = gap_resample(planet_wavelength(grid, rv), row, var, grid)
        planet.append(y)
        planet_variance.append(v)
    # Stack the shifted exposures into matrices.
    planet, planet_variance = map(np.asarray, (planet, planet_variance))
    # Average the selected in-transit rows with their continuum-S/N weights.
    coadd, coadd_var = weighted_mean(planet[inside], planet_variance[inside],
                                    exposure_weights=snr[inside])
    # Count surviving exposures at each planet-frame wavelength.
    coverage = np.isfinite(planet[inside]).sum(axis=0)
    # Initialize the optional full covariance and band light curve.
    covariance = None
    lightcurve, lightcurve_covariance = None, None
    band = None
    # Measure a fixed wavelength band after planet alignment for every exposure.
    if lightcurve_window_nm is not None:
        # Validate the requested vacuum-nm interval.
        bounds = np.asarray(lightcurve_window_nm, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Light-curve window requires two increasing positive wavelengths")
        # Select the common planet-frame pixels inside that interval.
        band = window_mask(grid, [bounds])
        if band.sum() < 2:
            raise ValueError("Light-curve window must contain at least two channels")
        # An arithmetic mean preserves missing coverage as NaN when a band is incomplete.
        lightcurve = np.mean(planet[:, band], axis=1)
    if calculate_covariance:
        from exoplore.pipelines.helium_noise_propagation import (
            interpolation_operator, transmission_coadd_covariance,
        )
        # Build the stellar-to-planet interpolation operator for each exposure.
        shifts = [interpolation_operator(planet_wavelength(grid, rv), grid,
                                         np.isfinite(row))
                  for rv, row in zip(planet_rv, ratio)]
        # Include division by the fixed stellar model before the planet-frame shift.
        if correction is not None:
            from scipy.sparse import diags
            # The fixed stellar correction precedes planetary interpolation.
            shifts = [operator @ diags(np.divide(1., row,
                out=np.zeros_like(row), where=np.isfinite(row) & (row > 0)))
                for operator, row in zip(shifts, correction)]
        # Use the actual surviving in-transit pixels when assigning coadd weights.
        valid = np.isfinite(planet[inside]) & np.isfinite(planet_variance[inside])
        # Apply continuum-S/N exposure weights separately at each output wavelength.
        coefficients = np.where(valid, snr[inside, None], 0.)
        # Calculate their normalization over the contributing exposures.
        total = coefficients.sum(axis=0)
        coefficients = np.divide(coefficients, total, out=np.zeros_like(coefficients),
                                 where=total > 0)
        # Combine native variances, interpolation and shared-reference terms into
        # the final wavelength-by-wavelength covariance matrix.
        covariance = transmission_coadd_covariance(
            stellar_operators, native_variances, shifts, normalized, master,
            weight, coefficients, refs, inside)
        # Also propagate correlations between the exposure band averages.
        if band is not None:
            from exoplore.pipelines.helium_noise_propagation import transmission_band_covariance
            lightcurve_covariance = transmission_band_covariance(
                stellar_operators, native_variances, shifts, normalized, master,
                weight, refs, band, np.isfinite(lightcurve))
    # Return the intermediate spectra alongside the final coadd for inspection.
    result = TransmissionResult(grid, ratio, marginal, conditional, master,
                                master_var, snr, grid.copy(), planet, coadd,
                                coadd_var, coverage, covariance, correction,
                                lightcurve, lightcurve_covariance)
    # When requested, estimate the final covariance by repeating the preparation
    # with native Gaussian draws. The measured point estimates remain in result.
    if noise_resamples:
        from exoplore.pipelines.seidel22_uncertainty import sample_preparation_covariance

        # Give each draw the same correction and exposure selection. Each call
        # re-estimates the continuum, exposure weights and common reference.
        def reduce(realization):
            return prepare_transmission(
                waves, realization, errors, stellar_rv_kms=star_rv,
                planet_rv_stellar_kms=planet_rv, reference_indices=refs,
                transit_indices=inside, settings=settings,
                telluric_corrected=True, sky_subtracted=True,
                stellar_model_flux=stellar_model_flux,
                lightcurve_window_nm=lightcurve_window_nm)

        # Calculate covariance across the repeated spectra and light curves.
        covariance, curve_covariance = sample_preparation_covariance(
            data, errors, noise_expectation, noise_resamples, noise_resample_seed, reduce)
        # Attach those covariance estimates and their recorded random-draw settings.
        result = replace(result, coadd_covariance=covariance,
                         lightcurve_covariance=curve_covariance,
                         covariance_method="resampled_continuum",
                         noise_resamples=noise_resamples,
                         noise_resample_seed=noise_resample_seed)
    return result


def validate_run(config):
    """Check the direct-transmission route before loading atmospheric tables."""
    # Distinguish the integrated sodium simulation from standard CCF preparation.
    atomic = config.atomic_transmission is not None and config.atomic_transmission.enabled
    # The integrated sodium runner explicitly requests the full resampling calculation.
    if config.pipeline.seidel22.noise_resamples and not atomic:
        raise ValueError('Configure noise_resamples on a direct atomic observation or pass it to prepare_transmission')
    # Check supported analysis combinations before loading expensive inputs.
    if config.retrieval.enabled:
        raise ValueError("Seidel22 retrieval requires propagation of the shared stellar reference")
    if config.pipeline.prepare_template:
        raise ValueError("Set pipeline.prepare_template=false for Seidel22")
    if config.pipeline.sysrem_robust_halt or config.pipeline.optimize_sysrem_order_by_order:
        raise ValueError("Seidel22 uses a stellar reference. Disable SYSREM optimisation and robust halt")
    if config.observation.event_type != 'transit':
        raise ValueError("Seidel22 requires a transit with out-of-transit reference exposures")
    # For measured spectra, require the input correction flags to be set by the user.
    if config.observation.use_real_data:
        choices = config.pipeline.seidel22
        if not choices.input_telluric_corrected or not choices.input_sky_subtracted:
            raise ValueError("Seidel22 observed inputs require explicit telluric and sky correction flags")
    elif config.tellurics.include_tellurics and not atomic:
        raise ValueError("Seidel22 synthetic runner currently requires include_tellurics=false")


def preparing_pipeline_adapter(inputs, data, noise, wave_um, mask,
                               without_signal, masks, correct_uncertainties,
                               retrieval):
    """Return prepared ratios on the runner's original observer-frame grid.

    CCF velocity tracks are evaluated in the observer frame by the runner.
    After forming the stellar reference, resample the ratios back to that grid
    so their planetary Doppler motion retains the existing CCF convention.
    Direct planet-frame spectra are available through prepare_transmission().
    """
    # The CCF adapter requires the observed stellar motion for every exposure.
    if retrieval or inputs.stellar_rv_kms is None:
        raise ValueError("Seidel22 needs per-exposure stellar velocities and forward preparation")
    # Read the continuum and masking choices from the active pipeline block.
    settings = inputs.pipeline.seidel22
    # Work on independent arrays so the runner's stored observation is preserved.
    flux = np.array(data, float, copy=True)
    error = np.array(noise, float, copy=True)
    # Apply the runner's existing pixel mask to both flux and error.
    flux[:, np.asarray(mask, int)] = np.nan
    error[:, np.asarray(mask, int)] = np.nan
    # The runner's without_signal list supplies the out-of-transit stellar reference.
    refs = _indices(without_signal, len(flux), 'without_signal')
    # The remaining exposures supply the selected in-transit set for this adapter.
    inside = np.setdiff1d(np.arange(len(flux)), refs)
    # For observed data, read the declared corrections. For this synthetic CCF
    # route, tellurics are disabled by validation and sky emission is absent.
    corrected = settings.input_telluric_corrected if inputs.use_real_data else not inputs.tellurics.include_tellurics
    sky = settings.input_sky_subtracted if inputs.use_real_data else True
    # Convert the CCF runner's micrometres to the preparation's vacuum nm.
    # This adapter uses the returned stellar-frame ratios for cross-correlation.
    result = prepare_transmission(
        np.asarray(wave_um) * 1000, flux, error,
        stellar_rv_kms=inputs.stellar_rv_kms,
        planet_rv_stellar_kms=0, reference_indices=refs,
        transit_indices=inside, settings=settings,
        telluric_corrected=corrected, sky_subtracted=sky,
        stellar_model_flux=inputs.stellar_model_flux)
    # Collect ratios and errors after restoring the observer wavelength frame.
    prepared, propagated = [], []
    # Restore each exposure's stellar shift before sampling the original CCF grid.
    for row, var, rv in zip(result.stellar_transmission,
                            result.stellar_variance, inputs.stellar_rv_kms):
        # Keep flux and variance on the same shifted wavelength coordinates.
        y, v = gap_resample(result.stellar_wave_nm * (1 + rv / C_KMS),
                            row, var, np.asarray(wave_um) * 1000)
        prepared.append(y)
        # Convert the resampled variance back to a standard deviation for the CCF runner.
        propagated.append(np.sqrt(v))
    prepared, propagated = map(np.asarray, (prepared, propagated))
    # The current CCF interface selects columns common to every exposure.
    # Preserve the individual moving masks in the direct-transmission API.
    # Select columns that have invalid data or errors in any exposure.
    bad = np.flatnonzero(np.any(~np.isfinite(prepared) | ~np.isfinite(propagated)
                                | (propagated <= 0), axis=0))
    # Retain the common valid wavelength columns used by the CCF interface.
    useful = np.setdiff1d(np.arange(prepared.shape[1]), bad)
    if useful.size == 0:
        raise ValueError("Seidel22 has no common valid wavelength columns after preparation")
    # Fill excluded columns with neutral values. The returned mask keeps them
    # out of the subsequent calculation.
    prepared[:, bad] = 1.0
    propagated[:, bad] = 1.0
    # Return either the requested arrays or the runner's full preparation record.
    if not masks:
        return (prepared, propagated) if correct_uncertainties else prepared
    return prepared, propagated, useful, bad, 0, bad.copy(), useful.copy(), None, None


def remove_wiggles(wave_nm, transmission, variance, *,
                   continuum_windows_nm, period_bounds_nm):
    """Fit a sinusoid to selected continuum windows of a transmission spectrum.

    Call this on the final planet-frame spectrum when interference is present.
    Choose continuum windows that exclude the planetary lines and inspect the
    returned baseline. Variances are conditional on the fitted baseline.
    The period search uses the explicit bounds in nm and several initial
    periods, since a sinusoid can have multiple local fitting minima.
    """
    # The optional wiggle fit uses weighted nonlinear least squares.
    from scipy.optimize import least_squares
    # Read one final transmission spectrum and its diagonal variance.
    wave, flux, var = map(lambda x: np.asarray(x, float),
                          (wave_nm, transmission, variance))
    if wave.ndim != 1 or wave.shape != flux.shape or wave.shape != var.shape:
        raise ValueError("Wiggle correction requires matching one-dimensional arrays")
    if np.any(~np.isfinite(wave)) or np.any(np.diff(wave) <= 0):
        raise ValueError("Wiggle wavelengths must be finite and increasing")
    # The user supplies the allowed interference periods in wavelength units.
    bounds = np.asarray(period_bounds_nm, float)
    if bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
        raise ValueError("Provide positive increasing sinusoid period bounds in nm")
    # Fit only finite positive-variance channels inside the declared continuum windows.
    good = window_mask(wave, continuum_windows_nm) & np.isfinite(flux) & np.isfinite(var) & (var > 0)
    if good.sum() < 10 or np.ptp(wave[good]) < bounds[0]:
        raise ValueError("Insufficient continuum coverage for the sinusoidal fit")
    # Centre wavelengths near zero to keep the sinusoidal fit numerically well scaled.
    x = wave - np.mean(wave[good])
    # Represent the baseline as an offset plus sine and cosine at a shared period.
    def baseline(parameters):
        offset, sine, cosine, period = parameters
        angle = 2 * np.pi * x / period
        return offset + sine * np.sin(angle) + cosine * np.cos(angle)
    # Divide continuum residuals by their standard deviations for the least-squares fit.
    def residual(parameters):
        return (baseline(parameters)[good] - flux[good]) / np.sqrt(var[good])
    # Collect converged fits from the different starting periods.
    fits = []
    # Search 32 initial periods because the objective can contain several minima.
    for period in np.linspace(bounds[0], bounds[1], 32):
        angle = 2 * np.pi * x[good] / period
        # At each starting period, solve the offset, sine and cosine coefficients linearly.
        design = np.column_stack([np.ones(good.sum()), np.sin(angle), np.cos(angle)])
        # Weight the linear starting fit by each continuum pixel's inverse standard deviation.
        coefficients = np.linalg.lstsq(design / np.sqrt(var[good, None]),
                                      flux[good] / np.sqrt(var[good]), rcond=None)[0]
        # Refine the three coefficients and period within the user's period bounds.
        fit = least_squares(residual, [*coefficients, period],
                            bounds=([-np.inf]*3 + [bounds[0]],
                                    [np.inf]*3 + [bounds[1]]))
        if fit.success:
            fits.append(fit)
    if not fits:
        raise ValueError("The sinusoidal continuum fit did not converge")
    # Choose the converged fit with the smallest weighted sum of squared residuals.
    best = min(fits, key=lambda fit: np.sum(fit.fun**2))
    # Evaluate the selected baseline over the full wavelength interval.
    model = baseline(best.x)
    if np.any(~np.isfinite(model)) or np.any(model <= 0):
        raise ValueError("Sinusoidal fit produced a non-positive continuum")
    # Divide by the fitted baseline and propagate the conditional variance
    # through the same fixed divisor. Return the baseline for visual inspection.
    return dict(transmission=flux/model, conditional_variance=var/model**2,
                baseline=model, period_nm=float(best.x[-1]))
