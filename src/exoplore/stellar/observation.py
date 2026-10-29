"""Connect factorized local stellar intensities to the observing calculation.

The model supplied to the runner is a continuum-normalized photospheric
distortion. Atmospheric transmission and BATMAN white-light weighting remain
separate factors. The spatial integrator in coupled.py shares the input model
and adds joint local stellar and atmospheric ray transport.


The standard simulator constructs TransitStellarModel for the factorized
calculation. Its output supplies a stellar reference and an exposure-dependent
RM+CLV ratio. Seidel22 can prepare the opaque-body spectra alongside the data
before applying their stellar correction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .surface import C_KMS, LocalIntensityGrid, StellarSurface
from exoplore.io.stellar_surface import load_local_intensities


def validate_route(config, route):
    """Reject unsupported observing routes before loading costly inputs."""
    # Read and validate the stellar-effect controls selected in the JSON.
    choices = config.rm_clv
    choices.__post_init__()
    # Leave the existing observing calculation in place when both switches are false.
    if not choices.active:
        return
    # Check the physical inputs needed for local stellar and planetary ray integration.
    if choices.coupling == "spatial":
        if choices.wavelength_medium != "vacuum":
            raise ValueError("Spatial pRT ray transfer currently requires vacuum wavelength grids")
        if not choices.simulate:
            raise ValueError("Spatial RM/CLV currently requires simulate=true for a forward simulation")
        if config.atmosphere.limb_asymmetries:
            raise ValueError("Spatial RM/CLV currently supports a 1D atmosphere. Set limb_asymmetries=false")
        if config.observation.external_planet_model:
            raise ValueError("Spatial RM/CLV requires pRT atmospheric columns. Disable external_planet_model")
        if config.retrieval.enabled:
            raise ValueError("Spatial RM/CLV retrieval needs joint trial spectra. Disable retrieval for this forward mode")
        if config.observation.scale_injection != 1:
            raise ValueError("Spatial RM/CLV requires scale_injection=1 for physical ray transmissions")
        if not config.observation.signal_uses_light_curve:
            raise ValueError("Spatial RM/CLV calculates transit geometry and requires signal_uses_light_curve=true")
    # Check that this observing runner can consume the selected stellar calculation.
    if route not in ("standard", "atomic_simulation"):
        raise ValueError("RM/CLV requires the standard CCF or direct atomic runner")
    # This surface occultation geometry describes a planetary transit.
    if config.observation.event_type != "transit":
        raise ValueError("RM/CLV currently requires observation.event_type='transit'")
    if config.observation.significant_eccentricity:
        raise ValueError("RM/CLV currently supports circular transit geometry")
    # Measured spectra use the correction switch. Injection belongs to synthetic observations.
    if choices.simulate and config.observation.use_real_data:
        raise ValueError("Use rm_clv.correct for measured data. rm_clv.simulate requires synthetic data")
    # Detector splitting needs matching wavelength handling in the stellar model.
    if config.instrument.split_detectors:
        raise ValueError("RM/CLV detector splitting requires a dedicated wavelength-grid adapter")
    if route == "standard" and config.cross_correlation.ccf_template_source != "planet":
        raise ValueError("RM/CLV currently requires cross_correlation.ccf_template_source='planet'")
    # Require the selected local stellar-intensity input before starting a run.
    if not Path(choices.model_file).is_file():
        raise FileNotFoundError(f"RM/CLV specific-intensity bundle not found: {choices.model_file}")




@dataclass(frozen=True)
class ObservedStellarModel:
    """Instrument-sampled line distortion and stationary stellar reference."""

    # Continuum-normalized stellar line ratio for each exposure and wavelength pixel.
    distortion: np.ndarray
    # Observer-frame unocculted stellar spectrum on the same wavelength grid.
    reference: np.ndarray


class TransitStellarModel:
    """Build the selected stellar effect once, then evaluate each order/night."""

    def __init__(self, choices, planet, resolving_power):
        # Load the specific intensities, prior instrumental resolution and synthesis metadata.
        local, source_resolution, self.metadata = load_local_intensities(
            choices.model_file, wavelength_medium=choices.wavelength_medium)
        # Retain the input broadening for the subsequent resolution calculation.
        self.source_resolving_power = source_resolution
        # Check orbit and instrument assumptions before building the surface.
        if planet.eccentricity != 0:
            raise ValueError("RM/CLV currently supports eccentricity=0")
        if resolving_power <= 0 or not np.isfinite(resolving_power):
            raise ValueError("RM/CLV needs a positive instrument resolving power")
        # An already broadened synthesis must have sufficient resolution for the instrument.
        if source_resolution > 0 and source_resolution < resolving_power:
            raise ValueError("RM/CLV synthesis resolution is lower than the instrument resolution")
        # The continuum-normalized correction requires local continuum intensities.
        if choices.normalization == "continuum" and local.continuum is None:
            raise ValueError("Supply local continuum intensities or explicitly select band_mean normalization")
        # For the RM-only experiment, keep the central normalized line shape
        # and retain the angular variation of continuum brightness.
        if choices.effects == "rm":
            if local.continuum is None:
                raise ValueError("RM-only mode needs continuum intensities to retain continuum limb darkening")
            if local.mu[-1] != 1:
                raise ValueError("RM-only mode needs a disk-centre spectrum at mu=1")
            # Keep the centre line shape everywhere while retaining each
            # angle's continuum brightness. This isolates rotation from line CLV.
            shape = local.intensity[-1] / local.continuum[-1]
            local = LocalIntensityGrid(local.wavelength_um, local.mu,
                                       local.continuum*shape, local.continuum)
        # The CLV-only experiment sets rotation to zero. Other choices use the catalog vsini.
        speed = 0.0 if choices.effects == "clv" else planet.v_rotsini_kms
        # Build projected surface coordinates and their rotation velocities once.
        self.surface = StellarSurface(
            local, vsini_kms=speed,
            projected_obliquity_deg=(choices.projected_obliquity_deg or 0.0),
            cells_across_diameter=choices.cells_across_diameter,
            subpixels_per_axis=choices.subpixels_per_axis)
        # Keep the geometry and JSON choices for order/night evaluations.
        self.choices, self.planet = choices, planet
        self.resolving_power = resolving_power
        # Add only the instrumental broadening missing from the input synthesis.
        # Gaussian broadening variances add. Calculate the extra width needed
        # to bring the synthesis to the requested instrumental resolution.
        inverse_r2 = 1.0/resolving_power**2
        if source_resolution > 0:
            inverse_r2 -= 1.0/source_resolution**2
        # Express the remaining Gaussian standard deviation in log wavelength.
        self.sigma_log_wave = np.sqrt(max(0.0, inverse_r2)) / np.sqrt(8*np.log(2))

    def evaluate(self, wave_um, phase, stellar_velocity_kms, exposure_seconds):
        """Average flux during exposures, convolve, then form observer-frame ratios.

        Phase zero is mid-transit for the circular orbit. Uniform midpoint
        samples cover each exposure. The stellar observer-frame velocity is
        supplied by the existing runner, including its systemic/BERV convention.
        """
        # Read one instrument order and its exposure phases in consistent floating-point arrays.
        wave = np.asarray(wave_um, dtype=float)
        phase = np.asarray(phase, dtype=float)
        # Allow either one common observed stellar velocity or one per exposure.
        velocity = np.broadcast_to(np.asarray(stellar_velocity_kms, dtype=float), phase.shape)
        # Validate the wavelength grid, exposure phases and observed velocities.
        if (wave.ndim != 1 or len(wave) < 2 or not np.isfinite(wave).all()
                or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
            raise ValueError("RM/CLV observing wavelengths must increase and be positive")
        if phase.ndim != 1 or not len(phase) or not np.isfinite(phase).all():
            raise ValueError("RM/CLV phases must be a nonempty finite vector")
        if not np.isfinite(velocity).all() or np.any(abs(velocity) >= C_KMS):
            raise ValueError("RM/CLV stellar velocities must be finite and subluminal")
        if not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
            raise ValueError("RM/CLV exposure time must be positive")
        # Read the circular transit geometry stored with the model.
        planet = self.planet
        # Use the configured number of temporal samples across each integration.
        count = self.choices.exposure_subsamples
        # Place midpoint samples uniformly across the exposure duration in seconds.
        offsets = ((np.arange(count) + 0.5)/count - 0.5)*exposure_seconds
        # Convert those time offsets to orbital phase and add each exposure midpoint.
        sampled_phase = phase[:, None] + offsets/(86400*planet.orbital_period_days)
        # Flatten the exposure/sample grid and convert orbital phase to radians.
        angle = 2*np.pi*sampled_phase.ravel()
        # Convert the semimajor axis from km to stellar-radius units using R_star in cm.
        scaled_a = planet.a/(planet.R_star/1e5)
        # The projected x coordinate follows the planet along its transit chord.
        x = scaled_a*np.sin(angle)
        # The projected y coordinate includes orbital inclination and the chosen sky orientation.
        y = -scaled_a*np.cos(angle)*np.cos(np.deg2rad(planet.inclination_deg))
        # The opaque body is behind the star during the other half of the orbit.
        x[np.cos(angle) <= 0] = scaled_a + 2.0
        # Use the opaque planet-to-star radius ratio for surface occultation.
        ratio = planet.R_pl/planet.R_star
        # Work on a log-wavelength grid so a Gaussian instrument profile has
        # constant resolving power across the order.
        # The local synthesis sets the finest available wavelength sampling.
        local_wave = self.surface.local.wavelength_um
        # Use the finer of the synthesis spacing and four samples per resolution element.
        log_step = min(float(np.min(np.diff(np.log(local_wave)))),
                       1.0/(4*self.resolving_power))
        # Find stellar-rest-frame coverage over the range of observer-frame stellar velocities.
        shifted_min = wave[0]/(1 + velocity.max()/C_KMS)
        shifted_max = wave[-1]/(1 + velocity.min()/C_KMS)
        # Add eight Gaussian standard deviations and two samples around the order.
        padding = 8*self.sigma_log_wave + 2*log_step
        low, high = np.log(shifted_min) - padding, np.log(shifted_max) + padding
        # Choose enough points to retain the requested fine-grid spacing.
        n_wave = int(np.ceil((high - low)/log_step)) + 1
        # Construct the uniform log-wavelength grid used for constant-R convolution.
        fine_wave = np.exp(np.linspace(low, high, n_wave))
        # Record its actual spacing after rounding the number of wavelength samples.
        step = (high - low)/(n_wave - 1)
        # Integrate the star at every instantaneous planet position.
        spectra = self.surface.integrate(fine_wave, x, y, ratio)

        # Apply the same remaining instrumental Gaussian to each integrated flux array.
        def convolved(values):
            if self.sigma_log_wave == 0:
                return values
            return gaussian_filter1d(values, self.sigma_log_wave/step, axis=-1,
                                     mode="nearest", truncate=8.0)

        # Average fluxes before normalization. Averaging already normalized
        # ratios would weight ingress and egress differently.
        # Average the subexposure fluxes at each wavelength for each exposure.
        inside = spectra.inside.reshape(len(phase), count, n_wave).mean(axis=1)
        # Convolve the averaged in-transit spectrum and unocculted reference separately.
        inside = convolved(inside)
        outside = convolved(spectra.outside)
        # Average and convolve the corresponding continua before normalizing the spectra.
        if self.choices.normalization == "continuum":
            continuum_in = spectra.continuum_inside.reshape(len(phase), count, n_wave).mean(axis=1)
            inside /= convolved(continuum_in)
            outside /= convolved(spectra.continuum_outside)
        if np.any(outside <= 0) or not np.isfinite(inside).all():
            raise ValueError("RM/CLV synthesis produced invalid stellar flux")
        # Allocate the stellar distortion and reference on the requested detector grid.
        model = np.empty((len(phase), len(wave)))
        reference = np.empty_like(model)
        # Apply the observed stellar velocity of each exposure to the integrated spectra.
        for exposure, speed in enumerate(velocity):
            # Map detector wavelengths into the stellar rest frame.
            rest = wave/(1 + speed/C_KMS)
            # Sample the normalized out-of-transit reference at those wavelengths.
            reference[exposure] = np.interp(rest, fine_wave, outside)
            # Sample this exposure's normalized occulted spectrum at the same wavelengths.
            observed_in = np.interp(rest, fine_wave, inside[exposure])
            # Form the photospheric line distortion as in-transit divided by reference.
            model[exposure] = observed_in/reference[exposure]
            # The alternative normalization divides each factor by its wavelength mean.
            if self.choices.normalization == "band_mean":
                model[exposure] /= np.mean(model[exposure])
                reference[exposure] /= np.mean(reference[exposure])
        if np.any(model <= 0) or not np.isfinite(model).all():
            raise ValueError("RM/CLV correction must be finite and positive")
        # Return the observer-frame factor and stellar reference to the caller.
        return ObservedStellarModel(model, reference)


def correct_stellar_distortion(flux, uncertainty, distortion):
    """Divide flux and its conditional uncertainty by the same fixed model.

    Uncertainty in the stellar model is outside this deterministic correction.
    The supplied uncertainty belongs to the flux before the division.
    """
    # Read matching exposure-by-wavelength flux, error and fixed correction arrays.
    flux, uncertainty, distortion = map(np.asarray, (flux, uncertainty, distortion))
    # Check that every datum has a corresponding error and correction factor.
    if flux.shape != uncertainty.shape or flux.shape != distortion.shape:
        raise ValueError("RM/CLV flux, uncertainty and model shapes must match")
    if not np.isfinite(distortion).all() or np.any(distortion <= 0):
        raise ValueError("RM/CLV distortion must be finite and positive")
    # Division by a fixed factor scales flux and standard deviation by that same factor.
    return flux/distortion, uncertainty/distortion
