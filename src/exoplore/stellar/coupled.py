"""Transport local stellar light through a spherical planetary atmosphere.

Atmospheric rays use the native hydrostatic pRT radii. Polar quadrature around
the planet resolves the thin absorbing annulus, including its overlap with
the stellar limb. The instrument response is applied to the integrated flux.


SpatialTransitModel is called by the molecular simulator and the direct
sodium observation builder. It inherits stellar-input checks from
TransitStellarModel. The integration below calculates the unocculted star,
the opaque-body transit and the atmosphere-absorbed flux on a common grid.

Sky-plane distances use stellar radii. Atmospheric input radii use cm.
Wavelengths use micrometres and velocities use km/s. Flux arrays have shape
(exposure, wavelength). Each integral retains local continuum brightness.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import gaussian_filter1d

from exoplore.atmosphere.transmission import ChordTransmission
from .observation import TransitStellarModel
from .surface import C_KMS


@njit(cache=True)
def _integrate_local_intensities(wave, active, upper, mu_fraction, velocity,
                                 weights, local_wave, values, continua, count):
    """Interpolate each ray and accumulate its original angular weight.

    The loops use the same wavelength and mu interpolation as the surface
    model. Accumulating immediately avoids allocating several arrays with
    one row per atmospheric ray for every wavelength chunk. Numba compiles
    these scalar operations with the specified arithmetic and accumulation order.
    """
    # Allocate one angularly integrated spectrum for each input ring.
    intensity = np.zeros((count, len(wave)))
    # Accumulate the continuum with the same ray geometry and weights.
    continuum = np.zeros_like(intensity)
    # Every illuminated ring uses the configured angular quadrature count.
    angles = weights.shape[1]
    # Visit the rings whose illuminated arcs overlap the stellar disk.
    for ring in range(len(active)):
        # Evaluate the stellar intensity at each angular quadrature point.
        for angle in range(angles):
            # Map this ring/angle pair to the flattened ray-coordinate arrays.
            ray = ring*angles+angle
            # Read the bracketing stellar synthesis angle and angular interpolation fraction.
            mu_upper = upper[ray]
            mu_weight = mu_fraction[ray]
            # Convert the local projected stellar rotation velocity to a wavelength factor.
            doppler = 1+velocity[ray]/C_KMS
            # This quadrature weight includes the illuminated arc width.
            weight = weights[ring, angle]
            # Start the wavelength bracket search at the first synthesis interval.
            right = 1
            # Sweep the requested wavelengths for this local stellar ray.
            for channel in range(len(wave)):
                # Evaluate the synthesis at the ray's local rest wavelength.
                rest = wave[channel]/doppler
                # Successive wavelengths usually share neighbouring interpolation
                # brackets, so continue searching from the preceding interval.
                # Start a full bracket search for the first channel or a decrease.
                if channel == 0 or rest < local_wave[right-1]:
                    right = min(max(np.searchsorted(local_wave, rest, side="right"),
                                    1), len(local_wave)-1)
                else:
                    while right < len(local_wave)-1 and rest >= local_wave[right]:
                        right += 1
                # Measure the wavelength interpolation fraction inside the selected bracket.
                fraction = ((rest-local_wave[right-1])
                            /(local_wave[right]-local_wave[right-1]))
                # Interpolate the line intensity at the lower synthesis mu.
                lower = (values[mu_upper-1, right-1]*(1-fraction)
                         + values[mu_upper-1, right]*fraction)
                # Repeat the wavelength interpolation at the upper synthesis mu.
                higher = (values[mu_upper, right-1]*(1-fraction)
                          + values[mu_upper, right]*fraction)
                # Interpolate between the two mu profiles, multiply by angular area weight,
                # and add to this ring's line spectrum.
                intensity[active[ring], channel] += (
                    ((1-mu_weight)*lower+mu_weight*higher)*weight)
                # Apply the same wavelength interpolation to the local continuum.
                lower = (continua[mu_upper-1, right-1]*(1-fraction)
                         + continua[mu_upper-1, right]*fraction)
                higher = (continua[mu_upper, right-1]*(1-fraction)
                          + continua[mu_upper, right]*fraction)
                # Add the angularly weighted continuum with the same mu interpolation.
                continuum[active[ring], channel] += (
                    ((1-mu_weight)*lower+mu_weight*higher)*weight)
    return intensity, continuum


@dataclass(frozen=True)
class IntrinsicCoupledObservation:
    """Disk-integrated spectra before the instrumental response, in common units.

    A telluric observation builder multiplies these spectra by Earth's
    transmission before convolving and sampling the detector spectrum.
    continuum is the out-of-transit stellar continuum used for normalization.
    """

    # Fine observer-frame wavelengths before the instrument response.
    wavelength_um: np.ndarray
    # Total flux including planetary atmospheric absorption at each exposure.
    flux: np.ndarray
    # Flux with the opaque planet blocking the star, for the stellar correction.
    opaque_flux: np.ndarray
    # Unocculted disk-integrated stellar spectrum at each exposure velocity.
    stellar_flux: np.ndarray
    # Unocculted stellar continuum used as the common normalization scale.
    continuum: np.ndarray


@dataclass(frozen=True)
class CoupledObservation:
    """Instrument-sampled joint flux and its opaque-body stellar diagnostic.

    flux contains the complete transit, normalized by the out-of-transit
    continuum. distortion describes the opaque-body photospheric line change.
    Dividing by distortion preserves the planetary atmospheric absorption.
    It is a fixed photospheric correction, conditional on the adopted model.
    The remaining atmospheric term retains its local stellar illumination.
    Its line shape can therefore differ from an isolated 1D planet template.
    """

    # Instrument-sampled total flux divided by the unocculted stellar continuum.
    flux: np.ndarray
    # Unocculted stellar spectrum on the same continuum-relative scale.
    reference: np.ndarray
    # Opaque-body line-profile ratio after removing its continuum transit depth.
    distortion: np.ndarray
    # Opaque-body continuum transit relative to the unocculted continuum.
    opaque_transmission: np.ndarray
    # Deepest opaque radius and outermost atmospheric radius represented by the model.
    base_radius_cm: float
    top_radius_cm: float
    # Optionally retain fine-grid fluxes for telluric absorption before convolution.
    intrinsic: IntrinsicCoupledObservation | None = None


def _disk_rings(radius, count):
    """Gauss quadrature for the radial area element r dr of an opaque disk."""
    # Obtain Gauss-Legendre nodes and weights on the interval [-1, 1].
    nodes, weights = leggauss(count)
    # Map the nodes into squared radius on [0, 1], where area integration is simple.
    squared_radius = (nodes + 1)/2
    # Convert back to radius. The factor radius squared / 4 follows from
    # r dr = d(r squared)/2 and the interval transformation.
    return radius*np.sqrt(squared_radius), radius**2*weights/4


def _occulting_disk_rings(radius, count, distance):
    """Split radial quadrature where rings first intersect the stellar edge.

    At ingress and egress, the visible angular arc changes behaviour at
    r=abs(1-distance). Integrating the two radial intervals separately improves
    convergence of the opaque-body flux near that boundary.
    """
    # A ring changes from wholly inside/outside the star to partial overlap here.
    transition = abs(1-distance)
    # Use a single disk quadrature when the transition lies outside the opaque planet.
    if transition <= 0 or transition >= radius:
        return _disk_rings(radius, count)
    nodes, weights = leggauss(count)
    # Collect radial nodes and area weights separately for the two overlap intervals.
    radii, area_weights = [], []
    # Integrate either side of the stellar-edge intersection with its own quadrature.
    for lower, upper in ((0., transition), (transition, radius)):
        # Map each interval's nodes into squared radius.
        squared = lower**2+(nodes+1)/2*(upper**2-lower**2)
        # Recover physical sky-plane radii from the squared-radius nodes.
        radii.append(np.sqrt(squared))
        # Scale the quadrature weights to the radial area element of this interval.
        area_weights.append((upper**2-lower**2)*weights/4)
    # Return one radial grid for the occulting body and its matching area weights.
    return np.concatenate(radii), np.concatenate(area_weights)


class SpatialTransitModel(TransitStellarModel):
    """Joint stellar and 1D atmospheric transfer for a circular transit.

    Stellar rotation is solid body. The atmosphere is spherical and its bulk
    velocity is supplied by the observing runner. The circular orbital velocity
    and stellar reflex velocity are sampled within each exposure. BERV and
    bulk wind are held at their supplied exposure values.
    """

    def __init__(self, choices, planet, resolving_power):
        # Reuse the local-intensity loading, effect selection and instrument checks.
        super().__init__(choices, planet, resolving_power)
        # The inherited surface holds the validated angular stellar synthesis.
        local = self.surface.local
        # Spatial absorption uses the continuum brightness across the stellar disk.
        if choices.normalization != "continuum" or local.continuum is None:
            raise ValueError("Spatial RM/CLV requires continuum normalization and local continuum spectra")
        if local.mu[-1] != 1:
            raise ValueError("Spatial RM/CLV needs a disk-centre spectrum at mu=1")
        # Local stellar light and atmospheric transmission are combined at
        # intrinsic resolution, requiring an unconvolved stellar synthesis.
        if self.source_resolving_power != 0:
            raise ValueError("Spatial RM/CLV requires stellar resolving_power=0 in the input bundle")
        # Choose the projected stellar rotation, or zero rotation for the CLV-only control.
        self.vsini_kms = 0.0 if choices.effects == "clv" else planet.v_rotsini_kms
        # Store the projected spin-orbit angle in radians for the sky-plane rotation.
        self.obliquity = np.deg2rad(choices.projected_obliquity_deg or 0)
        # Precompute the angular quadrature reused by every star and planet ring.
        self._angle_nodes, self._angle_weights = leggauss(choices.azimuth_points)

    def _rays(self, radii, planet_x, planet_y):
        """Sample only the visible angular arc of each planet-centred ring."""
        # Measure the ring centre's projected distance from stellar centre.
        distance = np.hypot(planet_x, planet_y)
        # For concentric rings, each radius is either fully illuminated or outside the star.
        if distance == 0:
            widths = np.where(radii < 1, 2*np.pi, 0.0)
            starts = np.zeros_like(radii)
        else:
            # Solve the two-circle intersection for the angular interval lying on the star.
            limit = (1-distance**2-radii**2)/(2*distance*radii)
            # Clip the cosine before arccos to cover fully inside and fully outside rings.
            cut = np.arccos(np.clip(limit, -1, 1))
            # Orient the illuminated arc according to the planet's position on the sky.
            starts = np.arctan2(planet_y, planet_x) + cut
            # The remaining angular width is the illuminated portion of each ring.
            widths = 2*(np.pi-cut)
        # Keep rings with a nonzero illuminated arc.
        active = np.flatnonzero(widths > 0)
        # Return an empty contribution when all rings lie outside the stellar disk.
        if not len(active):
            return None
        # Map the Gauss-Legendre nodes onto each surviving illuminated arc.
        angles = (starts[active, None] + widths[active, None]
                  *(self._angle_nodes[None, :]+1)/2)
        # Convert the planet-centred polar ray coordinates to stellar sky coordinates.
        x = planet_x + radii[active, None]*np.cos(angles)
        y = planet_y + radii[active, None]*np.sin(angles)
        # Recover the local stellar viewing angle at every illuminated ray.
        mu = np.sqrt(np.maximum(1-x*x-y*y, 0)).ravel()
        local = self.surface.local
        # Synthesis programs usually export a small positive minimum mu.
        # Require their angular coverage to include every sampled ray.
        if mu.min() < local.mu[0] or mu.max() > local.mu[-1]:
            raise ValueError(
                f"Stellar intensities cover mu={local.mu[0]:.6g} to {local.mu[-1]:.6g}, "
                f"but spatial integration samples {mu.min():.6g} to {mu.max():.6g}. "
                "Generate intensities covering these angles. Angular extrapolation is disabled.")
        # Bracket each ray's mu within the supplied synthesis angles.
        upper = np.clip(np.searchsorted(local.mu, mu, side="right"), 1, len(local.mu)-1)
        # Calculate its interpolation fraction between the two synthesis angles.
        fraction = (mu-local.mu[upper-1])/(local.mu[upper]-local.mu[upper-1])
        # Assign the local projected rotation velocity using the spin-orbit orientation.
        velocity = self.vsini_kms*(x*np.cos(self.obliquity)-y*np.sin(self.obliquity))
        # Scale angular quadrature weights by half of each illuminated arc width.
        weights = widths[active, None]/2*self._angle_weights[None, :]
        # Pass geometry and interpolation weights to the wavelength integration.
        return active, upper, fraction, velocity.ravel(), weights

    def _angular_flux(self, wave, rays, count):
        """Integrate intensity over each illuminated angular arc, wavelength by wavelength."""
        # Start with zero angular flux for every ring, including unilluminated rings.
        intensity = np.zeros((count, len(wave)))
        continuum = np.zeros_like(intensity)
        # A ring set entirely outside the star contributes zero line and continuum flux.
        if rays is None:
            return intensity, continuum
        # Unpack the geometry already calculated for this star/planet position.
        active, upper, mu_fraction, velocity, weights = rays
        local = self.surface.local
        # Check the local stellar wavelength range needed by every rotating ray.
        rest = wave[None, :]/(1+velocity[:, None]/C_KMS)
        if rest.min() < local.wavelength_um[0] or rest.max() > local.wavelength_um[-1]:
            raise ValueError("Spatial RM/CLV needs wider local stellar wavelength coverage")
        # Use the compiled interpolation loop to sum the same ray weights at each wavelength.
        return _integrate_local_intensities(
            wave, active, upper, mu_fraction, velocity, weights,
            local.wavelength_um, local.intensity, local.continuum, count)

    @staticmethod
    def _absorption(chords, planet_rest_wave):
        """Doppler-shift the ray transmissions on their supplied atmospheric radii."""
        # The pRT ray table provides an increasing planet-rest-frame wavelength axis.
        wave = chords.wavelength_um
        # Require pRT coverage after applying the planet's Doppler shift.
        if planet_rest_wave.min() < wave[0] or planet_rest_wave.max() > wave[-1]:
            raise ValueError("Spatial RM/CLV needs wider pRT wavelength coverage for planetary velocities")
        # Bracket each requested planet-rest wavelength in the pRT table.
        right = np.clip(np.searchsorted(wave, planet_rest_wave, side="right"), 1, len(wave)-1)
        # Compute the wavelength interpolation fraction for those ray transmissions.
        fraction = (planet_rest_wave-wave[right-1])/(wave[right]-wave[right-1])
        # Interpolate all atmospheric impact parameters with the same wavelength weights.
        transmission = (chords.transmission[right-1]*(1-fraction[:, None])
                        + chords.transmission[right]*fraction[:, None])
        # Convert transmission to absorbed fraction and return axes (radius, wavelength).
        return (1-transmission).T

    def evaluate(self, wave_um, phase, stellar_velocity_kms, exposure_seconds, *,
                 chords: ChordTransmission | None, planet_velocity_kms,
                 return_intrinsic: bool = False):
        """Compute complete fluxes, averaging exposure samples before convolution.

        chords=None selects the atmosphere-free opaque-body control.
        The reference-pressure radius and the opaque bottom radius of a pRT
        atmosphere remain distinct. The latter sets the occulting body here.
        """
        # Require the validated pRT ray product when atmospheric absorption is requested.
        if chords is not None and not isinstance(chords, ChordTransmission):
            raise TypeError("chords must be a ChordTransmission or None for the opaque control")
        # Read the observed wavelengths and exposure midpoint phases.
        wave = np.asarray(wave_um, dtype=float)
        phase = np.asarray(phase, dtype=float)
        if (wave.ndim != 1 or len(wave) < 2 or not np.isfinite(wave).all()
                or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
            raise ValueError("Spatial observing wavelengths must be positive and increasing")
        if phase.ndim != 1 or not len(phase) or not np.isfinite(phase).all():
            raise ValueError("Spatial phases must be a finite nonempty vector")
        # Accept a scalar or one observer-frame stellar and planetary velocity per exposure.
        star_velocity = np.broadcast_to(np.asarray(stellar_velocity_kms, dtype=float), phase.shape)
        planet_velocity = np.broadcast_to(np.asarray(planet_velocity_kms, dtype=float), phase.shape)
        if (not np.isfinite(star_velocity).all() or not np.isfinite(planet_velocity).all()
                or np.any(abs(star_velocity) >= C_KMS) or np.any(abs(planet_velocity) >= C_KMS)):
            raise ValueError("Spatial velocities must be finite and subluminal")
        if not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
            raise ValueError("Spatial exposure duration must be positive")
        # Use the selected number of midpoint samples per integration.
        count = self.choices.exposure_subsamples
        # Place those samples uniformly across the exposure duration in seconds.
        offsets = ((np.arange(count)+0.5)/count-0.5)*exposure_seconds
        # Convert sample time offsets to orbital phase using the period in days.
        subphase = phase[:, None] + offsets/(86400*self.planet.orbital_period_days)
        # Measure the orbital sine change relative to each exposure midpoint.
        sin_difference = np.sin(2*np.pi*subphase)-np.sin(2*np.pi*phase[:, None])
        # Update the stellar reflex velocity across the exposure with its negative orbital sign.
        star_subvelocity = star_velocity[:, None] - self.planet.stellar_rv_semiamplitude_kms*sin_difference
        # Update the planetary orbital velocity with its positive orbital sign.
        planet_subvelocity = planet_velocity[:, None] + self.planet.kp_kms*sin_difference
        local = self.surface.local
        # Choose a fine log-wavelength grid resolving both the stellar synthesis and instrument.
        step = min(float(np.min(np.diff(np.log(local.wavelength_um)))), 1/(4*self.resolving_power))
        # Also retain the atmospheric spectral sampling when it is finer.
        if chords is not None:
            step = min(step, float(np.min(np.diff(np.log(chords.wavelength_um)))))
        # Include convolution margins beyond the observing interval.
        padding = 8*self.sigma_log_wave+2*step
        low, high = np.log(wave[0])-padding, np.log(wave[-1])+padding
        # Construct the fine observer-frame wavelength grid.
        fine = np.exp(np.linspace(low, high, int(np.ceil((high-low)/step))+1))
        # Record its actual spacing for the Gaussian instrumental convolution.
        log_step = np.log(fine[1]/fine[0])
        # Find stellar-rest wavelengths needed by the full range of subexposure velocities.
        star_low = fine[0]/(1+star_subvelocity.max()/C_KMS)
        star_high = fine[-1]/(1+star_subvelocity.min()/C_KMS)
        # Build a padded stellar-rest grid on which the unocculted disk can be integrated once.
        star_grid = np.exp(np.linspace(np.log(star_low)-2*step, np.log(star_high)+2*step,
                           int(np.ceil(np.log(star_high/star_low)/step))+5))
        # Calculate the full disk once on a grid covering every exposure's
        # stellar velocity. Later samples interpolate this reference spectrum.
        # Set star-centred radial quadrature in projected stellar-radius units.
        star_radii, star_weights = _disk_rings(1.0, self.choices.stellar_radial_points)
        # Calculate viewing angles, rotation velocities and angular weights for the full star.
        star_rays = self._rays(star_radii, 0, 0)
        # Allocate the full-star line and continuum spectra.
        disk = np.empty(len(star_grid))
        continuum_disk = np.empty_like(disk)
        # Bound temporary local-intensity arrays by evaluating a small number
        # of wavelength channels at once, independently of the order length.
        chunk = 128
        for start in range(0, len(star_grid), chunk):
            section = slice(start, start+chunk)
            # Integrate the local intensities around each star-centred ring for this wavelength chunk.
            spectra, continua = self._angular_flux(star_grid[section], star_rays, len(star_radii))
            # Apply the radial area weights to obtain the complete stellar spectrum.
            disk[section] = star_weights @ spectra
            # Use the same radial integration for the stellar continuum.
            continuum_disk[section] = star_weights @ continua
        # Use the deepest pRT radius as the opaque bottom. The atmosphere-free control
        # uses the catalog planet radius when no ray table is supplied.
        base_cm = self.planet.R_pl if chords is None else chords.impact_parameter_cm[-1]
        # The outermost pRT impact parameter sets the atmosphere's extent.
        top_cm = base_cm if chords is None else chords.impact_parameter_cm[0]
        # Express the opaque radius in stellar-radius sky coordinates.
        base = base_cm/self.planet.R_star
        # Convert every native pRT impact parameter to stellar-radius coordinates.
        atmosphere_radii = (np.array([]) if chords is None else
                            chords.impact_parameter_cm/self.planet.R_star)
        # Allocate exposure-by-wavelength arrays for unocculted, opaque and atmospheric fluxes.
        shape = (len(phase), len(fine))
        outside, continuum_outside = np.zeros(shape), np.zeros(shape)
        inside, bare, continuum_bare = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        # The public a property uses km and R_star uses cm.
        scaled_a = self.planet.a/(self.planet.R_star/1e5)
        # Average the complete flux calculation over temporal samples in each exposure.
        for exposure in range(len(phase)):
            # Evaluate the projected geometry and local velocities at this subsample.
            for sample in range(count):
                # Read its orbital phase relative to mid-transit.
                phi = subphase[exposure, sample]
                # Compute the projected position along the transit chord.
                x = scaled_a*np.sin(2*np.pi*phi)
                # Compute the perpendicular position using the catalog inclination in radians.
                y = -scaled_a*np.cos(2*np.pi*phi)*np.cos(self.planet.inc)
                # Build opaque-body radial quadrature with the stellar-edge split when needed.
                body_radii, body_weights = _occulting_disk_rings(
                    base, self.choices.planet_radial_points, np.hypot(x, y))
                # Combine the opaque-body rings with the native atmospheric rays for local illumination.
                radii = np.concatenate((body_radii, atmosphere_radii))
                # Remember where the opaque-body rings end in the combined ray arrays.
                n_body = len(body_radii)
                # Only the orbital half with the planet in front of the star can remove stellar light.
                front = np.cos(2*np.pi*phi) > 0
                # Check whether the outer atmospheric radius overlaps the stellar disk.
                overlap = front and np.hypot(x, y) < 1+top_cm/self.planet.R_star
                # Calculate illuminated arcs once for this subsample, then reuse them across wavelengths.
                rays = self._rays(radii, x, y) if overlap else None
                for start in range(0, len(fine), chunk):
                    section = slice(start, start+chunk)
                    # Select this chunk of the fine observer-frame grid.
                    observed_wave = fine[section]
                    # Remove the observed stellar velocity to sample its disk-integrated reference.
                    star_rest = observed_wave/(1+star_subvelocity[exposure, sample]/C_KMS)
                    # Interpolate unocculted line and continuum flux at the required stellar-rest wavelengths.
                    full = np.interp(star_rest, star_grid, disk)
                    full_continuum = np.interp(star_rest, star_grid, continuum_disk)
                    # Start the opaque-body and atmospheric removed-light terms at zero.
                    blocked, blocked_continuum = np.zeros(len(full)), np.zeros(len(full))
                    atmosphere_blocked = np.zeros(len(full))
                    # Integrate the stellar light available behind the planet and its atmosphere.
                    if rays is not None:
                        # Angularly integrate the local stellar spectrum at every body and atmospheric radius.
                        ring_flux, ring_continuum = self._angular_flux(star_rest, rays, len(radii))
                        # Sum the opaque-body ring areas to obtain the blocked stellar line flux.
                        blocked = body_weights @ ring_flux[:n_body]
                        # Calculate the corresponding blocked continuum for the photospheric correction.
                        blocked_continuum = body_weights @ ring_continuum[:n_body]
                        if chords is not None:
                            # Evaluate atmospheric absorption in the planet's frame at this subsample velocity.
                            planet_rest = observed_wave/(1+planet_subvelocity[exposure, sample]/C_KMS)
                            # Interpolate the pRT absorbed fraction for every atmospheric radius.
                            absorbed = self._absorption(chords, planet_rest)
                            # Radial trapezoidal integration matches pRT. Angular
                            # integration weights each ray by its actual local
                            # stellar spectrum, including occultation at the limb.
                            # Multiply absorbed fraction, annular radius and the illuminating local stellar flux.
                            integrand = absorbed*atmosphere_radii[:, None]*ring_flux[n_body:]
                            # Sum radial trapezoids. The pRT radii run from top to
                            # bottom, so minus their differences gives positive widths.
                            atmosphere_blocked = np.sum(
                                (integrand[1:]+integrand[:-1])
                                *(-np.diff(atmosphere_radii))[:, None]/2, axis=0)
                    # Add this temporal sample with weight 1 / count to every flux component.
                    outside[exposure, section] += full/count
                    continuum_outside[exposure, section] += full_continuum/count
                    # The opaque-body transit removes the disk-blocked flux from the full star.
                    bare[exposure, section] += (full-blocked)/count
                    continuum_bare[exposure, section] += (full_continuum-blocked_continuum)/count
                    # The complete observation also removes light absorbed through the atmospheric annulus.
                    inside[exposure, section] += (full-blocked-atmosphere_blocked)/count

        # Optionally preserve the exposure-averaged intrinsic flux components.
        # The sodium builder uses them to multiply tellurics before instrumental convolution.
        intrinsic = (IntrinsicCoupledObservation(fine.copy(), inside.copy(),
            bare.copy(), outside.copy(), continuum_outside.copy())
            if return_intrinsic else None)

        # Apply the instrument's Gaussian response along wavelength in each exposure.
        def convolve_flux(values):
            return gaussian_filter1d(values, self.sigma_log_wave/log_step, axis=-1,
                                     mode="nearest", truncate=8)

        # Convolve all line and continuum components consistently before taking ratios.
        outside, continuum_outside, inside, bare, continuum_bare = map(
            convolve_flux, (outside, continuum_outside, inside, bare, continuum_bare))
        # Check positive finite fluxes before continuum division and detector sampling.
        if (not all(np.isfinite(values).all() for values in
                    (inside, outside, continuum_outside, bare, continuum_bare))
                or np.any(inside <= 0) or np.any(outside <= 0)
                or np.any(continuum_outside <= 0) or np.any(continuum_bare <= 0)):
            raise ValueError("Spatial transfer produced invalid flux. Check model and quadrature convergence")
        # Sample the convolved fluxes before forming normalized observables.
        # Keeping this order also preserves a common continuum for the planet,
        # the opaque-body diagnostic and the out-of-transit reference.
        # Interpolate each convolved exposure onto the requested observing wavelengths.
        def sample(values):
            return np.array([np.interp(wave, fine, row) for row in values])

        outside, continuum_outside, inside, bare, continuum_bare = map(
            sample, (outside, continuum_outside, inside, bare, continuum_bare))
        # Put the complete observation on the unocculted stellar-continuum scale.
        flux = inside/continuum_outside
        # Normalize the unocculted stellar spectrum by that same continuum.
        reference = outside/continuum_outside
        # Remove the opaque-body continuum transit from its line spectrum,
        # then divide by the unocculted reference to obtain the photospheric distortion.
        distortion = (bare/continuum_bare)/reference
        # Record the opaque-body continuum transit separately for diagnostics.
        opaque_transmission = continuum_bare/continuum_outside
        # Return the total observation, stellar correction components and atmospheric radius limits.
        return CoupledObservation(flux, reference, distortion, opaque_transmission,
                                  base_cm, top_cm, intrinsic)
