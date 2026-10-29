"""Integrate rotating stellar specific intensities during opaque occultations.

The disk integration follows Casasayas-Barris et al. (2020),
https://doi.org/10.1051/0004-6361/201937221.
Local intensity depends on wavelength and mu, the cosine
of the angle between the surface normal and the observer. Rotation shifts each
surface element before its contribution is integrated.

Coordinates are in stellar radii on the sky. Positive local velocity is a
redshift. Wavelengths are in microns and velocities are in km/s. The x direction
follows the projected orbital motion. The rotation coordinate is
x*cos(projected_obliquity) - y*sin(projected_obliquity).


io/stellar_surface.py constructs LocalIntensityGrid from the stellar bundle.
TransitStellarModel in observation.py uses StellarSurface for the factorized
RM+CLV calculation. The spatial calculation in coupled.py shares the intensity
grid and evaluates its own star-centred and planet-centred rings.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

# Speed of light in the km/s units used for all Doppler shifts here.
C_KMS = 299792.458


def _finite_array(value, name, ndim):
    """Check array shape and finite values at the model boundary."""
    # Convert coordinates to floating point before checking dimensions and finiteness.
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


@dataclass(frozen=True)
class LocalIntensityGrid:
    """Specific intensities with an explicit angular grid and continuum.

    intensity and optional continuum have shape (n_mu, n_wavelength). Both must have
    the same physical units and retain the relative brightness between angles.
    continuum describes the local continuum with the spectral lines removed.
    It allows continuum normalization after occultation and convolution.

    The caller supplies the continuum from the stellar synthesis. Normalizing
    each angular spectrum separately would discard the surface brightness
    information needed to calculate the blocked light.
    """

    # Increasing local-synthesis wavelengths, in micrometres.
    wavelength_um: np.ndarray
    # Increasing viewing-angle cosines, from the limb towards disk centre.
    mu: np.ndarray
    # Local specific intensity at each angle and wavelength, in shared physical units.
    intensity: np.ndarray
    # Matching local continuum needed for continuum-normalized stellar corrections.
    continuum: np.ndarray | None = None

    def __post_init__(self):
        # Validate the supplied coordinates and intensity arrays together.
        wave = _finite_array(self.wavelength_um, "wavelength_um", 1)
        mu = _finite_array(self.mu, "mu", 1)
        intensity = _finite_array(self.intensity, "intensity", 2)
        continuum = (None if self.continuum is None else
                     _finite_array(self.continuum, "continuum", 2))
        # Require ordered wavelengths so every later interpolation has a valid interval.
        if len(wave) < 2 or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError("wavelength_um must be positive and strictly increasing")
        # The viewing-angle grid must run from lower to higher mu within the stellar disk.
        if (len(mu) < 2 or np.any(np.diff(mu) <= 0)
                or mu[0] < 0 or mu[-1] > 1):
            raise ValueError("mu must increase strictly within [0, 1]")
        # Require one complete spectrum for each viewing angle.
        if (intensity.shape != (len(mu), len(wave))
                or (continuum is not None and continuum.shape != intensity.shape)):
            raise ValueError("intensity and continuum must have shape (n_mu, n_wavelength)")
        # Keep physical intensities nonnegative and supplied continua positive.
        if np.any(intensity < 0) or (continuum is not None and np.any(continuum <= 0)):
            raise ValueError("intensity must be nonnegative and continuum must be positive")
        # Store independent, fixed copies of the synthesis arrays in the model.
        for name, array in (("wavelength_um", wave), ("mu", mu),
                            ("intensity", intensity), ("continuum", continuum)):
            if array is None:
                continue
            # Detach each stored array from the caller's memory.
            array = array.copy()
            # Protect the input grid during repeated model evaluations.
            array.setflags(write=False)
            # Assign the validated copy while constructing this frozen dataclass.
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class StellarTransitSpectra:
    """Disk-integrated intensities before instrumental convolution.

    outside has shape (n_wavelength,), inside has shape (n_exposures,
    n_wavelength). Continuum arrays have the corresponding shapes. Projected
    cell areas are included, with the stellar radius set to unity.

    Keep these fluxes separate until the instrument response has been applied.
    Convolution and division generally give different results when reordered.
    """

    wavelength_um: np.ndarray
    # Disk-integrated spectrum with the planet away from the stellar disk.
    outside: np.ndarray
    # Disk-integrated spectrum at every supplied planet position.
    inside: np.ndarray
    # Equivalent flux arrays evaluated using the local continuum intensities.
    continuum_outside: np.ndarray | None
    continuum_inside: np.ndarray | None

    def line_distortion(self):
        """Return continuum-normalized in/out ratios on the current grid.

        Apply this after any required instrumental convolution of the four
        flux arrays. A wavelength-independent transit gives a ratio of one.
        """
        # Continuum normalization needs both the unocculted and occulted continua.
        if self.continuum_outside is None or self.continuum_inside is None:
            raise ValueError("Continuum-normalized line distortion needs local continuum intensities")
        if (np.any(self.outside <= 0) or np.any(self.continuum_outside <= 0)
                or np.any(self.continuum_inside <= 0)):
            raise ValueError("Cannot normalize zero stellar flux or a fully occulted disk")
        # Normalize in-transit and reference spectra by their own continua,
        # then take their ratio to isolate the changing stellar line profile.
        return ((self.inside / self.continuum_inside)
                / (self.outside / self.continuum_outside))


class StellarSurface:
    """Projected stellar disk with solid-body rotation and angular spectra.

    cells_across_diameter controls the spatial sampling. Each square cell is
    sampled at subpixels_per_axis points along each axis. Subpixels resolve
    the stellar edge and the planetary silhouette. Convergence must be checked
    for the planet size and spectral precision of the intended experiment.
    """

    def __init__(self, local, *, vsini_kms, projected_obliquity_deg,
                 cells_across_diameter, subpixels_per_axis=1):
        # Check the local synthesis and requested stellar surface sampling.
        if not isinstance(local, LocalIntensityGrid):
            raise TypeError("local must be a LocalIntensityGrid")
        for name, value in (("cells_across_diameter", cells_across_diameter),
                            ("subpixels_per_axis", subpixels_per_axis)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(vsini_kms) or not 0 <= vsini_kms < C_KMS:
            raise ValueError("vsini_kms must be finite and in [0, c)")
        if not np.isfinite(projected_obliquity_deg):
            raise ValueError("projected_obliquity_deg must be finite")
        # Keep the common local synthesis for all subsequent transit positions.
        self.local = local
        # Sample the projected stellar disk symmetrically. Each sample has the
        # same projected area on the sky.
        # Include the requested subpixel sampling along both projected axes.
        n = cells_across_diameter * subpixels_per_axis
        # Place equally spaced sample centres between projected stellar coordinates -1 and +1.
        axis = (np.arange(n) + 0.5) * (2.0 / n) - 1.0
        # Construct the square grid in the plane of the sky.
        x, y = np.meshgrid(axis, axis)
        # Keep sample centres inside the unit stellar disk.
        on_star = x*x + y*y < 1.0
        # Store the surviving projected coordinates in stellar-radius units.
        self.x, self.y = x[on_star], y[on_star]
        # Every sample represents this projected area on a unit-radius star.
        self.area = (2.0 / n)**2
        # For a spherical star, mu follows directly from projected distance to disk centre.
        self.mu = np.sqrt(1.0 - self.x**2 - self.y**2)
        if len(self.mu) == 0:
            raise ValueError("The surface grid contains no stellar samples")
        # Require synthesis coverage of every viewing angle used by the surface.
        if self.mu.min() < local.mu[0] or self.mu.max() > local.mu[-1]:
            raise ValueError("The supplied mu grid must cover every stellar surface sample")
        # Convert the projected spin-orbit angle from degrees for trigonometric functions.
        angle = np.deg2rad(projected_obliquity_deg)
        # This Cartesian expression remains defined on both rotation axes.
        # Project solid-body rotation onto the obliquity-rotated sky coordinate.
        # Positive values shift local line profiles towards longer wavelengths.
        self.velocity_kms = vsini_kms * (self.x*np.cos(angle) - self.y*np.sin(angle))
        # Locate the two synthesis angles around each surface sample once.
        self._upper = np.clip(np.searchsorted(local.mu, self.mu, side="right"),
                              1, len(local.mu) - 1)
        # The neighbouring lower synthesis angle brackets each sample with _upper.
        lower = self._upper - 1
        # Precompute the fraction between those two angles for repeated interpolation.
        self._weight = ((self.mu - local.mu[lower])
                        / (local.mu[self._upper] - local.mu[lower]))

    def _local_spectra(self, values, wavelength_um):
        """Interpolate local profiles in angle and then Doppler-shift them."""
        # Evaluate the bilinear interpolation for a wavelength chunk at once.
        # This avoids a Python loop over tens of thousands of surface samples.
        # For each surface element, map the output wavelength back to its local rest frame.
        rest_wave = wavelength_um[None, :] / (1.0 + self.velocity_kms[:, None]/C_KMS)
        # Use the synthesis wavelength axis as the interpolation reference.
        wave = self.local.wavelength_um
        # Locate the upper wavelength channel around each Doppler-shifted sample.
        right = np.clip(np.searchsorted(wave, rest_wave, side="right"), 1, len(wave) - 1)
        # Measure the distance between the neighbouring synthesis wavelengths.
        fraction = (rest_wave - wave[right - 1]) / (wave[right] - wave[right - 1])
        # Broadcast each surface element's angular bracket along wavelength.
        upper = self._upper[:, None]
        # Interpolate in wavelength at the lower viewing angle.
        low = (values[upper - 1, right - 1]*(1.0 - fraction)
               + values[upper - 1, right]*fraction)
        # Repeat at the upper viewing angle.
        high = (values[upper, right - 1]*(1.0 - fraction)
                + values[upper, right]*fraction)
        # Combine the two angular profiles with the stored mu interpolation weights.
        return (1.0 - self._weight[:, None])*low + self._weight[:, None]*high

    def integrate(self, wavelength_um, planet_x, planet_y, radius_ratio, *,
                  wavelength_chunk_size=128):
        """Integrate spectra for supplied instantaneous planet positions.

        planet_x and planet_y are one-dimensional arrays in stellar radii.
        radius_ratio is the opaque planet radius divided by the stellar radius.
        The caller supplies front-of-star positions, including out-of-transit
        positions where the planet does not overlap the stellar disk.

        Chunking wavelengths bounds memory use while retaining the complete
        disk for each integration. observation.py subsequently averages exposure
        samples, applies the instrument response and shifts the integrated spectra
        to the observer frame. coupled.py performs the atmospheric ray integration.
        """
        # Read the requested wavelength samples and projected planet positions.
        wave = _finite_array(wavelength_um, "wavelength_um", 1)
        px = _finite_array(planet_x, "planet_x", 1)
        py = _finite_array(planet_y, "planet_y", 1)
        if len(wave) == 0 or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError("wavelength_um must be nonempty, positive and strictly increasing")
        # Require one matching x,y position for every instantaneous occultation.
        if px.shape != py.shape or len(px) == 0:
            raise ValueError("planet_x and planet_y must have matching nonempty shapes")
        # The opaque-body radius is expressed relative to the stellar radius.
        if not np.isfinite(radius_ratio) or not 0 <= radius_ratio < 1:
            raise ValueError("radius_ratio must be finite and in [0, 1)")
        if (isinstance(wavelength_chunk_size, bool)
                or not isinstance(wavelength_chunk_size, (int, np.integer))
                or wavelength_chunk_size < 1):
            raise ValueError("wavelength_chunk_size must be a positive integer")
        # Require synthesis coverage of every rotationally shifted wavelength.
        # Use the most redshifted and blueshifted surface samples to check wavelength margins.
        factors = 1.0 + self.velocity_kms / C_KMS
        if (wave[0]/factors.max() < self.local.wavelength_um[0]
                or wave[-1]/factors.min() > self.local.wavelength_um[-1]):
            raise ValueError("Local spectra need wider wavelength coverage for rotational Doppler shifts")
        # Find which stellar surface samples lie behind the opaque planet at each position.
        occulted = ((self.x[None, :] - px[:, None])**2
                    + (self.y[None, :] - py[:, None])**2 < radius_ratio**2)
        # Each mask row selects the samples blocked in one exposure. Subtract
        # their light from the same full-disk sum used as the reference. This
        # also preserves the outside spectrum exactly when nothing is blocked.
        # A transiting planet covers a small fraction of the surface. Sparse
        # sums avoid multiplying all the zero weights in every wavelength chunk.
        # Store only occulted sample areas in a sparse exposure-by-surface matrix.
        weights = csr_matrix(occulted, dtype=float) * self.area
        # Allocate the full-disk spectrum and one occulted spectrum per planet position.
        outside = np.empty(len(wave))
        inside = np.empty((len(px), len(wave)))
        # Allocate continuum spectra when the synthesis supplies them.
        continuum_out = (None if self.local.continuum is None else np.empty_like(outside))
        continuum_in = (None if self.local.continuum is None else np.empty_like(inside))
        # Evaluate manageable wavelength chunks while retaining the same surface samples.
        for start in range(0, len(wave), wavelength_chunk_size):
            # Select this consecutive wavelength interval.
            section = slice(start, start + wavelength_chunk_size)
            # Evaluate all rotating local stellar line profiles in this chunk.
            spectra = self._local_spectra(self.local.intensity, wave[section])
            # Sum the projected area-weighted intensities over the whole visible star.
            outside[section] = spectra.sum(axis=0) * self.area
            # Subtract the local light blocked by the planet at each position.
            inside[:, section] = outside[section] - weights @ spectra
            # Repeat the identical disk and occultation calculation for the continuum.
            if self.local.continuum is not None:
                continuum = self._local_spectra(self.local.continuum, wave[section])
                continuum_out[section] = continuum.sum(axis=0) * self.area
                continuum_in[:, section] = continuum_out[section] - weights @ continuum
        # Pass the integrated flux components to exposure averaging and instrument sampling.
        return StellarTransitSpectra(wave.copy(), outside, inside, continuum_out, continuum_in)
