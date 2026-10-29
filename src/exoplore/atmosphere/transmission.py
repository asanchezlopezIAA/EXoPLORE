"""Resolve a hydrostatic pRT atmosphere into transmissions along individual rays.

The impact parameter is the distance of a ray from the planet centre. Its
transmission retains the altitude information needed to illuminate the
atmosphere with a spatially varying stellar spectrum.


stellar/coupled.py uses this ray table to weight planetary absorption by
the local stellar spectrum behind each atmospheric radius. The adapter
checks its integrated area against pRT before returning the rays.
"""

from dataclasses import dataclass
import importlib.metadata

import numpy as np


@dataclass(frozen=True)
class ChordTransmission:
    """Transmission on wavelength and decreasing impact-parameter axes.

    Radii are in cm and wavelengths in microns. The deepest radius defines
    the opaque body. The top radius bounds the supplied hydrostatic atmosphere.
    transmission has shape (n_wavelength, n_impact_parameters).
    """

    # Increasing vacuum wavelengths in micrometres.
    wavelength_um: np.ndarray
    # Ray distances from the planet centre, decreasing from the atmospheric top.
    impact_parameter_cm: np.ndarray
    # Fraction of incident light transmitted along each wavelength/radius chord.
    transmission: np.ndarray
    # Record the pRT version whose opacity layout supplied these transmissions.
    prt_version: str = ""

    def __post_init__(self):
        # Validate independent copies of the wavelength, radius and transmission arrays.
        arrays = {}
        for name in ("wavelength_um", "impact_parameter_cm", "transmission"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            arrays[name] = array
        # Use the declared field order to check the two coordinate axes together.
        wave, radius, transmission = arrays.values()
        # Wavelength interpolation needs a positive, increasing one-dimensional axis.
        if wave.ndim != 1 or len(wave) < 2 or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError("Chord wavelengths must be positive and strictly increasing")
        # The integration below uses radii ordered from atmospheric top to opaque bottom.
        if (radius.ndim != 1 or len(radius) < 2 or np.any(radius <= 0)
                or np.any(np.diff(radius) >= 0)):
            raise ValueError("Chord impact parameters must be positive and strictly decreasing")
        # Every wavelength must have one transmitted fraction for each atmospheric ray.
        if transmission.shape != (len(wave), len(radius)):
            raise ValueError("Chord transmission must have shape (n_wavelength, n_rays)")
        if np.any((transmission < 0) | (transmission > 1)):
            raise ValueError("Chord transmissions must lie in [0, 1]")
        # Keep the validated ray table fixed during stellar surface integration.
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def effective_radius_cm(self):
        """Integrate opaque area and annular absorption with pRT's quadrature."""
        # Use the same decreasing impact parameters returned by the hydrostatic calculation.
        radius = self.impact_parameter_cm
        # The annular area integrand is absorbed fraction multiplied by radius.
        absorbed = (1 - self.transmission)*radius
        # Add the opaque bottom area to the trapezoidal annular absorption integral.
        # The full circular annulus contributes 2 pi r dr, yielding this area / pi.
        area_over_pi = radius[-1]**2 + np.sum(
            (absorbed[:, 1:] + absorbed[:, :-1])*(-np.diff(radius)), axis=1)
        # The square root converts area / pi into effective transit radius in cm.
        return np.sqrt(area_over_pi)


def calculate_chord_transmission(radtrans, *, temperatures, mass_fractions,
                                 mean_molar_masses, reference_gravity,
                                 reference_pressure, planet_radius,
                                 opaque_cloud_top_pressure=None):
    """Export opacity and radius arrays through pRT's public transmission API.

    The adapter is validated against pRT 3.2.0 in line-by-line mode. That
    implementation stores the mixed absorption opacity in the first species
    slot and returns scattering separately. A reconstruction check against
    pRT's own effective radius verifies that layout on every call.

    The resulting radius-resolved transmissions supply the local stellar
    illumination calculation in stellar/coupled.py.
    """
    # Check the installed pRT version before interpreting its exported opacity axes.
    version = importlib.metadata.version("petitRADTRANS")
    if version != "3.2.0":
        raise NotImplementedError(
            f"Spatial opacity export is validated for petitRADTRANS 3.2.0, found {version}. "
            "Review its opacity layout and ray integration before enabling this adapter")
    from petitRADTRANS import physical_constants as cst

    # This ray reconstruction uses the line-by-line mixed-species opacity layout.
    if radtrans.line_opacity_mode != "lbl":
        raise ValueError("Spatial RM/CLV requires pRT line-by-line opacities")
    # Request the opacity arrays and hydrostatic radii from the same public call
    # that computes the effective transit radius.
    args = dict(temperatures=temperatures, mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses, reference_gravity=reference_gravity,
                reference_pressure=reference_pressure, planet_radius=planet_radius,
                return_opacities=True, return_radius_hydrostatic_equilibrium=True)
    # Pass through an explicitly supplied opaque cloud boundary.
    if opaque_cloud_top_pressure is not None:
        args["opaque_cloud_top_pressure"] = opaque_cloud_top_pressure
    # Compute the atmosphere once and retain pRT's effective radius for validation.
    wavelength_cm, expected_radius, output = radtrans.calculate_transit_radii(**args)
    # Read absorption opacity, scattering opacity and the matching hydrostatic radii.
    opacities = np.asarray(output["opacities"])
    scattering = np.asarray(output["continuum_opacities_scattering"])
    radius = np.asarray(output["radius_hydrostatic_equilibrium"])
    # Check the pRT absorption and scattering array layouts before selecting their axes.
    if opacities.ndim != 4 or opacities.shape[0] != 1 or opacities.shape[2] < 1:
        raise ValueError("Unsupported pRT opacity layout for spatial RM/CLV")
    if scattering.shape != (len(wavelength_cm), len(radius)):
        raise ValueError("Unsupported pRT scattering layout for spatial RM/CLV")
    # pRT's public pressure property uses CGS pressure. The input temperature
    # and molecular-mass profiles are the same ones used for its spectrum.
    density = (np.asarray(radtrans.pressures)*np.asarray(mean_molar_masses)
               * cst.amu/cst.kB/np.asarray(temperatures))
    # Convert opacity per unit mass to extinction per unit length using the gas density.
    extinction = (opacities[0, :, 0, :] + scattering)*density
    if not np.isfinite(extinction).all() or np.any(extinction < 0):
        raise ValueError("Spatial transfer requires finite nonnegative extinction")
    # A full chord crosses both sides of each spherical layer. The factor two
    # cancels the half in the trapezoidal mean of neighbouring extinctions.
    # Build sums of neighbouring layer extinctions for the chord trapezoidal rule.
    paired = extinction.copy()
    # Pair each layer with its preceding outer layer.
    paired[:, 1:] += extinction[:, :-1]
    # For every impact parameter, calculate half-chord distances to the spherical layer radii.
    lengths = np.sqrt(np.maximum(radius[None, :]**2-radius[:, None]**2, 0))
    # Allocate the path-length contribution of each layer to each ray.
    increments = np.zeros_like(lengths)
    # Subtract neighbouring half-chord lengths to obtain traversed layer distances.
    increments[1:, 1:] = lengths[1:, :-1] - lengths[1:, 1:]
    # Sum extinction times path length along every wavelength/ray chord.
    optical_depth = paired @ increments.T
    # Apply Beer-Lambert transmission exp(-tau) and convert pRT cm wavelengths to micrometres.
    chords = ChordTransmission(wavelength_cm*1e4, radius, np.exp(-optical_depth), version)
    # This catches changed library conventions, mixed-species double counting
    # and continuum/scattering mistakes before they affect an observed spectrum.
    if not np.allclose(chords.effective_radius_cm(), expected_radius, rtol=1e-10, atol=0):
        raise ValueError("Chord integration does not reproduce pRT's effective radius. "
                         "The spatial-transfer adapter needs review for this pRT setup")
    # Supply the validated radius-resolved transmissions to spatial stellar integration.
    return chords
