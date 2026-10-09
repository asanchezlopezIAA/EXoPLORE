"""Spherical helium outflows and transit spectra calculated with p-winds.

Physical profiles use the tidal Parker routines matching p-winds 2.0.1's
hydrogen solver. Radiative transfer uses explicit vacuum wavelengths in SI.
Atmospheric absorption follows projected overlap, including outside optical contacts.
"""
from __future__ import annotations

from importlib.metadata import version
import warnings

import numpy as np
from scipy.interpolate import interp1d

from exoplore.config.helium import HeliumPwindsConfig

RJ_M = 71492000.0
RS_M = 695700000.0
AU_M = 149597870700.0


def geometry(phase: float, planet) -> tuple[float, float, float]:
    """Return sky coordinates in stellar radii and Rp/Rstar for a circular orbit."""
    angle = 2 * np.pi * phase
    a = planet.semi_major_axis_au * AU_M / (planet.stellar_radius_rsun * RS_M)
    return (a * np.sin(angle), a * np.cos(np.deg2rad(planet.inclination_deg)) * np.cos(angle),
            planet.planet_radius_rjup * RJ_M / (planet.stellar_radius_rsun * RS_M))


def contact_phases(planet) -> np.ndarray:
    """Calculate first through fourth optical contacts; require a full transit."""
    _, b, p = geometry(0, planet)
    a = planet.semi_major_axis_au * AU_M / (planet.stellar_radius_rsun * RS_M)
    if abs(b) >= 1 - p or a <= 1 + p:
        raise ValueError("This first helium simulator requires a non-grazing transit")
    outer = np.arcsin(np.sqrt(((1 + p)**2 - b*b) / (a*a - b*b))) / (2*np.pi)
    inner = np.arcsin(np.sqrt(((1 - p)**2 - b*b) / (a*a - b*b))) / (2*np.pi)
    return np.array([-outer, -inner, inner, outer])


def solve_outflow(settings: HeliumPwindsConfig, planet) -> dict:
    """Solve hydrogen and helium populations, retaining solver warnings and units.

    The supplied irradiation is already at the planet. Spectrum mode reads two
    columns: Angstrom and erg s^-1 cm^-2 Angstrom^-1. No flux scaling is inferred.
    """
    try:
        installed = version("p-winds")
    except Exception as exc:
        raise ImportError("Install the optional helium dependencies: pip install '.[helium]'") from exc
    if installed != "2.0.1":
        raise RuntimeError(f"Validated backend requires p-winds 2.0.1; found {installed}")
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        from p_winds import hydrogen, helium, parker
        from astropy import units as u
        from astropy.constants import m_p
        s = settings
        spectrum = None
        if s.irradiation_mode == "spectrum":
            table = np.loadtxt(s.irradiation_spectrum_path)
            if table.ndim != 2 or table.shape[1] != 2 or not np.isfinite(table).all():
                raise ValueError("Irradiation file must contain two finite columns")
            if np.any(np.diff(table[:, 0]) <= 0) or np.any(table <= 0):
                raise ValueError("Irradiation wavelengths must increase and fluxes must be positive")
            if table[0, 0] >= 504 or table[-1, 0] < 2593:
                raise ValueError("Irradiation must cover EUV and metastable-helium photoionizing UV")
            spectrum = dict(wavelength=table[:, 0], flux_lambda=table[:, 1],
                            wavelength_unit=u.angstrom,
                            flux_unit=u.erg / u.s / u.cm**2 / u.angstrom)
        radius = np.geomspace(1, s.radial_max_rp, s.radial_points)
        he_h = (1-s.hydrogen_number_fraction)/s.hydrogen_number_fraction
        mu0 = (1+4*he_h)/(1+he_h+s.initial_h_ion_fraction)
        fh, mu = hydrogen.ion_fraction(
            radius, planet.planet_radius_rjup, s.temperature_K, s.hydrogen_number_fraction,
            s.mass_loss_rate_g_s, planet.planet_mass_mjup, mean_molecular_weight_0=mu0,
            star_mass=planet.stellar_mass_msun, semimajor_axis=planet.semi_major_axis_au,
            spectrum_at_planet=spectrum, flux_euv=s.flux_h_euv_erg_s_cm2,
            initial_f_ion=s.initial_h_ion_fraction, relax_solution=True,
            convergence=s.solver_convergence, max_n_relax=s.solver_max_iterations,
            exact_phi=s.hydrogen_exact_photoionization, return_mu=True,
            method="RK45", rtol=s.solver_rtol, atol=s.solver_atol)
        cs = parker.sound_speed(s.temperature_K, mu)
        rs = parker.radius_sonic_point_tidal(planet.planet_mass_mjup, cs,
                                           planet.stellar_mass_msun, planet.semi_major_axis_au)
        rhos = parker.density_sonic_point(s.mass_loss_rate_g_s, rs, cs)
        velocity, density = parker.structure_tidal(radius*planet.planet_radius_rjup/rs,
            cs, rs, planet.planet_mass_mjup, planet.stellar_mass_msun, planet.semi_major_axis_au)
        f1, f3 = helium.population_fraction(
            radius, velocity, density, fh, planet.planet_radius_rjup, s.temperature_K,
            s.hydrogen_number_fraction, cs, rs, rhos, spectrum_at_planet=spectrum,
            flux_euv=s.flux_he_euv_erg_s_cm2, flux_fuv=s.flux_he_fuv_erg_s_cm2,
            initial_state=np.asarray(s.initial_he_fractions), relax_solution=True,
            convergence=s.solver_convergence, max_n_relax=s.solver_max_iterations,
            method=s.helium_solver_method, rtol=s.solver_rtol, atol=s.solver_atol)
        rho = density*rhos
        nhe = rho*(1-s.hydrogen_number_fraction)/(4-3*s.hydrogen_number_fraction)/m_p.cgs.value
        result = dict(radius_rp=radius, radius_m=radius*planet.planet_radius_rjup*RJ_M,
                      velocity_m_s=velocity*cs*1000, density_g_cm3=rho,
                      hydrogen_ion_fraction=fh, helium_singlet_fraction=f1,
                      helium_triplet_fraction=f3, helium_triplet_m3=f3*nhe*1e6,
                      mean_molecular_weight=float(mu), p_winds_version=installed,
                      warnings=[str(item.message) for item in messages])
    for key in ("radius_m", "velocity_m_s", "density_g_cm3", "helium_triplet_m3", "hydrogen_ion_fraction"):
        if not np.isfinite(result[key]).all() or np.any(result[key] < 0):
            raise RuntimeError(f"Unphysical p-winds output: {key}")
    if np.any(f1+f3 > 1+1e-6) or np.any(fh > 1+1e-6) or not np.isfinite(f1+f3).all():
        raise RuntimeError("Invalid ionization populations")
    return result


class HeliumTransit:
    """Cache p-winds optical depths and integrate them over the stellar disk."""

    def __init__(self, settings: HeliumPwindsConfig, planet, science, profile: dict):
        from p_winds import transit, lines
        from astropy.constants import m_p
        self.settings, self.planet = settings, planet
        self.wave_nm = np.linspace(*settings.model_wavelength_range_nm, settings.model_wavelength_points)
        properties = lines.he_3_properties()
        self.p_winds_air_lines_m = list(properties[:3])
        if np.any(profile["helium_triplet_m3"] > 0):
            tau = transit.optical_depth_2d(
                profile["radius_m"], profile["helium_triplet_m3"], profile["velocity_m_s"],
                np.asarray(science.helium_vacuum_lines_nm)*1e-9,
                np.asarray(science.helium_oscillator_strengths), np.full(3, properties[-1]),
                self.wave_nm*1e-9, settings.temperature_K, 4*m_p.value,
                settings.line_of_sight_points, bulk_los_velocity=settings.bulk_velocity_kms*1000,
                planet_radial_velocity=0, wind_broadening_method=settings.wind_broadening_method,
                voigt_method=settings.voigt_method, turbulence_broadening=settings.turbulence_broadening)
        else:
            tau = np.zeros((len(profile["radius_m"]), len(self.wave_nm)))
        if not np.isfinite(tau).all() or np.any(tau < 0):
            raise RuntimeError("Invalid helium optical depths")
        self.tau = tau
        self.radius_m = profile["radius_m"]

    def spectrum(self, phase: float) -> tuple[np.ndarray, float]:
        """Return helium-only transmission and the opaque-planet continuum.

        Optical contacts do not truncate the atmosphere. For a spherical wind
        and radially limb-darkened star, rotating the projected separation onto
        the x axis preserves the integrated flux and avoids flatstar's chord
        singularity when the planet lies beyond the stellar limb in y.
        The atmosphere is limited only by the supplied radial profile.
        """
        from p_winds import transit
        x, y, p = geometry(phase, self.planet)
        if np.cos(2*np.pi*phase) <= 0:
            return np.ones_like(self.wave_nm), 1.0
        separation = np.hypot(x, y)
        stellar_radius_m = self.planet.stellar_radius_rsun * RS_M
        if separation >= 1 + max(p, self.radius_m[-1] / stellar_radius_m):
            return np.ones_like(self.wave_nm), 1.0
        s = self.settings
        coefficients = s.limb_darkening_coefficients
        coefficient = coefficients[0] if s.limb_darkening_law == "linear" else coefficients or None
        intensity, depth, distance = transit.draw_transit(
            p, self.planet.planet_radius_rjup*RJ_M, impact_parameter=0.0,
            phase=separation/(2*(1+p)),
            grid_size=s.stellar_grid_size, supersampling=s.stellar_supersampling,
            limb_darkening_law=None if s.limb_darkening_law == "none" else s.limb_darkening_law,
            ld_coefficient=coefficient)
        # Supersampling/resizing may slightly alter the reported depth. Use the
        # actual ray weights for both continuum and helium to preserve unity.
        intensity = np.asarray(intensity, dtype=float)
        opaque = float(np.sum(intensity))
        # Blocking wavelengths limits working memory without approximating rays.
        illuminated = intensity > 0
        rays, weights = distance[illuminated], intensity[illuminated]
        helium = np.empty_like(self.wave_nm)
        for start in range(0, len(helium), 64):
            stop = min(start+64, len(helium))
            radial_tau = interp1d(self.radius_m, self.tau[:, start:stop], axis=0,
                                 bounds_error=False, fill_value=0)
            helium[start:stop] = np.sum(weights[:, None]*np.exp(-radial_tau(rays)), axis=0)/opaque
        if np.max(np.abs(1-helium[[0, -1]])) > s.model_edge_absorption_tolerance:
            raise ValueError("Helium model wavelength range truncates appreciable absorption")
        return helium, opaque
