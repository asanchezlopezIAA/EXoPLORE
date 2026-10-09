"""Explicit configuration for spherical p-winds helium simulations.

The stellar irradiation is supplied at the planet. No target's irradiation,
mass-loss rate, temperature, or limb darkening is inferred from its name.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeliumPwindsConfig:
    """Optional helium backend, including observing and solver choices.

    Template NPZ files contain ``wave_nm`` and ``flux`` (stellar rest frame)
    or ``transmission`` (topocentric frame). Irradiation spectrum text files
    contain Angstrom and erg s^-1 cm^-2 Angstrom^-1 at the planet.
    The current backend requires p-winds 2.0.1 and a circular orbit.
    """
    backend: str
    temperature_K: float
    mass_loss_rate_g_s: float
    hydrogen_number_fraction: float
    irradiation_mode: str
    irradiation_source: str
    irradiation_spectrum_path: str
    flux_h_euv_erg_s_cm2: float | None
    flux_he_euv_erg_s_cm2: float | None
    flux_he_fuv_erg_s_cm2: float | None
    initial_h_ion_fraction: float
    initial_he_fractions: list[float]
    radial_max_rp: float
    radial_points: int
    solver_convergence: float
    solver_max_iterations: int
    solver_rtol: float
    solver_atol: float
    hydrogen_exact_photoionization: bool
    helium_solver_method: str
    wind_broadening_method: str
    voigt_method: str
    turbulence_broadening: bool
    line_of_sight_points: int
    stellar_grid_size: int
    stellar_supersampling: int
    limb_darkening_law: str
    limb_darkening_coefficients: list[float]
    optical_transit_only: bool
    bulk_velocity_kms: float
    model_wavelength_range_nm: list[float]
    model_wavelength_points: int
    model_edge_absorption_tolerance: float
    observing_wavelength_range_nm: list[float]
    observing_pixels: int
    instrumental_resolving_power: float
    instrumental_oversampling: int
    exposure_quadrature_points: int
    phase_midpoints: list[float]
    berv_kms: list[float]
    stellar_mode: str
    stellar_template_path: str
    telluric_mode: str
    telluric_template_path: str
    telluric_correction: str
    noise_mode: str
    snr_per_pixel: float
    noise_seed: int
    observing_grid_mode: str = "generated"
    uncertainty_mode: str = "monte_carlo"
    airmass: list[float] | None = None
    molecfit_config_path: str = ""
    resume_simulation_path: str = ""
    molecfit_workers: int = 1
    molecfit_reuse_path: str = ""
    completed_night_paths: list[str] | None = None
    sunbather: dict | None = None
    allart2023_significance: dict | None = None

    def __post_init__(self) -> None:
        """Reject incomplete inputs and unsupported physical assumptions."""
        if self.allart2023_significance is not None:
            from exoplore.pipelines.helium_significance import Allart2023SignificanceConfig
            Allart2023SignificanceConfig(**self.allart2023_significance)
        if self.completed_night_paths is not None and (not isinstance(self.completed_night_paths, list) or any(not isinstance(p, str) or not p for p in self.completed_night_paths)):
            raise ValueError("Completed helium nights must be a list of nonempty paths")
        if self.backend not in ("p-winds", "sunbather"):
            raise ValueError("Select sunbather or the legacy p-winds helium backend")
        if self.backend == "sunbather":
            from exoplore.config.sunbather import SunbatherConfig
            if self.sunbather is None:
                raise ValueError("Provide explicit Sunbather solver settings")
            SunbatherConfig(**self.sunbather)
            if self.wind_broadening_method != 'formal' or self.turbulence_broadening or self.bulk_velocity_kms != 0:
                raise ValueError('Sunbather uses resolved wind velocities; select formal broadening, no turbulence and zero bulk shift')
        if self.optical_transit_only is not False:
            raise ValueError("Set optical_transit_only=false: retain atmospheric absorption outside optical contacts")
        if type(self.molecfit_workers) is not int or not 1 <= self.molecfit_workers <= 4:
            raise ValueError("Choose one to four independent synthetic molecfit workers")
        if self.observing_grid_mode not in ("generated", "instrument"):
            raise ValueError("Select generated pixels or the EXoPLORE instrument wavelength grid")
        if self.uncertainty_mode not in ("monte_carlo", "conditional"):
            raise ValueError("Select monte_carlo or conditional marginal uncertainties")
        for name in ("temperature_K", "mass_loss_rate_g_s", "radial_max_rp",
                     "solver_convergence", "solver_rtol", "solver_atol",
                     "instrumental_resolving_power", "snr_per_pixel", "model_edge_absorption_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.radial_max_rp <= 1 or not 0 < self.hydrogen_number_fraction < 1:
            raise ValueError("Require outer radius > Rp and a mixed H/He atmosphere")
        if not 0 <= self.initial_h_ion_fraction <= 1 or len(self.initial_he_fractions) != 2:
            raise ValueError("Invalid initial ionization fractions")
        if any(not math.isfinite(x) or x < 0 for x in self.initial_he_fractions) or sum(self.initial_he_fractions) > 1:
            raise ValueError("Initial helium singlet/triplet fractions must sum to at most one")
        for name, minimum in (("radial_points", 30), ("solver_max_iterations", 2),
                              ("line_of_sight_points", 20), ("stellar_grid_size", 20),
                              ("stellar_supersampling", 1), ("model_wavelength_points", 100),
                              ("observing_pixels", 100), ("instrumental_oversampling", 2),
                              ("exposure_quadrature_points", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.irradiation_mode not in ("spectrum", "monochromatic") or not self.irradiation_source:
            raise ValueError("Document a spectrum or monochromatic irradiation source")
        if self.irradiation_mode == "spectrum":
            if not self.irradiation_spectrum_path:
                raise ValueError("Provide an irradiation spectrum at the planet")
            if any(x is not None for x in (self.flux_h_euv_erg_s_cm2, self.flux_he_euv_erg_s_cm2, self.flux_he_fuv_erg_s_cm2)):
                raise ValueError("Do not mix spectrum and monochromatic irradiation")
        else:
            values = (self.flux_h_euv_erg_s_cm2, self.flux_he_euv_erg_s_cm2, self.flux_he_fuv_erg_s_cm2)
            if any(x is None or not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("Supply all three monochromatic fluxes at the planet")
            if self.hydrogen_exact_photoionization or self.irradiation_spectrum_path:
                raise ValueError("Exact spectral photoionization requires spectrum mode")
        if self.helium_solver_method not in ("RK45", "Radau", "BDF", "LSODA"):
            raise ValueError("Choose an explicit solve_ivp helium solver")
        if self.wind_broadening_method not in ("formal", "average") or self.voigt_method not in ("formal", "fast"):
            raise ValueError("Unsupported p-winds radiative-transfer method")
        if self.limb_darkening_law not in ("none", "linear", "quadratic"):
            raise ValueError("Use none, linear or quadratic limb darkening")
        count = {"none": 0, "linear": 1, "quadratic": 2}[self.limb_darkening_law]
        if len(self.limb_darkening_coefficients) != count or not all(math.isfinite(x) for x in self.limb_darkening_coefficients):
            raise ValueError("Supply coefficients for the selected limb-darkening law")
        if len(self.phase_midpoints) < 3 or len(self.berv_kms) != len(self.phase_midpoints):
            raise ValueError("Provide phase and BERV for every exposure")
        if not all(math.isfinite(x) for x in self.phase_midpoints + self.berv_kms + [self.bulk_velocity_kms]):
            raise ValueError("Phases and velocities must be finite")
        if any(a >= b for a, b in zip(self.phase_midpoints, self.phase_midpoints[1:])) or any(abs(x) >= .25 for x in self.phase_midpoints):
            raise ValueError("Provide increasing phases around a single primary transit")
        for window in (self.model_wavelength_range_nm, self.observing_wavelength_range_nm):
            if len(window) != 2 or not all(math.isfinite(x) and x > 0 for x in window) or window[0] >= window[1]:
                raise ValueError("Supply increasing vacuum wavelength bounds in nm")
        if self.stellar_mode not in ("flat", "template") or self.telluric_mode not in ("none", "template"):
            raise ValueError("Choose explicit stellar and telluric inputs")
        if self.stellar_mode == "template" and not self.stellar_template_path:
            raise ValueError("Stellar template path required")
        if self.telluric_mode == "template" and not self.telluric_template_path:
            raise ValueError("Telluric template path required")
        if self.airmass is not None and (len(self.airmass) != len(self.phase_midpoints) or
                any(not math.isfinite(x) or x < 1 for x in self.airmass)):
            raise ValueError("Supply a valid airmass for every exposure")
        if self.telluric_correction == "molecfit" and (not self.molecfit_config_path or self.telluric_mode == "none"):
            raise ValueError("Molecfit requires injected tellurics and explicit fitting settings")
        if self.telluric_correction not in ("known_model", "none", "molecfit") or self.noise_mode not in ("none", "gaussian"):
            raise ValueError("Select the correction and noise experiment explicitly")
        if type(self.noise_seed) is not int or self.noise_seed < 0:
            raise ValueError("Noise seed must be a non-negative integer")
