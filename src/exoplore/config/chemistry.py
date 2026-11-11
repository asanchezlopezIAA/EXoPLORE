"""Chemistry choices for one atmospheric column.

The surrounding atmosphere block supplies its pressure and temperature profile.
VULCAN settings describe chemical transport and irradiation on that column.
"""
from dataclasses import dataclass, field
import math


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be a finite {'non-negative' if zero else 'positive'} number")


@dataclass
class EddyDiffusionConfig:
    """Use a constant Kzz or a pressure[bar], Kzz[cm²/s] text table."""
    mode: str = "constant"
    value_cm2_s: float | None = None
    profile_file: str = ""

    def __post_init__(self):
        if self.mode not in ("constant", "file", "off"):
            raise ValueError("eddy_diffusion.mode must be constant, file or off")
        if self.mode == "constant":
            _positive(self.value_cm2_s, "eddy_diffusion.value_cm2_s", zero=True)
        if self.mode == "file" and not self.profile_file:
            raise ValueError("Supply eddy_diffusion.profile_file")
        if self.mode != "constant" and self.value_cm2_s is not None:
            raise ValueError("value_cm2_s requires constant eddy diffusion")
        if self.mode != "file" and self.profile_file:
            raise ValueError("profile_file requires file eddy diffusion")


@dataclass
class ChemicalBoundaryConfig:
    """Zero flux, a native VULCAN flux table, or fixed bottom mole fractions."""
    type: str = "zero_flux"
    file: str = ""
    mixing_ratios: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in ("zero_flux", "flux_file", "fixed_mixing_ratios"):
            raise ValueError("Unknown chemical boundary type")
        if self.type == "flux_file" and not self.file:
            raise ValueError("Supply the chemical boundary file")
        if self.type == "fixed_mixing_ratios":
            if not self.mixing_ratios or sum(self.mixing_ratios.values()) > 1:
                raise ValueError("Supply fixed boundary mole fractions with sum <= 1")
            for key, value in self.mixing_ratios.items():
                _positive(value, f"boundary {key}", zero=True)
        elif self.mixing_ratios:
            raise ValueError("mixing_ratios requires fixed_mixing_ratios boundary")


@dataclass
class CondensationConfig:
    """Native gas depletion and settling. Particle optical properties are separate."""
    enabled: bool = False
    settling: bool = False
    gas_species: list[str] = field(default_factory=list)
    condensed_species: list[str] = field(default_factory=list)
    particle_radius_cm: dict[str, float] = field(default_factory=dict)
    particle_density_g_cm3: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.settling) is not bool:
            raise ValueError("Condensation enabled and settling must be true or false")
        if not self.enabled and (self.gas_species or self.condensed_species):
            raise ValueError("Condensation species require condensation.enabled")
        if self.settling and not self.enabled:
            raise ValueError("Condensate settling requires condensation.enabled")
        if len(set(self.gas_species)) != len(self.gas_species) or len(set(self.condensed_species)) != len(self.condensed_species):
            raise ValueError("Condensation species lists must contain unique names")
        if set(self.gas_species) & set(self.condensed_species):
            raise ValueError("Gas and condensate species must have distinct names")
        if self.enabled:
            if not self.gas_species or not self.condensed_species:
                raise ValueError("Supply gas_species and condensed_species for condensation")
            for species in self.condensed_species:
                _positive(self.particle_radius_cm.get(species), f"{species} particle radius")
                _positive(self.particle_density_g_cm3.get(species), f"{species} particle density")


# These settings affect the numerical integration and its convergence checks.
# Physical settings above remain named fields in the master JSON.
SOLVER_OPTIONS = frozenset({
    "dttry", "trun_min", "runtime", "dt_min", "dt_max", "dt_var_max", "dt_var_min",
    "count_min", "count_max", "atol", "mtol", "mtol_conv", "loss_eps", "rtol",
    "yconv_cri", "slope_cri", "yconv_min", "flux_cri", "flux_atol", "conv_step",
    "use_adapt_rtol", "update_frq",
})


@dataclass
class VulcanConfig:
    """Inputs for the original shami-EEG/VULCAN script interface.

    elemental_abundances contains number ratios to hydrogen nuclei, including
    H=1 and He. It explicitly records the composition used by this network.
    """
    network_file: str = ""
    initialization: str = "equilibrium"
    initial_mixing_ratios: dict[str, float] = field(default_factory=dict)
    initial_element_tolerance: float = 1e-3
    elemental_abundances: dict[str, float] = field(default_factory=dict)
    photochemistry: bool = False
    stellar_flux_file: str = ""
    stellar_flux_reference: str = "stellar_surface"
    orbit_radius_au: float | None = None
    zenith_angle_deg: float | None = None
    diurnal_factor: float | None = None
    eddy_diffusion: EddyDiffusionConfig | dict = field(default_factory=lambda: EddyDiffusionConfig(mode="off"))
    molecular_diffusion: bool = True
    molecular_diffusion_scheme: str = "central"
    vertical_velocity_cm_s: float = 0.0
    background_gas: str = "H2"
    top_boundary: ChemicalBoundaryConfig | dict = field(default_factory=ChemicalBoundaryConfig)
    bottom_boundary: ChemicalBoundaryConfig | dict = field(default_factory=ChemicalBoundaryConfig)
    condensation: CondensationConfig | dict = field(default_factory=CondensationConfig)
    solver_options: dict[str, float | bool] = field(default_factory=dict)
    timeout_seconds: float = 7200.0
    max_atom_loss: float = 0.01
    species_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for key, cls in (("eddy_diffusion", EddyDiffusionConfig), ("top_boundary", ChemicalBoundaryConfig),
                         ("bottom_boundary", ChemicalBoundaryConfig), ("condensation", CondensationConfig)):
            if isinstance(getattr(self, key), dict):
                setattr(self, key, cls(**getattr(self, key)))
        if self.initialization not in ("equilibrium", "constant_mixing_ratios"):
            raise ValueError("Choose equilibrium or constant_mixing_ratios initialization")
        _positive(self.initial_element_tolerance, "initial_element_tolerance")
        if self.initialization == "constant_mixing_ratios":
            if not self.initial_mixing_ratios:
                raise ValueError("Supply initial_mixing_ratios for constant initialization")
            for name, value in self.initial_mixing_ratios.items():
                _positive(value, f"initial_mixing_ratios.{name}", zero=True)
            if not math.isclose(sum(self.initial_mixing_ratios.values()), 1., rel_tol=0, abs_tol=1e-10):
                raise ValueError("Initial gas mixing ratios must sum to one")
            if set(self.initial_mixing_ratios) & set(self.condensation.condensed_species):
                raise ValueError("Initial mixing ratios describe the gas phase")
        elif self.initial_mixing_ratios:
            raise ValueError("initial_mixing_ratios requires constant_mixing_ratios initialization")
        if not self.network_file:
            raise ValueError("Supply vulcan.network_file")
        if self.elemental_abundances.get("H") != 1 or "He" not in self.elemental_abundances:
            raise ValueError("VULCAN elemental_abundances requires H=1 and an explicit He/H ratio")
        for key, value in self.elemental_abundances.items():
            if not key.isalpha():
                raise ValueError("Use element symbols in elemental_abundances")
            _positive(value, f"elemental_abundances.{key}")
        if self.photochemistry:
            if not self.stellar_flux_file or self.stellar_flux_reference != "stellar_surface":
                raise ValueError("Photochemistry requires a stellar-surface UV flux file")
            _positive(self.orbit_radius_au, "orbit_radius_au")
            if self.zenith_angle_deg is None or not 0 <= self.zenith_angle_deg < 90:
                raise ValueError("Supply zenith_angle_deg in [0, 90)")
            if self.diurnal_factor is None or not 0 < self.diurnal_factor <= 1:
                raise ValueError("Supply diurnal_factor in (0, 1]")
        if self.background_gas not in ("H2", "N2", "O2", "CO2"):
            raise ValueError("Unsupported VULCAN background_gas")
        if self.condensation.enabled and not self.molecular_diffusion:
            raise ValueError("Native direct condensation requires molecular_diffusion=true to calculate its diffusion-limited growth rates")
        if self.molecular_diffusion_scheme not in ("central", "upwind", "hybrid"):
            raise ValueError("Choose central, upwind or hybrid molecular diffusion")
        _positive(abs(self.vertical_velocity_cm_s), "vertical_velocity_cm_s magnitude", zero=True)
        if self.top_boundary.type == "fixed_mixing_ratios":
            raise ValueError("Fixed mixing ratios are supported at the bottom boundary")
        for name in ("photochemistry", "molecular_diffusion"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        for key, value in self.solver_options.items():
            if key not in SOLVER_OPTIONS:
                raise ValueError(f"Unsupported VULCAN solver option: {key}")
            if key == "use_adapt_rtol":
                if type(value) is not bool:
                    raise ValueError("solver_options.use_adapt_rtol must be true or false")
                continue
            _positive(value, f"solver_options.{key}")
            if key in ("count_min", "count_max", "conv_step", "update_frq") and type(value) is not int:
                raise ValueError(f"solver_options.{key} must be an integer")
        for key in ("timeout_seconds", "max_atom_loss"):
            _positive(getattr(self, key), key)


@dataclass
class ChemistryConfig:
    """An explicit selection takes precedence over the legacy use_easychem flag."""
    backend: str
    vulcan: VulcanConfig | dict | None = None

    def __post_init__(self):
        if self.backend not in ("easychem", "manual", "vulcan"):
            raise ValueError("chemistry.backend must be easychem, manual or vulcan")
        if isinstance(self.vulcan, dict):
            self.vulcan = VulcanConfig(**self.vulcan)
        if (self.backend == "vulcan") != (self.vulcan is not None):
            raise ValueError("Supply chemistry.vulcan only for backend=vulcan")
