"""Explicit settings for the Sunbather physical spectrum provider."""
from dataclasses import dataclass
import math

SUNBATHER_METAL_ELEMENTS = (
    'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
)


@dataclass(frozen=True)
class SunbatherConfig:
    """Cloudy installation and energy-balance choices; T0 remains temperature_K."""
    cloudy_path: str
    project_path: str
    heating_cooling_factor: float
    maximum_iterations: int
    starting_temperature: str
    metal_scale: float = 0.0
    cosmic_rays: bool | None = None
    stellar_tidal_gravity: bool = False
    parker_mu_tolerance: float | None = None
    parker_mu_max_iterations: int | None = None
    parker_initialization: str | None = None
    metal_elements: list[str] | None = None
    include_metals: bool | None = None

    @property
    def metals_enabled(self) -> bool:
        """Read the explicit switch, retaining older metal_scale-only files."""
        return self.metal_scale>0 if self.include_metals is None else self.include_metals

    @property
    def effective_metal_scale(self) -> float:
        """Metal/H scale used by both physical solvers (zero for H/He only)."""
        return self.metal_scale if self.metals_enabled else 0.

    def __post_init__(self):
        if type(self.stellar_tidal_gravity) is not bool:
            raise ValueError("stellar_tidal_gravity must be explicitly true or false")
        if not self.cloudy_path or not self.project_path:
            raise ValueError("Supply Cloudy and a new Sunbather project directory")
        if not math.isfinite(self.heating_cooling_factor) or self.heating_cooling_factor <= 1:
            raise ValueError("Heating/cooling convergence factor must exceed one")
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 3:
            raise ValueError("Supply at least three temperature iterations")
        if self.starting_temperature not in ('constant', 'free'):
            raise ValueError("Choose constant or free initial temperature")
        if isinstance(self.metal_scale,bool) or not isinstance(self.metal_scale,(int,float)) or not math.isfinite(self.metal_scale) or self.metal_scale < 0:
            raise ValueError("metal_scale must be a finite non-negative linear factor relative to solar")
        if self.include_metals is not None and type(self.include_metals) is not bool:
            raise ValueError('include_metals must be true or false')
        if self.metals_enabled and self.metal_scale<=0:
            raise ValueError('include_metals=true requires an explicit positive metal_scale (1 means solar metal/H)')
        if self.metal_elements is not None:
            if (not isinstance(self.metal_elements,list) or not self.metal_elements
                    or any(not isinstance(x,str) or x not in SUNBATHER_METAL_ELEMENTS for x in self.metal_elements)
                    or len(set(self.metal_elements))!=len(self.metal_elements)):
                raise ValueError('metal_elements must be a nonempty list of unique supported element symbols (Li to Zn, excluding H and He)')
            if self.metal_scale==0:
                raise ValueError('metal_elements requires metal_scale > 0. Set metal_scale=0 without a list for H/He only')
        if self.metals_enabled:
            if (isinstance(self.parker_mu_tolerance,bool) or not isinstance(self.parker_mu_tolerance,(int,float))
                    or not math.isfinite(self.parker_mu_tolerance) or not 0 < self.parker_mu_tolerance < 1):
                raise ValueError("Nonzero metallicity requires explicit parker_mu_tolerance between zero and one")
            if type(self.parker_mu_max_iterations) is not int or self.parker_mu_max_iterations < 2:
                raise ValueError("Nonzero metallicity requires parker_mu_max_iterations >= 2")
            if self.parker_initialization not in ('pwinds','neutral'):
                raise ValueError("Nonzero metallicity requires parker_initialization: pwinds or neutral")
        if self.cosmic_rays is not True:
            raise ValueError("Sunbather 1.1.0 run_s enables cosmic rays; explicitly acknowledge this")
