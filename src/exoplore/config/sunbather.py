"""Explicit settings for the Sunbather physical spectrum provider."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SunbatherConfig:
    """Cloudy installation and energy-balance choices; T0 remains temperature_K."""
    cloudy_path: str
    project_path: str
    heating_cooling_factor: float
    maximum_iterations: int
    starting_temperature: str
    metal_scale: float
    cosmic_rays: bool
    stellar_tidal_gravity: bool = True

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
        if self.metal_scale != 0:
            raise ValueError("This first adapter preserves the adopted H/He-only composition")
        if self.cosmic_rays is not True:
            raise ValueError("Sunbather 1.1.0 run_s enables cosmic rays; explicitly acknowledge this")
