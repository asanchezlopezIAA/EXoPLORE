"""exoplore.config, typed simulation configuration objects."""

# Public imports
# Export the configuration classes and load_simulation_config(). The loader
# below calls SimulationConfig.from_json().


from exoplore.config.models import (
    PlanetConfig,
    InstrumentConfig,
    ObservationConfig,
    NoiseConfig,
    AtmosphereRegionConfig,
    AtmosphereConfig,
    TelluricConfig,
    PipelineConfig,
    CrossCorrelationConfig,
    RetrievalConfig,
    StatisticsConfig,
    PathConfig,
    SimulationConfig,
)
from exoplore.config.rm_clv import RMCLVConfig
from exoplore.config.seidel22 import Seidel22Config
from exoplore.config.atomic_transmission import AtomicTransmissionConfig
from exoplore.config.stellar_surface import (
    StellarSurfaceConfig, StellarComponentConfig, SurfaceRegionConfig,
)

# Convenience alias, load_simulation_config(path) == SimulationConfig.from_json(path)
def load_simulation_config(path):
    """Load a SimulationConfig from a JSON file.  Alias for SimulationConfig.from_json."""
    return SimulationConfig.from_json(path)


__all__ = [
    "PlanetConfig",
    "InstrumentConfig",
    "ObservationConfig",
    "NoiseConfig",
    "AtmosphereRegionConfig",
    "AtmosphereConfig",
    "TelluricConfig",
    "PipelineConfig",
    "CrossCorrelationConfig",
    "RetrievalConfig",
    "StatisticsConfig",
    "PathConfig",
    "SimulationConfig",
    "RMCLVConfig",
    "StellarSurfaceConfig", "StellarComponentConfig", "SurfaceRegionConfig",
    "Seidel22Config",
    "AtomicTransmissionConfig",
    "load_simulation_config",
]
