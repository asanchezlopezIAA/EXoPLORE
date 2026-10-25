"""exoplore.config, typed simulation configuration objects."""

# Reading these imports
# This is the current public configuration entry point. Its loader below
# delegates to SimulationConfig.from_json. An older file named config/load.py
# also exists; do not infer that it is the implementation behind this public
# function merely because it has a similar filename.


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
    "load_simulation_config",
]
