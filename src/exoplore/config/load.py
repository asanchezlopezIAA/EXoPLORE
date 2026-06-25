"""Read EXoPLORE configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exoplore.config.models import (
    AtmosphereConfig,
    CrossCorrelationConfig,
    InstrumentConfig,
    ObservationConfig,
    PathConfig,
    PipelineConfig,
    PlanetConfig,
    RetrievalConfig,
    SimulationConfig,
    TelluricConfig,
)


def _as_tuple(value):
    """Return ``None`` or convert a JSON list to a tuple."""
    if value is None:
        return None
    return tuple(value)


def _as_path(value):
    """Return ``None`` or convert a string path to ``Path``."""
    if value is None:
        return None
    return Path(value).expanduser()


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Read a JSON file and return a ``SimulationConfig`` object."""
    path = Path(path)

    with path.open("r") as file:
        data: dict[str, Any] = json.load(file)

    planet_data = data["planet"]
    instrument_data = data["instrument"]
    observation_data = data["observation"]
    atmosphere_data = data["atmosphere"]
    paths_data = data.get("paths")

    return SimulationConfig(
        planet=PlanetConfig(
            name=planet_data["name"],
            parameter_file=_as_path(planet_data.get("parameter_file")),
        ),
        instrument=InstrumentConfig(
            name=instrument_data["name"],
            observatory=instrument_data.get("observatory", "paranal"),
            pixels_per_resolution_element=instrument_data.get(
                "pixels_per_resolution_element",
                2.5,
            ),
            convolve_to_instrument_resolution=instrument_data.get(
                "convolve_to_instrument_resolution",
                True,
            ),
            order_start=instrument_data.get("order_start"),
            order_stop=instrument_data.get("order_stop"),
            orders=_as_tuple(instrument_data.get("orders")),
        ),
        observation=ObservationConfig(
            event=observation_data.get("event", "transit"),
            number_of_nights=observation_data.get("number_of_nights", 1),
            exposure_time_seconds=observation_data.get("exposure_time_seconds", 30.0),
            readout_fraction=observation_data.get("readout_fraction", 0.2),
            overhead_seconds=observation_data.get("overhead_seconds", 0.0),
            use_real_observations=observation_data.get("use_real_observations", False),
        ),
        atmosphere=AtmosphereConfig(
            species=_as_tuple(atmosphere_data.get("species", ["H2", "He", "H2O"])),
            use_easychem=atmosphere_data.get("use_easychem", True),
            metallicity_wrt_solar=atmosphere_data.get("metallicity_wrt_solar", 0.0),
            carbon_to_oxygen_ratio=atmosphere_data.get(
                "carbon_to_oxygen_ratio",
                0.55,
            ),
            mean_molecular_weight=atmosphere_data.get("mean_molecular_weight", 2.33),
            reference_pressure_bar=atmosphere_data.get("reference_pressure_bar", 0.01),
            pressure_min_bar=atmosphere_data.get("pressure_min_bar", 1e-6),
            pressure_max_bar=atmosphere_data.get("pressure_max_bar", 1e2),
            pressure_grid_size=atmosphere_data.get("pressure_grid_size", 100),
            isothermal=atmosphere_data.get("isothermal", False),
            isothermal_temperature_k=atmosphere_data.get("isothermal_temperature_k"),
            wind_velocity_kms=atmosphere_data.get("wind_velocity_kms", 0.0),
        ),
        tellurics=TelluricConfig(**data.get("tellurics", {})),
        pipeline=PipelineConfig(**data.get("pipeline", {})),
        cross_correlation=CrossCorrelationConfig(**data.get("cross_correlation", {})),
        retrieval=RetrievalConfig(**data.get("retrieval", {})),
        paths=(
            PathConfig(
                output_root=_as_path(paths_data["output_root"]),
                planet_parameter_directory=_as_path(
                    paths_data.get("planet_parameter_directory"),
                ),
                petit_radtrans_input_data_path=_as_path(
                    paths_data.get("petit_radtrans_input_data_path"),
                ),
            )
            if paths_data is not None
            else None
        ),
    )


load_config = load_simulation_config
