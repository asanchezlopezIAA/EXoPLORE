"""Tests for exoplore.config, config loading and serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from exoplore.config import SimulationConfig
from exoplore.config.models import (
    AtmosphereRegionConfig,
    CrossCorrelationConfig,
    PlanetConfig,
)
from exoplore.core.simulator import ExoploreSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return SimulationConfig()


@pytest.fixture
def hd189733b_config():
    config_path = Path(__file__).parent.parent / "configs" / "hd189733b_andes_transit_clean.json"
    if not config_path.exists():
        pytest.skip("Example config not found.")
    return SimulationConfig.from_json(config_path)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_default_config_creates():
    cfg = SimulationConfig()
    assert cfg.planet.name == "HD189733b"
    assert cfg.instrument.name == "ANDES_YJHK"
    assert cfg.observation.event_type == "transit"


def test_planet_config_fields():
    p = PlanetConfig(name="55Cnce", parameter_file="planet_params/55Cnce.json")
    assert p.name == "55Cnce"


def test_ccf_config_defaults():
    ccf = CrossCorrelationConfig()
    assert ccf.velocity_max_kms == 325.0
    assert ccf.velocity_step_kms == 1.0
    assert ccf.use_inverse_variance_weighting is True


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_to_dict_round_trip():
    cfg = SimulationConfig()
    d = cfg.to_dict()
    cfg2 = SimulationConfig.from_dict(d)
    assert cfg2.planet.name == cfg.planet.name
    assert cfg2.cross_correlation.velocity_max_kms == cfg.cross_correlation.velocity_max_kms


def test_json_round_trip():
    cfg = SimulationConfig()
    cfg.planet.name = "WASP-121b"
    cfg.cross_correlation.velocity_max_kms = 400.0
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as fh:
        tmp_path = Path(fh.name)
    cfg.to_json(tmp_path)
    cfg2 = SimulationConfig.from_json(tmp_path)
    assert cfg2.planet.name == "WASP-121b"
    assert cfg2.cross_correlation.velocity_max_kms == 400.0
    tmp_path.unlink()


# ---------------------------------------------------------------------------
# Load from the real example config
# ---------------------------------------------------------------------------

def test_load_example_config(hd189733b_config):
    cfg = hd189733b_config
    assert cfg.planet.name == "HD189733b"
    assert cfg.instrument.name == "ANDES_YJHK"
    assert cfg.observation.event_type == "transit"
    assert cfg.atmosphere.limb_asymmetries is True
    assert cfg.atmosphere.planet_model.use_easychem is True
    assert cfg.cross_correlation.velocity_max_kms == 325.0
    assert cfg.retrieval.enabled is False


def test_limb_config_loaded(hd189733b_config):
    cfg = hd189733b_config
    assert cfg.atmosphere.morning_day is not None
    assert cfg.atmosphere.evening_day is not None
    assert cfg.atmosphere.morning_day.wind_velocity_kms == -3.9


# ---------------------------------------------------------------------------
# Simulator validation
# ---------------------------------------------------------------------------

def test_simulator_validates_default():
    cfg = SimulationConfig()
    sim = ExoploreSimulator(cfg)
    assert sim is not None


def test_simulator_bad_event_raises():
    cfg = SimulationConfig()
    cfg.observation.event_type = "nightside"
    with pytest.raises(ValueError, match="event_type"):
        ExoploreSimulator(cfg)


def test_simulator_summary(default_config):
    sim = ExoploreSimulator(default_config)
    s = sim.summary()
    assert s.planet_name == "HD189733b"
    assert s.instrument == "ANDES_YJHK"


def test_simulator_str_summary(default_config):
    sim = ExoploreSimulator(default_config)
    text = str(sim.summary())
    assert "EXoPLORE" in text
    assert "HD189733b" in text


def test_simulator_output_dirs(default_config):
    sim = ExoploreSimulator(default_config)
    dirs = sim.output_dirs(simulation_name="test_run")
    assert "matrices" in dirs
    assert "plots" in dirs
    assert dirs["matrices"].name == "matrices"
