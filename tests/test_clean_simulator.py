"""Tests for ExoploreSimulator, replaces the stale version from a prior session."""

from pathlib import Path

from exoplore.config import SimulationConfig, load_simulation_config
from exoplore.core.simulator import ExoploreSimulator, SimulationSummary


def test_clean_simulator_summary():
    config = load_simulation_config(
        Path("configs/hd189733b_andes_transit_clean.json")
    )

    simulator = ExoploreSimulator(config)
    summary = simulator.summarize()

    assert isinstance(summary, SimulationSummary)
    assert summary.planet_name == "HD189733b"   # clean name
    assert summary.planet == "HD189733b"        # backward-compat alias
    assert summary.instrument == "ANDES_YJHK"
    assert summary.event_type == "transit"      # clean name
    assert summary.n_nights == 1
    assert summary.pipeline == "Blain24"
    assert summary.retrieval_enabled is False
