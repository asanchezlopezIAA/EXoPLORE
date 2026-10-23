"""Configuration and active-simulator routing of throughput random streams."""

import numpy as np
import pytest

from exoplore.config import SimulationConfig
from exoplore.config.models import ObservationConfig
from exoplore.core import simulator as simulator_module


@pytest.mark.parametrize("settings, expected", [
    ({}, None), ({"throughput_seed": None}, None),
    ({"throughput_seed": 12345}, 12345), ({"throughput_seed": 0}, 0),
])
def test_config_load_and_json_roundtrip(tmp_path, settings, expected):
    config = SimulationConfig.from_dict({"observation": settings})
    assert config.observation.throughput_seed == expected
    path = tmp_path / "config.json"
    config.to_json(path)
    restored = SimulationConfig.from_json(path)
    assert restored.to_dict() == config.to_dict()


@pytest.mark.parametrize("seed", [-1, 1.5, True, "12345"])
def test_invalid_throughput_seed_rejected(seed):
    with pytest.raises(ValueError, match="throughput_seed"):
        SimulationConfig.from_dict({"observation": {"throughput_seed": seed}})


def test_reference_config_sets_two_separate_seeds():
    config = SimulationConfig.from_json("configs/hd189733b_andes_transit_clean.json")
    assert config.observation.throughput_seed == config.noise.noise_seed == 12345


def test_seeded_throughput_repeats_without_using_global_rng():
    observation = ObservationConfig(throughput_seed=12345)
    flux = np.arange(128 * 3, dtype=float).reshape(128, 3) + 1
    first = simulator_module._apply_throughput(flux, observation, 35, 0)
    state_before = np.random.get_state()
    second = simulator_module._apply_throughput(flux, observation, 35, 0)
    state_after = np.random.get_state()
    np.testing.assert_array_equal(first, second)
    for a, b in zip(state_before, state_after):
        np.testing.assert_equal(a, b)
    assert not np.array_equal(first, flux)


def test_order_night_and_base_seed_have_distinct_streams():
    flux = np.ones((128, 3))
    def draw(seed, order, night):
        return simulator_module._apply_throughput(
            flux, ObservationConfig(throughput_seed=seed), order, night)
    reference = draw(12345, 35, 0)
    for identity in [(12345, 36, 0), (12345, 35, 1), (54321, 35, 0)]:
        assert not np.array_equal(reference, draw(*identity))


def test_order_selection_invariance():
    flux = np.ones((128, 2))
    observation = ObservationConfig(throughput_seed=12345)
    forward = {order: simulator_module._apply_throughput(flux, observation, order, 0)
               for order in [34, 35, 36]}
    reordered = {order: simulator_module._apply_throughput(flux, observation, order, 0)
                 for order in [36, 34, 35]}
    alone = simulator_module._apply_throughput(flux, observation, 35, 0)
    for order in forward:
        np.testing.assert_array_equal(forward[order], reordered[order])
    np.testing.assert_array_equal(forward[35], alone)


def test_seed_only_change_preserves_exact_original_throughput_operation():
    """No scaling/renormalisation: same existing routine, same 2% normal draw."""
    observation = ObservationConfig(throughput_seed=12345)
    flux = np.arange(128 * 3, dtype=float).reshape(128, 3) + 1
    child_seed = int(np.random.SeedSequence([12345, 35, 0, 1]).generate_state(1)[0])
    factors = np.random.default_rng(child_seed).normal(
        loc=1.0, scale=0.02, size=flux.shape[0])
    expected = flux * factors[:, None]
    actual = simulator_module._apply_throughput(flux, observation, 35, 0)
    np.testing.assert_array_equal(actual, expected)


def test_none_keeps_unseeded_call_and_original_jitter(monkeypatch):
    calls = []
    def spy(**kwargs):
        calls.append(kwargs)
        return kwargs["F"] * 1.02
    monkeypatch.setattr(simulator_module, "add_throughput", spy)
    flux = np.ones((4, 3))
    output = simulator_module._apply_throughput(flux, ObservationConfig(), 35, 0)
    assert calls[0]["seed"] is None
    assert calls[0]["jitter_frac"] == 0.02
    assert calls[0]["F"] is flux
    np.testing.assert_array_equal(output, flux * 1.02)


@pytest.mark.parametrize("settings", [
    {"add_throughput_variations": False}, {"noiseless": True},
])
def test_disabled_throughput_skips_rng_and_copies_flux(monkeypatch, settings):
    def forbidden(**kwargs):
        pytest.fail("Disabled/noiseless throughput must not draw randomness")
    monkeypatch.setattr(simulator_module, "add_throughput", forbidden)
    flux = np.ones((4, 3))
    observation = ObservationConfig(throughput_seed=12345, **settings)
    output = simulator_module._apply_throughput(flux, observation, 35, 0)
    np.testing.assert_array_equal(output, flux)
    assert not np.shares_memory(output, flux)


def test_gaussian_and_throughput_seeds_are_separate(monkeypatch):
    config = SimulationConfig()
    config.observation.throughput_seed = 12345
    config.noise.noise_seed = 12345
    gaussian_rng = np.random.default_rng(config.noise.noise_seed)
    expected_rng = np.random.default_rng(config.noise.noise_seed)
    gaussian_rng.normal(size=10)
    expected_rng.normal(size=10)
    calls = []
    original = simulator_module.add_throughput
    def record(**kwargs):
        calls.append(kwargs["seed"])
        return original(**kwargs)
    monkeypatch.setattr(simulator_module, "add_throughput", record)
    flux = np.ones((32, 2))
    first = simulator_module._apply_throughput(flux, config.observation, 35, 0)
    config.noise.noise_seed = 54321
    second = simulator_module._apply_throughput(flux, config.observation, 35, 0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(gaussian_rng.normal(size=10), expected_rng.normal(size=10))
    assert calls[0] == calls[1] != 12345
    config.observation.throughput_seed = 54321
    changed = simulator_module._apply_throughput(flux, config.observation, 35, 0)
    assert not np.array_equal(first, changed)
    assert config.noise.noise_seed == 54321
