"""Check preparation bookkeeping without changing the normalized spectra."""

import numpy as np
import pytest

from exoplore.config.models import PipelineConfig, TelluricConfig
from exoplore.pipelines.models import PreparationInputs
from exoplore.pipelines.prepare import preparing_pipeline
from exoplore.pipelines.runner import prepare_observation


@pytest.fixture
def inputs_and_arrays():
    inputs = PreparationInputs(PipelineConfig(name='Blain24'),
                               TelluricConfig(include_tellurics=False), False, 2)
    wave = np.linspace(1.5, 1.51, 20)
    flux = (1 + np.linspace(-0.01, 0.01, 5)[:, None])*np.linspace(0.9, 1, 20)
    noise = np.full_like(flux, 0.01)
    return (inputs, flux, noise, wave, np.arange(20), np.array([], dtype=int),
            np.linspace(1.1, 1.3, 5), np.linspace(-0.02, 0.02, 5), np.array([0, 4]), None)


def test_fixed_preparation_preserves_numeric_pair_and_supplies_runner_masks(inputs_and_arrays):
    args = inputs_and_arrays
    pair = preparing_pipeline(*args)
    result = prepare_observation(*args)
    assert len(pair) == 2
    assert len(result) == 9
    np.testing.assert_array_equal(result[0], pair[0])
    np.testing.assert_array_equal(result[1], pair[1])
    np.testing.assert_array_equal(result[2], args[4])
    np.testing.assert_array_equal(result[3], args[5])
    assert result[4] == 0
    assert result[8].shape == (5, 2)
    assert np.all(result[8] == 0)


def test_model_only_fixed_preparation_returns_the_flux_array(inputs_and_arrays):
    args = inputs_and_arrays
    expected, _ = preparing_pipeline(*args, masks=False, correct_uncertainties=False)
    actual = prepare_observation(*args, masks=False, correct_uncertainties=False)
    np.testing.assert_array_equal(actual, expected)


def test_fixed_preparation_robust_halt_bookkeeping(inputs_and_arrays):
    args = inputs_and_arrays
    args[0].pipeline.sysrem_robust_halt = True
    result = prepare_observation(*args)
    assert len(result) == 5
    assert result[-1] == 0


def test_standard_preparation_result_is_passed_through(monkeypatch, inputs_and_arrays):
    from exoplore.pipelines import runner
    args = inputs_and_arrays
    args[0].tellurics.include_tellurics = True
    result = object()
    monkeypatch.setattr(runner, 'preparing_pipeline', lambda *a, **k: result)
    assert prepare_observation(*args) is result
