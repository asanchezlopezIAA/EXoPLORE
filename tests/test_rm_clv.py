"""Configuration, instrument sampling and fixed RM/CLV correction checks."""

from dataclasses import replace

import numpy as np
import pytest

from exoplore.config import RMCLVConfig, SimulationConfig
from exoplore.planets import load_planet
from exoplore.stellar.observation import (
    TransitStellarModel, correct_stellar_distortion,
    validate_route,
)
from exoplore.io.stellar_surface import load_local_intensities


@pytest.fixture
def bundle(tmp_path):
    """An analytic specific-intensity line with known angular dependence."""
    wave = np.linspace(0.588, 0.592, 2001)
    mu = np.linspace(0, 1, 11)
    continuum = (0.4 + 0.6*mu[:, None])*np.ones_like(wave)
    depth = 0.2 + 0.5*mu[:, None]
    intensity = continuum*(1 - depth*np.exp(-0.5*((wave-0.59)/0.00002)**2))
    path = tmp_path/'intensities.npz'
    np.savez(path, wavelength_um=wave, mu=mu, intensity=intensity,
             continuum=continuum, intensity_kind='specific_intensity',
             wavelength_medium='vacuum', intensity_unit='arbitrary_common_unit',
             source='analytic test profile', resolving_power=0.0)
    return path


def choices(bundle, **kwargs):
    return RMCLVConfig(model_file=str(bundle), projected_obliquity_deg=0,
                       cells_across_diameter=30, **kwargs)


def planet():
    return load_planet('planet_params/HD189733b.json')


def evaluate(bundle, **kwargs):
    model = TransitStellarModel(choices(bundle, **kwargs), planet(), 100000)
    return model.evaluate(np.linspace(0.5897, 0.5903, 101), [-0.1, -0.01, 0, 0.01, 0.1],
                          np.full(5, -2.361), 30.0)


def test_omitted_controls_are_disabled():
    config = SimulationConfig.from_dict({})
    assert not config.rm_clv.active
    assert not config.rm_clv.simulate
    assert not config.rm_clv.correct


def test_json_round_trip_preserves_all_controls(bundle, tmp_path):
    config = SimulationConfig(rm_clv=choices(bundle, simulate=True, correct=True))
    path = tmp_path/'run.json'
    config.to_json(path)
    assert SimulationConfig.from_json(path).to_dict() == config.to_dict()


@pytest.mark.parametrize('fields', [
    {'simulate': 'false'}, {'correct': 1}, {'effects': 'other'},
    {'normalization': 'guess'}, {'cells_across_diameter': True},
    {'subpixels_per_axis': 0}, {'exposure_subsamples': 1.5},
    {'projected_obliquity_deg': float('nan')}, {'wavelength_medium': 'guess'},
])
def test_invalid_controls_fail(fields):
    with pytest.raises(ValueError):
        RMCLVConfig(**fields)


def test_unknown_option_fails():
    with pytest.raises(TypeError):
        SimulationConfig.from_dict({'rm_clv': {'simualte': True}})


def test_active_model_requires_source_and_rm_geometry(bundle):
    with pytest.raises(ValueError, match='model_file'):
        RMCLVConfig(simulate=True)
    with pytest.raises(ValueError, match='obliquity'):
        RMCLVConfig(simulate=True, model_file=str(bundle))
    assert RMCLVConfig(simulate=True, effects='clv', model_file=str(bundle)).active


@pytest.mark.parametrize('simulate,correct', [(False, True), (True, False), (True, True)])
def test_independent_simulation_and_correction_switches(bundle, simulate, correct):
    config = SimulationConfig(rm_clv=choices(bundle, simulate=simulate, correct=correct))
    validate_route(config, 'standard')


def test_disabled_effects_leave_existing_routes_untouched():
    config = SimulationConfig()
    for route in ['standard', 'helium_simulation', 'helium_observed', 'helium_grid']:
        validate_route(config, route)


def test_unsupported_routes_and_observations_fail_early(bundle):
    config = SimulationConfig(rm_clv=choices(bundle, simulate=True))
    with pytest.raises(ValueError, match='standard'):
        validate_route(config, 'helium_simulation')
    config.observation.event_type = 'dayside'
    with pytest.raises(ValueError, match='transit'):
        validate_route(config, 'standard')
    config.observation.event_type = 'transit'
    config.observation.use_real_data = True
    with pytest.raises(ValueError, match='measured data'):
        validate_route(config, 'standard')
    config.rm_clv = choices(bundle, correct=True)
    validate_route(config, 'standard')


def test_air_vacuum_mismatch_is_reported(bundle):
    with pytest.raises(ValueError, match='wavelength convention'):
        load_local_intensities(bundle, wavelength_medium='air')


def test_no_telluric_simulation_can_select_stellar_effects(bundle):
    config = SimulationConfig(rm_clv=choices(bundle, simulate=True))
    config.tellurics.include_tellurics = False
    validate_route(config, 'standard')


def test_active_stellar_choices_are_in_the_public_summary(bundle):
    from exoplore.core import ExoploreSimulator
    config = SimulationConfig(rm_clv=choices(bundle, simulate=True, correct=True))
    summary = str(ExoploreSimulator(config).summary())
    assert 'Stellar effects' in summary
    assert 'Simulate RM/CLV     : True' in summary
    assert 'Correct RM/CLV      : True' in summary


def test_zero_rotation_rm_control_has_no_line_distortion(bundle):
    system = replace(planet(), v_rotsini_kms=0.)
    model = TransitStellarModel(choices(bundle, effects='rm'), system, 100000)
    result = model.evaluate(np.linspace(0.5897, 0.5903, 101), [-0.01, 0, 0.01], [0, 0, 0], 30.)
    np.testing.assert_allclose(result.distortion, 1, atol=1e-13)


def test_observer_velocity_moves_reference_and_distortion_together(bundle):
    from exoplore.stellar.surface import C_KMS
    model = TransitStellarModel(choices(bundle), planet(), 100000)
    wave = np.linspace(0.5897, 0.5903, 101)
    rest = model.evaluate(wave, [0], [0], 30.)
    shifted = model.evaluate(wave*(1+20/C_KMS), [0], [20], 30.)
    np.testing.assert_allclose(shifted.reference, rest.reference, atol=1e-10)
    np.testing.assert_allclose(shifted.distortion, rest.distortion, atol=1e-10)


def test_incomplete_model_metadata_fails(tmp_path):
    path = tmp_path/'undocumented.npz'
    np.savez(path, intensity=np.ones((2, 3)))
    with pytest.raises(ValueError, match='lacks'):
        load_local_intensities(path, wavelength_medium='vacuum')


def test_observed_spectra_have_expected_shape_and_outside_unity(bundle):
    result = evaluate(bundle)
    assert result.distortion.shape == result.reference.shape == (5, 101)
    assert np.isfinite(result.distortion).all()
    np.testing.assert_allclose(result.distortion[[0, -1]], 1, atol=1e-13)
    assert np.ptp(result.distortion[2]) > 0.001


@pytest.mark.parametrize('effects', ['rm', 'clv', 'rm_clv'])
def test_all_effect_controls_execute(bundle, effects):
    result = evaluate(bundle, effects=effects)
    assert np.isfinite(result.distortion).all()
    assert np.all(result.distortion > 0)


def test_correction_recovers_injected_flux_and_propagates_noise(bundle):
    distortion = evaluate(bundle).distortion
    flux = np.full_like(distortion, 0.98)
    error = np.full_like(distortion, 0.001)
    corrected, propagated = correct_stellar_distortion(flux*distortion, error, distortion)
    np.testing.assert_allclose(corrected, flux, atol=1e-15)
    np.testing.assert_allclose(propagated, error/distortion)
    # The input arrays remain available as the raw-data diagnostics.
    np.testing.assert_array_equal(error, np.full_like(distortion, 0.001))


def test_correction_rejects_invalid_model():
    with pytest.raises(ValueError, match='positive'):
        correct_stellar_distortion(np.ones((2, 3)), np.ones((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match='shapes'):
        correct_stellar_distortion(np.ones((2, 3)), np.ones(3), np.ones((2, 3)))


def test_exposure_averaging_has_stable_short_exposure_limit(bundle):
    short = choices(bundle, exposure_subsamples=1)
    model_a = TransitStellarModel(short, planet(), 100000)
    model_b = TransitStellarModel(replace(short, exposure_subsamples=5), planet(), 100000)
    wave = np.linspace(0.5897, 0.5903, 51)
    a = model_a.evaluate(wave, [0], [0], 0.0001)
    b = model_b.evaluate(wave, [0], [0], 0.0001)
    np.testing.assert_allclose(a.distortion, b.distortion, atol=1e-13)


def test_synthesis_resolution_below_instrument_fails(bundle, tmp_path):
    with np.load(bundle) as source:
        contents = dict(source)
    contents['resolving_power'] = 10000.
    path = tmp_path/'low_resolution.npz'
    np.savez(path, **contents)
    with pytest.raises(ValueError, match='lower'):
        TransitStellarModel(choices(path), planet(), 100000)


def test_band_mean_is_an_explicit_finite_window_normalization(bundle, tmp_path):
    with np.load(bundle) as source:
        contents = dict(source)
    contents.pop('continuum')
    path = tmp_path/'no_continuum.npz'
    np.savez(path, **contents)
    with pytest.raises(ValueError, match='continuum intensities'):
        evaluate(path)
    result = evaluate(path, normalization='band_mean')
    np.testing.assert_allclose(np.mean(result.distortion, axis=1), 1, atol=1e-13)
