"""Area, sector geometry and Doppler limits of spatial two-column transfer."""
from dataclasses import replace

import numpy as np
import pytest

from exoplore.atmosphere.transmission import TwoColumnTransmission
from exoplore.config.models import AtmosphereConfig
from exoplore.stellar.coupled import SpatialTransitModel
from exoplore.stellar.planet_columns import sector_rays, absorbed_rays
from tests.test_spatial_rm_clv import system, stellar_bundle, choices, atmosphere, evaluate


def test_identical_columns_recover_uniform_single_column(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    column = atmosphere(system, spectral=True)
    _, single = evaluate(model, column)
    _, double = evaluate(model, TwoColumnTransmission(column, column))
    np.testing.assert_allclose(double.flux, single.flux, atol=3e-14, rtol=0)
    np.testing.assert_allclose(double.opaque_transmission, single.opaque_transmission, atol=3e-14)


def test_different_native_radii_recover_half_area_sum(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    morning = atmosphere(system)
    evening = replace(morning, impact_parameter_cm=morning.impact_parameter_cm*1.025)
    _, result = evaluate(model, TwoColumnTransmission(morning, evening))
    expected = 1-(morning.effective_radius_cm()[0]**2+
                   evening.effective_radius_cm()[0]**2)/(2*system.R_star**2)
    np.testing.assert_allclose(result.flux, expected, atol=3e-14, rtol=0)


def test_sector_weights_and_projected_rotation_coordinate(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    for leading in (True, False):
        rays = sector_rays(model, np.array([.1, .15]), 0., 0., None, leading)
        np.testing.assert_allclose(np.bincount(rays.rings, weights=rays.weights), np.pi)
        assert np.all(rays.x > 0) if leading else np.all(rays.x < 0)
        # The projected rotation coordinate approaches zero towards each pole.
        assert np.min(abs(rays.x)) < .005


def test_ingress_and_egress_distinguish_columns(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    absorbing = atmosphere(system)
    clear = atmosphere(system, transparent=True)
    _, morning = evaluate(model, TwoColumnTransmission(absorbing, clear), phase=(-.017, .017))
    _, evening = evaluate(model, TwoColumnTransmission(clear, absorbing), phase=(-.017, .017))
    assert morning.flux[0].mean() < evening.flux[0].mean()
    assert morning.flux[1].mean() > evening.flux[1].mean()


def test_local_velocity_moves_absorption_redward():
    wave = np.linspace(.5895, .5905, 1001)
    transmission = (1-.5*np.exp(-.5*((wave-.59)/.00002)**2))[:, None]
    flux = absorbed_rays(wave, np.array([10., -10.]), np.array([0, 0]), wave, transmission)
    peaks = wave[np.argmax(flux, axis=1)]
    np.testing.assert_allclose(peaks, .59*(1+np.array([10., -10.])/299792.458), atol=1e-6)


def test_rotation_and_wind_change_resolved_line(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    column = atmosphere(system, spectral=True)
    _, baseline = evaluate(model, TwoColumnTransmission(column, column))
    _, rotating = evaluate(model, TwoColumnTransmission(column, column, angular_speed_s=4e-4))
    assert np.max(abs(rotating.flux-baseline.flux)) > 1e-5
    # A common bulk wind moves the line without changing the occulted body.
    _, windy = evaluate(model, TwoColumnTransmission(column, column,
                                                    morning_wind_kms=10., evening_wind_kms=10.))
    assert np.argmin(windy.flux[0]) > np.argmin(baseline.flux[0])
    np.testing.assert_allclose(windy.opaque_transmission, baseline.opaque_transmission, atol=3e-14)


def test_configuration_rotation_validation():
    assert AtmosphereConfig().spatial_planet_rotation == 'synchronous_aligned'
    assert AtmosphereConfig(spatial_planet_rotation='off').spatial_planet_rotation == 'off'
    with pytest.raises(ValueError, match='spatial_planet_rotation'):
        AtmosphereConfig(spatial_planet_rotation='arbitrary')


def test_invalid_column_velocities_fail(system):
    column = atmosphere(system)
    with pytest.raises(ValueError):
        TwoColumnTransmission(column, column, morning_wind_kms=np.nan)
    with pytest.raises(ValueError):
        TwoColumnTransmission(column, column, angular_speed_s=-1.)


def test_native_wavelength_and_radial_grids_may_differ(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    morning = atmosphere(system)
    evening = replace(morning, wavelength_um=morning.wavelength_um[::2],
                      impact_parameter_cm=morning.impact_parameter_cm[::2],
                      transmission=morning.transmission[::2, ::2])
    _, result = evaluate(model, TwoColumnTransmission(morning, evening))
    expected = 1-(morning.effective_radius_cm()[0]**2+
                   evening.effective_radius_cm()[0]**2)/(2*system.R_star**2)
    np.testing.assert_allclose(result.flux, expected, atol=3e-14)


def test_identical_columns_with_stellar_clv_converge(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True),
                                       azimuth_points=64), system, 100000)
    column = atmosphere(system, spectral=True)
    _, single = evaluate(model, column, phase=(-.01, 0., .01))
    _, double = evaluate(model, TwoColumnTransmission(column, column), phase=(-.01, 0., .01))
    np.testing.assert_allclose(double.flux, single.flux, atol=1e-8, rtol=0)


def test_wavelength_coverage_fails_before_extrapolation(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    column = atmosphere(system)
    with pytest.raises(ValueError, match='wider wavelength coverage'):
        evaluate(model, TwoColumnTransmission(column, column, morning_wind_kms=1000.))


def test_spatial_builder_retains_each_column_inputs(monkeypatch, system):
    import sys
    from types import ModuleType, SimpleNamespace
    from exoplore.atmosphere import spatial_columns
    from exoplore.config.models import AtmosphereRegionConfig

    # Replace the expensive opacity engine while checking every hand-off to it.
    package = ModuleType('petitRADTRANS')
    radtrans = ModuleType('petitRADTRANS.radtrans')
    radtrans.Radtrans = lambda **kwargs: SimpleNamespace(**kwargs)
    package.physical_constants = SimpleNamespace(G=6.6743e-8)
    monkeypatch.setitem(sys.modules, 'petitRADTRANS', package)
    monkeypatch.setitem(sys.modules, 'petitRADTRANS.radtrans', radtrans)
    seen, exported = [], []
    def forward(*args, **kwargs):
        seen.append((args, kwargs))
        n = len(args[1])
        return None, None, {'H2': np.ones(n)}, np.full(n, args[5]), None, np.full(n, args[8])
    def rays(rt, **kwargs):
        exported.append(kwargs)
        return atmosphere(system)
    monkeypatch.setattr(spatial_columns, 'call_pRT', forward)
    monkeypatch.setattr(spatial_columns, 'calculate_chord_transmission', rays)
    morning = AtmosphereRegionConfig(mean_molecular_weight=2.4, wind_velocity_kms=3.)
    evening = replace(morning, mean_molecular_weight=3.1, wind_velocity_kms=-2.,
                      isothermal_temperature_K=1700., cloud_pressure_bar=.01)
    config = AtmosphereConfig(limb_asymmetries=True, morning_day=morning, evening_day=evening)
    pressures = (np.logspace(-6, 2, 8), np.logspace(-7, 2, 10))
    tables, wave, depth = spatial_columns.build_spatial_columns(
        object(), config, pressures, np.array([.589, .591]), system, object())
    assert [a[0][5] for a in seen] == [2.4, 3.1]
    assert seen[1][1]['P_cloud'] == .01
    assert [len(a['temperatures']) for a in exported] == [8, 10]
    assert exported[0]['reference_pressure'] == exported[1]['reference_pressure']
    assert tables.morning_wind_kms == 3.
    assert tables.evening_wind_kms == -2.
    assert tables.angular_speed_s == pytest.approx(2*np.pi/(system.orbital_period_days*86400))
    np.testing.assert_allclose(depth, (tables.morning.effective_radius_cm()/system.R_star)**2)


def test_two_columns_illuminate_heterogeneous_stellar_surface(tmp_path, system):
    from exoplore.config import StellarSurfaceConfig, StellarComponentConfig, SurfaceRegionConfig
    quiet = stellar_bundle(tmp_path)
    with np.load(quiet) as data:
        values = {key: data[key] for key in data.files}
    values['intensity'] = values['intensity']*.3
    values['continuum'] = values['continuum']*.3
    spot = tmp_path/'spot.npz'
    np.savez(spot, **values)
    surface = StellarSurfaceConfig(components={
        'quiet': StellarComponentConfig(str(quiet)),
        'spot': StellarComponentConfig(str(spot))},
        regions=[SurfaceRegionConfig('spot', x=.1, y=-.08, radius=.15)])
    model = SpatialTransitModel(choices(quiet), system, 100000, surface_config=surface)
    column = atmosphere(system, spectral=True)
    _, single = evaluate(model, column)
    _, two = evaluate(model, TwoColumnTransmission(column, column))
    np.testing.assert_allclose(two.flux, single.flux, atol=2e-7, rtol=0)
    # The same ray tables see different incident spectra when the spot is removed.
    _, quiet_result = evaluate(SpatialTransitModel(choices(quiet), system, 100000),
                               TwoColumnTransmission(column, column))
    assert np.max(abs(two.flux-quiet_result.flux)) > 1e-4


@pytest.mark.parametrize('invalid', ['weights', 'anchor', 'atomic'])
def test_two_column_route_checks_physical_inputs(tmp_path, invalid):
    from exoplore.config import SimulationConfig
    from exoplore.config.models import AtmosphereRegionConfig
    from exoplore.stellar.observation import validate_route
    config = SimulationConfig(rm_clv=choices(stellar_bundle(tmp_path)))
    config.atmosphere.limb_asymmetries = True
    config.atmosphere.morning_day = AtmosphereRegionConfig()
    config.atmosphere.evening_day = AtmosphereRegionConfig()
    route = 'standard'
    if invalid == 'weights':
        config.atmosphere.limb_divisions = 'simplified_step'
    elif invalid == 'anchor':
        config.atmosphere.evening_day.reference_pressure_bar = 1.
    else:
        route = 'atomic_simulation'
    with pytest.raises(ValueError):
        validate_route(config, route)


def test_two_column_route_accepts_existing_unused_night_blocks(tmp_path):
    from exoplore.config import SimulationConfig
    from exoplore.config.models import AtmosphereRegionConfig
    from exoplore.stellar.observation import validate_route
    config = SimulationConfig(rm_clv=choices(stellar_bundle(tmp_path)))
    config.atmosphere.limb_asymmetries = True
    for name in ('morning_day', 'evening_day', 'morning_night', 'evening_night'):
        setattr(config.atmosphere, name, AtmosphereRegionConfig())
    validate_route(config, 'standard')
    restored = SimulationConfig.from_dict(config.to_dict())
    assert restored.atmosphere.spatial_planet_rotation == 'synchronous_aligned'
    validate_route(restored, 'standard')
