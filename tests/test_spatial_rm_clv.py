"""Physical limiting cases for joint stellar and atmospheric transfer."""

from dataclasses import replace

import numpy as np
import pytest

from exoplore.atmosphere.transmission import ChordTransmission
from exoplore.config import RMCLVConfig, SimulationConfig
from exoplore.planets import load_planet
from exoplore.stellar.coupled import SpatialTransitModel
from exoplore.stellar.observation import validate_route


@pytest.fixture
def system():
    return load_planet('planet_params/HD189733b.json')


def stellar_bundle(tmp_path, *, line=False, clv=False, resolution=0):
    """A uniform continuum with optional stellar absorption and line CLV."""
    wave = np.linspace(.5885, .5915, 1001)
    mu = np.array([0., .5, 1.])
    continuum = np.ones((3, len(wave)))
    intensity = continuum.copy()
    if line:
        depth = (.2+.5*mu[:, None]) if clv else .5
        intensity *= 1-depth*np.exp(-.5*((wave-.59)/.000035)**2)
    path = tmp_path/'local.npz'
    np.savez(path, wavelength_um=wave, mu=mu, intensity=intensity, continuum=continuum,
             intensity_kind='specific_intensity', wavelength_medium='vacuum',
             intensity_unit='common arbitrary unit', source='analytic test spectrum',
             resolving_power=resolution)
    return path


def choices(path, **overrides):
    args = dict(simulate=True, coupling='spatial', model_file=str(path),
                projected_obliquity_deg=0., stellar_radial_points=12,
                planet_radial_points=12, azimuth_points=32, exposure_subsamples=1)
    args.update(overrides)
    return RMCLVConfig(**args)


def atmosphere(system, *, transparent=False, spectral=False):
    wave = np.linspace(.589, .591, 1001)
    radius = np.linspace(system.R_pl*1.06, system.R_pl*.98, 20)
    depth = 2.0
    if spectral:
        depth = .05+3*np.exp(-.5*((wave[:, None]-.59)/.00004)**2)
    optical_depth = depth*np.linspace(0, 1, len(radius))[None, :]
    transmission = np.broadcast_to(np.exp(-optical_depth), (len(wave), len(radius))).copy()
    if transparent:
        transmission[:] = 1
    return ChordTransmission(wave, radius, transmission)


def evaluate(model, chords, *, phase=(0.,), velocity=0., star_velocity=0., seconds=.001):
    wave = np.linspace(.5896, .5904, 61)
    result = model.evaluate(wave, phase, star_velocity, seconds,
                            chords=chords, planet_velocity_kms=velocity)
    return wave, result


def test_spatial_option_round_trips_and_factorized_is_default():
    assert SimulationConfig().rm_clv.coupling == 'factorized'
    config = SimulationConfig(rm_clv=RMCLVConfig(coupling='spatial'))
    assert SimulationConfig.from_dict(config.to_dict()).rm_clv.coupling == 'spatial'


def test_unknown_coupling_is_rejected():
    with pytest.raises(ValueError, match='coupling'):
        RMCLVConfig(coupling='unknown')


def test_config_changes_after_preview_are_checked_before_execution(tmp_path):
    from exoplore.core import ExoploreSimulator
    config = SimulationConfig(rm_clv=choices(stellar_bundle(tmp_path)))
    simulator = ExoploreSimulator(config)
    config.atmosphere.limb_asymmetries = True
    with pytest.raises(ValueError, match='1D atmosphere'):
        simulator.run()


def test_unreviewed_prt_version_stops_before_opacity_export(monkeypatch):
    from exoplore.atmosphere import transmission
    monkeypatch.setattr(transmission.importlib.metadata, 'version', lambda name: '99.0.0')
    with pytest.raises(NotImplementedError, match='3.2.0'):
        transmission.calculate_chord_transmission(object(), temperatures=None,
            mass_fractions=None, mean_molar_masses=None, reference_gravity=None,
            reference_pressure=None, planet_radius=None)


@pytest.mark.parametrize('setting', ['limbs', 'retrieval', 'external', 'scale', 'boxcar', 'air'])
def test_unsupported_spatial_combinations_fail_early(tmp_path, setting):
    config = SimulationConfig(rm_clv=choices(stellar_bundle(tmp_path)))
    if setting == 'limbs':
        config.atmosphere.limb_asymmetries = True
    elif setting == 'retrieval':
        config.retrieval.enabled = True
    elif setting == 'external':
        config.observation.external_planet_model = True
    elif setting == 'scale':
        config.observation.scale_injection = 10
    elif setting == 'air':
        config.rm_clv.wavelength_medium = 'air'
    else:
        config.observation.signal_uses_light_curve = False
    with pytest.raises(ValueError):
        validate_route(config, 'standard')


def test_uniform_star_recovers_native_atmospheric_area(tmp_path, system):
    chords = atmosphere(system)
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    _, result = evaluate(model, chords)
    # A grey atmosphere has the same effective area at every wavelength.
    # This independently checks angular weights, radial weights, the opaque
    # base radius and the absence of a second white-light transit factor.
    expected = 1-(chords.effective_radius_cm()/system.R_star)**2
    np.testing.assert_allclose(result.flux, expected[0], atol=3e-14, rtol=0)
    np.testing.assert_allclose(result.reference, 1, atol=1e-14)
    np.testing.assert_allclose(result.distortion, 1, atol=1e-14)


def test_transparent_atmosphere_retains_only_opaque_base(tmp_path, system):
    chords = atmosphere(system, transparent=True)
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    _, result = evaluate(model, chords)
    expected = 1-(chords.impact_parameter_cm[-1]/system.R_star)**2
    np.testing.assert_allclose(result.flux, expected, atol=3e-14, rtol=0)
    np.testing.assert_allclose(result.opaque_transmission, expected, atol=3e-14)


def test_opaque_control_uses_configured_planet_radius(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    _, result = evaluate(model, None)
    np.testing.assert_allclose(result.flux, 1-(system.R_pl/system.R_star)**2, atol=3e-14)


def test_no_overlap_and_secondary_conjunction_are_unocculted(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True)), system, 100000)
    _, result = evaluate(model, atmosphere(system), phase=(-.1, .1, .5))
    np.testing.assert_allclose(result.flux, result.reference, atol=1e-14)
    np.testing.assert_allclose(result.distortion, 1, atol=1e-14)


def test_stellar_line_and_atmosphere_are_combined_locally(tmp_path, system):
    path = stellar_bundle(tmp_path, line=True, clv=True)
    model = SpatialTransitModel(choices(path), system, 100000)
    wave, result = evaluate(model, atmosphere(system, spectral=True))
    _, transparent = evaluate(model, atmosphere(system, transparent=True))
    assert np.max(transparent.flux-result.flux) > 1e-4
    # The opaque-body correction is independent of the atmospheric line and
    # therefore leaves the atmospheric absorption available to the analysis.
    np.testing.assert_allclose(result.distortion, transparent.distortion, atol=1e-14)
    assert (result.flux/result.distortion)[0, np.argmin(abs(wave-.59))] < result.reference[0, 30]


def test_local_line_clv_changes_atmospheric_absorption_relative_to_factorization(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True)), system, 100000)
    chords = atmosphere(system)
    _, coupled = evaluate(model, chords)
    _, bare = evaluate(model, atmosphere(system, transparent=True))
    base_depth = (chords.impact_parameter_cm[-1]/system.R_star)**2
    extra_depth = (chords.effective_radius_cm()[0]/system.R_star)**2-base_depth
    # This controlled factorization illuminates the annulus with the mean
    # unocculted spectrum. The joint calculation uses the actual background
    # stellar line at each annular position, giving a different line shape.
    factorized = bare.flux*(1-extra_depth/(1-base_depth))
    assert np.max(abs(coupled.flux-factorized)) > 2e-6


def test_photospheric_correction_retains_local_illumination_in_the_atmosphere(tmp_path, system):
    # The atmospheric transmission is grey in this controlled case. The local
    # stellar line varies with mu, so its annular illumination can still leave
    # spectral structure after correcting the opaque body's photospheric signal.
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True)),
                                system, 100000)
    _, result = evaluate(model, atmosphere(system))
    remaining = result.flux / (result.distortion * result.reference
                               * result.opaque_transmission)
    assert np.ptp(remaining[0]) > 2e-6

    # A uniform stellar continuum gives the wavelength-independent atmospheric
    # attenuation expected for the same grey chord transmissions.
    uniform = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    _, control = evaluate(uniform, atmosphere(system))
    residual = control.flux / (control.distortion * control.reference
                               * control.opaque_transmission)
    assert np.ptp(residual[0]) < 1e-12


def test_opaque_ingress_and_egress_match_circle_overlap(tmp_path, system):
    distances = np.array([.94, 1.0, 1.08])
    scaled_a = system.a/(system.R_star/1e5)
    cosi = np.cos(system.inc)
    phase = np.arcsin(np.sqrt((distances**2/scaled_a**2-cosi**2)/(1-cosi**2)))/(2*np.pi)
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path), planet_radial_points=48,
                                        azimuth_points=64), system, 100000)
    _, result = evaluate(model, None, phase=phase)
    radius = system.R_pl/system.R_star
    overlap = (radius**2*np.arccos((distances**2+radius**2-1)/(2*distances*radius))
               + np.arccos((distances**2+1-radius**2)/(2*distances))
               - .5*np.sqrt((-distances+radius+1)*(distances+radius-1)
                             *(distances-radius+1)*(distances+radius+1)))
    np.testing.assert_allclose(result.flux[:, 0], 1-overlap/np.pi, atol=2e-7, rtol=0)


def test_planet_bulk_velocity_shifts_atmospheric_feature(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path)), system, 100000)
    wave, rest = evaluate(model, atmosphere(system, spectral=True))
    _, moving = evaluate(model, atmosphere(system, spectral=True), velocity=20.)
    assert wave[np.argmin(moving.flux[0])] > wave[np.argmin(rest.flux[0])]


def test_short_exposure_limit_is_stable(tmp_path, system):
    path = stellar_bundle(tmp_path)
    one = SpatialTransitModel(choices(path), system, 100000)
    three = SpatialTransitModel(choices(path, exposure_subsamples=3), system, 100000)
    _, a = evaluate(one, atmosphere(system), seconds=.0001)
    _, b = evaluate(three, atmosphere(system), seconds=.0001)
    np.testing.assert_allclose(a.flux, b.flux, atol=1e-13)


def test_exposure_averaging_broadens_a_moving_atmospheric_line(tmp_path, system):
    path = stellar_bundle(tmp_path)
    original = atmosphere(system)
    profile = 3*np.exp(-.5*((original.wavelength_um[:, None]-.59)/.000004)**2)
    optical_depth = profile*np.linspace(0, 1, len(original.impact_parameter_cm))
    chords = replace(original, transmission=np.exp(-optical_depth))
    wave = np.linspace(.58995, .59005, 121)
    midpoint = SpatialTransitModel(choices(path), system, 100000)
    averaged = SpatialTransitModel(choices(path, exposure_subsamples=9), system, 100000)
    a = midpoint.evaluate(wave, [0], [0], 1800., chords=chords, planet_velocity_kms=[0])
    b = averaged.evaluate(wave, [0], [0], 1800., chords=chords, planet_velocity_kms=[0])
    assert b.flux.min() > a.flux.min()
    # The entire atmosphere remains in front of the uniform star during this
    # exposure. Doppler smearing redistributes its line absorption in velocity.
    continuum = 1-(chords.impact_parameter_cm[-1]/system.R_star)**2
    area_a = np.trapz(continuum-a.flux[0], wave)
    area_b = np.trapz(continuum-b.flux[0], wave)
    np.testing.assert_allclose(area_b, area_a, rtol=.01)


def test_spatial_mode_rejects_preconvolved_stellar_spectra(tmp_path, system):
    with pytest.raises(ValueError, match='resolving_power=0'):
        SpatialTransitModel(choices(stellar_bundle(tmp_path, resolution=1000000)), system, 100000)


def test_spatial_mode_accepts_positive_mu_minimum_when_rays_are_covered(tmp_path, system):
    path = stellar_bundle(tmp_path)
    with np.load(path) as source:
        content = dict(source)
    content['mu'][0] = 1e-4
    np.savez(path, **content)
    model = SpatialTransitModel(choices(path), system, 100000)
    _, result = evaluate(model, None)
    assert np.isfinite(result.flux).all()


def test_spatial_rays_reject_uncovered_limb_angles(tmp_path, system):
    path = stellar_bundle(tmp_path)
    with np.load(path) as source:
        content = dict(source)
    content['mu'][0] = .005
    np.savez(path, **content)
    model = SpatialTransitModel(choices(path), system, 100000)
    # This grazing ring samples closer to the limb than the Cartesian preview.
    with pytest.raises(ValueError, match='Angular extrapolation is disabled'):
        model._rays(np.array([.1]), .9, 0.)


def test_chord_radius_identity_for_transparent_and_opaque_limits(system):
    clear = atmosphere(system, transparent=True)
    np.testing.assert_allclose(clear.effective_radius_cm(), clear.impact_parameter_cm[-1])
    opaque = replace(clear, transmission=np.zeros_like(clear.transmission))
    np.testing.assert_allclose(opaque.effective_radius_cm(), clear.impact_parameter_cm[0], rtol=1e-14)


@pytest.mark.parametrize('kind', ['wavelength', 'radius', 'range', 'shape', 'nan'])
def test_invalid_chord_data_are_rejected(system, kind):
    source = atmosphere(system)
    wave, radius, transmission = source.wavelength_um.copy(), source.impact_parameter_cm.copy(), source.transmission.copy()
    if kind == 'wavelength':
        wave = wave[::-1]
    elif kind == 'radius':
        radius = radius[::-1]
    elif kind == 'range':
        transmission[0, 0] = 1.1
    elif kind == 'shape':
        transmission = transmission.T
    else:
        transmission[0, 0] = np.nan
    with pytest.raises(ValueError):
        ChordTransmission(wave, radius, transmission)


def _reference_angular_flux(model, wave, rays, count):
    """Independent interpolation of each ray for the acceleration checks."""
    intensity = np.zeros((count, len(wave)))
    continuum = np.zeros_like(intensity)
    if rays is None:
        return intensity, continuum
    active, upper, fraction, velocity, weights = rays
    local = model.surface.local
    for ring, destination in enumerate(active):
        for angle in range(model.choices.azimuth_points):
            ray = ring*model.choices.azimuth_points+angle
            rest = wave/(1+velocity[ray]/299792.458)
            for values, output in ((local.intensity, intensity),
                                   (local.continuum, continuum)):
                lower = np.interp(rest, local.wavelength_um, values[upper[ray]-1])
                higher = np.interp(rest, local.wavelength_um, values[upper[ray]])
                output[destination] += ((1-fraction[ray])*lower
                                        + fraction[ray]*higher)*weights[ring, angle]
    return intensity, continuum


@pytest.mark.parametrize('position', [(0., -.6), (.8, -.6), (1.4, -.6)])
@pytest.mark.parametrize('effects', ['rm', 'clv', 'rm_clv'])
def test_accelerated_angular_flux_matches_independent_ray_interpolation(
        tmp_path, system, position, effects):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True),
                                      effects=effects), system, 94600.)
    radii = np.linspace(.01, .2, 19)
    rays = model._rays(radii, *position)
    # Include interpolation knots, endpoints of chunks, and unordered channels.
    wave = np.r_[np.linspace(.5895, .5905, 137), .59, .5897]
    actual = model._angular_flux(wave, rays, len(radii))
    expected = _reference_angular_flux(model, wave, rays, len(radii))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=3e-14, atol=3e-14)


def test_accelerated_full_transfer_preserves_reference_fluxes(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True),
        exposure_subsamples=3), system, 94600.)
    chords = atmosphere(system, spectral=True)
    phase = (-.018, 0., .018)
    _, actual = evaluate(model, chords, phase=phase, seconds=198.)
    # Replace only the interpolation implementation. The same geometry,
    # atmospheric rays, quadrature, exposure sampling and convolution run again.
    model._angular_flux = lambda wave, rays, count: _reference_angular_flux(
        model, wave, rays, count)
    _, expected = evaluate(model, chords, phase=phase, seconds=198.)
    for field in ('flux', 'reference', 'distortion', 'opaque_transmission'):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                   rtol=3e-14, atol=3e-14)


def test_intrinsic_export_preserves_the_default_flux(tmp_path, system):
    model = SpatialTransitModel(choices(stellar_bundle(tmp_path, line=True, clv=True)),
                                system, 94600.)
    chords = atmosphere(system, spectral=True)
    wave = np.linspace(.5896, .5904, 61)
    args = dict(chords=chords, planet_velocity_kms=0.)
    plain = model.evaluate(wave, [0.], 0., 198., **args)
    exported = model.evaluate(wave, [0.], 0., 198., return_intrinsic=True, **args)
    for field in ('flux', 'reference', 'distortion', 'opaque_transmission'):
        np.testing.assert_array_equal(getattr(plain, field), getattr(exported, field))
    assert plain.intrinsic is None
    assert exported.intrinsic.flux.shape == (1, len(exported.intrinsic.wavelength_um))
    assert np.isfinite(exported.intrinsic.flux).all()
