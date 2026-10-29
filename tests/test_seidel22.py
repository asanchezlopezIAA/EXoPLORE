"""Check reference subtraction, velocity frames and masks for atomic spectra."""

import numpy as np
import pytest

from exoplore.config.models import PipelineConfig
from exoplore.config.seidel22 import Seidel22Config
from exoplore.pipelines.seidel22 import prepare_transmission
from exoplore.pipelines.helium_math import C_KMS


def example(centre=589.16, absorption=True, **changes):
    wave = np.linspace(centre - .4, centre + .4, 801)
    star = np.array([-3., -2., -1., 0., 1., 2.])
    planet = np.array([0., 0., -20., 5., 20., 0.])
    stellar_wave = wave[None, :] / (1 + star[:, None] / C_KMS)
    flux = 1 - .5 * np.exp(-.5 * ((stellar_wave - centre) / .015)**2)
    if absorption:
        for i in (2, 3, 4):
            local = stellar_wave[i] / (1 + planet[i] / C_KMS)
            flux[i] *= 1 - .01 * np.exp(-.5 * ((local - centre) / .008)**2)
    choices = dict(wave_nm=wave, flux=flux, error=np.full_like(flux, .002),
                   stellar_rv_kms=star, planet_rv_stellar_kms=planet,
                   reference_indices=[0, 1, 5], transit_indices=[2, 3, 4],
                   settings=Seidel22Config([[centre - .35, centre - .2],
                                            [centre + .2, centre + .35]]),
                   telluric_corrected=True, sky_subtracted=True)
    choices.update(changes)
    return choices


@pytest.mark.parametrize('centre', [589.16, 656.46])
def test_moving_line_is_recovered_in_planet_frame(centre):
    result = prepare_transmission(**example(centre))
    peak = np.nanargmin(result.planet_coadd)
    assert abs(result.planet_wave_nm[peak] - centre) < .002
    assert .009 < 1 - result.planet_coadd[peak] < .011
    assert result.contributing_exposures[peak] == 3


def test_stellar_reference_removes_stationary_lines():
    result = prepare_transmission(**example(absorption=False))
    # Two linear interpolations leave small curvature errors on a sampled line.
    assert np.nanmax(abs(result.stellar_transmission - 1)) < .001


def test_shared_reference_variance():
    values = example(absorption=False)
    values.update(flux=np.ones((6, 801)), stellar_rv_kms=0,
                  planet_rv_stellar_kms=0)
    result = prepare_transmission(**values)
    np.testing.assert_allclose(result.stellar_variance[2, 100], .002**2 * (1 + 1/3))
    np.testing.assert_allclose(result.stellar_variance[0, 100], .002**2 * (1 - 1/3))


def test_native_masks_survive_interpolation():
    values = example()
    values['flux'][:, 200:210] = np.nan
    result = prepare_transmission(**values)
    assert np.isnan(result.stellar_transmission[:, 204]).all()


def test_stellar_core_mask_moves_with_planet_alignment():
    values = example()
    values['settings'].stellar_mask_windows_nm = [[589.15, 589.17]]
    result = prepare_transmission(**values)
    assert np.isnan(result.stellar_transmission[:, 400]).all()
    assert np.isfinite(result.planet_transmission[2, 400])


@pytest.mark.parametrize('changes', [dict(telluric_corrected=False),
                                   dict(sky_subtracted=False),
                                   dict(reference_indices=[0, 2]),
                                   dict(reference_indices=[]),
                                   dict(stellar_rv_kms=np.nan)])
def test_rejects_ambiguous_inputs(changes):
    with pytest.raises(ValueError):
        prepare_transmission(**example(**changes))


def test_config_round_trip():
    from dataclasses import asdict
    original = PipelineConfig(name='Seidel22', prepare_template=False,
                              seidel22=dict(continuum_windows_nm=[[588., 588.1]]))
    restored = PipelineConfig(**asdict(original))
    assert isinstance(restored.seidel22, Seidel22Config)
    assert asdict(original) == asdict(restored)


def test_input_arrays_are_preserved():
    values = example()
    before = values['flux'].copy()
    prepare_transmission(**values)
    np.testing.assert_array_equal(values['flux'], before)


def test_existing_helium_weighted_mean_is_unchanged():
    from exoplore.pipelines.helium_math import weighted_mean
    flux = np.array([[1., 2., np.nan], [3., 4., np.nan]])
    var = np.array([[1., 4., 1.], [4., 1., 1.]])
    result, error = weighted_mean(flux, var)
    np.testing.assert_allclose(result[:2], [1.4, 3.6])
    np.testing.assert_allclose(error[:2], [.8, .8])
    assert np.isnan(result[2]) and np.isnan(error[2])
    result, error = weighted_mean(flux, var, exposure_weights=np.array([1., 3.]))
    np.testing.assert_allclose(result[:2], [2.5, 3.5])
    np.testing.assert_allclose(error[:2], [37/16, 13/16])


def test_wiggle_fit_excludes_planet_line():
    from exoplore.pipelines.seidel22 import remove_wiggles
    wave = np.linspace(588., 590., 1001)
    baseline = 1 + .003*np.sin(2*np.pi*(wave - 588.)/.42)
    line = 1 - .01*np.exp(-.5*((wave - 589.)/.01)**2)
    result = remove_wiggles(wave, baseline*line, np.full_like(wave, 1e-6),
                            continuum_windows_nm=[[588., 588.9], [589.1, 590.]],
                            period_bounds_nm=[.3, .6])
    np.testing.assert_allclose(result['transmission'], line, atol=1e-7)
    assert abs(result['period_nm'] - .42) < 1e-5


def test_runner_adapter_preserves_observer_frame_and_return_layout():
    from exoplore.config.models import TelluricConfig
    from exoplore.pipelines.models import PreparationInputs
    from exoplore.pipelines.runner import prepare_observation
    values = example(absorption=False)
    pipeline = PipelineConfig(name='Seidel22', prepare_template=False,
                              seidel22=values['settings'])
    inputs = PreparationInputs(pipeline, TelluricConfig(include_tellurics=False),
                               False, 0, stellar_rv_kms=values['stellar_rv_kms'])
    result = prepare_observation(
        inputs, values['flux'], values['error'], values['wave_nm']/1000,
        np.arange(801), np.array([], int), np.ones(6), np.zeros(6),
        np.array(values['reference_indices']), None)
    assert len(result) == 9
    prepared, error, good, bad, *_ = result
    assert prepared.shape == error.shape == (6, 801)
    assert good.size > 700
    assert bad.size > 0
    assert np.isfinite(prepared).all() and np.isfinite(error).all()
    assert np.max(abs(prepared[:, good] - 1)) < .001


def test_detector_gap_implementation_is_shared_with_helium():
    from exoplore.pipelines.carmenes_helium import gap_resample as helium
    from exoplore.pipelines.seidel22 import gap_resample as atomic
    assert helium is atomic
    wave = np.r_[np.arange(5.), np.arange(10., 15.)]
    flux, var = atomic(wave, np.ones(10), np.ones(10), np.arange(15.))
    assert np.isnan(flux[5:10]).all() and np.isnan(var[5:10]).all()


def test_coadd_covariance_keeps_the_shared_reference_noise():
    values = example(absorption=False)
    values.update(flux=np.ones((6, 801)), stellar_rv_kms=0,
                  planet_rv_stellar_kms=0, calculate_covariance=True)
    result = prepare_transmission(**values)
    # Three independent transit exposures share one three-exposure master.
    # The master's variance contributes once to the final mean.
    np.testing.assert_allclose(result.coadd_covariance.diagonal(), .002**2*(1/3+1/3))
    np.testing.assert_allclose(result.conditional_coadd_variance, .002**2/3)


@pytest.mark.parametrize('with_stellar_correction', [False, True])
def test_interpolation_covariance_agrees_with_flux_derivatives(with_stellar_correction):
    values = example(calculate_covariance=True)
    if with_stellar_correction:
        values['stellar_model_flux'] = example(absorption=False)['flux']
    result = prepare_transmission(**values)
    sample = np.arange(395, 406)
    calculated = result.coadd_covariance[sample][:, sample].toarray()
    numerical = np.zeros_like(calculated)
    # Perturb only native pixels that can contribute to this short output
    # interval. Continuum anchors and their measured S/N stay unchanged.
    for exposure in range(6):
        for pixel in range(330, 470):
            modified = values['flux'].copy()
            modified[exposure, pixel] += 1e-6
            changed = prepare_transmission(**{**values, 'flux': modified,
                                               'calculate_covariance': False})
            derivative = (changed.planet_coadd[sample] - result.planet_coadd[sample])/1e-6
            numerical += np.outer(derivative, derivative)*values['error'][exposure, pixel]**2
    np.testing.assert_allclose(calculated, numerical, rtol=5e-6, atol=1e-12)
    assert np.any(np.abs(np.triu(calculated, 1)) > 1e-10)


def test_covariance_preserves_the_default_prepared_flux():
    first = prepare_transmission(**example())
    second = prepare_transmission(**example(calculate_covariance=True))
    np.testing.assert_array_equal(first.planet_coadd, second.planet_coadd)
    np.testing.assert_array_equal(first.stellar_transmission, second.stellar_transmission)


def test_common_stellar_continuum_avoids_window_boundary_level_jumps():
    # A fixed stellar spectrum drifts by a fraction of a native pixel. Its
    # broad absorption wings cross the continuum-window boundaries, as in Na.
    wave = np.linspace(589., 590., 328)
    velocity = np.linspace(-2.40, -2.32, 19)
    rest = wave[None, :] / (1 + velocity[:, None]/C_KMS)
    flux = (1 - .8*np.exp(-.5*((rest-589.16)/.06)**2)
            - .78*np.exp(-.5*((rest-589.76)/.06)**2)
            + .02*(rest-589.5))
    result = prepare_transmission(
        wave, flux, np.full_like(flux, .003), stellar_rv_kms=velocity,
        planet_rv_stellar_kms=0, reference_indices=[0, 1, 2, 16, 17, 18],
        transit_indices=list(range(4, 15)),
        settings=Seidel22Config([[589.2, 589.4], [589.8, 589.95]]),
        telluric_corrected=True, sky_subtracted=True)
    row_levels = np.nanmedian(result.stellar_transmission, axis=1)
    assert np.ptp(row_levels) < 1e-5


def test_matched_stellar_control_is_flat_after_preparation():
    values = example(absorption=False)
    # A time-dependent line distortion mimics an occulted rotating surface.
    for i in values['transit_indices']:
        values['flux'][i] *= 1+.02*np.exp(-.5*((values['wave_nm']-589.16)/.02)**2)
    values['stellar_model_flux'] = values['flux'].copy()
    result = prepare_transmission(**values)
    good = np.isfinite(result.stellar_transmission)
    np.testing.assert_allclose(result.stellar_transmission[good], 1., atol=5e-16)
    good = np.isfinite(result.planet_coadd)
    np.testing.assert_allclose(result.planet_coadd[good], 1., atol=5e-16)


def test_matched_correction_preserves_injected_planet_line():
    values = example()
    values['stellar_model_flux'] = example(absorption=False)['flux']
    result = prepare_transmission(**values)
    peak = np.nanargmin(result.planet_coadd)
    assert abs(result.planet_wave_nm[peak]-589.16) < .002
    assert .009 < 1-result.planet_coadd[peak] < .011
    assert result.stellar_correction is not None


def test_fixed_correction_scales_covariance_before_planet_shift():
    values = example(absorption=False, calculate_covariance=True)
    values['flux'] = np.tile(values['flux'][0], (6, 1))
    values['stellar_rv_kms'] = 0
    model = values['flux'].copy()
    for i in values['transit_indices']:
        model[i] *= 1+.02*np.exp(-.5*((values['wave_nm']-589.16)/.02)**2)
    plain = prepare_transmission(**values)
    changed = prepare_transmission(**values, stellar_model_flux=model)
    # With zero relative planetary velocity there is no additional shift.
    # A common divisor on transit spectra scales both covariance axes.
    zero = {**values, 'planet_rv_stellar_kms': 0}
    plain = prepare_transmission(**zero)
    changed = prepare_transmission(**zero, stellar_model_flux=model)
    divisor = changed.stellar_correction[values['transit_indices'][0]]
    sample = np.arange(390, 411)
    expected = plain.coadd_covariance[sample][:, sample].toarray()
    expected /= divisor[sample, None]*divisor[None, sample]
    np.testing.assert_allclose(changed.coadd_covariance[sample][:, sample].toarray(),
                               expected, rtol=2e-14, atol=1e-18)


@pytest.mark.parametrize('model', [np.ones((2, 3)), np.zeros((6, 801))])
def test_bad_stellar_correction_model_is_rejected(model):
    with pytest.raises(ValueError, match='Stellar model'):
        prepare_transmission(**example(), stellar_model_flux=model)


def test_correction_model_selection_round_trips():
    from exoplore.config import SimulationConfig
    config = SimulationConfig()
    config.rm_clv.correction_method = 'factorized'
    config.rm_clv.correction_model_file = 'stellar_alternative.npz'
    restored = SimulationConfig.from_dict(config.to_dict())
    assert restored.rm_clv.correction_method == 'factorized'
    assert restored.rm_clv.correction_model_file == 'stellar_alternative.npz'


def test_matched_correction_rejects_an_unrelated_model_file():
    from exoplore.config import RMCLVConfig
    with pytest.raises(ValueError, match='correction_method=factorized'):
        RMCLVConfig(correction_model_file='other.npz')


def test_lightcurve_covariance_retains_shared_reference_and_self_terms():
    values = example(absorption=False)
    values.update(flux=np.ones((6, 801)), stellar_rv_kms=0,
                  planet_rv_stellar_kms=0, calculate_covariance=True,
                  lightcurve_window_nm=(589.15, 589.17))
    result = prepare_transmission(**values)
    band = (result.planet_wave_nm >= 589.15) & (result.planet_wave_nm <= 589.17)
    variance = .002**2/band.sum()
    refs, inside = np.array([0, 1, 5]), np.array([2, 3, 4])
    expected = np.zeros((6, 6))
    expected[np.ix_(inside, inside)] = variance/3
    expected[inside, inside] += variance
    expected[np.ix_(refs, refs)] = -variance/3
    expected[refs, refs] += variance
    np.testing.assert_allclose(result.lightcurve_covariance, expected,
                               rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(result.planet_lightcurve, 1.)
