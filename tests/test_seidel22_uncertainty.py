"""Check continuum-noise propagation against an independent small-spectrum limit."""
import numpy as np
import pytest

from exoplore.config.seidel22 import Seidel22Config
from exoplore.pipelines.seidel22 import prepare_transmission


def inputs(**overrides):
    wave = np.linspace(589., 589.1, 11)
    values = dict(wave_nm=wave, flux=np.ones((4, 11)),
                  error=np.full((4, 11), .002), stellar_rv_kms=0.,
                  planet_rv_stellar_kms=0., reference_indices=[0, 1],
                  transit_indices=[2, 3],
                  settings=Seidel22Config([[588.999, 589.011]],
                                          minimum_continuum_pixels=2),
                  telluric_corrected=True, sky_subtracted=True,
                  calculate_covariance=True,
                  lightcurve_window_nm=(589.04, 589.06))
    values.update(overrides)
    return values


def test_resampling_includes_the_common_continuum_error():
    values = inputs(noise_resamples=1200, noise_resample_seed=42)
    result = prepare_transmission(**values)
    # Two continuum pixels give a mean (the even-sample median). At flat
    # flux, reference division adds equal in-transit and reference terms.
    # Each contributes pixel noise plus the variance of that continuum mean.
    variance = .002**2
    expected_diagonal = variance * (1/2 + 1/2) * (1 + 1/2)
    expected_cross_pixel = variance * (1/2 + 1/2) / 2
    covariance = result.coadd_covariance.toarray()
    assert covariance[5, 5] == pytest.approx(expected_diagonal, rel=.12)
    assert covariance[5, 7] == pytest.approx(expected_cross_pixel, rel=.20)
    assert result.covariance_method == 'resampled_continuum'
    assert result.noise_resamples == 1200
    assert np.isfinite(result.lightcurve_covariance).all()
    wave = np.asarray(values['wave_nm'])
    pixels = np.count_nonzero((wave >= 589.04) & (wave <= 589.06))
    row_variance = variance * (1 / pixels + 1 / 2)
    assert result.lightcurve_covariance[2, 2] == pytest.approx(1.5 * row_variance, rel=.15)
    assert result.lightcurve_covariance[2, 3] == pytest.approx(.5 * row_variance, rel=.25)


def test_resampling_preserves_the_observed_point_estimates_and_inputs():
    values = inputs()
    values['flux'][2:, 5] -= .01
    original = values['flux'].copy()
    conditional = prepare_transmission(**values)
    sampled = prepare_transmission(**values, noise_resamples=20, noise_resample_seed=8)
    for field in ['planet_coadd', 'planet_lightcurve', 'master', 'stellar_transmission']:
        np.testing.assert_array_equal(getattr(sampled, field), getattr(conditional, field))
    np.testing.assert_array_equal(values['flux'], original)


def test_seed_and_supplied_expectation_are_used():
    values = inputs(noise_resamples=12, noise_resample_seed=7,
                    noise_expectation=np.ones((4, 11)))
    first = prepare_transmission(**values)
    second = prepare_transmission(**values)
    np.testing.assert_array_equal(first.coadd_covariance.toarray(),
                                  second.coadd_covariance.toarray())
    values['noise_resample_seed'] = 8
    third = prepare_transmission(**values)
    assert not np.array_equal(first.coadd_covariance.toarray(), third.coadd_covariance.toarray())


@pytest.mark.parametrize('changes', [dict(noise_resamples=1), dict(noise_resamples=-1),
                                   dict(noise_resamples=True),
                                   dict(noise_resamples=2, calculate_covariance=False),
                                   dict(noise_expectation=np.ones((4, 11))),
                                   dict(noise_resamples=2, noise_expectation=np.ones((2, 11)))])
def test_rejects_ambiguous_resampling_options(changes):
    with pytest.raises(ValueError):
        prepare_transmission(**inputs(**changes))
