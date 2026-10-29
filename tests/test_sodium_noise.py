"""Flux-dependent sodium errors with fixed ETC continuum sensitivity."""

import numpy as np
import pytest

from exoplore.observation.carmenes_noise import CarmenesETC
from exoplore.observation.sodium_noise import sodium_photon_uncertainty


@pytest.fixture
def inputs():
    wave = np.linspace(.585, .595, 101)
    shape = np.sinc(np.linspace(-.5, .5, 101))**2
    table = CarmenesETC(wave[None, :]*1000,
                        np.array([[200*shape], [160*shape]]),
                        np.array([1., 1.5]), 198., 'VIS')
    return table, wave


def estimate(inputs, flux, tell=None):
    table, wave = inputs
    if tell is None:
        tell = np.ones_like(flux)
    return sodium_photon_uncertainty(table, 0, wave, flux, tell,
                                     [1., 1.5], 198., 94600.)


def test_continuum_reproduces_etc_pixel_anchor(inputs):
    result = estimate(inputs, np.ones((2, 101)))
    table, wave = inputs
    m = wave[50]/(94600*np.gradient(wave)[50])
    np.testing.assert_allclose(result.local_snr_pixel[:, 50], np.array([200, 160])/np.sqrt(m))
    np.testing.assert_allclose(result.std_noise, 1/result.continuum_snr_pixel)


def test_stellar_core_has_fewer_photons_and_smaller_absolute_sigma(inputs):
    continuum = estimate(inputs, np.ones((2, 101)))
    flux = np.ones((2, 101))
    flux[:, 30] = .1
    result = estimate(inputs, flux)
    np.testing.assert_allclose(result.std_noise[:, 30]/continuum.std_noise[:, 30], np.sqrt(.1))
    np.testing.assert_allclose(result.local_snr_pixel[:, 30]/continuum.local_snr_pixel[:, 30], np.sqrt(.1))
    np.testing.assert_allclose(result.std_noise*result.local_snr_pixel, flux)


def test_telluric_absorption_is_counted_once(inputs):
    continuum = estimate(inputs, np.ones((2, 101)))
    tell = np.ones((2, 101))
    tell[:, 30] = .25
    result = estimate(inputs, tell, tell)
    np.testing.assert_allclose(result.photon_response, continuum.photon_response)
    np.testing.assert_allclose(result.std_noise[:, 30], .5*continuum.std_noise[:, 30])
    # Dividing by tellurics also divides sigma. Corrected errors grow as 1/sqrt(T).
    np.testing.assert_allclose(result.std_noise[:, 30]/tell[:, 30],
                               2*continuum.std_noise[:, 30])


def test_attenuated_anchor_retains_the_supplied_etc_snr(inputs):
    # The ETC maximum includes atmospheric attenuation at its reference point.
    tell = np.full((2, 101), .8)
    result = estimate(inputs, tell, tell)
    np.testing.assert_allclose(result.local_snr_pixel, result.continuum_snr_pixel)
    np.testing.assert_allclose(result.std_noise, tell/result.continuum_snr_pixel)


def test_throughput_and_transit_propagate_from_the_expectation(inputs):
    continuum = estimate(inputs, np.ones((2, 101)))
    result = estimate(inputs, np.ones((2, 101))*np.array([.9, 1.1])[:, None])
    np.testing.assert_allclose(result.std_noise/continuum.std_noise,
                               np.broadcast_to(np.sqrt([.9, 1.1])[:, None], (2, 101)))


def test_cropped_order_preserves_the_full_order_blaze(inputs):
    table, wave = inputs
    flux = np.ones((2, 101))
    tell = flux.copy()
    tell[:, 42] = .4
    full = estimate(inputs, flux, tell)
    indices = np.arange(30, 80)
    crop = sodium_photon_uncertainty(table, 0, wave, flux[:, indices], tell[:, indices],
                                     [1., 1.5], 198., 94600., indices)
    np.testing.assert_array_equal(crop.std_noise, full.std_noise[:, indices])
    with pytest.raises(ValueError, match='anchor wavelengths'):
        sodium_photon_uncertainty(table, 0, wave, flux[:, :20], tell[:, :20],
                                  [1., 1.5], 198., 94600., np.arange(20))


def test_gaussian_draws_have_the_prescribed_variance_and_seed(inputs):
    flux = np.full((2, 101), .1)
    sigma = estimate(inputs, flux).std_noise
    first = np.random.default_rng(12345).normal(0, sigma, size=(10000, *sigma.shape))
    second = np.random.default_rng(12345).normal(0, sigma, size=(10000, *sigma.shape))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.var(axis=0), sigma**2, rtol=.05)


@pytest.mark.parametrize('kind', ['negative_flux', 'nan_flux', 'zero_transmission'])
def test_invalid_inputs_are_rejected(inputs, kind):
    flux, tell = np.ones((2, 101)), np.ones((2, 101))
    if kind == 'negative_flux':
        flux[0, 30] = -.1
    elif kind == 'nan_flux':
        flux[0, 30] = np.nan
    else:
        tell[0, 30] = 0
    with pytest.raises(ValueError, match='expected flux'):
        estimate(inputs, flux, tell)


def test_ccf_and_helium_snr_method_is_unchanged(inputs):
    table, wave = inputs
    tell = np.ones((2, 101))
    before = table.pixel_snr(0, wave, tell, [1., 1.5], 198., 94600.)
    estimate(inputs, np.full((2, 101), .1))
    np.testing.assert_array_equal(table.pixel_snr(0, wave, tell, [1., 1.5], 198., 94600.), before)
