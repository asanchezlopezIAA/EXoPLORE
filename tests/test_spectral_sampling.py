"""Flux and equivalent-width checks for the shared native-pixel sampler."""

import numpy as np
from exoplore.observation.spectral_sampling import sample_native_pixels


def test_native_sampling_preserves_constant_flux():
    edges = np.linspace(np.log(588.), np.log(590.), 101)
    fine = np.exp(np.linspace(np.log(587.9), np.log(590.1), 20001))
    result = sample_native_pixels(np.ones(len(fine)), fine, edges,
                                  np.log(fine[1]/fine[0]), 94600.)
    np.testing.assert_allclose(result, 1., rtol=0, atol=5e-12)


def test_native_sampling_conserves_an_isolated_absorption_area():
    edges = np.linspace(np.log(588.), np.log(590.), 501)
    fine = np.exp(np.linspace(np.log(587.9), np.log(590.1), 20001))
    spectrum = 1-.01*np.exp(-.5*((fine-589.)/.003)**2)
    result = sample_native_pixels(spectrum, fine, edges,
                                  np.log(fine[1]/fine[0]), 94600.)
    measured = np.sum((1-result)*np.diff(np.exp(edges)))
    expected = .01*.003*np.sqrt(2*np.pi)
    np.testing.assert_allclose(measured, expected, rtol=1e-6)
