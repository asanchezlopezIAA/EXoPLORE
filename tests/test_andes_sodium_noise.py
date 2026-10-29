"""Check ANDES photon response against the shared ETC order model."""
import numpy as np
import pytest
from exoplore.observation.andes_sodium_noise import andes_sodium_photon_uncertainty
from exoplore.instruments.wavegrid import pixel_snr_one_order


def inputs():
    wave = np.linspace(.5882,.593,101)
    indices = np.arange(10,91)
    tell = np.ones((2,len(indices)))*np.array([.9,.8])[:,None]
    flux = tell*np.array([1.,.1])[:,None]
    return wave, indices, flux, tell


def test_response_uses_shared_order_envelope_and_expected_photons():
    wave, indices, flux, tell = inputs()
    result = andes_sodium_photon_uncertainty(wave,indices,flux,tell,800.,2.5,wave[50])
    full_tell = np.ones((2,len(wave)))
    full_tell[:,indices] = tell
    continuum = pixel_snr_one_order(wave,full_tell,800.,2.5,50)[:,indices]
    np.testing.assert_allclose(result.continuum_snr_pixel,continuum)
    np.testing.assert_allclose(flux/result.std_noise,result.local_snr_pixel)
    np.testing.assert_allclose(result.std_noise**2,flux*tell/continuum**2)
    # At equal illumination, a fourfold ETC S/N gain divides sigma by four.
    stronger = andes_sodium_photon_uncertainty(wave,indices,flux,tell,3200.,2.5,wave[50])
    np.testing.assert_allclose(stronger.std_noise,result.std_noise/4)


def test_crop_preserves_blaze_and_input_arrays():
    wave, indices, flux, tell = inputs()
    original = [a.copy() for a in (wave,indices,flux,tell)]
    result = andes_sodium_photon_uncertainty(wave,indices,flux,tell,800.,2.5,wave[50])
    full_flux = np.ones((2,len(wave)))
    full_tell = np.ones_like(full_flux)
    full_flux[:,indices] = flux
    full_tell[:,indices] = tell
    full = andes_sodium_photon_uncertainty(wave,np.arange(len(wave)),full_flux,full_tell,800.,2.5,wave[50])
    np.testing.assert_array_equal(result.std_noise,full.std_noise[:,indices])
    for a,b in zip((wave,indices,flux,tell),original):np.testing.assert_array_equal(a,b)


def test_requires_covered_etc_anchor():
    wave, indices, flux, tell = inputs()
    with pytest.raises(ValueError,match='ETC centre'):
        andes_sodium_photon_uncertainty(wave,indices[:10],flux[:,:10],tell[:,:10],800.,2.5,wave[50])
