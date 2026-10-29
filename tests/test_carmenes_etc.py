"""ETC wavelength arrays and conversion into observation flux units."""

import json
import numpy as np
from exoplore.core.helium_etc import etc_pixel_noise
from exoplore.io.carmenes_etc import extract_etc_arrays


def test_full_vis_reader_preserves_order_and_wavelength_dependent_snr():
    sources=[]
    for order in range(61):
        wave=np.linspace(5000+order*100,5090+order*100,4096)
        snr=np.linspace(10+order,40+order,4096)
        sources.append(dict(name='ColumnDataSource',attributes=dict(data=dict(entries=[
            ['x',dict(array=wave.tolist())],['y',dict(array=snr.tolist())]]))))
    response='<script type="application/json">'+json.dumps(sources)+'</script>'
    wave,snr=extract_etc_arrays(response,'VIS')
    assert wave.shape==snr.shape==(61,4096)
    np.testing.assert_allclose(wave[14,[0,-1]],[640.,649.])
    np.testing.assert_allclose(snr[14,[0,-1]],[24.,54.])


def test_etc_noise_uses_the_existing_helium_conversion():
    pixel, sigma = etc_pixel_noise(np.array([100., 200., 300.]), 4.)
    np.testing.assert_allclose(pixel, [50., 100., 150.])
    np.testing.assert_allclose(sigma, 1/pixel)


def test_gaussian_draw_uses_the_etc_standard_deviation():
    from exoplore.observation.noise import photon_noise
    resel = np.tile(np.array([100., 200., 300.]), (10000, 1))
    pixel, sigma = etc_pixel_noise(resel, 4.)
    noise = photon_noise(pixel, rng=np.random.default_rng(12345))
    standardized = noise/sigma
    assert abs(standardized.mean()) < .02
    assert abs(standardized.std()-1) < .02
