"""ETC maximum anchors, relative tellurics and master-JSON routing."""

from types import SimpleNamespace

import numpy as np
import pytest

from exoplore.config import SimulationConfig
from exoplore.instruments.wavegrid import get_WaveGrid, pixel_snr_one_order
from exoplore.observation.carmenes_noise import CarmenesETC, configured_carmenes_etc


@pytest.fixture
def bundle(tmp_path):
    wave = np.array([np.linspace(580, 590, 101), np.linspace(590, 600, 101)])
    shape = np.sinc(np.linspace(-.5, .5, 101))**2
    curves = np.array([[100*shape, 200*shape], [80*shape, 160*shape]])
    path = tmp_path/'etc.npz'
    np.savez(path, wave_nm=wave, snr_per_resel=curves,
             airmass=[1., 1.5], exposure_seconds=180., channel='VIS')
    return path, CarmenesETC.load(path)


def test_same_andes_blaze_and_tellurics_with_native_pixel_sampling(bundle):
    _, table = bundle
    wave = table.wave_nm[1]/1000
    tell = np.ones((2, 101))
    tell[0, 20:25] = .25
    tell[1, 20:25] = .0625
    snr = table.pixel_snr(1, wave, tell, [1., 1.5], 180., 94600.)
    m = wave/(94600*np.gradient(wave))
    expected = pixel_snr_one_order(wave, tell, [200., 160.],
                                  px_per_resel=1, center_pix=50)/np.sqrt(m)
    np.testing.assert_allclose(snr, expected, rtol=1e-14)
    # The telluric contribution to S/N follows the square root of transmission.
    plain = table.pixel_snr(1, wave, np.ones_like(tell), [1., 1.5], 180., 94600.)
    np.testing.assert_allclose(snr[:, 22]/plain[:, 22], [.5, .25])


def test_airmass_interpolation_and_physical_order_invariance(bundle):
    _, table = bundle
    wave = table.wave_nm[1]/1000
    tell = np.ones((1, 101))
    evaluate = lambda x: table.pixel_snr(1, wave, tell, [x], 180., 94600.)
    expected = (evaluate(1.)+evaluate(1.5))/2
    np.testing.assert_allclose(evaluate(1.25), expected)
    table.pixel_snr(0, table.wave_nm[0]/1000, tell, [1.25], 180., 94600.)
    np.testing.assert_array_equal(evaluate(1.25), expected)


@pytest.mark.parametrize('airmass,duration,match', [
    ([.99], 180., 'airmass'), ([1.51], 180., 'airmass'),
    ([1.2], 198., 'exposure'),
])
def test_conditions_are_checked(bundle, airmass, duration, match):
    _, table = bundle
    with pytest.raises(ValueError, match=match):
        table.pixel_snr(1, table.wave_nm[1]/1000, np.ones((1, 101)),
                        airmass, duration, 94600.)


def test_unavailable_order_and_missing_anchor_are_rejected(bundle):
    _, table = bundle
    with pytest.raises(ValueError, match='Physical order'):
        table.pixel_snr(2, table.wave_nm[1]/1000, np.ones((1, 101)), [1.], 180., 94600.)
    with pytest.raises(ValueError, match='complete native order'):
        table.pixel_snr(1, table.wave_nm[1, :20]/1000, np.ones((1, 20)), [1.], 180., 94600.)
    zeros = CarmenesETC(table.wave_nm, np.zeros_like(table.snr_per_resel),
                        table.airmass, 180., 'VIS')
    with pytest.raises(ValueError, match='positive S/N'):
        zeros.pixel_snr(1, table.wave_nm[1]/1000, np.ones((1, 101)), [1.], 180., 94600.)


def test_single_condition_bundle_and_opaque_pixels(bundle):
    _, table = bundle
    single = CarmenesETC(table.wave_nm, table.snr_per_resel[:1],
                         table.airmass[:1], 180., 'VIS')
    tell = np.ones((2, 101))
    tell[:, 20] = 0
    result = single.pixel_snr(1, table.wave_nm[1]/1000, tell, [1., 1.], 180., 94600.)
    assert np.all(result[:, 20] == 0)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match='airmass'):
        single.pixel_snr(1, table.wave_nm[1]/1000, tell, [1., 1.01], 180., 94600.)


def test_missing_etc_edge_samples_do_not_change_the_maximum_anchor(bundle):
    _, table = bundle
    curves = table.snr_per_resel.copy()
    curves[:, :, :5] = np.nan
    incomplete = CarmenesETC(table.wave_nm, curves, table.airmass, 180., 'VIS')
    args = (1, table.wave_nm[1]/1000, np.ones((1, 101)), [1.], 180., 94600.)
    np.testing.assert_array_equal(incomplete.pixel_snr(*args), table.pixel_snr(*args))


def test_config_round_trip_and_legacy_default(bundle, tmp_path):
    path, _ = bundle
    assert configured_carmenes_etc(SimulationConfig()) is None
    config = SimulationConfig.from_dict(dict(instrument=dict(name='CARMENES_VIS'),
        observation=dict(exposure_time_seconds=180.),
        noise=dict(carmenes_etc_file=str(path))))
    target = tmp_path/'config.json'
    config.to_json(target)
    loaded = SimulationConfig.from_json(target)
    assert configured_carmenes_etc(loaded).channel == 'VIS'
    assert loaded.noise.carmenes_etc_file == str(path)
    config.noise.fixed_snr = 300
    with pytest.raises(ValueError, match='fixed_snr'):
        configured_carmenes_etc(config)


def test_etc_wave_reader_bypasses_measured_noise_and_preserves_cadence(tmp_path):
    from astropy.io import fits
    wave = np.array([np.linspace(.58, .59, 101), np.linspace(.59, .60, 101)])
    fits.writeto(tmp_path/'wave.fits', wave)
    fits.writeto(tmp_path/'jd.fits', np.array([1., 2.]))
    fits.writeto(tmp_path/'am.fits', np.array([1.1, 1.2]))
    info = SimpleNamespace(wave_file=tmp_path/'wave.fits', sig_file='missing.fits',
                           snr_file='missing.fits', JD_file=tmp_path/'jd.fits',
                           airmass_file=tmp_path/'am.fits')
    config = SimulationConfig()
    config.instrument.name = 'CARMENES_VIS'
    result = get_WaveGrid(config.instrument, config.observation, info, 2, use_etc=True)
    np.testing.assert_array_equal(result[0], wave.T)
    np.testing.assert_array_equal(result[-2], [1., 2.])
    np.testing.assert_array_equal(result[-1], [1.1, 1.2])
    assert result[2] is None


def test_helium_uses_same_shared_uncertainty_model(bundle):
    from exoplore.core.carmenes_helium_simulation import carmenes_observation_error
    path, table = bundle
    config = SimulationConfig()
    config.noise.carmenes_etc_file = str(path)
    config.observation.exposure_time_seconds = 180.
    config.atmosphere.helium = SimpleNamespace(airmass=[1., 1.5])
    arrays = dict(expected_flux=np.ones((2, 101)), telluric_transmission=np.ones((2, 101)))
    order = dict(etc=table, order_index=1, wave_nm=table.wave_nm[1], resolving_power=94600.)
    error = carmenes_observation_error(config, order, arrays)
    snr = table.pixel_snr(1, order['wave_nm']/1000, arrays['telluric_transmission'],
                          [1., 1.5], 180., 94600.)
    np.testing.assert_array_equal(error, 1/np.maximum(snr, 1.))
    config.noise.carmenes_etc_file = None
    legacy = dict(snr=np.full(101, 90.))
    np.testing.assert_array_equal(carmenes_observation_error(config, legacy, arrays),
                                  np.full((2, 101), 1/90.))


def test_bundle_writer_retains_conditions_and_refuses_overwrite(bundle, tmp_path, monkeypatch):
    import json
    from exoplore.io import carmenes_etc
    _, table = bundle
    one, two = tmp_path/'x1.html', tmp_path/'x1p5.html'
    one.write_text('first response')
    two.write_text('second response')
    monkeypatch.setattr(carmenes_etc, 'extract_etc_arrays',
        lambda html, channel: (table.wave_nm, table.snr_per_resel[0 if html.startswith('first') else 1]))
    output = tmp_path/'packed.npz'
    carmenes_etc.write_etc_bundle([(1.5, two), (1., one)], 'VIS', 180., output)
    actual = CarmenesETC.load(output)
    np.testing.assert_array_equal(actual.airmass, [1., 1.5])
    np.testing.assert_array_equal(actual.snr_per_resel, table.snr_per_resel)
    with np.load(output) as data:
        assert len(json.loads(str(data['source_records']))[0]['sha256']) == 64
    with pytest.raises(FileExistsError):
        carmenes_etc.write_etc_bundle([(1., one)], 'VIS', 180., output)


def test_reference_nights_keep_separate_cadences_with_etc(tmp_path):
    from astropy.io import fits
    wave = np.array([np.linspace(.58, .59, 101)])
    fits.writeto(tmp_path/'wave.fits', wave)
    jd, am = [], []
    for night, count in enumerate([2, 3]):
        jd.append(tmp_path/f'jd{night}.fits')
        am.append(tmp_path/f'am{night}.fits')
        fits.writeto(jd[-1], np.arange(count, dtype=float))
        fits.writeto(am[-1], np.linspace(1., 1.2, count))
    info = SimpleNamespace(wave_file=tmp_path/'wave.fits', sig_file='missing',
                           snr_file='missing', JD_file=jd, airmass_file=am)
    config = SimulationConfig()
    config.instrument.name = 'CARMENES_VIS'
    config.observation.different_nights = True
    config.observation.n_nights = 2
    result = get_WaveGrid(config.instrument, config.observation, info, 1, use_etc=True)
    assert [len(row) for row in result[-2]] == [2, 3]
    assert [len(row) for row in result[-1]] == [2, 3]


@pytest.mark.parametrize('change', ['observed', 'channel', 'duration', 'mean', 'sig'])
def test_master_config_rejects_incompatible_etc_selections(bundle, change):
    from exoplore.core import ExoploreSimulator
    path, _ = bundle
    config = SimulationConfig()
    config.instrument.name = 'CARMENES_VIS'
    config.observation.exposure_time_seconds = 180.
    config.noise.carmenes_etc_file = str(path)
    if change == 'observed':
        config.observation.use_real_data = True
    elif change == 'channel':
        config.instrument.name = 'CARMENES_NIR'
    elif change == 'duration':
        config.observation.exposure_time_seconds = 198.
    elif change == 'mean':
        config.noise.use_mean_snr = True
    else:
        config.noise.noise_choice = 'sig'
    with pytest.raises(ValueError, match='CARMENES|carmenes_etc_file'):
        ExoploreSimulator(config)
