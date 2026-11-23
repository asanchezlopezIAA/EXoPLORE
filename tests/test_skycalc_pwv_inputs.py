"""Controlled transmissions separate cache/interpolation tests from ESO access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from exoplore.config.models import TelluricConfig, ObservationConfig
from exoplore.observation.skycalc_inputs import (
    SkyCalcCache, exposure_pwv, pwv_bracket, skycalc_directory,
)


def model(pwv, wave=None):
    wave = np.array([1000., 1001., 1002.]) if wave is None else wave
    return fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([
        fits.Column(name="lam", format="D", array=wave),
        fits.Column(name="trans", format="D", array=np.array([1., .8, .6])-pwv*.01),
    ])])


def test_time_knots():
    series = dict(hours_from_observation_start=[0., 4., 8.], pwv_mm=[2.5, 5., 3.5])
    np.testing.assert_allclose(exposure_pwv([0, 2, 4, 8], series), [2.5, 3.75, 5., 3.5])
    with pytest.raises(ValueError, match="cover"):
        exposure_pwv([9], series)


@pytest.mark.parametrize("series", [
    dict(hours_from_observation_start=[1, 0], pwv_mm=[2, 3]),
    dict(hours_from_observation_start=[0, 1], pwv_mm=[2]),
    dict(hours_from_observation_start=[0, 1], pwv_mm=[0, 3]),
    dict(hours_from_observation_start=[0, 1], pwv_mm=[2, np.nan]),
])
def test_bad_time_knots(series):
    with pytest.raises(ValueError):
        exposure_pwv([.5], series)


def test_brackets():
    assert pwv_bracket(3.5) == (3.5, 3.5, 0.)
    assert pwv_bracket(4.25) == (3.5, 5., .5)
    assert pwv_bracket(.05) == (.05, .05, 0.)
    assert pwv_bracket(30) == (30., 30., 0.)
    with pytest.raises(ValueError):
        pwv_bracket(31)


def test_cache_reuses_models_between_exposures_and_runs(tmp_path):
    calls = []
    def query(am, pwv):
        calls.append((am, pwv))
        return model(pwv)
    settings = dict(recipe="test", arguments={})
    cache = SkyCalcCache(tmp_path, query, settings)
    with cache.transmission(1.4, 4.25) as result:
        np.testing.assert_allclose(result[1].data['trans'], [1-.0425, .8-.0425, .6-.0425])
        assert result[0].header['PWVWGHT'] == .5
    with cache.transmission(1.4, 4.):
        pass
    cache = SkyCalcCache(tmp_path, query, settings)
    with cache.transmission(1.4, 3.5):
        pass
    assert calls == [(1.4, 3.5), (1.4, 5.)]
    with cache.transmission(1.5, 3.5):
        pass
    assert len(calls) == 3
    cache = SkyCalcCache(tmp_path, query, dict(recipe="changed", arguments={}))
    with cache.transmission(1.4, 3.5):
        pass
    assert len(calls) == 4


def test_failed_queries_are_not_cached(tmp_path):
    cache = SkyCalcCache(tmp_path, lambda *a: None, dict(arguments={}))
    with pytest.raises(RuntimeError):
        cache.transmission(1.4, 3.5)
    assert not list(tmp_path.glob('*.fits'))


def test_different_wavelength_grids_fail(tmp_path):
    cache = SkyCalcCache(tmp_path, lambda am, pwv: model(pwv, np.array([1, 2, pwv])), dict(arguments={}))
    with pytest.raises(ValueError, match="wavelength"):
        cache.transmission(1.4, 4.)


def test_config_and_directory(tmp_path):
    cfg = TelluricConfig(use_full_skycalc=True, skycalc_source="external", skycalc_directory="inputs/custom")
    assert skycalc_directory(cfg, tmp_path, 'full_event') == Path('inputs/custom')
    assert skycalc_directory(cfg, tmp_path, 'full_event', 2) == Path('inputs/custom/night_2')
    assert skycalc_directory(TelluricConfig(), tmp_path, 'full_event') == tmp_path/'Skycalc_full_event/Fixed_PWV'
    with pytest.raises(ValueError, match="use_full"):
        TelluricConfig(skycalc_source="external")


def generator():
    spec = importlib.util.spec_from_file_location("skycalc_generator", "scripts/generate_skycalc_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_only_uses_simulator_dates_and_never_queries(tmp_path, monkeypatch):
    from exoplore.observation.timing import get_event
    from exoplore.planets import load_planet
    import csv
    cfg = json.loads(Path('configs/hd189733b_andes_transit_clean.json').read_text())
    cfg['paths']['inputs_dir'] = str(tmp_path)
    cfg['tellurics'].update(use_full_skycalc=True, skycalc_source='external', constant_pwv=False)
    cfg['tellurics']['airmass_limits'] = [1.4, 1.7]
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    script = generator()
    monkeypatch.setattr(script, '_query_skycalc', lambda *a, **k: pytest.fail('Unexpected network query'))
    script.generate(str(path), mode='synthetic', plan_only=True)
    rows = list(csv.DictReader((tmp_path/'Skycalc_full_event/Variable_PWV/exposure_plan.csv').open()))
    jd = get_event(ObservationConfig(**cfg['observation']), load_planet(cfg['planet']['parameter_file']), None)[0]
    np.testing.assert_array_equal([float(row['jd']) for row in rows], jd)
    assert rows[-1]['filename'] == f'tell_spec_{len(jd)-1}.fits'
    assert all(row['pwv_mm'] == '' for row in rows)


def test_generator_interpolates_and_reuses_persistent_cache(tmp_path, monkeypatch):
    cfg = json.loads(Path('configs/hd189733b_andes_transit_clean.json').read_text())
    cfg['paths']['inputs_dir'] = str(tmp_path)
    cfg['observation'].update(exposure_time_seconds=3600.)
    cfg['tellurics'].update(use_full_skycalc=True, constant_pwv=False,
        airmass_limits=[1.4, 1.4], pwv_time_series={
            'hours_from_observation_start': [0., 20.], 'pwv_mm': [4., 4.5]})
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    calls = []
    def query(am, pwv, **kwargs):
        calls.append((am, pwv))
        return model(pwv)
    script = generator()
    monkeypatch.setattr(script, '_query_skycalc', query)
    script.generate(str(path), mode='synthetic')
    count = len(calls)
    assert calls.count((1.4, 3.5)) == 1
    assert calls.count((1.4, 5.)) == 1
    script.generate(str(path), mode='synthetic', overwrite=True)
    assert len(calls) == count
    cfg['tellurics']['pwv_time_series']['pwv_mm'] = [4.4, 4.5]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='matching request'):
        script.generate(str(path), mode='synthetic')


def test_multinight_plan_matches_runner_cadence(tmp_path):
    from exoplore.observation.skycalc_inputs import configured_exposure_dates
    from exoplore.planets import load_planet
    planet = load_planet('planet_params/HD189733b.json')
    obs = ObservationConfig(different_nights=True, n_nights=2,
                            pre_event_hours=2., post_event_hours=2.,
                            exposure_time_seconds_per_night=[180., 300.])
    tell = TelluricConfig(use_accurate_airmass=False)
    for night, exp in enumerate([180., 300.]):
        jd = configured_exposure_dates(obs, planet, tmp_path, night, tell)
        midpoint = planet.transit_epoch_bjd+night*planet.orbital_period_days
        expected = np.arange(midpoint-2/24., midpoint+2/24.,
                             (exp+obs.readout_time_seconds+obs.overhead_time_seconds)/86400.)
        np.testing.assert_array_equal(jd, expected)


def test_file_loading_full_mode_needs_no_unused_reference(tmp_path):
    from exoplore.pipelines.tellurics import Load_Telluric_Transmittances
    jd = np.array([2450000., 2450000.01])
    for n in range(2):
        with model(2.5+n) as hd:
            hd.writeto(tmp_path/f'tell_spec_{n}.fits')
    fits.PrimaryHDU(jd).writeto(tmp_path/'exposure_jd.fits')
    wave = np.linspace(1.0001, 1.0019, 20)
    ref, trans = Load_Telluric_Transmittances(np.ones(20), True, True,
        str(tmp_path/'missing_reference.fits'), str(tmp_path)+'/', 100000.,
        jd, wave, np.ones((2, 20)), None)
    assert ref is None
    assert np.all(trans[0] > trans[1])
    with pytest.raises(ValueError, match='simulation dates'):
        Load_Telluric_Transmittances(np.ones(20), True, True, '', str(tmp_path)+'/',
            100000., jd+.1, wave, np.ones((2, 20)), None)
