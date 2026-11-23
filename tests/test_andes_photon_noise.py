"""Photon-count and flux-unit checks for the opt-in ANDES ETC prescription."""
import numpy as np
import pytest
from exoplore.config.models import NoiseConfig
from exoplore.observation.andes_photon_noise import andes_photon_noise


def example(star=None, tell=None, factor=None, divided=False):
    wave=np.array([np.linspace(1.,1.01,101)])
    star=np.ones((2,101)) if star is None else star
    tell=np.ones_like(star) if tell is None else tell
    factor=np.ones_like(star) if factor is None else factor
    flux=(1 if divided else star)*tell*factor
    return andes_photon_noise(wave,0,flux,star,tell,np.array([200.]),
                              np.array([1005.]),pixels_per_resel=4.,stellar_divided=divided)


def test_anchor_and_square_root_of_photon_count():
    star=np.ones((2,101));star[:,30]=.25
    reference=example();line=example(star=star)
    np.testing.assert_allclose(reference.snr[:,50],100.)
    np.testing.assert_allclose(line.snr[:,30]/reference.snr[:,30],.5)
    np.testing.assert_allclose(line.sigma[:,30]/reference.sigma[:,30],.5)


def test_arbitrary_flux_units_rescale_sigma_and_preserve_snr():
    base=example();scaled=example(star=np.full((2,101),17.))
    np.testing.assert_allclose(scaled.sigma,17*base.sigma)
    np.testing.assert_allclose(scaled.snr,base.snr)


def test_dividing_by_star_propagates_same_photon_uncertainty():
    star=np.ones((2,101));star[:,30]=.25
    full=example(star=star);divided=example(star=star,divided=True)
    np.testing.assert_allclose(divided.sigma,full.sigma/star)
    np.testing.assert_allclose(divided.snr,full.snr)


def test_telluric_and_throughput_counting():
    tell=np.ones((2,101));tell[:,30]=.36
    base=example();absorbed=example(tell=tell)
    np.testing.assert_allclose(absorbed.sigma[:,30]/base.sigma[:,30],.6)
    np.testing.assert_allclose(absorbed.snr[:,30]/base.snr[:,30],.6)
    bright=example(factor=np.full((2,101),4.))
    np.testing.assert_allclose(bright.sigma,2*base.sigma)
    np.testing.assert_allclose(bright.snr,2*base.snr)


def test_zero_transmission_is_zero_signal_and_zero_snr():
    tell=np.ones((2,101));tell[:,30]=0
    result=example(tell=tell)
    assert (result.snr[:,30]==0).all()
    assert (result.sigma[:,30]==0).all()
    assert np.isfinite(result.response).all()


def test_nontransparent_anchor_is_calibrated_to_etc():
    star=np.full((2,101),.7);tell=np.full((2,101),.8)
    result=example(star=star,tell=tell)
    np.testing.assert_allclose(result.snr[:,50],100.)
    np.testing.assert_allclose(result.sigma[:,50],.56/100)


def test_each_exposure_uses_its_own_stellar_spectrum():
    star=np.ones((2,101));star[0,30]=.25;star[1,35]=.25
    result=example(star=star)
    np.testing.assert_allclose(result.snr[0,30]/result.snr[1,30],.5)


@pytest.mark.parametrize('field,value',[('star',0.),('star',np.nan),('tell',-1.),('factor',-1.)])
def test_invalid_photon_inputs_fail(field,value):
    with pytest.raises(ValueError):example(**{field:np.full((2,101),value)})


def test_noise_model_is_explicit_and_old_default_is_preserved():
    assert NoiseConfig().andes_etc_noise_model=='legacy'
    assert NoiseConfig(andes_etc_noise_model='photon').andes_etc_noise_model=='photon'
    with pytest.raises(ValueError):NoiseConfig(andes_etc_noise_model='unknown')


@pytest.mark.parametrize('atomic_block', ['omitted', None, {'enabled': False}])
def test_molecular_config_accepts_inactive_atomic_transmission(atomic_block):
    """Existing CCF JSON files may omit the optional sodium settings."""
    import json
    from pathlib import Path
    from exoplore.config.models import SimulationConfig
    from exoplore.core.simulator import ExoploreSimulator
    from exoplore.core.spectroscopy_mode import spectroscopy_route

    raw = json.loads(Path('configs/hd189733b_andes_retrieval_blain24_noisy.json').read_text())
    raw['noise']['andes_etc_noise_model'] = 'photon'
    raw.pop('atomic_transmission', None)
    if atomic_block != 'omitted':
        raw['atomic_transmission'] = atomic_block
    cfg = SimulationConfig.from_dict(raw)
    simulator = ExoploreSimulator(cfg)
    assert spectroscopy_route(simulator.config) == 'standard'


def test_molecular_photon_noise_rejects_enabled_atomic_transmission():
    import json
    from pathlib import Path
    from exoplore.config.models import SimulationConfig
    from exoplore.core.simulator import ExoploreSimulator

    raw = json.loads(Path('configs/hd189733b_andes_retrieval_blain24_noisy.json').read_text())
    raw['noise']['andes_etc_noise_model'] = 'photon'
    sodium = json.loads(Path('configs/hd189733b_andes_sodium_rm_clv.json').read_text())
    raw['atomic_transmission'] = sodium['atomic_transmission']
    with pytest.raises(ValueError, match='synthetic molecular transit'):
        ExoploreSimulator(SimulationConfig.from_dict(raw))
