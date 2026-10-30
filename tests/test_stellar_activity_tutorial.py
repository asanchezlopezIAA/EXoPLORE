"""Check Tutorial 14's fixed atmosphere, controlled spectra and paired reductions."""
from pathlib import Path
import sys
import numpy as np
import pytest
from exoplore.config import SimulationConfig
from exoplore.core.atomic_simulation import telluric_input_hash, reduce_atomic_observation

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from run_stellar_activity_tutorial import (case_config, fixed_settings, make_components,
                                           array_hash, CASES)
from plot_stellar_activity_tutorial import line_metrics


@pytest.fixture
def config():
    return SimulationConfig.from_json('configs/hd189733b_andes_stellar_activity.json')


def test_cases_keep_tutorial13_science(config):
    reference=SimulationConfig.from_json('configs/hd189733b_andes_sodium_rm_clv.json')
    assert fixed_settings(config)==fixed_settings(reference)
    for name in CASES:
        case=case_config(config,name)
        assert fixed_settings(case)==fixed_settings(reference)
        assert case.atmosphere.planet_model==reference.atmosphere.planet_model


def test_same_spot_spectrum_only_moves_between_geometries(config):
    a,b=(case_config(config,name).stellar_surface for name in ['unocculted_spot','crossed_spot'])
    assert a.components==b.components
    assert a.regions[0].component==b.regions[0].component=='spot'
    assert a.regions[0].x==b.regions[0].x
    assert a.regions[0].radius==b.regions[0].radius
    assert a.regions[0].y>0>b.regions[0].y


def test_controlled_grids_keep_absolute_contrasts(config,tmp_path):
    centre=config.atomic_transmission.line_wavelengths_nm[0]/1000
    wave=centre*(1+np.linspace(-60,60,121)/299792.458)
    mu=np.array([0.,.5,1.])
    continuum=(.3+.7*mu[:,None])*np.ones_like(wave)
    intensity=continuum*(1-.5*np.exp(-.5*((wave/centre-1)*299792.458/8)**2))
    quiet=tmp_path/'quiet.npz'
    np.savez(quiet,wavelength_um=wave,mu=mu,intensity=intensity,continuum=continuum,
        intensity_kind='specific_intensity',wavelength_medium='vacuum',intensity_unit='shared units',
        source='test',resolving_power=0.)
    config.rm_clv.model_file=str(quiet)
    for name in ['spot','facula']:
        config.stellar_surface.components[name].intensity_model_file=str(tmp_path/f'{name}.npz')
    make_components(config)
    with np.load(tmp_path/'spot.npz') as spot, np.load(tmp_path/'facula.npz') as facula:
        np.testing.assert_allclose(spot['continuum']/continuum,.60,atol=2e-15)
        np.testing.assert_allclose(facula['continuum']/continuum,
                                  np.broadcast_to(1.10+.20*(1-mu[:,None]),continuum.shape))
        # At line centre, the spot multiplies normalized intensity by 0.60.
        np.testing.assert_allclose(spot['intensity'][:,60]/spot['continuum'][:,60],.30)
        np.testing.assert_allclose(facula['intensity'][:,60]/facula['continuum'][:,60],.70)
        np.testing.assert_array_equal(spot['mu'],mu)
        np.testing.assert_array_equal(spot['wavelength_um'],wave)


def observation():
    shape=(3,8)
    return dict(raw_flux=np.ones(shape),raw_error=np.full(shape,.01),
                wave_nm=np.tile(np.linspace(588,593,8),(3,1)),
                bjd=np.arange(3)+2450000.,airmass=np.ones(3))


def test_telluric_cache_tracks_data_and_fit_settings(config):
    data=observation()
    original=telluric_input_hash(config,data)
    config.stellar_surface.correction_surface='matched'
    assert telluric_input_hash(config,data)==original
    changed={k:v.copy() for k,v in data.items()}
    changed['raw_flux'][0,0]+=.001
    assert telluric_input_hash(config,changed)!=original
    config.atomic_transmission.molecfit['continuum_degree']=2
    assert telluric_input_hash(config,data)!=original


def test_reduction_rejects_wrong_observation_cache(config,tmp_path):
    with pytest.raises(ValueError,match='different observation'):
        reduce_atomic_observation(config,tmp_path,observation(),None,None,100000.,0.,0.,
            corrections=dict(input_sha256=np.array('wrong')))


def test_fixed_window_line_metrics():
    centre=589.158326
    velocity=np.linspace(-30,30,301)
    wave=centre*(1+velocity/299792.458)
    transmission=1-.01*np.exp(-.5*((velocity-2)/4)**2)
    inside=abs(299792.458*(wave/centre-1))<=20
    covariance=np.eye(inside.sum())*1e-8
    result=line_metrics(wave,transmission,centre,covariance)
    assert abs(result['centroid_kms']-2)<.001
    assert abs(result['rms_width_kms']-4)<.002
    assert .81<result['core_depth_percent']<.83
    assert result['centroid_kms_error']>0


def test_signed_moments_do_not_clip_emission_samples():
    centre=589.158326
    velocity=np.linspace(-20,20,101)
    wave=centre*(1+velocity/299792.458)
    transmission=1-.01*np.exp(-.5*(velocity/4)**2)
    transmission[velocity>10]+=.002
    result=line_metrics(wave,transmission,centre)
    assert result['centroid_kms']<0


def test_array_identity_includes_coordinate_shape():
    x=np.arange(8.)
    assert array_hash(dict(x=x))!=array_hash(dict(x=x.reshape(2,4)))


def test_shared_atmosphere_uses_tutorial13_prt_inputs(config,monkeypatch):
    from exoplore.core.atomic_simulation import calculate_atomic_atmosphere
    from exoplore.atmosphere.transmission import ChordTransmission
    from exoplore.planets import load_planet
    from petitRADTRANS import radtrans
    import exoplore.atmosphere.transmission as transmission
    planet=load_planet(config.planet.parameter_file)
    saved={}
    class Model:
        def __init__(self,**kwargs):
            saved['setup']=kwargs
    rays=ChordTransmission(np.array([.588,.594]),np.array([1.1,1.])*planet.R_pl,
                           np.ones((2,2)))
    def calculate(model,**kwargs):
        saved['calculation']=kwargs
        return rays
    monkeypatch.setattr(radtrans,'Radtrans',Model)
    monkeypatch.setattr(transmission,'calculate_chord_transmission',calculate)
    assert calculate_atomic_atmosphere(config,planet) is rays
    setup,calculation=saved['setup'],saved['calculation']
    assert setup['line_species']==['Na__LorCut']
    assert setup['wavelength_boundaries']==[.5882,.5932]
    np.testing.assert_array_equal(setup['pressures'],np.logspace(-15,2,445))
    np.testing.assert_array_equal(calculation['temperatures'],np.full(445,2750.))
    np.testing.assert_array_equal(calculation['mass_fractions']['Na__LorCut'],np.full(445,.001))
    np.testing.assert_array_equal(calculation['mean_molar_masses'],np.full(445,2.3))
    assert calculation['planet_radius']==planet.R_pl
    assert calculation['reference_pressure']==.01
