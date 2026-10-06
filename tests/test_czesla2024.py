"""Scientific and integration tests for the opt-in direct helium preparation."""
from dataclasses import asdict,replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from exoplore.config import SimulationConfig
from exoplore.config.czesla2024 import Czesla2024Config
from exoplore.core.simulator import ExoploreSimulator
from exoplore.instruments.crires_czesla2024 import protected_windows
from exoplore.pipelines.czesla2024 import exposure_indices,fit_helium_multiplet,prepare_transmission
from exoplore.pipelines.helium_math import C_KMS,air_to_vacuum_nm,multiplet,resample,stellar_wavelength
from exoplore.pipelines.prepare import preparing_pipeline


def example() -> SimulationConfig:
    return SimulationConfig.from_json(Path(__file__).parents[1]/'configs/wasp121b_crires_czesla2024.json')


def small_config() -> Czesla2024Config:
    return replace(example().pipeline.czesla2024,reference_running_numbers=[1,2,8,9],in_transit_running_numbers=[3,4,5,6,7],monte_carlo_draws=100)


def synthetic(config):
    wave=np.linspace(1077.5,1084.7,2400);berv=np.linspace(-12,-11,9);rv=np.linspace(-70,70,9)
    native=np.array([wave*(1+config.gamma_kms/C_KMS)/(1+b/C_KMS) for b in berv])
    flux=np.array([(1+.01*(wave-1083.3))*(1-.12*np.exp(-.5*((wave-1083.327)/.04)**2)) for _ in berv])
    for i in range(2,7):flux[i]*=1-.02*np.exp(-.5*((C_KMS*(wave/1083.327-1)-rv[i])/10)**2)
    return native,flux,np.full_like(flux,.001),berv,rv


def test_config_round_trip_and_existing_defaults():
    config=example();round_trip=SimulationConfig.from_dict(config.to_dict())
    assert isinstance(round_trip.pipeline.czesla2024,Czesla2024Config)
    assert asdict(round_trip.pipeline.czesla2024)==asdict(config.pipeline.czesla2024)
    assert SimulationConfig().pipeline.czesla2024 is None
    assert SimulationConfig().pipeline.name=='BL19'


def test_explicit_conventions_and_selections():
    config=small_config()
    with pytest.raises(ValueError):replace(config,wavelength_frame='air')
    with pytest.raises(ValueError):replace(config,oh_mode='model')
    with pytest.raises(ValueError):replace(config,reference_running_numbers=[1,1])
    with pytest.raises(ValueError):replace(config,in_transit_running_numbers=[1,3])
    with pytest.raises(ValueError):exposure_indices([10],9)
    assert exposure_indices([1,2,8,9],9).tolist()==[0,1,7,8]


def test_stellar_then_planet_alignment_preserves_moving_line():
    config=small_config();native,flux,error,berv,rv=synthetic(config);before=flux.copy()
    result=prepare_transmission(native,flux,error,berv,rv,config)
    np.testing.assert_array_equal(flux,before)
    center=np.argmin(abs(result['planet_wave_nm']-1083.327))
    assert .977<result['planet_coadd'][center]<.984
    ref=exposure_indices(config.reference_running_numbers,9)
    weights=1/result['normalized_stellar_variance'][ref]
    averaged=np.sum(weights*result['stellar_transmission'][ref],axis=0)/weights.sum(axis=0)
    np.testing.assert_allclose(averaged,1,atol=1e-12)


def test_frame_sign_and_conversion():
    rest=1083.327;gamma=38.35;berv=-12.
    observed=rest*(1+gamma/C_KMS)/(1+berv/C_KMS)
    assert stellar_wavelength(observed,berv,gamma)==pytest.approx(rest)
    assert 75<C_KMS*(air_to_vacuum_nm(np.array([rest]))[0]/rest-1)<90


def test_masked_native_pixel_is_not_interpolated_over():
    wave=np.arange(6,dtype=float);flux=np.ones(6);flux[2]=np.nan
    values,_=resample(wave,flux,np.ones(6),np.array([1.5,2.5,3.5]))
    assert np.isnan(values[:2]).all();assert values[2]==1


def test_guard_drops_contaminated_anchor_in_observer_frame():
    windows=[[1083.29,1083.36],[1084.,1084.1]]
    retained=protected_windows(windows,np.array([1083.2057472,1083.3216751,1083.3306444]),0,0,0,50)
    assert retained==[[1084.,1084.1]]


def test_shared_multiplet_shift_and_width_recovery():
    wave=np.linspace(1082.95,1083.6,500);lines=np.array([1083.2057472,1083.3216751,1083.3306444]);strength=np.array([.0599,.179693,.299482])
    # Independent Gaussian-cgs cross-section expression, no call to fitted model.
    sigma=np.hypot(10.,C_KMS/61600/np.sqrt(8*np.log(2)))
    velocity=C_KMS*(wave[:,None]/lines[None,:]-1)-2.6
    tau=np.sum(4.7e11*.02654029*strength*lines*1e-7/(np.sqrt(2*np.pi)*sigma*1e5)*np.exp(-.5*(velocity/sigma)**2),axis=1)
    flux=.95+.05*np.exp(-tau)
    fit=fit_helium_multiplet(wave,flux,np.full_like(wave,.001),lines,strength,.05,61600,[11.5,0.,15.],[[9,-30,1],[15,30,40]])
    assert fit['success'];assert fit['velocity_shift_kms']==pytest.approx(2.6,abs=.001)
    assert fit['intrinsic_sigma_kms']==pytest.approx(10.,abs=.001)


def test_runner_routes_only_explicit_czesla_choice():
    config=example()
    with patch('exoplore.pipelines.czesla2024.run_czesla2024') as run:
        ExoploreSimulator(config).run();run.assert_called_once_with(config)
    invalid=SimulationConfig.from_dict(config.to_dict());invalid.retrieval.enabled=True
    with pytest.raises(ValueError):ExoploreSimulator(invalid)


def test_dispatcher_exposes_recipe_without_mutating_input():
    config=small_config();native,flux,error,berv,rv=synthetic(config);before=flux.copy()
    inp={'preparing_pipeline':'czesla2024','telluric_corrected':True,'czesla2024_config':asdict(config),'berv_kms':berv,'planet_rv_kms':rv}
    result=preparing_pipeline(inp,flux,error,native,np.arange(flux.shape[1]),np.array([],int),np.ones(9),np.zeros(9),np.array([0,1,7,8]),None)
    assert len(result)==9;assert result[4]==0;assert result[8] is None
    np.testing.assert_array_equal(flux,before)
    inp['telluric_corrected']=False
    with pytest.raises(ValueError):preparing_pipeline(inp,flux,error,native,np.arange(flux.shape[1]),np.array([],int),np.ones(9),np.zeros(9),np.array([0,1,7,8]),None)


def test_existing_outputs_refused_before_loading_spectra(tmp_path):
    from exoplore.pipelines.czesla2024 import run_czesla2024
    config=example();config.paths.output_root=str(tmp_path)
    directory=tmp_path/config.planet.name/'czesla2024';directory.mkdir(parents=True)
    preserved=directory/'old_result.txt';preserved.write_text('preserve this result')
    with patch('exoplore.pipelines.czesla2024.read_corrected_night') as reader:
        with pytest.raises(FileExistsError):run_czesla2024(config)
        reader.assert_not_called()
    assert preserved.read_text()=='preserve this result'


def test_large_wavelength_displacement_is_rejected(tmp_path):
    import hashlib
    from astropy.io import fits
    from exoplore.pipelines.czesla2024 import read_corrected_night
    config=replace(small_config(),input_path=str(tmp_path))
    raw=tmp_path/'raw.fits';fits.PrimaryHDU().writeto(raw)
    extracted=tmp_path/'extracted.fits';wave=np.linspace(1077.5,1084.7,128)
    table=fits.BinTableHDU.from_columns([
        fits.Column(name='02_01_WL',format='D',array=wave),
        fits.Column(name='02_01_SPEC',format='D',array=np.ones(128)),
        fits.Column(name='02_01_ERR',format='D',array=np.full(128,.01))],name='CHIP1.INT1')
    fits.HDUList([fits.PrimaryHDU(),table]).writeto(extracted)
    directory=tmp_path/'exposure_01_A';directory.mkdir();correction=directory/'correction.npz'
    np.savez(correction,native_wave_nm=wave,refined_wave_nm=wave*(1+50/C_KMS),transmittance=np.ones(128),flux=np.ones(128),error=np.full(128,.01),valid=np.ones(128,bool))
    report={'exposure':1,'nod':'A','accepted':True,'frame':'topocentric vacuum nm','raw':str(raw),'extracted':str(extracted),
            'raw_header_sha256':hashlib.sha256(fits.getheader(raw).tostring().encode()).hexdigest(),
            'extracted_sha256':hashlib.sha256(extracted.read_bytes()).hexdigest(),
            'correction_sha256':hashlib.sha256(correction.read_bytes()).hexdigest(),'he_shift_kms':50.}
    (tmp_path/'night_report.json').write_text(json.dumps([report]))
    with pytest.raises(ValueError,match='displacement'):read_corrected_night(config)
    report['he_shift_kms']=0.;report['correction_sha256']='invalid'
    (tmp_path/'night_report.json').write_text(json.dumps([report]))
    with pytest.raises(ValueError,match='provenance hash'):read_corrected_night(config)
