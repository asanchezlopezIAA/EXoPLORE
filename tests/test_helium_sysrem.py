"""Check helium-only SYSREM filtering and its masked pixel contract."""
from types import SimpleNamespace
import numpy as np
from exoplore.pipelines.helium_sysrem import correct_helium_sysrem


def test_removes_stationary_time_variable_absorption():
    wave=np.linspace(1080,1085,120)
    amplitude=np.linspace(.2,1,12)
    line=np.exp(-.5*((wave-1083)/.08)**2)
    flux=1-.1*amplitude[:,None]*line
    settings=SimpleNamespace(_reference_indices=[0,1,2],oh_mode='ignore',
        continuum_stellar_windows_nm=[[1080,1081]],minimum_continuum_pixels=10)
    result=correct_helium_sysrem(np.tile(wave,(12,1)),flux,np.full_like(flux,.01),settings,1)
    assert np.nanmax(abs(result['flux']-1))<1e-8
    assert result['components'].shape==(1,12,120)
    assert np.isfinite(result['error']).all()


def test_invalid_native_column_remains_masked():
    wave=np.linspace(1080,1085,120)
    flux=np.ones((12,120));flux[:,60]=np.nan
    settings=SimpleNamespace(_reference_indices=[0,1,2],oh_mode='ignore',
        continuum_stellar_windows_nm=[[1080,1081]],minimum_continuum_pixels=10)
    result=correct_helium_sysrem(np.tile(wave,(12,1)),flux,np.full_like(flux,.01),settings,1)
    assert np.isnan(result['flux'][:,60]).all()
    assert np.nanmax(abs(result['flux']-1))==0
