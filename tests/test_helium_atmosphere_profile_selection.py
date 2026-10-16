"""Retained mean-mass iterations must not obscure the final Parker profile."""
from types import SimpleNamespace

import numpy as np

from exoplore.plotting.helium_atmosphere import write_atmosphere_diagnostics


def test_only_final_parker_profile_is_plotted(tmp_path):
    final=tmp_path/'parker_profiles/example/exoplore'
    intermediate=final/'mu_iterations/iteration_001/temp'
    intermediate.mkdir(parents=True)
    (intermediate/'pprof_intermediate.txt').write_text('not the final profile')
    parker=np.array([[7.1492e9,1e-12,1e4,1.],[1.42984e10,1e-13,2e4,1.]])
    np.savetxt(final/'pprof_final.txt',parker)
    profile=dict(radius_rp=np.array([1.,2.]),temperature_K=np.array([3000.,5000.]),
        density_g_cm3=parker[:,1],velocity_m_s=parker[:,2]/100,
        helium_triplet_m3=np.array([1e8,1e7]),cloudy_simulation='saved_converged')
    settings=SimpleNamespace(temperature_K=3000.,mass_loss_rate_g_s=1e10,
        hydrogen_number_fraction=.998,radial_max_rp=2.,sunbather={'stellar_tidal_gravity':False})
    planet=SimpleNamespace(name='example',planet_radius_rjup=1.)
    write_atmosphere_diagnostics(tmp_path,settings,planet,profile)
    with np.load(tmp_path/'parker_profiles.npz') as data:
        np.testing.assert_allclose(data['density_g_cm3'],parker[:,1])
    assert (intermediate/'pprof_intermediate.txt').read_text()=='not the final profile'
