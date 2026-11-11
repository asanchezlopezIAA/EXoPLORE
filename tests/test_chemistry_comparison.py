"""Check the tutorial comparison without starting external chemical solvers."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from exoplore.analysis.chemistry import (
    SPECIES, comparison_settings, equilibrium_reference, _vmr_from_profile,
)
from exoplore.config.chemistry import VulcanConfig, EddyDiffusionConfig
from exoplore.plotting.chemistry import plot_chemistry_profiles


def photo_settings():
    return VulcanConfig(network_file='thermo/NCHO_photo_network.txt',
        elemental_abundances={'H': 1., 'He': .08}, photochemistry=True,
        stellar_flux_file='stellar_uv.txt', orbit_radius_au=.031,
        zenith_angle_deg=48., diurnal_factor=1.,
        eddy_diffusion=EddyDiffusionConfig(mode='constant', value_cm2_s=1e9))


def test_comparison_changes_processes_with_fixed_network_and_inventory():
    original = photo_settings()
    cases = comparison_settings(original)
    assert cases['photochemistry'] is original
    assert original.photochemistry
    for case in cases.values():
        assert case.network_file == original.network_file
        assert case.elemental_abundances == original.elemental_abundances
        assert not case.condensation.enabled
    assert not cases['mixing'].photochemistry
    assert cases['mixing'].molecular_diffusion
    assert cases['mixing'].eddy_diffusion.value_cm2_s == 1e9
    assert not cases['thermal_control'].photochemistry
    assert not cases['thermal_control'].molecular_diffusion
    assert cases['thermal_control'].eddy_diffusion.mode == 'off'
    assert cases['thermal_control'].vertical_velocity_cm_s == 0.


@pytest.mark.parametrize('changes', [
    {'photochemistry': False},
    {'eddy_diffusion': EddyDiffusionConfig(mode='off')},
])
def test_incomplete_comparison_fails(changes):
    with pytest.raises(ValueError):
        comparison_settings(replace(photo_settings(), **changes))


def test_vmrs_use_native_gas_masses(tmp_path):
    np.savez(tmp_path/'gas_profile.npz', species=np.array(['H2','H']),
             molecular_masses=np.array([2.,1.]))
    profile = SimpleNamespace(provenance_file=tmp_path/'manifest.json',
        mean_molecular_weight=np.array([1.5,1.5]),
        mass_fractions={'H2':np.array([2/3,2/3]),'H':np.array([1/3,1/3])})
    np.testing.assert_allclose(_vmr_from_profile(profile, ('H2','H')), .5)


def test_equilibrium_uses_full_gas_denominator_and_explicit_atoms(tmp_path, monkeypatch):
    composition = tmp_path/'compose.txt'
    composition.write_text('species H He C e mass\nH2 2 0 0 0 2\n'
        'He 0 1 0 0 4\nCH4 4 0 1 0 16\n')
    thermo = tmp_path/'thermo.txt'
    thermo.write_text('controlled thermodynamic input')
    calls = []

    class FakeEquilibrium:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.reactants = kwargs.get('reactants', ['H2','He','CH4','H+','C(c)'])
            self.thermofpath = thermo

        def solve(self, p, t):
            self.layers = len(p)

        def result_mol(self):
            return {'H2':np.full(self.layers,.9), 'He':np.full(self.layers,.1)}

    fake = SimpleNamespace(ExoAtmos=FakeEquilibrium)
    monkeypatch.setitem(sys.modules, 'easychem', SimpleNamespace(easychem=fake))
    monkeypatch.setitem(sys.modules, 'easychem.easychem', fake)
    result, info = equilibrium_reference(np.array([.1,1.]), np.array([1000.,1000.]),
        {'H':1., 'He':1/18}, composition, species=('H2',))
    np.testing.assert_allclose(result, [[.9,.9]])
    assert calls[-1]['reactants'] == ['H2','He']
    assert calls[-1]['atoms'] == ['H','He']
    np.testing.assert_allclose(calls[-1]['atomAbunds'], [1.,1/18])
    assert info['max_relative_element_error'] < 1e-14
    assert len(info['thermodata_sha256']) == 64
    selected, _, gas = equilibrium_reference(
        np.array([.1,1.]), np.array([1000.,1000.]), {'H':1., 'He':1/18},
        composition, species=('H2',), return_gas=True)
    np.testing.assert_allclose(selected, [[.9,.9]])
    assert gas['species'].tolist() == ['H2','He']
    np.testing.assert_allclose(gas['mole_fractions'].sum(axis=1), 1.)
    np.testing.assert_allclose(gas['mole_fractions'] @ gas['molecular_masses'], 2.2)


def figure_arrays():
    return dict(pressure_bar=np.geomspace(1e-8,100,4), species=np.array(SPECIES),
        easychem=np.full((9,4),.01), mixing=np.full((9,4),.02),
        photochemistry=np.full((9,4),.03))


def test_figure_contains_nine_species_and_unmodified_profiles():
    arrays = figure_arrays()
    fig = plot_chemistry_profiles(arrays, 'HD 189733 b')
    assert len(fig.axes) == 9
    for index, ax in enumerate(fig.axes):
        assert len(ax.lines) == 3
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), arrays['easychem'][index])
        assert ax.yaxis_inverted()
    plt.close(fig)


def test_figure_requires_photochemistry():
    arrays = figure_arrays()
    del arrays['photochemistry']
    with pytest.raises(ValueError, match='photochemical'):
        plot_chemistry_profiles(arrays, 'WASP-76 b')


def test_zero_abundances_remain_masked():
    arrays = figure_arrays()
    for method in ('easychem','mixing','photochemistry'):
        arrays[method][0] = 0.
    fig = plot_chemistry_profiles(arrays, 'controlled zero profile')
    assert all(np.ma.getmaskarray(line.get_xdata()).all() for line in fig.axes[0].lines)
    plt.close(fig)


def test_complete_comparison_preserves_column_and_reports_cache(tmp_path, monkeypatch):
    import json
    import exoplore.analysis.chemistry as comparison
    from exoplore.config.models import AtmosphereRegionConfig
    from exoplore.config.chemistry import ChemistryConfig

    region = AtmosphereRegionConfig(use_easychem=False,
        chemistry=ChemistryConfig('vulcan', photo_settings()))
    config = SimpleNamespace(atmosphere=SimpleNamespace(morning_day=region),
        planet=SimpleNamespace(parameter_file='planet.json'),
        paths=SimpleNamespace(vulcan_cache_dir=str(tmp_path/'cache'),
                              vulcan_directory=str(tmp_path/'VULCAN')))
    planet = SimpleNamespace(name='controlled planet', M_pl=1e30, R_pl=1e10,
                             R_star=1e11, stellar_teff_K=5000., t_int_K=200.)
    monkeypatch.setattr(comparison, 'load_planet', lambda _: planet)
    p = np.array([1e-8,1.,100.])
    t = np.array([900.,1000.,1200.])
    monkeypatch.setattr(comparison, 'create_log_pressure_grid', lambda *args: p)
    monkeypatch.setattr(comparison, 'calculate_temperature_structure', lambda *args: t)
    monkeypatch.setattr(comparison, 'equilibrium_reference',
        lambda *args: (np.full((9,3),.01), {'max_relative_element_error':0.}))
    seen = []

    def native(settings, pressures, temperatures, species, physical, paths, **kwargs):
        np.testing.assert_array_equal(pressures, p)
        np.testing.assert_array_equal(temperatures, t)
        assert settings.elemental_abundances == region.chemistry.vulcan.elemental_abundances
        seen.append(settings)
        label = ('photo' if settings.photochemistry else
                 'thermal' if settings.eddy_diffusion.mode == 'off' else 'mixing')
        folder = tmp_path/'cache'/label
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder/'manifest.json'
        manifest.write_text(json.dumps(dict(runtime_seconds=.05, cache_key=label, atom_loss={'H':0.})))
        alignment = folder/'alignment.json'
        alignment.write_text(json.dumps({'iterations':[{'chemistry_manifest':str(manifest)}]}))
        np.savez(folder/'gas_profile.npz', species=np.array(SPECIES), molecular_masses=np.ones(9))
        return SimpleNamespace(provenance_file=manifest, radius_alignment_file=alignment,
            mean_molecular_weight=np.ones(3), mass_fractions={name:np.full(3,.01) for name in SPECIES})

    monkeypatch.setattr(comparison, 'run_vulcan', native)
    arrays, first = comparison.compare_profiles(config, 'morning_day', tmp_path/'results')
    assert len(seen) == 3
    assert arrays['photochemistry'].shape == (9,3)
    assert first['cases']['photochemistry']['new_columns'] == 1
    assert first['cases']['photochemistry']['reused_columns'] == 0
    fresh = first['cases']['photochemistry']['fresh_wall_seconds']
    _, second = comparison.compare_profiles(config, 'morning_day', tmp_path/'results')
    assert second['cases']['photochemistry']['cache_hit']
    assert second['cases']['photochemistry']['fresh_wall_seconds'] == fresh
    assert second['thermal_control_max_abs_log10_ratio'] == {name:0. for name in SPECIES}
    assert (tmp_path/'results/comparison.npz').is_file()
