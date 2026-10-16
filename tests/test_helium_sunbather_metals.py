"""Composition validation and retained native Parker/Cloudy iterations."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exoplore.config.sunbather import SunbatherConfig
from exoplore.atmosphere.helium_sunbather_metals import save_metal_parker_profile, _select_metal_abundances


def settings():
    return SunbatherConfig('cloudy', 'new_project', 1.1, 20, 'constant', 100.,
                           True, True, .01, 7, 'pwinds')


@pytest.mark.parametrize('value', [0., 1., 100.])
def test_linear_metal_scale(value):
    assert replace(settings(), metal_scale=value).metal_scale == value


def test_tidal_gravity_defaults_off():
    config = SunbatherConfig('cloudy', 'new_project', 1.1, 20, 'constant', 0., True)
    assert config.stellar_tidal_gravity is False


def test_default_composition_is_hydrogen_and_helium_only():
    config=SunbatherConfig('cloudy','new_project',1.1,20,'constant',cosmic_rays=True)
    assert config.metal_scale==0.
    assert config.metal_elements is None
    assert config.parker_mu_tolerance is None
    assert config.metals_enabled is False
    assert config.effective_metal_scale==0.
    with pytest.raises(ValueError,match='cosmic rays'):
        SunbatherConfig('cloudy','new_project',1.1,20,'constant')


def test_main_json_metal_switch_preserves_configured_abundance():
    config=replace(settings(),include_metals=False,parker_mu_tolerance=None,
                   parker_mu_max_iterations=None,parker_initialization=None)
    assert config.metal_scale==100.
    assert config.effective_metal_scale==0.
    assert not config.metals_enabled
    enabled=replace(settings(),include_metals=True)
    assert enabled.metals_enabled
    assert enabled.effective_metal_scale==100.
    assert replace(settings(),include_metals=None).metals_enabled  # Archived JSON.
    with pytest.raises(ValueError,match='positive metal_scale'):
        replace(settings(),include_metals=True,metal_scale=0.)


@pytest.mark.parametrize('value',['false',0,1,[]])
def test_metal_switch_requires_boolean(value):
    with pytest.raises(ValueError,match='include_metals'):
        replace(settings(),include_metals=value)


@pytest.mark.parametrize('enabled',[False,True])
def test_bundled_main_json_choices_validate_without_launching_solver(enabled):
    path=Path(__file__).resolve().parents[1]/'configs/helium_sunbather_options.json'
    block=json.loads(path.read_text())['atmosphere']['helium']['sunbather']
    block['include_metals']=enabled
    config=SunbatherConfig(**block)
    assert config.metals_enabled is enabled
    assert config.effective_metal_scale==(1. if enabled else 0.)
    assert config.metal_elements is None


@pytest.mark.parametrize('elements', [[], ['C','C'], ['H'], ['He'], ['X'], 'C', [None]])
def test_invalid_element_selection(elements):
    with pytest.raises(ValueError,match='metal_elements'):
        replace(settings(),metal_elements=elements)


def test_selected_abundances_preserve_he_and_do_not_redistribute_metals():
    config=replace(settings(),metal_elements=['C','O'])
    original={'He':.02,'C':100.,'O':100.,'Fe':100.}
    assert _select_metal_abundances(original,config.metal_elements)=={'He':.02,'C':100.,'O':100.,'Fe':0.}
    assert original['Fe']==100.
    assert _select_metal_abundances(original,None)==original
    with pytest.raises(ValueError,match='metal_elements'):
        replace(config,metal_scale=0.)


@pytest.mark.parametrize('value', [-1., True, np.nan, np.inf])
def test_invalid_metal_scale(value):
    with pytest.raises(ValueError, match='metal_scale'):
        replace(settings(), metal_scale=value)


@pytest.mark.parametrize('field', ['parker_mu_tolerance', 'parker_mu_max_iterations',
                                  'parker_initialization'])
def test_metals_require_explicit_iteration_choices(field):
    with pytest.raises(ValueError):
        replace(settings(), **{field: None})
    replace(settings(), metal_scale=0., **{field: None})


class NativeSolver:
    """A file-producing stand-in for the upstream numerical solver."""
    def __init__(self, masses):
        self.masses = iter(masses)
        self.calls = []

    def save_temp_parker_profile(self, planet, logmass, temperature, spectrum,
                                 zdict, pdir, **kwargs):
        self.calls.append((zdict, kwargs))
        root = Path(kwargs['projectpath'])/'parker_profiles'/planet.name/pdir/'temp'
        root.mkdir(parents=True, exist_ok=False)
        path = root/'profile.txt'
        path.write_text('retained profile')
        return str(path), kwargs['mu_bar'] or 1., .01

    def run_parker_with_cloudy(self, filename, temperature, planet, zdict):
        stem = str(Path(filename).with_suffix(''))
        for suffix in ('.in', '.out', '.ovr'):
            Path(stem + suffix).write_text('retained Cloudy output')
        return stem, SimpleNamespace(alt=np.array([1., 2.]), v=np.ones(2))

    def calc_mu_bar(self, sim):
        return next(self.masses)

    def calc_neutral_mu(self, zdict):
        return 2.


def tools():
    class Simulation:
        def __init__(self, *args, **kwargs):
            self.ovr = SimpleNamespace(alt=np.array([20., 1.]), mu=np.ones(2))

        def addv(self, *args):
            pass
    return SimpleNamespace(Sim=Simulation)


def run(root, native, config=None):
    return save_metal_parker_profile(native, tools(), SimpleNamespace(name='test', R=1.),
                                    10.4, 5000, {}, {'C': 100.}, config or settings(), 20, root)


def test_converged_native_iterations_and_preservation(tmp_path):
    native = NativeSolver([1.5, 1.505])
    result = run(tmp_path, native)
    assert result.read_text() == 'retained profile'
    assert len(native.calls) == 3  # Two Cloudy evaluations and the updated final wind.
    assert len(list(tmp_path.rglob('*.ovr'))) == 2
    assert (tmp_path/'parker_mu_convergence.json').is_file()
    assert all(zdict == {'C': 100.} and args['no_tidal'] is False
               and args['altmax'] == 20 for zdict, args in native.calls)
    with pytest.raises(FileExistsError):
        run(tmp_path, native)
    assert result.read_text() == 'retained profile'


def test_unconverged_mean_mass_is_not_accepted(tmp_path):
    with pytest.raises(RuntimeError, match='did not converge'):
        run(tmp_path, NativeSolver([2., 3.]), replace(settings(), parker_mu_max_iterations=2))
    assert len(list(tmp_path.rglob('*.ovr'))) == 2
    assert not (tmp_path/'parker_mu_convergence.json').exists()


def test_neutral_initialization_uses_supplied_composition(tmp_path):
    native = NativeSolver([2.001])
    run(tmp_path, native, replace(settings(), parker_initialization='neutral'))
    assert native.calls[0][1]['mu_bar'] == 2.
    np.testing.assert_array_equal(native.calls[0][1]['mu_struc'], [[1., 2.], [20., 2.]])


def test_existing_iteration_is_refused_before_native_write(tmp_path):
    path = tmp_path/'parker_profiles/test/exoplore/mu_iterations/iteration_001'
    path.mkdir(parents=True)
    sentinel = path/'original.txt'
    sentinel.write_text('original')
    native = NativeSolver([])
    with pytest.raises(FileExistsError):
        run(tmp_path, native)
    assert not native.calls
    assert sentinel.read_text() == 'original'
