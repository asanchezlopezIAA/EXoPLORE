"""Small chemistry columns test units, cache isolation and the pRT connection."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from exoplore.config import SimulationConfig
from exoplore.config.models import AtmosphereRegionConfig, PathConfig
from exoplore.config.chemistry import ChemistryConfig, VulcanConfig
from exoplore.atmosphere.models import RadiativeTransferInputs
from exoplore.atmosphere.vulcan import gas_composition, regrid_composition, _run_native_column as run_vulcan


@pytest.fixture
def settings():
    return VulcanConfig(network_file='thermo/network.txt',
                        elemental_abundances={'H':1., 'He':.08, 'O':.001}, molecular_diffusion=False)


@pytest.fixture
def physical():
    return RadiativeTransferInputs('transit',1.9e30,7.e9,7.e10,5000.,100000.,100.,None,True)


def native(*, condensed=False):
    names=['H2','He','H2O']+(['water_ice'] if condensed else [])
    density=np.tile([8.,1.,1.]+([20.] if condensed else []),(3,1))
    return {'parameter':{'end_case':1}, 'variable':{'species':names,'y':density,'atom_loss':{'H':1e-8,'O':1e-9}},
            'atm':{'pco':np.array([1e6,1e3,1.]),'Tco':np.full(3,1000.),'mu':np.full(3,3.8)}}


def test_legacy_configs_and_roundtrip():
    config=SimulationConfig.from_dict({'atmosphere':{'planet_model':{'use_easychem':False}}})
    assert config.atmosphere.planet_model.chemistry is None
    assert config.atmosphere.planet_model.chemistry_backend=='manual'
    assert AtmosphereRegionConfig().chemistry_backend=='easychem'
    assert SimulationConfig.from_dict(config.to_dict()).to_dict()==config.to_dict()


@pytest.mark.parametrize('backend',['manual','easychem'])
def test_explicit_selection_overrides_legacy_boolean(backend):
    region=AtmosphereRegionConfig(use_easychem=backend=='manual', chemistry={'backend':backend})
    assert region.chemistry_backend==backend


def test_vulcan_json_roundtrip(settings,tmp_path):
    payload={'backend':'vulcan','vulcan':asdict(settings)}
    config=SimulationConfig.from_dict({'atmosphere':{'planet_model':{'chemistry':payload},
            'morning_day':{'chemistry':payload}},'paths':{'vulcan_directory':'external/VULCAN','vulcan_python':'env/bin/python'}})
    path=tmp_path/'run.json';config.to_json(path)
    assert SimulationConfig.from_json(path).to_dict()==config.to_dict()
    assert config.atmosphere.morning_day.chemistry.vulcan.eddy_diffusion.mode=='off'


@pytest.mark.parametrize('kw',[
    {'backend':'unknown'}, {'backend':'vulcan'}, {'backend':'easychem','vulcan':{}},
])
def test_invalid_backend(kw):
    with pytest.raises((ValueError,TypeError)):
        ChemistryConfig(**kw)


@pytest.mark.parametrize('kw',[
    {'elemental_abundances':{'H':1}}, {'elemental_abundances':{'H':2,'He':.08}},
    {'elemental_abundances':{'H':1,'He':-.1}},
    {'eddy_diffusion':{'mode':'constant','value_cm2_s':-1}},
    {'eddy_diffusion':{'mode':'file'}}, {'photochemistry':True},
    {'photochemistry':True,'stellar_flux_file':'uv','orbit_radius_au':.03,'zenith_angle_deg':90.,'diurnal_factor':1.},
    {'molecular_diffusion_scheme':'mystery'}, {'solver_options':{'use_photo':1}},
    {'solver_options':{'rtol':0}}, {'top_boundary':{'type':'fixed_mixing_ratios','mixing_ratios':{'H2':1.}}},
    {'condensation':{'enabled':True}}, {'condensation':{'settling':True}},
])
def test_invalid_settings(settings,kw):
    d=asdict(settings);d.update(kw)
    with pytest.raises((ValueError,TypeError)):
        VulcanConfig(**d)


def test_mole_to_mass_includes_unselected_gases():
    p,t,names,x,m=gas_composition(native(),{'H2':2.,'He':4.,'H2O':18.})
    mf,mu=regrid_composition(p,t,p,t,names,x,m,['H2O'])
    np.testing.assert_allclose(mu,3.8)
    np.testing.assert_allclose(mf['H2O'],1.8/3.8)
    assert set(mf)=={'H2O'}


def test_condensates_excluded_from_gas_mu():
    d=native(condensed=True)
    d['atm']['mu']=np.full(3,39.8)
    result=gas_composition(d,{'H2':2.,'He':4.,'H2O':18.,'water_ice':18.},['water_ice'])
    p,t,names,x,m=result
    assert 'water_ice' not in names
    np.testing.assert_allclose(x @ m,3.8)


@pytest.mark.parametrize('change', ['unconverged','negative','nonfinite','shape','duplicate','atom_loss','missing_loss','nonmonotonic'])
def test_invalid_native_output(change):
    d=native()
    if change=='unconverged':d['parameter']['end_case']=3
    elif change=='negative':d['variable']['y'][0,0]=-1
    elif change=='nonfinite':d['variable']['y'][0,0]=np.nan
    elif change=='shape':d['variable']['y']=np.ones((2,3))
    elif change=='duplicate':d['variable']['species']=['H2','H2','H2O']
    elif change=='atom_loss':d['variable']['atom_loss']['H']=.3
    elif change=='missing_loss':del d['variable']['atom_loss']
    elif change=='nonmonotonic':d['atm']['pco']=[1.,1.,1.]
    with pytest.raises(ValueError):gas_composition(d,{'H2':2.,'He':4.,'H2O':18.})


def test_mapping_grid_direction_and_temperature_check():
    p,t,names,x,m=gas_composition(native(),{'H2':2.,'He':4.,'H2O':18.})
    grid=np.logspace(0,-6,7)
    mf,mu=regrid_composition(grid,np.full(7,1000.),p,t,names,x,m,['1H2-16O__POKAZATEL'])
    np.testing.assert_allclose(mf['1H2-16O__POKAZATEL'],1.8/3.8)
    with pytest.raises(ValueError,match='temperature'):
        regrid_composition(grid,np.full(7,1100.),p,t,names,x,m,['H2O'])
    with pytest.raises(ValueError,match='coverage'):
        regrid_composition(np.logspace(-8,0,7),np.full(7,1000.),p,t,names,x,m,['H2O'])
    with pytest.raises(ValueError,match='absent'):
        regrid_composition(grid,np.full(7,1000.),p,t,names,x,m,['CO'])


@pytest.fixture
def installation(tmp_path):
    source=tmp_path/'VULCAN'
    for name in ['thermo','fastchem_vulcan']:(source/name).mkdir(parents=True)
    (source/'thermo/all_compose.txt').write_text('species mass\nH2 2\nHe 4\nH2O 18\n')
    (source/'thermo/network.txt').write_text('controlled test network')
    (source/'fastchem_vulcan/fastchem').write_text('test executable')
    (source/'vulcan_cfg.py').write_text('# Test configuration\n')
    (source/'vulcan.py').write_text('''import pickle, pathlib, numpy as np
import vulcan_cfg as c
p=np.logspace(np.log10(c.P_b),np.log10(c.P_t),c.nz)
a=np.loadtxt(c.atm_file,skiprows=2)
t=np.interp(p,a[::-1,0],a[::-1,1])
d={'parameter':{'end_case':1},'variable':{'species':['H2','He','H2O'],
'y':np.tile([8.,1.,1.],(c.nz,1)),'atom_loss':{'H':0.,'O':0.}},'atm':{'pco':p,'Tco':t,'pico':np.array([1e4]),'pref_indx':0}}
with open('output/column.vul','wb') as f:pickle.dump(d,f)
''')
    return PathConfig(vulcan_directory=str(source),vulcan_python=sys.executable,vulcan_cache_dir=str(tmp_path/'cache'))


def test_subprocess_cache_and_provenance(settings,installation,physical):
    p=np.logspace(-6,0,8);t=np.full(8,1000.)
    first=run_vulcan(settings,p,t,['H2O'],physical,installation)
    second=run_vulcan(settings,p,t,['He','H2O'],physical,installation)
    assert first.provenance_file==second.provenance_file
    np.testing.assert_array_equal(first.mass_fractions['H2O'],second.mass_fractions['H2O'])
    record=json.loads(first.provenance_file.read_text())
    assert record['end_case']==1 and record['runtime_seconds']>0
    assert (first.provenance_file.parent/'run.log').is_file()
    assert (Path(installation.vulcan_directory)/'vulcan_cfg.py').read_text()=='# Test configuration\n'
    assert not list(Path(installation.vulcan_cache_dir).glob('*.lock'))
    assert list(first.provenance_file.parent.glob('gas_profile.npz'))


def test_cache_changes_with_transport_and_input(settings,installation,physical):
    p=np.logspace(-6,0,6);t=np.full(6,1000.)
    a=run_vulcan(settings,p,t,['H2O'],physical,installation)
    settings.vertical_velocity_cm_s=1.
    b=run_vulcan(settings,p,t,['H2O'],physical,installation)
    assert a.provenance_file!=b.provenance_file
    (Path(installation.vulcan_directory)/'thermo/network.txt').write_text('new reaction coefficients')
    c=run_vulcan(settings,p,t,['H2O'],physical,installation)
    assert c.provenance_file!=b.provenance_file


def test_cache_corruption_is_rejected(settings,installation,physical):
    p=np.logspace(-6,0,6);t=np.full(6,1000.)
    a=run_vulcan(settings,p,t,['H2O'],physical,installation)
    (a.provenance_file.parent/'gas_profile.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='integrity'):
        run_vulcan(settings,p,t,['H2O'],physical,installation)


def test_bad_uv_or_kzz_coverage_stops_before_run(settings,installation,physical,tmp_path):
    p=np.logspace(-6,0,6);t=np.full(6,1000.)
    uv=tmp_path/'uv.txt';np.savetxt(uv,[[100,1],[200,1]])
    settings.photochemistry=True;settings.stellar_flux_file=str(uv)
    settings.orbit_radius_au=.03;settings.zenith_angle_deg=48.;settings.diurnal_factor=1.
    with pytest.raises(ValueError,match='2–700'):
        run_vulcan(settings,p,t,['H2O'],physical,installation)
    settings.photochemistry=False
    settings.eddy_diffusion=SimpleNamespace(mode='file',profile_file=str(uv))
    with pytest.raises(ValueError,match='Kzz profile must span'):
        run_vulcan(settings,p,t,['H2O'],physical,installation)


def test_failed_solver_keeps_log_and_releases_lock(settings,installation,physical):
    (Path(installation.vulcan_directory)/'vulcan.py').write_text('raise RuntimeError("test failed kinetics")')
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        run_vulcan(settings,np.logspace(-6,0,6),np.full(6,1000.),['H2O'],physical,installation)
    assert not list(Path(installation.vulcan_cache_dir).glob('*.lock'))
    assert 'test failed kinetics' in next(Path(installation.vulcan_cache_dir).glob('*/run.log')).read_text()


def test_prt_explicit_manual_matches_legacy(physical,monkeypatch):
    from exoplore.atmosphere import prt
    received=[]
    class RT:
        def calculate_transit_radii(self,**kwargs):
            received.append(kwargs)
            return np.linspace(1e-4,1.1e-4,10),np.full(10,7e9),{}
    kwargs=dict(model_inputs=physical,pressures=np.logspace(-6,0,8),prt_object=RT(),
        species=['H2','He','H2O'],vmr=[.74,.259,.001],MMW=2.33,p0=.01,
        isothermal=True,iso_T_value=1000.,two_point_T=False,p_points=None,t_points=None,
        kappa=.01,gamma=.4,T_equil=1000.,metallicity=0.,C_to_O=.55,use_easyCHEM=False)
    a=prt.call_pRT(**kwargs)
    b=prt.call_pRT(**kwargs,chemistry=ChemistryConfig('manual'))
    for index in [0,1,3,4,5]:np.testing.assert_array_equal(a[index],b[index])
    for name in kwargs['species']:np.testing.assert_array_equal(a[2][name],b[2][name])


def test_prt_receives_vulcan_profile(physical,settings,installation):
    from exoplore.atmosphere import prt
    received={}
    class RT:
        def calculate_transit_radii(self,**kwargs):
            received.update(kwargs)
            return np.linspace(1e-4,1.1e-4,10),np.full(10,7e9),{}
    prt.call_pRT(physical,np.logspace(-6,0,8),RT(),['H2','He','H2O'],[],None,.01,
        True,1000.,False,None,None,.01,.4,1000.,0.,.55,
        chemistry=ChemistryConfig('vulcan',settings),chemistry_paths=installation)
    np.testing.assert_allclose(received['mean_molar_masses'],3.8)
    np.testing.assert_allclose(received['mass_fractions']['H2O'],1.8/3.8)


def test_conflicting_explicit_json_flag_is_rejected():
    with pytest.raises(ValueError,match='conflicts'):
        SimulationConfig.from_dict({'atmosphere':{'planet_model':{'use_easychem':True,'chemistry':{'backend':'manual'}}}})


def test_explicit_easychem_matches_legacy_call(physical,monkeypatch):
    from exoplore.atmosphere import prt
    calls=[]
    def equilibrium(metal,co,p,t,species,symmetric=False):
        calls.append((metal,co,species,symmetric))
        return {name:np.full(len(p),i*.001+.01) for i,name in enumerate(species)},np.full(len(p),2.31)
    monkeypatch.setattr(prt,'call_easyCHEM',equilibrium)
    class RT:
        def calculate_transit_radii(self,**kwargs):
            return np.linspace(1e-4,1.1e-4,10),np.full(10,7e9),{}
    args=(physical,np.logspace(-6,0,8),RT(),['H2','He','1H2-16O__POKAZATEL'],[],None,.01,
          True,1000.,False,None,None,.01,.4,1000.,.3,.7)
    a=prt.call_pRT(*args,use_easyCHEM=True,easychem_CtoO_ret=True)
    b=prt.call_pRT(*args,chemistry=ChemistryConfig('easychem'),easychem_CtoO_ret=True)
    assert calls[0]==calls[1]
    for name in a[2]:np.testing.assert_array_equal(a[2][name],b[2][name])
    np.testing.assert_array_equal(a[3],b[3])


def test_two_limb_vulcan_uses_each_column(settings,physical,installation):
    from exoplore.config.models import AtmosphereConfig
    from exoplore.atmosphere.prt import call_pRT_limbs
    region=AtmosphereRegionConfig(species=['H2','He','H2O'],isothermal=True,isothermal_temperature_K=1000.,
        chemistry=ChemistryConfig('vulcan',settings))
    atmosphere=AtmosphereConfig(limb_asymmetries=True,limb_divisions='gradual',morning_day=region,evening_day=region)
    class RT:
        def calculate_transit_radii(self,**kwargs):
            return np.linspace(1e-4,1.1e-4,10),np.full(10,7e9),{}
    p=np.logspace(-6,0,8)
    output=call_pRT_limbs(physical,atmosphere,p,None,p,None,RT(),None,RT(),None,chemistry_paths=installation)
    np.testing.assert_allclose(output[5]['H2O'],1.8/3.8)
    np.testing.assert_allclose(output[9]['H2O'],1.8/3.8)
    assert len(list(Path(installation.vulcan_cache_dir).glob('*/manifest.json')))==1


def test_timeout_preserves_log_and_releases_lock(settings,installation,physical):
    import subprocess
    (Path(installation.vulcan_directory)/'vulcan.py').write_text('import time; time.sleep(5)')
    settings.timeout_seconds=.1
    with pytest.raises(subprocess.TimeoutExpired):
        run_vulcan(settings,np.logspace(-6,0,6),np.full(6,1000.),['H2O'],physical,installation)
    assert not list(Path(installation.vulcan_cache_dir).glob('*.lock'))


def test_concurrent_same_column_has_one_completed_workspace(settings,installation,physical):
    from concurrent.futures import ThreadPoolExecutor
    p=np.logspace(-6,0,6);t=np.full(6,1000.)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_vulcan,settings,p,t,['H2O'],physical,installation) for _ in range(2)]
        results=[job.result() for job in jobs]
    assert results[0].provenance_file==results[1].provenance_file
    assert len(list(Path(installation.vulcan_cache_dir).glob('*/manifest.json')))==1


def test_direct_atomic_cannot_silently_ignore_chemistry():
    from exoplore.core.simulator import ExoploreSimulator
    config=SimulationConfig.from_json('configs/hd189733b_andes_sodium_rm_clv.json')
    config.atmosphere.planet_model.chemistry=ChemistryConfig('manual')
    with pytest.raises(ValueError,match='Explicit chemistry is supported'):
        ExoploreSimulator(config)


def test_retrieval_chemistry_cannot_be_silently_ignored():
    from exoplore.core.simulator import ExoploreSimulator
    config=SimulationConfig.from_json('configs/hd189733b_andes_retrieval_blain24_noisy.json')
    config.retrieval.atmosphere=AtmosphereRegionConfig(chemistry=ChemistryConfig('manual'))
    with pytest.raises(ValueError,match='Explicit retrieval chemistry'):
        ExoploreSimulator(config)


def test_inconsistent_condensed_hydrostatic_mass_is_rejected():
    d=native(condensed=True)
    d['atm']['mu']=np.full(3,3.8)
    with pytest.raises(ValueError,match='hydrostatic mass'):
        gas_composition(d,{'H2':2.,'He':4.,'H2O':18.,'water_ice':18.},['water_ice'])


def test_virtual_environment_executable_path_is_preserved(settings,installation,physical,tmp_path):
    import subprocess
    env=tmp_path/'environment'
    subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(env)],check=True)
    installation.vulcan_python=str(env/'bin/python')
    script=Path(installation.vulcan_directory)/'vulcan.py'
    script.write_text(script.read_text()+"\npathlib.Path('environment_prefix.txt').write_text(__import__('sys').prefix)\n")
    result=run_vulcan(settings,np.logspace(-6,0,6),np.full(6,1000.),['H2O'],physical,installation)
    assert (result.provenance_file.parent/'environment_prefix.txt').read_text()==str(env)


def test_chemistry_preparation_cli_uses_master_json(settings,installation,tmp_path):
    import os,subprocess
    config=SimulationConfig()
    config.atmosphere.limb_asymmetries=False
    config.atmosphere.planet_model=AtmosphereRegionConfig(
        species=['H2','He','H2O'],isothermal=True,isothermal_temperature_K=1000.,
        pressure_grid_size=6,chemistry=ChemistryConfig('vulcan',settings))
    config.paths=installation
    filename=tmp_path/'simulation.json';config.to_json(filename)
    result=subprocess.run([sys.executable,'scripts/prepare_vulcan_chemistry.py',str(filename)],
        text=True,capture_output=True,env={**os.environ,'MPLBACKEND':'Agg'})
    assert result.returncode==0, result.stderr
    assert 'Converged VULCAN gas profile' in result.stdout
    assert len(list(Path(installation.vulcan_cache_dir).glob('*/manifest.json')))==1


def test_summary_names_explicit_chemistry(settings,installation):
    from exoplore.core.simulator import ExoploreSimulator
    config=SimulationConfig()
    config.atmosphere.limb_asymmetries=False
    config.atmosphere.planet_model=AtmosphereRegionConfig(chemistry=ChemistryConfig('vulcan',settings))
    config.paths=installation
    summary=ExoploreSimulator(config).summary()
    assert summary.chemistry_backend=='vulcan'
    assert 'Chemistry           : vulcan' in str(summary)
    assert 'Metallicity (log Z)' not in str(summary)


@pytest.mark.skipif(__import__('os').name != 'posix',reason='POSIX process groups')
def test_timeout_stops_native_child_processes(settings,installation,physical,tmp_path):
    import subprocess,time
    marker=tmp_path/'orphan_finished.txt'
    child="import time,pathlib;time.sleep(.8);pathlib.Path("+repr(str(marker))+").write_text('orphan')"
    script="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(5)"
    (Path(installation.vulcan_directory)/'vulcan.py').write_text(script)
    settings.timeout_seconds=.3
    with pytest.raises(subprocess.TimeoutExpired):
        run_vulcan(settings,np.logspace(-6,0,6),np.full(6,1000.),['H2O'],physical,installation)
    time.sleep(.9)
    assert not marker.exists()


@pytest.mark.parametrize('p0,p1',[(.01,1.),(1.,.01),(.01,.01)])
def test_radius_conversion_isothermal(p0,p1):
    from exoplore.atmosphere.vulcan import radius_at_pressure
    p=np.logspace(-6,2,32)
    radius,gravity,temp,mu=8e9,2200.,1300.,2.3
    expected=1/(1/radius+1.380649e-16*temp/(gravity*radius**2*mu*1.6605390666e-24)*np.log(p1/p0))
    actual=radius_at_pressure(p,np.full(32,temp),np.full(32,mu),radius,gravity,p0,p1)
    assert actual == pytest.approx(expected,abs=1.)


def test_radius_reference_coverage():
    from exoplore.atmosphere.vulcan import radius_at_pressure
    with pytest.raises(ValueError,match='inside'):
        radius_at_pressure([.01,.1,1.],[1000.]*3,[2.3]*3,8e9,2200.,.001,1.)


def test_coupled_radius_preserves_mass_and_provenance(monkeypatch,tmp_path,settings,physical):
    import exoplore.atmosphere.vulcan as adapter
    p=np.logspace(-6,2,32)
    t=np.full(32,1200.)
    mu=np.full(32,2.3)
    calls=[]
    def native_column(settings,p,t,species,inputs,paths):
        calls.append(inputs)
        work=tmp_path/str(len(calls));work.mkdir()
        np.savez(work/'gas_profile.npz',pressure_bar=p,temperature_K=t,
                 mole_fractions=np.ones((32,1)),molecular_masses=[2.3],
                 condensed_mass_per_gas_particle=np.zeros(32))
        manifest=work/'manifest.json'
        manifest.write_text(json.dumps({'native_radius_reference_pressure_bar':1.}))
        return adapter.ChemistryProfile({'H2':np.ones(32)},mu,manifest)
    monkeypatch.setattr(adapter,'_run_native_column',native_column)
    from dataclasses import replace
    inputs=replace(physical,gravity_cgs=2200.)
    result=adapter.run_vulcan(settings,p,t,['H2'],inputs,None,reference_pressure_bar=.01)
    assert len(calls)==2
    assert calls[1].planet_radius_cm < inputs.planet_radius_cm
    assert all(c.gravity_cgs is None for c in calls)
    expected_mass=2200.*inputs.planet_radius_cm**2/6.67430e-8
    assert all(c.planet_mass_g==expected_mass for c in calls)
    record=json.loads(next(result.provenance_file.parent.glob('radius_alignment_*.json')).read_text())
    assert record['reference_radius_cm']==inputs.planet_radius_cm
    assert len(record['iterations'])==2


def test_each_limb_passes_its_reference_pressure(monkeypatch,settings,physical):
    import exoplore.atmosphere.vulcan as adapter
    from exoplore.atmosphere.chemistry import selected_chemistry
    seen=[]
    def calculate(settings,p,t,species,inputs,paths,*,reference_pressure_bar):
        seen.append((reference_pressure_bar,np.array(t).copy()))
        return adapter.ChemistryProfile({'H2':np.ones(3)},np.full(3,2.3),Path('manifest.json'))
    monkeypatch.setattr(adapter,'run_vulcan',calculate)
    config=ChemistryConfig(backend='vulcan',vulcan=settings)
    for temp,p0 in [(1000.,.01),(1500.,.1)]:
        selected_chemistry(config,[1e-6,.01,1.],[temp]*3,['H2'],[],2.3,0.,.55,
                           physical,object(),reference_pressure_bar=p0)
    assert [x[0] for x in seen]==[.01,.1]
    assert np.all(seen[0][1]==1000.)
    assert np.all(seen[1][1]==1500.)


@pytest.mark.parametrize('mu',[[2.3,2.3],[2.3,-1.,2.3],[2.3,float('nan'),2.3]])
def test_radius_rejects_invalid_molecular_weight(mu):
    from exoplore.atmosphere.vulcan import radius_at_pressure
    with pytest.raises(ValueError,match='molecular weights'):
        radius_at_pressure([1e-6,.01,1.],[1000.]*3,mu,8e9,2200.,.01,1.)


def test_radius_requires_explicit_region_reference(settings,physical):
    from exoplore.atmosphere.vulcan import run_vulcan
    with pytest.raises(ValueError,match='reference_pressure_bar'):
        run_vulcan(settings,[1e-6,.01,1.],[1000.]*3,['H2'],physical,None)


def test_master_example_keeps_existing_chemistry_choices():
    data=json.loads(Path('configs/hd189733b_andes_transit_clean.json').read_text())
    config=SimulationConfig.from_dict(data)
    for name in ('planet_model','ccf_template','morning_day','morning_night','evening_day','evening_night'):
        region=getattr(config.atmosphere,name)
        assert region.chemistry is None
        assert region.chemistry_backend==('easychem' if data['atmosphere'][name]['use_easychem'] else 'manual')


def test_documented_limb_example_loads():
    text=Path('docs/chemistry.md').read_text().split('## Different chemistry in the two limbs')[1]
    block=text.split('```json\n',1)[1].split('```',1)[0]
    config=SimulationConfig.from_dict({'atmosphere':json.loads('{'+block+'}')})
    for name,temp in [('morning_day',1200.),('evening_day',1500.)]:
        region=getattr(config.atmosphere,name)
        assert region.chemistry_backend=='vulcan'
        assert region.isothermal_temperature_K==temp
        assert region.chemistry.vulcan.eddy_diffusion.value_cm2_s==1e9


@pytest.mark.parametrize('loading',[0.,.038,3.8])
def test_prt_composition_uses_gas_mass_with_condensates_present(loading):
    p,t,names,x,m=gas_composition(native(),{'H2':2.,'He':4.,'H2O':18.})
    plain,gas_mu=regrid_composition(p,t,p,t,names,x,m,names)
    loaded,mu=regrid_composition(p,t,p,t,names,x,m,names,
                                condensed_mass_per_gas_particle=np.full(3,loading))
    np.testing.assert_array_equal(mu,gas_mu)
    for i,name in enumerate(names):
        # At fixed P,T, n_i/n_gas equals X_i * mu / molecular_mass_i.
        np.testing.assert_allclose(loaded[name]*mu/m[i],x[:,i])
        np.testing.assert_array_equal(loaded[name],plain[name])
    np.testing.assert_allclose(sum(loaded.values()),1.)


def test_condensate_loading_interpolation():
    p=np.array([1e-6,1e-3,1.]);t=np.full(3,1000.)
    x=np.tile([.9,.1],(3,1));m=np.array([2.,4.])
    target=np.array([1e-6,1e-4,1.])
    _,mu=regrid_composition(target,t,p,t,['H2','He'],x,m,['H2'],
                            condensed_mass_per_gas_particle=np.array([0.,.3,.6]))
    np.testing.assert_allclose(mu,[2.2,2.2,2.2])


@pytest.mark.parametrize('loading',[[0.,0.],[0.,-1.,0.],[0.,float('nan'),0.]])
def test_invalid_condensate_loading(loading):
    p,t,names,x,m=gas_composition(native(),{'H2':2.,'He':4.,'H2O':18.})
    with pytest.raises(ValueError,match='Condensate loading'):
        regrid_composition(p,t,p,t,names,x,m,['H2'],condensed_mass_per_gas_particle=loading)


def test_native_loading_is_separate_from_prt_gas_mmw():
    from exoplore.atmosphere.vulcan import condensate_mass_per_gas_particle
    data=native(condensed=True)
    masses={'H2':2.,'He':4.,'H2O':18.,'water_ice':18.}
    data['atm']['mu']=np.full(3,39.8)
    p,t,names,x,m=gas_composition(data,masses,['water_ice'])
    loading=condensate_mass_per_gas_particle(data,masses,['water_ice'])
    mf,mu=regrid_composition(p,t,p,t,names,x,m,names,
                            condensed_mass_per_gas_particle=loading)
    np.testing.assert_allclose(mu,3.8)
    np.testing.assert_allclose(loading,36.)
    np.testing.assert_allclose(mu+loading,data['atm']['mu'])
    np.testing.assert_allclose(sum(mf.values()),1.)


@pytest.mark.parametrize('kwargs',[
    {'initialization':'mystery'},
    {'initialization':'constant_mixing_ratios'},
    {'initial_mixing_ratios':{'H2':1.}},
    {'initialization':'constant_mixing_ratios','initial_mixing_ratios':{'H2':.8}},
    {'initialization':'constant_mixing_ratios','initial_mixing_ratios':{'H2':float('nan')}},
    {'initial_element_tolerance':0},
])
def test_invalid_initialization_settings(settings,kwargs):
    payload=asdict(settings);payload.update(kwargs)
    with pytest.raises(ValueError):VulcanConfig(**payload)


def initial_state_fixture():
    composition=np.array([('H2',2,0,0,2.),('He',0,1,0,4.),('H2O',2,0,1,18.)],
                          dtype=[('species','U10'),('H',int),('He',int),('O',int),('mass',float)])
    var=SimpleNamespace(y=np.tile([8.,1.,1.],(3,1)))
    atm=SimpleNamespace(pco=np.array([1e6,1e3,1.]),Tco=np.full(3,1000.))
    return var,atm,['H2','He','H2O'],composition,{'H':1.,'He':1/18,'O':1/18}


def test_initial_state_element_checks():
    from exoplore.atmosphere.vulcan_initialisation import check_initial_state
    args=initial_state_fixture()
    result=check_initial_state(*args)
    assert result['status']=='valid'
    np.testing.assert_allclose(result['gas_mmw_range'],[3.8,3.8])
    args[0].y[1,2]=20.
    with pytest.raises(ValueError,match='differs from configured'):
        check_initial_state(*args)


def test_initial_state_rejects_nan():
    from exoplore.atmosphere.vulcan_initialisation import check_initial_state
    args=initial_state_fixture();args[0].y[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        check_initial_state(*args)


@pytest.mark.parametrize('valid',[True,False])
def test_native_launcher_checks_before_solver(tmp_path,valid):
    from exoplore.atmosphere import vulcan
    import shutil,subprocess
    helper=Path(vulcan.__file__).with_name('vulcan_initialisation.py')
    shutil.copy2(helper,tmp_path/'exoplore_initialisation.py')
    (tmp_path/'exoplore_initialisation_inputs.json').write_text(json.dumps({
        'elemental_abundances':{'H':1.,'He':1/18,'O':1/18},
        'non_gas_species':[],'tolerance':1e-3}))
    water=1. if valid else 20.
    (tmp_path/'build_atm.py').write_text(f'''import numpy as np
species=['H2','He','H2O']
compo=np.array([('H2',2,0,0,2.),('He',0,1,0,4.),('H2O',2,0,1,18.)],dtype=[('species','U10'),('H',int),('He',int),('O',int),('mass',float)])
class InitialAbun:
    def ini_y(self,var,atm):
        var.y=np.tile([8.,1.,{water}],(3,1))
        return var
''')
    (tmp_path/'vulcan.py').write_text('''from types import SimpleNamespace
from pathlib import Path
import numpy as np
import build_atm
v=SimpleNamespace()
a=SimpleNamespace(pco=np.array([1e6,1e3,1.]),Tco=np.full(3,1000.))
build_atm.InitialAbun().ini_y(v,a)
Path('solver_started').write_text('yes')
''')
    if valid:vulcan._execute_vulcan(Path(sys.executable),tmp_path,30.)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            vulcan._execute_vulcan(Path(sys.executable),tmp_path,30.)
    assert (tmp_path/'solver_started').exists()==valid
    record=json.loads((tmp_path/'initialisation_checks.json').read_text())
    assert record['status']==('valid' if valid else 'rejected')


@pytest.mark.parametrize('corrected', [False, True])
def test_ammonia_work_copy_corrections(tmp_path, corrected):
    from exoplore.atmosphere.vulcan import _repair_ammonia_condensation
    from exoplore.config.chemistry import CondensationConfig
    table = tmp_path/'thermo/all_compose.txt'
    table.parent.mkdir()
    table.write_text('species H N mass\nNH3 3 1 17.031\nNH3_l_s 3 1 '+('17.031' if corrected else '16.023')+'\n')
    operator = tmp_path/'op.py'
    operator.write_text("var.Rf[re] == 'NH3 -> "+('NH3_l_s' if corrected else 'NH3_l')+"'\n")
    settings = SimpleNamespace(condensation=CondensationConfig(enabled=True,
        gas_species=['NH3'], condensed_species=['NH3_l_s'],
        particle_radius_cm={'NH3_l_s':5e-5}, particle_density_g_cm3={'NH3_l_s':.8}))
    records = _repair_ammonia_condensation(tmp_path, settings)
    assert len(records) == (0 if corrected else 2)
    assert '17.031' in table.read_text().splitlines()[-1]
    assert "NH3 -> NH3_l_s" in operator.read_text()
    assert _repair_ammonia_condensation(tmp_path, settings) == []
    for record in records:
        assert record['before_sha256'] != record['after_sha256']


def test_ammonia_unknown_mass_fails(tmp_path):
    from exoplore.atmosphere.vulcan import _repair_ammonia_condensation
    table = tmp_path/'thermo/all_compose.txt'
    table.parent.mkdir()
    table.write_text('species H N mass\nNH3 3 1 17.031\nNH3_l_s 3 1 99.0\n')
    settings = SimpleNamespace(condensation=SimpleNamespace(enabled=True, condensed_species=['NH3_l_s']))
    with pytest.raises(ValueError, match='masses disagree'):
        _repair_ammonia_condensation(tmp_path, settings)


def test_direct_condensation_requires_native_molecular_diffusion(settings):
    payload=asdict(settings)
    payload['condensation']={'enabled':True, 'gas_species':['H2O'], 'condensed_species':['H2O_l_s'],
        'particle_radius_cm':{'H2O_l_s':5e-5}, 'particle_density_g_cm3':{'H2O_l_s':.9}}
    with pytest.raises(ValueError, match='diffusion-limited growth'):
        VulcanConfig(**payload)
    payload['molecular_diffusion']=True
    assert VulcanConfig(**payload).condensation.enabled


def test_network_generation_failure_stops_native_launch(tmp_path):
    import shutil, subprocess
    from exoplore.atmosphere import vulcan
    shutil.copy2(Path(vulcan.__file__).with_name('vulcan_initialisation.py'), tmp_path/'exoplore_initialisation.py')
    (tmp_path/'exoplore_initialisation_inputs.json').write_text('{}')
    (tmp_path/'make_chem_funs.py').write_text('raise RuntimeError("network generation failed")')
    (tmp_path/'vulcan.py').write_text("from pathlib import Path\nPath('solver_started').touch()")
    with pytest.raises(subprocess.CalledProcessError):
        vulcan._execute_vulcan(Path(sys.executable), tmp_path, 30.)
    assert not (tmp_path/'solver_started').exists()
    assert 'network generation failed' in (tmp_path/'run.log').read_text()


@pytest.mark.parametrize('options', [{'use_adapt_rtol':1}, {'update_frq':1.5}, {'update_frq':0}])
def test_invalid_native_numerical_controls(settings,options):
    payload=asdict(settings);payload['solver_options']=options
    with pytest.raises(ValueError):VulcanConfig(**payload)


def test_fixed_tolerance_controls_roundtrip(settings):
    payload=asdict(settings)
    payload['solver_options']={'use_adapt_rtol':False,'rtol':.001,'update_frq':1,'dt_max':100.}
    parsed=VulcanConfig(**payload)
    assert parsed.solver_options==payload['solver_options']
    assert VulcanConfig(**asdict(parsed)).solver_options==payload['solver_options']


def test_four_limb_columns_keep_their_own_temperatures(settings,physical,installation):
    from exoplore.config.models import AtmosphereConfig
    from exoplore.atmosphere.prt import call_pRT_limbs
    names=['morning_day','morning_night','evening_day','evening_night']
    temperatures=[1000.,1100.,1200.,1300.]
    regions={name:AtmosphereRegionConfig(species=['H2','He','H2O'],isothermal=True,
        isothermal_temperature_K=temp,chemistry=ChemistryConfig('vulcan',settings))
        for name,temp in zip(names,temperatures)}
    atmosphere=AtmosphereConfig(limb_asymmetries=True,limb_divisions='quarters',**regions)
    class RT:
        def calculate_transit_radii(self,**kwargs):
            return np.linspace(1e-4,1.1e-4,10),np.full(10,7e9),{}
    p=np.logspace(-6,0,8)
    output=call_pRT_limbs(physical,atmosphere,p,p,p,p,RT(),RT(),RT(),RT(),chemistry_paths=installation)
    for index,temp in enumerate(temperatures):np.testing.assert_array_equal(output[14+index],np.full(8,temp))
    for index in [5,6,9,10]:np.testing.assert_allclose(output[index]['H2O'],1.8/3.8)
    assert len(list(Path(installation.vulcan_cache_dir).glob('*/manifest.json')))==4


@pytest.mark.parametrize('division',['quarters','gradual'])
@pytest.mark.parametrize('backend',['manual','easychem'])
def test_explicit_limb_chemistry_alias_preserves_existing_calculation(division,backend,physical,monkeypatch):
    from exoplore.config.models import AtmosphereConfig
    from exoplore.atmosphere import prt
    calls=[]
    def equilibrium(metal,co,p,t,species,symmetric=False):
        calls.append(symmetric)
        return {name:np.full(len(p),.001*(i+1)) for i,name in enumerate(species)},np.full(len(p),2.2 if symmetric else 2.4)
    monkeypatch.setattr(prt,'call_easyCHEM',equilibrium)
    def atmosphere(explicit):
        regions={name:AtmosphereRegionConfig(species=['H2','He','H2O'],mass_fractions=[.7,.299,.001],
            mean_molecular_weight=3.,isothermal=True,isothermal_temperature_K=1000.+100*i,
            use_easychem=backend=='easychem',chemistry=ChemistryConfig(backend) if explicit else None)
            for i,name in enumerate(['morning_day','morning_night','evening_day','evening_night'])}
        return AtmosphereConfig(limb_asymmetries=True,limb_divisions=division,
            planet_model=AtmosphereRegionConfig(mean_molecular_weight=2.33),**regions)
    class RT:
        def calculate_transit_radii(self,**kwargs):
            return np.linspace(1e-4,1.1e-4,10),np.full(10,7e9),{}
    p=np.logspace(-6,0,8)
    args=(physical,p,p,p,p,RT(),RT(),RT(),RT())
    before=prt.call_pRT_limbs(physical,atmosphere(False),*args[1:],easychem_CtoO_ret=True)
    first_calls=calls.copy();calls.clear()
    after=prt.call_pRT_limbs(physical,atmosphere(True),*args[1:],easychem_CtoO_ret=True)
    assert calls==first_calls
    if backend=='easychem':assert calls==([False]*4 if division=='quarters' else [True]*2)
    for a,b in zip(before,after):
        if isinstance(a,dict):
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        elif a is None:assert b is None
        else:np.testing.assert_array_equal(a,b)
    if backend=='manual':np.testing.assert_array_equal(after[7],np.full(8,2.33))


def test_regional_chemistry_summary_labels_active_columns(settings):
    from exoplore.config.models import AtmosphereConfig
    assert AtmosphereConfig().explicit_chemistry_label is None
    region=AtmosphereRegionConfig(chemistry=ChemistryConfig('vulcan',settings))
    assert AtmosphereConfig(planet_model=region).explicit_chemistry_label=='vulcan'
    atmosphere=AtmosphereConfig(limb_asymmetries=True,limb_divisions='gradual',
        morning_day=region,evening_day=AtmosphereRegionConfig(use_easychem=False))
    assert atmosphere.explicit_chemistry_label=='morning_day: vulcan, evening_day: manual'
    atmosphere.limb_divisions='quarters'
    atmosphere.morning_night=AtmosphereRegionConfig()
    atmosphere.evening_night=region
    assert atmosphere.explicit_chemistry_label=='morning_day: vulcan, morning_night: easychem, evening_day: manual, evening_night: vulcan'
