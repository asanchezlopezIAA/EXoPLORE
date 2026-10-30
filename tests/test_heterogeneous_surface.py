"""Controlled surface spectra test geometry, flux scales and local ray transport."""
from dataclasses import replace
import json
import numpy as np
import pytest
from exoplore.config import (SimulationConfig, RMCLVConfig, StellarSurfaceConfig,
                             StellarComponentConfig, SurfaceRegionConfig)
from exoplore.planets import load_planet
from exoplore.stellar.surface import LocalIntensityGrid, StellarSurface
from exoplore.stellar.observation import TransitStellarModel
from exoplore.stellar.coupled import SpatialTransitModel
from exoplore.stellar.components import StellarIntensityField
from exoplore.stellar.regions import component_map, cap_centre
from exoplore.io.stellar_surface import load_local_intensities
from exoplore.atmosphere.transmission import ChordTransmission


@pytest.fixture
def setup(tmp_path):
    # All components share absolute brightness units. Their line depths and CLV differ.
    w = np.linspace(.5885, .5915, 601)
    mu = np.array([0., .3, .7, 1.])
    for name, scale, depth, limb in [('quiet', 1., .5, .5), ('spot', .4, .8, .5),
                                      ('facula', 1.2, .2, .1)]:
        continuum = scale*(limb+(1-limb)*mu[:, None])*np.ones_like(w)
        intensity = continuum*(1-depth*np.exp(-.5*((w-.59)/.00004)**2))
        np.savez(tmp_path/f'{name}.npz', wavelength_um=w, mu=mu, intensity=intensity,
                 continuum=continuum, resolving_power=0., intensity_kind='specific_intensity',
                 intensity_unit='shared test units', wavelength_medium='vacuum', source='analytic',
                 metadata_json=json.dumps({'dimensionality': 'test', 'citation': 'analytic fixture'}))
    components = {n: StellarComponentConfig(str(tmp_path/f'{n}.npz')) for n in ['quiet','spot','facula']}
    surface = StellarSurfaceConfig(components=components,
        regions=[SurfaceRegionConfig('spot', x=0., y=0., radius=.25)])
    choices = RMCLVConfig(simulate=True, model_file=components['quiet'].intensity_model_file,
        projected_obliquity_deg=0., cells_across_diameter=40, exposure_subsamples=1, azimuth_points=24,
        stellar_radial_points=16, planet_radial_points=12)
    return surface, choices, load_planet('planet_params/HD189733b.json')


def evaluate(cls, choices, planet, surface, phases=(0.,), **kwargs):
    model = cls(choices, planet, 100000, surface_config=surface)
    wave = np.linspace(.5897, .5903, 41)
    args = dict(chords=None, planet_velocity_kms=0.) if cls is SpatialTransitModel else {}
    args.update(kwargs)
    return model.evaluate(wave, np.array(phases), 0., .001, **args)


@pytest.mark.parametrize('cls', [TransitStellarModel, SpatialTransitModel])
def test_empty_regions_preserve_all_arrays(setup, cls):
    surface, c, p = setup
    original = evaluate(cls,c,p,None)
    empty = evaluate(cls,c,p,replace(surface,regions=[]))
    for name in ['reference','distortion']+(['flux','opaque_transmission'] if cls is SpatialTransitModel else []):
        np.testing.assert_array_equal(getattr(original,name),getattr(empty,name))


@pytest.mark.parametrize('cls', [TransitStellarModel, SpatialTransitModel])
def test_entire_disk_quiet_and_alternative(setup, cls):
    surface,c,p = setup
    for component in ['quiet','spot']:
        full = replace(surface,regions=[SurfaceRegionConfig(component,x=0.,y=0.,radius=2.)])
        actual = evaluate(cls,c,p,full)
        expected = evaluate(cls,replace(c,model_file=surface.components[component].intensity_model_file),p,None)
        np.testing.assert_allclose(actual.reference,expected.reference,atol=5e-6)
        np.testing.assert_allclose(actual.distortion,expected.distortion,atol=5e-6)


def test_config_round_trip(setup):
    s,_,_ = setup
    cfg=SimulationConfig(stellar_surface=s)
    saved=json.loads(json.dumps(cfg.to_dict()))
    assert SimulationConfig.from_dict(saved).stellar_surface == s
    assert SimulationConfig().stellar_surface is None
    assert SimulationConfig.from_dict({}).stellar_surface is None


@pytest.mark.parametrize('region', [dict(component='missing',x=0,y=0,radius=.2),
    dict(component='spot',geometry='bad'),dict(component='spot',x=0,y=0,radius=-1),
    dict(component='spot',geometry='stellar',latitude_deg=91,longitude_deg=0,angular_radius_deg=10),
    dict(component='spot',x=0,y=0,radius=.2,typo=1)])
def test_invalid_regions(setup,region):
    s,_,_=setup
    with pytest.raises(ValueError):
        StellarSurfaceConfig.from_dict(dict(components=s.components,regions=[region]))


def test_overlap_precedence(setup):
    s,_,_=setup
    s.regions += [SurfaceRegionConfig('facula',x=0,y=0,radius=.5,priority=1),
                  SurfaceRegionConfig('quiet',x=0,y=0,radius=.1,priority=1)]
    assert component_map(np.array([0.,.2,.8]),np.zeros(3),s).tolist()==['quiet','facula','quiet']


def test_surface_absolute_brightness_and_area(setup):
    s,c,p=setup
    field=StellarIntensityField(s,c,p)
    surface=StellarSurface(field.default.grid,vsini_kms=0,projected_obliquity_deg=0,
                           cells_across_diameter=160)
    w=np.array([.5908,.5909])
    quiet=surface.integrate(w,[2.],[0.],.1)
    surface.field=field
    spot=surface.integrate(w,[2.],[0.],.1)
    # A dark central region removes exactly the sum of its continuum contrast.
    labels=field.labels(surface.x,surface.y)
    contrast=.6*(.5+.5*surface.mu[labels=='spot']).sum()*surface.area
    np.testing.assert_allclose(quiet.outside-spot.outside[0],contrast,atol=1e-12)
    assert np.all(spot.continuum_outside < quiet.continuum_outside)


@pytest.mark.parametrize('cls',[TransitStellarModel,SpatialTransitModel])
def test_occulted_and_unocculted_regions(setup,cls):
    s,c,p=setup
    # The actual HD189733 transit chord crosses y about -0.66 at mid-transit.
    y=-p.a/(p.R_star/1e5)*np.cos(p.inc)
    crossing=replace(s,regions=[SurfaceRegionConfig('spot',x=0.,y=y,radius=.23)])
    off=replace(s,regions=[SurfaceRegionConfig('spot',x=0.,y=-y,radius=.23)])
    a,b=evaluate(cls,c,p,crossing),evaluate(cls,c,p,off)
    np.testing.assert_allclose(a.reference,b.reference,atol=2e-6)
    assert np.max(abs(a.distortion-b.distortion)) > 1e-4
    if cls is SpatialTransitModel:
        assert a.opaque_transmission.mean() > b.opaque_transmission.mean()


def test_facula_preserves_own_clv(setup):
    s,c,p=setup
    f=StellarIntensityField(s,c,p)
    _,continuum=f.sample(np.array([.5908]),np.array([0.,1.]),np.zeros(2),np.array(['facula','facula']))
    np.testing.assert_allclose(continuum[:,0],[.12,1.2])


@pytest.mark.parametrize('speed',[0.,4.,-4.])
def test_component_velocity_shift(setup,speed):
    s,c,p=setup
    s.components['spot']=replace(s.components['spot'],velocity_offset_kms=speed)
    f=StellarIntensityField(s,c,p)
    wave=np.linspace(.58995,.59005,2001)
    intensity,_=f.sample(wave,np.array([1.]),np.array([3.]),np.array(['spot']))
    measured=wave[np.argmin(intensity[0])]
    assert abs(measured-.59*(1+(speed+3.)/299792.458)) < 3e-6


def test_tabulated_velocity(setup,tmp_path):
    s,c,p=setup
    path=tmp_path/'velocity.npz'
    np.savez(path,mu=[0.,.5,1.],velocity_kms=[-2.,-1.,0.])
    s.components['spot']=replace(s.components['spot'],velocity_model_file=str(path))
    f=StellarIntensityField(s,c,p)
    np.testing.assert_allclose(f.components['spot'].velocity(np.array([.25,.75])),[-1.5,-.5])
    assert len(f.provenance()['components']['spot']['velocity_sha256'])==64


def physical(s,p,lon=0.,alpha=15.):
    period=2*np.pi*p.R_star/(p.v_rotsini_kms*1e5)/86400
    return replace(s,stellar_inclination_deg=90.,rotation_period_days=period,reference_bjd=2450000.,
        regions=[SurfaceRegionConfig('spot',geometry='stellar',latitude_deg=0.,longitude_deg=lon,angular_radius_deg=alpha)])


def test_physical_projection_rotation_and_far_side(setup):
    s,c,p=setup
    s=physical(s,p)
    r=s.regions[0]
    np.testing.assert_allclose(cap_centre(r,s,0.,s.reference_bjd),[0,0,1],atol=1e-15)
    np.testing.assert_allclose(cap_centre(r,s,0.,s.reference_bjd+s.rotation_period_days/4),[1,0,0],atol=1e-9)
    far=replace(s,regions=[replace(r,longitude_deg=180.)])
    assert component_map(np.array([0.,.9]),np.zeros(2),far,0.,s.reference_bjd).tolist()==['quiet','quiet']
    limb=replace(s,regions=[replace(r,longitude_deg=90.)])
    assert component_map(np.array([.99,0.]),np.zeros(2),limb,0.,s.reference_bjd).tolist()==['spot','quiet']


@pytest.mark.parametrize('cls',[TransitStellarModel,SpatialTransitModel])
def test_rotating_spectra_and_explicit_times(setup,cls):
    s,c,p=setup
    s=physical(s,p)
    with pytest.raises(ValueError,match='times_bjd'):
        evaluate(cls,c,p,s)
    a=evaluate(cls,c,p,s,times_bjd=[s.reference_bjd])
    b=evaluate(cls,c,p,s,times_bjd=[s.reference_bjd+s.rotation_period_days/2])
    assert np.max(abs(a.reference-b.reference)) > .005
    assert np.isfinite(a.distortion).all()


def test_rotation_consistency(setup):
    s,c,p=setup
    s=replace(physical(s,p),rotation_period_days=1.)
    with pytest.raises(ValueError,match='inconsistent'):
        StellarIntensityField(s,c,p)


def test_different_native_grids_and_coverage(setup,tmp_path):
    s,c,p=setup
    path=tmp_path/'spot.npz'
    with np.load(path) as d: data={k:d[k] for k in d.files}
    data.update(wavelength_um=data['wavelength_um'][::2],mu=data['mu'][[0,3]],
                intensity=data['intensity'][[0,3],::2],continuum=data['continuum'][[0,3],::2])
    np.savez(path,**data)
    f=StellarIntensityField(s,c,p)
    assert f.sample(np.array([.59]),np.array([.2]),np.array([0.]),np.array(['spot']))[0].shape==(1,1)
    with pytest.raises(ValueError,match='wavelength coverage'):
        f.sample(np.array([.58]),np.array([.2]),np.array([0.]),np.array(['spot']))
    with pytest.raises(ValueError,match='mu coverage'):
        f.sample(np.array([.59]),np.array([-.1]),np.array([0.]),np.array(['spot']))


@pytest.mark.parametrize('bad', ['wave','mu','shape','nan','continuum'])
def test_invalid_intensity_grid(setup,tmp_path,bad):
    s,_,_=setup
    with np.load(s.components['spot'].intensity_model_file) as d: data={k:d[k] for k in d.files}
    if bad=='wave': data['wavelength_um']=data['wavelength_um'][::-1]
    if bad=='mu': data['mu']=np.array([0.,.3,.7,1.1])
    if bad=='shape': data['intensity']=data['intensity'].T
    if bad=='nan': data['intensity'][0,0]=np.nan
    if bad=='continuum': data['continuum']=data['continuum'][:1]
    path=tmp_path/'bad.npz'
    np.savez(path,**data)
    with pytest.raises(ValueError): load_local_intensities(path,wavelength_medium='vacuum')


@pytest.mark.parametrize('cls',[TransitStellarModel,SpatialTransitModel])
def test_quiet_correction_does_not_replace_injection(setup,cls):
    s,c,p=setup
    matched=evaluate(cls,c,p,s)
    incomplete=evaluate(cls,c,p,replace(s,correction_surface='quiet'))
    np.testing.assert_array_equal(matched.reference,incomplete.reference)
    np.testing.assert_array_equal(matched.distortion,incomplete.distortion)
    assert incomplete.correction_flux is not None
    assert np.max(abs(incomplete.correction_distortion-incomplete.distortion)) > 1e-5


def test_spatial_atmosphere_uses_local_component(setup):
    s,c,p=setup
    w=np.linspace(.589,.591,201)
    radii=np.linspace(p.R_pl*1.08,p.R_pl,12)
    transmission=np.ones((len(w),len(radii)))*.5
    chords=ChordTransmission(w,radii,transmission)
    y=-p.a/(p.R_star/1e5)*np.cos(p.inc)
    results=[]
    for component in ['quiet','spot','facula']:
        case=replace(s,regions=[SurfaceRegionConfig(component,x=0.,y=y,radius=.3)])
        on=evaluate(SpatialTransitModel,c,p,case,chords=chords)
        off=evaluate(SpatialTransitModel,c,p,case)
        results.append(np.mean(off.flux-on.flux))
    assert results[1] < results[0] < results[2]


def test_factorized_stellar_assembly_preserves_opaque_transit(setup):
    from exoplore.stellar.observation import apply_factorized_surface
    s,c,p=setup
    result=evaluate(TransitStellarModel,c,p,s,phases=(-.02,0.,.02))
    fraction=np.array([0.,1.,0.])
    baseline=np.broadcast_to((1-(p.R_pl/p.R_star)**2*fraction)[:,None],result.reference.shape).copy()
    apply_factorized_surface(baseline,result,radius_ratio=p.R_pl/p.R_star,
        transit_fraction=fraction,planet_present=True,include_reference=True)
    np.testing.assert_allclose(baseline,result.reference*result.distortion*result.opaque_transmission)
    absent=np.ones_like(baseline)
    apply_factorized_surface(absent,result,radius_ratio=p.R_pl/p.R_star,
        transit_fraction=fraction,planet_present=False,include_reference=True)
    np.testing.assert_allclose(absent,baseline)


def test_uniform_disk_analytic_integral(setup):
    s,c,p=setup
    f=StellarIntensityField(s,c,p)
    w=f.default.grid.wavelength_um
    grid=LocalIntensityGrid(w,np.array([0.,1.]),np.ones((2,len(w))),np.ones((2,len(w))))
    from exoplore.stellar.components import StellarComponent
    f.components={name:StellarComponent(name,grid,0.,{}) for name in f.components}
    f.default=f.components['quiet']
    model=SpatialTransitModel(c,p,100000,surface_config=s)
    model.field=f
    from exoplore.stellar.quadrature import radial_rule
    radii,weights=radial_rule(1.,16,f)
    flux,_=model._angular_flux(np.array([.59]),model._rays(radii,0,0),len(radii))
    np.testing.assert_allclose(weights@flux,np.pi,atol=2e-14)


@pytest.mark.parametrize('stellar_geometry',[False,True])
def test_central_region_area_quadrature(setup,stellar_geometry):
    s,c,p=setup
    from exoplore.stellar.components import StellarComponent
    from exoplore.stellar.quadrature import radial_rule
    if stellar_geometry:
        s=physical(s,p,alpha=20.)
        radius=np.sin(np.deg2rad(20.))
        time=s.reference_bjd
    else:
        radius=.25
        time=None
    model=SpatialTransitModel(c,p,100000,surface_config=s)
    f=model.field
    wave=f.default.grid.wavelength_um
    for name,scale in [('quiet',1.),('spot',.4),('facula',1.2)]:
        grid=LocalIntensityGrid(wave,np.array([0.,1.]),np.full((2,len(wave)),scale),np.full((2,len(wave)),scale))
        f.components[name]=StellarComponent(name,grid,0.,{})
    radii,weights=radial_rule(1.,16,f,time=time)
    flux,_=model._angular_flux(np.array([.59]),model._rays(radii,0,0,time),len(radii))
    np.testing.assert_allclose(weights@flux,np.pi*(1-.6*radius**2),atol=2e-12)


def test_projected_disk_diagnostic(setup):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from exoplore.plotting.stellar_surface import plot_stellar_surface
    s,c,p=setup
    model=TransitStellarModel(c,p,100000,surface_config=s)
    fig=plot_stellar_surface(model,planet_position=(0.,-.65))
    assert len(fig.axes)==1
    assert len(fig.axes[0].images)==1
    plt.close(fig)


def test_component_kernel_matches_explicit_ray_sum(setup):
    s,c,p=setup
    model=SpatialTransitModel(c,p,100000,surface_config=s)
    radii=np.linspace(.01,.99,17)
    rays=model._rays(radii,0.,0.)
    wave=np.linspace(.5899,.5901,13)
    intensity,continuum=model.field.sample(wave,rays.mu,rays.velocity,rays.labels)
    expected=np.zeros((len(radii),len(wave)))
    expected_c=np.zeros_like(expected)
    np.add.at(expected,rays.rings,intensity*rays.weights[:,None])
    np.add.at(expected_c,rays.rings,continuum*rays.weights[:,None])
    actual,actual_c=model._angular_flux(wave,rays,len(radii))
    np.testing.assert_allclose(actual,expected,atol=3e-14)
    np.testing.assert_allclose(actual_c,expected_c,atol=3e-14)


def test_offcentre_physical_cap_projected_area(setup):
    s,c,p=setup
    from exoplore.stellar.components import StellarComponent
    from exoplore.stellar.quadrature import radial_rule
    s=physical(s,p,lon=30.,alpha=15.)
    model=SpatialTransitModel(c,p,100000,surface_config=s)
    f=model.field
    w=f.default.grid.wavelength_um
    for name,scale in [('quiet',1.),('spot',.4),('facula',1.2)]:
        grid=LocalIntensityGrid(w,np.array([0.,1.]),np.full((2,len(w)),scale),np.full((2,len(w)),scale))
        f.components[name]=StellarComponent(name,grid,0.,{})
    radii,weights=radial_rule(1.,64,f,time=s.reference_bjd)
    flux,_=model._angular_flux(np.array([.59]),model._rays(radii,0,0,s.reference_bjd),len(radii))
    area=np.pi*np.sin(np.deg2rad(15.))**2*np.cos(np.deg2rad(30.))
    np.testing.assert_allclose(weights@flux,np.pi-.6*area,atol=5e-5)


def test_heterogeneous_spectra_pass_through_seidel22(setup):
    from exoplore.config.seidel22 import Seidel22Config
    from exoplore.pipelines.seidel22 import prepare_transmission
    s,c,p=setup
    phase=np.array([-.03,-.025,-.01,0.,.01,.025,.03])
    wave=np.linspace(.5897,.5903,301)
    model=SpatialTransitModel(c,p,100000,surface_config=s)
    result=model.evaluate(wave,phase,0.,30.,chords=None,planet_velocity_kms=0.)
    bare=result.reference*result.distortion*result.opaque_transmission
    prepared=prepare_transmission(wave*1000,result.flux,np.full_like(result.flux,.001),
        stellar_rv_kms=0.,planet_rv_stellar_kms=p.kp_kms*np.sin(2*np.pi*phase),
        reference_indices=[0,1,5,6],transit_indices=[2,3,4],
        settings=Seidel22Config([[589.71,589.8],[590.2,590.29]]),
        telluric_corrected=True,sky_subtracted=True,stellar_model_flux=bare)
    np.testing.assert_allclose(prepared.planet_coadd[np.isfinite(prepared.planet_coadd)],1.,atol=3e-14)


def test_limb_cap_clipping_analytic_area(setup):
    s,c,p=setup
    from exoplore.stellar.components import StellarComponent
    from exoplore.stellar.quadrature import radial_rule
    s=physical(s,p,lon=90.,alpha=20.)
    model=SpatialTransitModel(c,p,100000,surface_config=s)
    f=model.field
    w=f.default.grid.wavelength_um
    for name,scale in [('quiet',1.),('spot',.4),('facula',1.2)]:
        grid=LocalIntensityGrid(w,np.array([0.,1.]),np.full((2,len(w)),scale),np.full((2,len(w)),scale))
        f.components[name]=StellarComponent(name,grid,0.,{})
    radii,weights=radial_rule(1.,48,f,time=s.reference_bjd)
    flux,_=model._angular_flux(np.array([.59]),model._rays(radii,0,0,s.reference_bjd),len(radii))
    alpha=np.deg2rad(20.)
    area=alpha-np.sin(alpha)*np.cos(alpha)
    np.testing.assert_allclose(weights@flux,np.pi-.6*area,atol=3e-6)


@pytest.mark.parametrize('bad',['continuum','unit','resolution'])
def test_component_contract_consistency(setup,tmp_path,bad):
    s,c,p=setup
    with np.load(s.components['spot'].intensity_model_file) as d: data={k:d[k] for k in d.files}
    if bad=='continuum': data.pop('continuum')
    if bad=='unit': data['intensity_unit']='different units'
    if bad=='resolution': data['resolving_power']=100000.
    path=tmp_path/'inconsistent.npz'
    np.savez(path,**data)
    s.components['spot']=replace(s.components['spot'],intensity_model_file=str(path))
    with pytest.raises(ValueError): StellarIntensityField(s,c,p)


@pytest.mark.parametrize('mu,velocity',[([0.,1.],[0.]),([1.,0.],[0.,1.]),([0.,1.],[0.,np.nan])])
def test_invalid_velocity_tables(setup,tmp_path,mu,velocity):
    s,c,p=setup
    path=tmp_path/'invalid_velocity.npz'
    np.savez(path,mu=mu,velocity_kms=velocity)
    s.components['spot']=replace(s.components['spot'],velocity_model_file=str(path))
    with pytest.raises(ValueError,match='Velocity tables'): StellarIntensityField(s,c,p)


def test_physical_geometry_requires_explicit_spin_parameters(setup):
    s,_,_=setup
    with pytest.raises(ValueError,match='stellar_inclination_deg'):
        replace(s,regions=[SurfaceRegionConfig('spot',geometry='stellar',
            latitude_deg=0.,longitude_deg=0.,angular_radius_deg=10.)])


def test_intrinsic_quiet_correction_is_exported_on_same_grid(setup):
    s,c,p=setup
    result=evaluate(SpatialTransitModel,c,p,replace(s,correction_surface='quiet'),return_intrinsic=True)
    assert result.correction_intrinsic is not None
    np.testing.assert_array_equal(result.intrinsic.wavelength_um,result.correction_intrinsic.wavelength_um)
    assert result.correction_intrinsic.opaque_flux.shape == result.intrinsic.opaque_flux.shape
