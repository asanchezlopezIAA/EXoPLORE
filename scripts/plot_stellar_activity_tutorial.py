"""Measure and plot the completed resolved-Na comparison in Tutorial 14."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exoplore.config import SimulationConfig
from exoplore.planets import load_planet
from exoplore.stellar.coupled import SpatialTransitModel
from exoplore.atmosphere.helium_pwinds import contact_phases
from run_stellar_activity_tutorial import case_config, load_arrays, file_hash, array_hash, CASES

C_KMS=299792.458
LABELS={'quiet':'Quiet star','unocculted_spot':'Unocculted spot',
        'crossed_spot':'Spot crossing','crossed_facula':'Facular crossing'}


def line_metrics(wave_nm,transmission,centre_nm,covariance=None):
    """Measure the core, band and absorption-weighted moments in fixed velocity windows.

    Signed absorption enters the moments, preserving noise contributions on
    both sides of unity. The RMS width describes only the selected +-20 km/s
    interval. Covariance propagation is linearized around the measured profile.
    """
    velocity=C_KMS*(np.asarray(wave_nm)/centre_nm-1)
    selected=abs(velocity)<=20
    v=velocity[selected]
    signal=1-np.asarray(transmission)[selected]
    if len(v)<4 or not np.isfinite(signal).all():
        raise ValueError('The full measurement window requires finite transmission samples')
    core=abs(v)<=3
    band_weights=np.ones(len(v))/len(v)
    core_weights=core.astype(float)/core.sum()
    result=dict(band_depth_percent=float(100*band_weights@signal),
                core_depth_percent=float(100*core_weights@signal))
    gradients=dict(band_depth_percent=-100*band_weights,core_depth_percent=-100*core_weights)
    dw=np.gradient(v)
    area=dw@signal
    result.update(centroid_kms=None,rms_width_kms=None)
    if area>0:
        mean=float((dw*v)@signal/area)
        result['centroid_kms']=mean
        gradients['centroid_kms']=-dw*(v-mean)/area
        variance=float((dw*(v-mean)**2)@signal/area)
        if variance>0:
            width=np.sqrt(variance)
            result.update(centroid_kms=mean,rms_width_kms=float(width))
            gradients['centroid_kms']=-dw*(v-mean)/area
            gradients['rms_width_kms']=-dw*((v-mean)**2-variance)/(2*width*area)
    if covariance is not None:
        covariance=np.asarray(covariance,float)
        if covariance.shape!=(len(v),len(v)):
            raise ValueError('Supply the covariance on the same +-20 km/s band')
        for key,gradient in gradients.items():
            result[key+'_error']=float(np.sqrt(max(0.,gradient@covariance@gradient)))
    return result


def compare_results(root):
    config=SimulationConfig.from_json(root/'campaign.json')
    centre=config.atomic_transmission.line_wavelengths_nm[0]
    results={}
    for name in CASES:
        results[name]={}
        identities=[]
        for assumed in ('matched',) if name=='quiet' else ('matched','quiet'):
            folder=root/name/assumed
            data=load_arrays(folder/'Na_D2_transmission.npz')
            measured=line_metrics(data['planet_wave_nm'],data['planet_coadd'],centre,data['band_covariance'])
            truth=line_metrics(data['planet_wave_nm'],data['truth_planet_coadd'],centre)
            record=json.loads((folder/'paired_observation.json').read_text())
            identities.append(record['observation_sha256'])
            results[name][assumed]=dict(measured=measured,prepared_noiseless=truth,
                observation_sha256=record['observation_sha256'],
                planetary_rays_sha256=record['planetary_rays_sha256'])
        if len(set(identities))!=1:
            raise AssertionError('The two reductions used different observations')
        if name!='quiet':
            a,b=results[name]['matched']['measured'],results[name]['quiet']['measured']
            results[name]['quiet_minus_matched']={k:None if a[k] is None or b[k] is None else b[k]-a[k]
                for k in ('band_depth_percent','core_depth_percent','centroid_kms','rms_width_kms')}
    return results


def validate_observations(root,config):
    """Check atmosphere identity, paired data, region geometry and brightness scales."""
    from exoplore.io.stellar_surface import load_local_intensities
    from run_stellar_activity_tutorial import fixed_settings, telluric_input_hash
    planet=load_planet(config.planet.parameter_file)
    rays=load_arrays(root/'planetary_rays.npz')
    reference=load_arrays(root/'quiet/observations.npz')
    unit_noise=reference['gauss_noise']/reference['raw_error']
    with np.load(root/'quiet/intrinsic_spectra.npz') as saved:
        reference_velocities={k:saved[k] for k in ('stellar_velocity_kms','planet_velocity_kms')}
    report=dict(cases={},planetary_rays_sha256=file_hash(root/'planetary_rays.npz'),
                pressure_layers=len(rays['impact_parameter_cm']))
    for name in CASES:
        current=SimulationConfig.from_json(root/name/'resolved.json')
        if fixed_settings(current)!=fixed_settings(config):
            raise AssertionError('A case changed non-stellar settings')
        data=load_arrays(root/name/'observations.npz')
        for axis in ('bjd','phase','airmass','throughput','continuum_snr_pixel','photon_response','telluric_sampled'):
            np.testing.assert_array_equal(data[axis],reference[axis])
        np.testing.assert_array_equal(data['wave_nm'],reference['wave_nm'])
        with np.load(root/name/'intrinsic_spectra.npz') as saved:
            for key,values in reference_velocities.items():
                np.testing.assert_array_equal(saved[key],values)
        noise_difference=float(np.max(abs(data['gauss_noise']/data['raw_error']-unit_noise)))
        if noise_difference>3e-15:
            raise AssertionError('Gaussian component draws differ between surfaces')
        correction=load_arrays(root/name/'corrected_observations.npz')
        if str(correction['input_sha256'])!=telluric_input_hash(current,data):
            raise AssertionError('Telluric correction belongs to another observation')
        identity=array_hash({k:data[k] for k in ('raw_flux','raw_error','wave_nm','bjd')})
        checks=dict(exposures=len(data['bjd']),unit_gaussian_max_difference=noise_difference,
                    observation_sha256=identity)
        if name!='quiet':
            region=current.stellar_surface.regions[0]
            angle=2*np.pi*data['phase']
            a=planet.a/(planet.R_star/1e5)
            x,y=a*np.sin(angle),-a*np.cos(angle)*np.cos(planet.inc)
            separation=np.hypot(x-region.x,y-region.y)
            base=rays['impact_parameter_cm'][-1]/planet.R_star
            top=rays['impact_parameter_cm'][0]/planet.R_star
            blocked=(separation<region.radius+base)&(np.cos(angle)>0)
            checks.update(opaque_overlap_exposures=int(blocked.sum()),
                          minimum_atmospheric_separation=float(np.min(separation-region.radius-top)))
            if name=='unocculted_spot' and (blocked.any() or checks['minimum_atmospheric_separation']<=0):
                raise AssertionError('The unocculted region intersects the planet or atmosphere')
            if name!='unocculted_spot' and not blocked.any():
                raise AssertionError('The intended active-region crossing is absent')
        for assumed in ('matched',) if name=='quiet' else ('matched','quiet'):
            metadata=json.loads((root/name/assumed/'paired_observation.json').read_text())
            if metadata['observation_sha256']!=identity or metadata['planetary_rays_sha256']!=report['planetary_rays_sha256']:
                raise AssertionError('A paired analysis changed the observation or planetary atmosphere')
        report['cases'][name]=checks
    quiet,_,_=load_local_intensities(config.rm_clv.model_file,wavelength_medium='vacuum')
    report['continuum_ratios']={}
    for name in ('spot','facula'):
        component,_,_=load_local_intensities(config.stellar_surface.components[name].intensity_model_file,wavelength_medium='vacuum')
        np.testing.assert_array_equal(component.mu,quiet.mu)
        np.testing.assert_array_equal(component.wavelength_um,quiet.wavelength_um)
        ratio=component.continuum/quiet.continuum
        expected=.60 if name=='spot' else np.broadcast_to(1.10+.20*(1-quiet.mu[:,None]),ratio.shape)
        np.testing.assert_allclose(ratio,expected,rtol=3e-15,atol=0)
        report['continuum_ratios'][name]=[float(ratio.min()),float(ratio.max())]
    return report


def geometry_figure(config,planet,destination):
    fig,axes=plt.subplots(2,2,figsize=(9,7.6))
    axes=axes.ravel()
    axis=np.linspace(-1,1,241)
    x,y=np.meshgrid(axis,axis)
    colors=['#e7e2d9','#51403b','#edc96d']
    for ax,name in zip(axes[:3],CASES[1:]):
        current=case_config(config,name)
        model=SpatialTransitModel(current.rm_clv,planet,100000,surface_config=current.stellar_surface)
        # Component selection comes from the same field used by the spatial integration.
        names=list(model.field.components)
        labels=model.field.labels(x,y)
        from matplotlib.colors import ListedColormap
        values=np.vectorize(names.index)(labels).astype(float)
        values[x*x+y*y>1]=np.nan
        ax.imshow(values,origin='lower',extent=(-1,1,-1,1),cmap=ListedColormap(colors),vmin=-.5,vmax=2.5)
        ax.add_patch(Circle((0,0),1,fill=False,color='black',lw=.8))
        t=np.linspace(-.025,.025,300)
        a=planet.a/(planet.R_star/1e5)
        px,py=a*np.sin(2*np.pi*t),-a*np.cos(2*np.pi*t)*np.cos(planet.inc)
        ax.plot(px,py,':',color='black',lw=1)
        index=np.argmin(abs(px+.35))
        ax.add_patch(Circle((px[index],py[index]),planet.R_pl/planet.R_star,color='black',alpha=.55))
        ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),aspect='equal',title=LABELS[name],xlabel=r'$x/R_\star$')
    axes[0].set_ylabel(r'$y/R_\star$')
    axes[2].set_ylabel(r'$y/R_\star$')
    ax=axes[3]
    centre=config.atomic_transmission.line_wavelengths_nm[0]/1000
    wave=centre*(1+np.linspace(-35,35,801)/C_KMS)
    model=SpatialTransitModel(config.rm_clv,planet,100000,surface_config=config.stellar_surface)
    quiet=model.field.components['quiet'].grid
    mu=.75
    upper=np.searchsorted(quiet.mu,mu)
    fraction=(mu-quiet.mu[upper-1])/(quiet.mu[upper]-quiet.mu[upper-1])
    reference=(1-fraction)*quiet.continuum[upper-1]+fraction*quiet.continuum[upper]
    reference=np.interp(wave,quiet.wavelength_um,reference)
    for name,color in zip(model.field.components,['black','#75503b','#cc8e11']):
        intensity,_=model.field.sample(wave,np.array([mu]),np.array([0.]),np.array([name]))
        ax.plot(C_KMS*(wave/centre-1),intensity[0]/reference,color=color,label=name)
    ax.set(xlabel='Local velocity (km/s)',ylabel=r'$I/C_{\rm quiet}$',title=r'Local spectra, $\mu=0.75$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def stellar_spectrum_figure(root,config,destination):
    """Compare the unocculted stellar disks at the same observing time."""
    centre=config.atomic_transmission.line_wavelengths_nm[0]/1000
    reference=load_arrays(root/'quiet/intrinsic_spectra.npz')
    wave=reference['wavelength_um']
    velocity=C_KMS*(wave/centre-1-reference['stellar_velocity_kms'][0]/C_KMS)
    selected=abs(velocity)<=30
    # The first exposure lies outside transit. stellar_flux is the unobscured
    # disk integral, before telluric absorption and instrumental convolution.
    quiet_flux=reference['stellar_flux'][0]
    quiet_continuum=reference['continuum'][0]
    fig,axes=plt.subplots(2,1,figsize=(8,5.5),sharex=True,
                          gridspec_kw={'height_ratios':[2,1]})
    for name,color in zip(CASES,['black','#75503b','#1768a6','#cc8e11']):
        current=load_arrays(root/name/'intrinsic_spectra.npz')
        np.testing.assert_array_equal(current['wavelength_um'],wave)
        flux=current['stellar_flux'][0]
        axes[0].plot(velocity[selected],(flux/quiet_continuum)[selected],
                     color=color,label=LABELS[name],lw=1.2)
        if name!='quiet':
            axes[1].plot(velocity[selected],100*(flux/quiet_flux-1)[selected],
                         color=color,lw=1.2)
    axes[0].set_ylabel(r'$F_\star/C_{\rm quiet}$')
    axes[0].legend(fontsize=8)
    axes[1].axhline(0,color='.5',lw=.6)
    axes[1].set(xlabel='Stellar-frame velocity (km/s)',
                ylabel='Change from quiet (%)',xlim=(-30,30))
    fig.tight_layout()
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def phase_figure(root,config,planet,destination):
    fig,axes=plt.subplots(2,2,figsize=(9,7.5),sharex=True,sharey=True)
    axes=axes.ravel()
    centre=config.atomic_transmission.line_wavelengths_nm[0]
    panels=[]
    for name in CASES:
        assumed='matched' if name=='quiet' else 'quiet'
        data=load_arrays(root/name/assumed/'Na_D2_transmission.npz')
        velocity=C_KMS*(data['stellar_wave_nm']/centre-1)
        select=abs(velocity)<=45
        panels.append((data,velocity[select],100*(data['stellar_transmission'][:,select]-1)))
    # Use the same percentile display rule as the existing atomic diagnostics.
    deviations=np.concatenate([p[2][np.isfinite(p[2])] for p in panels])
    limit=np.percentile(abs(deviations),98)
    for ax,name,(data,velocity,image) in zip(axes,CASES,panels):
        m=ax.pcolormesh(velocity,data['phase'],image,cmap='RdBu',vmin=-limit,vmax=limit,shading='auto',rasterized=True)
        track=(planet.kp_kms+planet.stellar_rv_semiamplitude_kms)*np.sin(2*np.pi*data['phase'])
        ax.plot(track,data['phase'],'k--',lw=.8)
        for contact in contact_phases(planet):
            ax.axhline(contact,color='black',ls=':',lw=.5)
        ax.set(title=LABELS[name],xlabel='Stellar-frame velocity (km/s)',xlim=(-45,45))
    for ax in axes[:2]:
        ax.set_xlabel('')
    axes[0].set_ylabel('Orbital phase')
    axes[2].set_ylabel('Orbital phase')
    fig.colorbar(m,ax=axes,location='right',label='Transmission excess (%)',
                 fraction=.025,pad=.02,extend='both')
    fig.subplots_adjust(left=.10,right=.88,bottom=.10,top=.94,wspace=.12,hspace=.28)
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def profiles_figure(root,config,destination):
    fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True)
    centre=config.atomic_transmission.line_wavelengths_nm[0]
    baseline=load_arrays(root/'quiet/matched/Na_D2_transmission.npz')
    velocity=C_KMS*(baseline['planet_wave_nm']/centre-1)
    window=abs(velocity)<=30
    for ax,name in zip(axes,CASES[1:]):
        ax.plot(velocity[window],100*(baseline['planet_coadd'][window]-1),color='.65',lw=1,label='Quiet-star reference')
        for assumed,color,label in [('matched','#1768a6','Matched surface'),('quiet','#ca5727','Quiet-star assumption')]:
            data=load_arrays(root/name/assumed/'Na_D2_transmission.npz')
            v=C_KMS*(data['planet_wave_nm']/centre-1)
            selected=abs(v)<=30
            excess=100*(data['planet_coadd']-1)
            ax.plot(v[selected],excess[selected],color=color,lw=1.4,label=label)
            ax.fill_between(v[selected],excess[selected]-100*data['coadd_error'][selected],
                excess[selected]+100*data['coadd_error'][selected],color=color,alpha=.12,lw=0)
            if assumed=='matched':
                ax.plot(v[selected],100*(data['truth_planet_coadd'][selected]-1),color=color,ls=':',lw=1,label='Prepared noiseless model')
        ax.axvline(0,color='black',ls=':',lw=.7)
        ax.axhline(0,color='.5',lw=.6)
        ax.set(title=LABELS[name],xlabel='Planet-frame velocity (km/s)',xlim=(-30,30))
    for ax in axes[:-1]:
        ax.set_xlabel('')
    for ax in axes:
        ax.set_ylabel('Transmission excess (%)')
    axes[0].legend(fontsize=7,loc='lower left')
    fig.tight_layout()
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    parser.add_argument('figure_directory',type=Path)
    args=parser.parse_args()
    args.figure_directory.mkdir(parents=True,exist_ok=True)
    config=SimulationConfig.from_json(args.run_directory/'campaign.json')
    planet=load_planet(config.planet.parameter_file)
    validation=validate_observations(args.run_directory,config)
    (args.run_directory/'comparison_validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False))
    results=compare_results(args.run_directory)
    (args.run_directory/'line_measurements.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    geometry_figure(config,planet,args.figure_directory/'tutorial14_surface_components.png')
    stellar_spectrum_figure(args.run_directory,config,args.figure_directory/'tutorial14_stellar_Na_D2.png')
    phase_figure(args.run_directory,config,planet,args.figure_directory/'tutorial14_Na_D2_phase_maps.png')
    profiles_figure(args.run_directory,config,args.figure_directory/'tutorial14_Na_D2_profiles.png')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
