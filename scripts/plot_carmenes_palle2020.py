#!/usr/bin/env python3
"""Plot saved Palle2020 corrections and transmission products without refitting.

Each output must be new. Observed FITS files and saved arrays are read-only.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.io import fits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def archive(path):
    """Read all arrays from a saved product."""
    with np.load(path) as data:
        return {k:data[k].copy() for k in data.files}


def save(fig,path):
    """Save a new figure without replacing an earlier file."""
    with path.open('xb') as stream:
        fig.savefig(stream,format=path.suffix.lstrip('.'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def exposure(directory,output,intervals):
    """Show selected saved molecfit intervals on the original fitting grid."""
    report=json.loads((directory/'report.json').read_text())
    model=fits.getdata(directory/'BEST_FIT_MODEL.fits',1)
    full=fits.getdata(directory/'calctrans/TELLURIC_DATA.fits',1)
    if len(full)!=len(model) or not np.allclose(full['lambda'],model['lambda']):
        raise ValueError('Saved telluric model grids differ')
    selected=[report['windows_nm'][i-1] for i in intervals]
    with plt.rc_context({'font.size':16,'axes.labelsize':18,'legend.fontsize':14}):
        fig,axes=plt.subplots(3,len(selected),figsize=(5*len(selected),10),
            squeeze=False,sharex='col',gridspec_kw={'hspace':0})
        wave=model['lambda']*1000
        for j,(lo,hi) in enumerate(selected):
            valid=(wave>=lo)&(wave<=hi)&(model['weight']>0)&(model['mscal']>0)&(full['mtrans']>0)
            x=wave[valid]/1000;f=model['flux'][valid];continuum=model['mscal'][valid]
            axes[0,j].plot(x,f,'k.-',label='Observed')
            axes[0,j].plot(x,model['mflux'][valid],color='tab:orange',lw=2,label='molecfit model')
            axes[0,j].set_title(f'{lo/1000:.5f}–{hi/1000:.5f} μm')
            axes[1,j].plot(x,f/continuum,color='.6',label='Before correction')
            axes[1,j].plot(x,f/continuum/full['mtrans'][valid],color='tab:blue',label='After correction')
            axes[1,j].axhline(1,color='.6',ls=':')
            axes[2,j].plot(x,(f-model['mflux'][valid])*model['weight'][valid],'k.-')
            axes[2,j].axhline(0,color='.6',ls=':')
            axes[2,j].set_xlabel('Wavelength (μm)')
            for ax in axes[:,j]:
                ax.ticklabel_format(axis='x',useOffset=False)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
                ax.tick_params(length=6,width=1.2)
        axes[0,0].set_ylabel('Flux')
        axes[1,0].set_ylabel('Normalized flux')
        axes[2,0].set_ylabel('Residual / uncertainty')
        axes[0,0].legend();axes[1,0].legend()
        fig.subplots_adjust(wspace=.3,left=.08,right=.99,bottom=.10,top=.92)
        save(fig,output)


def science_config(config):
    """Read the explicitly configured CARMENES transmission settings."""
    from exoplore.config.models import SimulationConfig
    cfg=SimulationConfig.from_json(config)
    if cfg.pipeline.name!='Palle2020':
        raise ValueError('Use a Palle2020 configuration')
    return cfg,cfg.pipeline.carmenes_helium


def sky_plot(directory,output,science,number):
    """Use the standard CARMENES OH plot, including its A/B calibration."""
    from exoplore.plotting.carmenes_oh import plot_oh_correction
    plot_oh_correction(directory,output,science,number)


def steps(directory,output_directory,cfg,science):
    """Use EXoPLORE's standard observed-data pipeline-stages renderer."""
    from exoplore.plotting.steps import plot_pipeline_steps
    from exoplore.pipelines.carmenes_helium import gap_resample
    from exoplore.pipelines.helium_math import stellar_wavelength
    obs=archive(directory/'observations.npz');data=archive(directory/'transmission.npz')
    native=obs['native_wave_nm'][0]
    residual=np.array([gap_resample(data['stellar_wave_nm'],row-1,np.ones_like(row),
        stellar_wavelength(native,b,science.gamma_kms))[0]
        for row,b in zip(data['stellar_transmission'],data['berv_kms'])])
    corrected=np.array([gap_resample(w,f,e**2,native)[0]
        for w,f,e in zip(obs['wave_nm'],obs['corrected_flux'],obs['corrected_error'])])
    output_directory.mkdir(exist_ok=False)
    title='Telluric and OH correction'
    final='Divided by master out-of-transit (Earth-frame display)'
    plot_pipeline_steps(f'{cfg.planet.name}_Palle2020',str(output_directory)+'/',native/1000,
        data['phase'],data['with_signal'],data['without_signal'],
        np.flatnonzero(np.isfinite(residual).any(axis=0)),obs['raw_flux'],obs['raw_flux'],residual,
        order_label=science.order_segment,xlim_1d=tuple(np.array(science.plot_stellar_window_nm)/1000),
        sysrem_stages={title:corrected,final:residual},sysrem_iters=[title,final],
        use_real_data=True,save_plot=True,show_plot=False)


def compare(directories,output):
    """Compare saved planetary coadds with a three-point display mean."""
    fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
    labels=['Calibrated fibre-B profile','OOT fibre-A profile']
    for directory,label in zip(directories,labels):
        data=archive(directory/'transmission.npz');w=data['planet_wave_nm'];f=data['planet_coadd']
        smoothed=(f[:-2]+f[1:-1]+f[2:])/3;x=w[1:-1]
        b=(x>=1083.1)&(x<=1083.48)&np.isfinite(smoothed)
        ax.plot(x[b]/1000,100*(smoothed[b]-1),lw=2,label=label)
    ax.axhline(0,color='.6',lw=.8)
    ax.set(xlabel='Wavelength (μm)',ylabel='Transmission excess (%)')
    ax.ticklabel_format(axis='x',useOffset=False);ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.legend(title='Three-point running mean',fontsize=14,title_fontsize=14)
    save(fig,output)


def main():
    """Expose saved exposure, sky, transmission and standard-stages figures."""
    plt.rcParams.update({'font.size':16,'axes.labelsize':18,'legend.fontsize':14})
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='kind',required=True)
    p=sub.add_parser('exposure');p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--intervals',type=int,nargs='+',default=[1,4,5])
    p=sub.add_parser('night');p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
    for name in ('sky','transmission','steps'):
        p=sub.add_parser(name);p.add_argument('config',type=Path);p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
        if name=='sky':p.add_argument('--exposure',type=int,default=17)
    p=sub.add_parser('compare');p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output path; existing plots are preserved')
    if args.kind=='night':
        from exoplore.plotting.helium_molecfit import plot_molecfit_diagnostics
        plot_molecfit_diagnostics(args.directory,args.output,'CARMENES_NIR')
    elif args.kind=='exposure':exposure(args.directory,args.output,args.intervals)
    elif args.kind=='compare':compare([args.first,args.second],args.output)
    else:
        cfg,science=science_config(args.config)
        if args.kind=='sky':sky_plot(args.directory,args.output,science,args.exposure)
        elif args.kind=='steps':steps(args.directory,args.output,cfg,science)
        else:
            from exoplore.plotting.helium import plot_helium_summary
            d=archive(args.directory/'transmission.npz')
            plot_helium_summary(args.output,d,d['phase'],d['planet_rv_kms'],d['berv_kms'],science)
    print(args.output)


if __name__=='__main__':
    main()
