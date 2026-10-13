"""OH sky-subtraction diagnostic plots for observed CARMENES helium runs.

Display the measured B-to-A factor with the science-spectrum correction.
Input correction products are read-only and the figure output must be new.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def archive(path: Path) -> dict:
    """Read saved per-exposure arrays without modifying the archive."""
    with np.load(path) as data:
        return {key:data[key].copy() for key in data.files}


def plot_oh_correction(directory: Path, output: Path, science, number: int) -> None:
    """Compare the measured sky model and its subtraction from fibre A."""
    obs=archive(directory/'observations.npz');oh=archive(directory/'oh_model.npz')
    i=number-1
    if not 0<=i<len(obs['raw_flux']):
        raise ValueError('Exposure number lies outside this night')
    report=json.loads((directory/'oh_report.json').read_text())
    transfer=report['transfer']
    if 'fiber_b_scale' in transfer:
        scale=float(transfer['fiber_b_scale'][i])
    else:
        scale=float(transfer['transfer_scale'])
    scale_label=f'B-to-A factor = {scale:.4g}'
    if 'transfer_scale_error' in transfer:
        scale_label+=f" ± {transfer['transfer_scale_error']:.2g}"
    if 'sky_flux' in oh:
        sky=oh['sky_flux'][i];sw=oh['master_wave_nm']
        sky_label=f'Fibre B, exposure {number}'
        sky_model=oh['model_b'][i]
    else:
        sky=oh['master_flux'];sw=oh['master_wave_nm']
        sky_model=oh['model_b'].sum(axis=0);sky_label='Summed fibre B'
    w=obs['native_wave_nm'][i]
    band=(w>=science.continuum_stellar_windows_nm[0][0])&(w<=science.continuum_stellar_windows_nm[0][1])
    continuum=np.nanmean(obs['raw_flux'][i,band])
    with plt.rc_context({'font.size':16,'axes.labelsize':18,'legend.fontsize':13}):
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col',gridspec_kw={'hspace':0})
        for j,(lo,hi) in enumerate(((1083.17,1083.28),(1083.39,1083.48))):
            b=(sw>=lo)&(sw<=hi)
            axes[0,j].plot(sw[b]/1000,sky[b],'k.-',label=sky_label)
            axes[0,j].plot(sw[b]/1000,sky_model[b],color='tab:orange',lw=2,label='OH model')
            # Isolated sky-detector spikes remain in the data. Limit the view
            # to the resolved fitted line rather than autoscaling to spikes.
            axes[0,j].set_ylim(min(0,float(np.min(sky[b]))),1.5*float(np.max(sky_model[b])))
            a=(w>=lo)&(w<=hi)
            axes[1,j].plot(w[a]/1000,obs['raw_flux'][i,a]/continuum,'k.-',label='Fibre A before OH correction')
            corrected=obs['raw_flux'][i]-oh['model_a'][i]
            axes[1,j].plot(w[a]/1000,corrected[a]/continuum,'.-',color='tab:blue',label='After OH correction')
            axes[1,j].set_xlabel('Wavelength (μm)')
            for ax in axes[:,j]:
                ax.ticklabel_format(axis='x',useOffset=False);ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(length=6,width=1.2)
            axes[0,j].set_title('Weak OH pair' if j==0 else 'Strong OH line')
        axes[0,0].set_ylabel('Fibre B flux');axes[1,0].set_ylabel('Normalized fibre A flux')
        axes[0,0].legend();axes[1,0].legend(title=scale_label,title_fontsize=13)
        fig.subplots_adjust(wspace=.25,left=.09,right=.99,bottom=.11,top=.93)
        try:
            with Path(output).open('xb') as stream:
                fig.savefig(stream,format=Path(output).suffix.lstrip('.'),dpi=180,bbox_inches='tight')
        finally:
            plt.close(fig)

