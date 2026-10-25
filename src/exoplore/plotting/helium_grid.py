"""Standard viridis grid-fit and atmosphere-temperature figures for He I."""

# Reading this file
# The grid runner has already evaluated model points. This file displays
# those fit results and temperature information; it does not infer a detection
# from the colour scale or silently run an extra physical grid.

from pathlib import Path
import numpy as np


def plot_helium_grid(output: Path, settings, results: list[dict]) -> None:
    """Write new fit figures; preserve existing plots when a grid is resumed."""
    import matplotlib.pyplot as plt
    from exoplore.config.helium_grid import axis_values
    successful=[r for r in results if r['status']=='complete']
    if not successful:
        print('  No converged fitted model; outcomes retained in grid_results.json',flush=True)
        return
    temperatures=np.asarray(axis_values(settings.temperature_grid_K,'temperature_grid_K',True))
    masses=np.asarray(axis_values(settings.log10_mass_loss_grid_g_s,'log10_mass_loss_grid_g_s'))
    maps={name:np.full((len(masses),len(temperatures)),np.nan) for name in ('reduced_chi2','Tmax_K','THe_K')}
    for result in successful:
        i=int(np.argmin(abs(masses-result['log10_mass_loss_rate_g_s'])))
        j=int(np.argmin(abs(temperatures-result['T0_K'])))
        for name in maps:maps[name][i,j]=np.nan if result[name] is None else result[name]
    best=min(successful,key=lambda r:r['chi2'])
    with np.load(output/'fit_input.npz') as saved:
        wave,flux,error,indices=(saved[k] for k in ('wave_nm','transmission','error','fit_indices'))
    with np.load(Path(best['path'])/'fitted_spectrum.npz') as saved:model=saved['transmission']
    with plt.rc_context({'font.size':14,'axes.labelsize':16,'legend.fontsize':12}):
        fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
        mesh=axes[0].pcolormesh(temperatures,masses,np.ma.masked_invalid(maps['reduced_chi2']),cmap='viridis',shading='nearest')
        for j in range(len(temperatures)):
            if np.isfinite(maps['reduced_chi2'][:,j]).any():
                i=int(np.nanargmin(maps['reduced_chi2'][:,j]));axes[0].plot(temperatures[j],masses[i],'wo',ms=4)
        axes[0].plot(best['T0_K'],best['log10_mass_loss_rate_g_s'],'*',color='white',markeredgecolor='black',ms=15)
        axes[0].set(xlabel='Parker T₀ (K)',ylabel='log₁₀ Ṁ (g s⁻¹)')
        fig.colorbar(mesh,ax=axes[0],label=r'$\chi_R^2$')
        x=wave[indices]/1000;y=100*(flux[indices]-1);e=100*error[indices]
        axes[1].fill_between(x,y-e,y+e,color='0.7',alpha=.4)
        axes[1].plot(x,y,'k.-',lw=1,label='Measured transmission')
        axes[1].plot(x,100*(model[indices]-1),lw=2,color='tab:orange',label=f'Sunbather ({best["velocity_shift_kms"]:+.1f} km s⁻¹)')
        axes[1].set(xlabel='Wavelength (μm)',ylabel='Absorption excess (%)')
        axes[1].ticklabel_format(axis='x',useOffset=False);axes[1].legend()
        for extension in ('png','pdf'):
            path=output/f'helium_grid_fit.{extension}'
            if not path.exists():
                with path.open('xb') as stream:fig.savefig(stream,format=extension,dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True,constrained_layout=True)
        for ax,name,label in zip(axes,('Tmax_K','THe_K'),('Tmax (K)','THe (K)')):
            mesh=ax.pcolormesh(temperatures,masses,np.ma.masked_invalid(maps[name]),cmap='viridis',shading='nearest')
            fig.colorbar(mesh,ax=ax,label=label);ax.set_xlabel('Parker T₀ (K)')
        axes[0].set_ylabel('log₁₀ Ṁ (g s⁻¹)')
        for extension in ('png','pdf'):
            path=output/f'helium_grid_temperatures.{extension}'
            if not path.exists():
                with path.open('xb') as stream:fig.savefig(stream,format=extension,dpi=180)
        plt.close(fig)
