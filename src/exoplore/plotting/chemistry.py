"""Abundance-profile comparisons for the atmospheric chemistry tutorial."""
import numpy as np
import matplotlib.pyplot as plt


def plot_chemistry_profiles(arrays, planet):
    """Show all nine requested gases with the same pressure axis.

    Abundance limits follow the positive values in each panel. Zero abundances
    are masked. The plotted curves retain their calculated values.
    """
    labels={'H2':r'H$_2$','H':'H','H2O':r'H$_2$O','CO':'CO','CO2':r'CO$_2$',
            'CH4':r'CH$_4$','NH3':r'NH$_3$','HCN':'HCN','OH':'OH'}
    methods=[('easychem','EasyChem equilibrium','black','--'),
             ('mixing','VULCAN + mixing','tab:blue','-'),
             ('photochemistry','VULCAN + mixing + UV','tab:orange','-')]
    for method,_,_,_ in methods:
        if method not in arrays:
            raise ValueError('Complete the photochemical comparison before plotting')
    species=arrays['species'].tolist()
    if len(species)!=9:
        raise ValueError('This tutorial figure expects nine gas profiles')
    fig,axes=plt.subplots(3,3,figsize=(11,10),sharey=True,layout='constrained')
    for i,(ax,name) in enumerate(zip(axes.flat,species)):
        positive=[]
        for method,label,color,style in methods:
            x=np.asarray(arrays[method][i])
            ax.plot(np.ma.masked_less_equal(x,0),arrays['pressure_bar'],
                    color=color,ls=style,lw=1.6,alpha=.85,label=label)
            positive.extend(x[x>0])
        ax.set_xscale('log')
        ax.set_yscale('log')
        lower=max(1e-20,10**np.floor(np.log10(min(positive))-.4)) if positive else 1e-20
        upper=min(1.,10**np.ceil(np.log10(max(positive))+.1)) if positive else 1.
        ax.set_xlim(min(lower,upper/100),upper)
        ax.set_ylim(max(arrays['pressure_bar']),min(arrays['pressure_bar']))
        ax.set_title(labels.get(name,name))
        ax.grid(alpha=.15)
        if i%3==0:ax.set_ylabel('Pressure [bar]')
        if i>=6:ax.set_xlabel('Gas volume mixing ratio')
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside upper center',ncol=3,frameon=False,
               title=planet)
    return fig


def plot_chemistry_transmission(arrays, planet, resolving_power=3500):
    """Compare intrinsic transit depths and differences from equilibrium in ppm."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    methods = [('easychem', 'EasyChem equilibrium', 'black', '--'),
               ('mixing', 'VULCAN + mixing', 'tab:blue', '-'),
               ('photochemistry', 'VULCAN + mixing + UV', 'tab:orange', '-')]
    wave = arrays['wavelength_um']
    for name, label, color, style in methods:
        axes[0].plot(wave, 100*arrays[name], label=label, color=color,
                     ls=style, lw=1.2, alpha=.85)
        if name != 'easychem':
            axes[1].plot(wave, 1e6*(arrays[name]-arrays['easychem']), color=color,
                         lw=1.2, alpha=.85)
    axes[0].set_ylabel('Transit depth [%]')
    axes[0].set_title(f'{planet}, R = {resolving_power:g} (for illustration)')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].axhline(0, color='0.4', lw=.8, ls=':')
    axes[1].set_ylabel('Difference from\nequilibrium [ppm]')
    axes[1].set_xlabel(r'Wavelength [$\mu$m]')
    for ax in axes:
        ax.set_xlim(wave.min(), wave.max())
        ax.grid(alpha=.15)
    return fig
