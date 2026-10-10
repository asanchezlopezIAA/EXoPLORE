"""Consistent diagnostics for the opt-in direct helium simulation branches.

The pipeline-steps figure calls EXoPLORE's existing plotting routine. The
transmission-map layout follows Fig. 4 of Czesla et al. (2024),
https://doi.org/10.1051/0004-6361/202451003, with residuals explicitly measured
against the prepared injected model, not a fitted empirical slab. Plotting
does not smooth, detrend, fit or change the saved scientific arrays.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exoplore.pipelines.carmenes_helium import gap_resample
from exoplore.pipelines.helium_math import C_KMS, stellar_wavelength


def plot_transmission_maps(
    directory: Path, wave_nm: np.ndarray, transmission: np.ndarray,
    model_transmission: np.ndarray, phase: np.ndarray, rv_kms: np.ndarray,
    berv_kms: np.ndarray, science, *, planetary: bool = False,
) -> None:
    """Save observed/model-residual maps with common scales and frame markers.

Both inputs are dimensionless reference-divided transmission on the supplied
grid. No smoothing is applied. OH markers indicate configured wavelengths;
they do not imply simulated emission or a correction for airglow.
    """
    import matplotlib.pyplot as plt
    frame = 'planetary' if planetary else 'stellar'
    stem = directory/f'helium_{frame}_transmission_map'
    if any(stem.with_suffix(ext).exists() for ext in ('.pdf', '.png')):
        raise FileExistsError(f'Preserve existing figures; choose a new directory: {stem}')
    lo, hi = science.plot_stellar_window_nm
    band = (wave_nm >= lo)&(wave_nm <= hi)
    observed = 100*(transmission[:, band]-1)
    residual = 100*(transmission[:, band]-model_transmission[:, band])
    limit = max(float(np.nanpercentile(abs(observed), 98)), .01)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             sharey=True, constrained_layout=True)
    for axis, values, title in zip(axes, (observed, residual),
            ('Prepared transmission', 'Residuals relative to the prepared injected model')):
        mesh = axis.pcolormesh(wave_nm[band], phase, values, cmap='RdBu_r',
                              vmin=-limit, vmax=limit, shading='nearest', rasterized=True)
        for contact in science.optical_contact_phases:
            axis.axhline(contact, color='black', ls='--', lw=.8)
        for index, line in enumerate(science.helium_vacuum_lines_nm):
            axis.axvline(line, color='magenta', ls='-.', lw=.8,
                         label='He I rest wavelengths' if index == 0 else None)
            if not planetary:
                axis.plot(line*(1+rv_kms/C_KMS), phase, color='red', ls='--', lw=.9,
                          label='Planet orbital track' if index == 0 else None)
        for index, line in enumerate(science.oh_topocentric_lines_nm):
            marker = stellar_wavelength(np.full_like(phase, line), berv_kms, science.gamma_kms)
            if planetary: marker = marker/(1+rv_kms/C_KMS)
            axis.plot(marker, phase, color='gold', ls=':', lw=1,
                      label='Configured OH wavelengths' if index == 0 else None)
        axis.set(title=title, ylabel='Orbital phase', xlim=(lo, hi))
        axis.ticklabel_format(axis='x', useOffset=False)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel(f'{frame.capitalize()}-frame vacuum wavelength (nm)')
    fig.colorbar(mesh, ax=axes.tolist(), label='Transmission excess / residual (%)')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_helium_run_plots(output: Path, arrays: dict, recovered: dict,
                          truth: dict, phase: np.ndarray, rv_kms: np.ndarray,
                          science, *, config: dict | None = None) -> Path:
    """Use the standard CCF steps renderer and save direct-transmission maps.

The steps residual is transformed back to the Earth-frame display grid so
that every panel has the same wavelength coordinate. Master-out division was
performed in the stellar frame; this display interpolation does not replace
the saved stellar or planetary arrays. Existing files are never overwritten.
    """
    from exoplore.plotting.steps import plot_pipeline_steps
    import matplotlib.pyplot as plt
    output = Path(output)
    directory = output/'plots'
    directory.mkdir(exist_ok=False)
    if config is None:
        config = json.loads((output/'run_config.json').read_text())
    plotting = config.get('plotting', {})
    run_name = str(config.get('output', {}).get('simulation_name') or config['planet']['name'])
    wave = np.asarray(arrays['wave_nm'][0])
    berv = np.asarray(config['atmosphere']['helium']['berv_kms'])
    raw_flux = arrays.get('raw_flux')
    if raw_flux is None:
        with np.load(output/'observations.npz') as saved:
            raw_flux = saved['raw_flux'].copy()
    residual = []
    for row, b in zip(recovered['stellar_transmission'], berv):
        destination = stellar_wavelength(wave, b, science.gamma_kms)
        display, _ = gap_resample(recovered['stellar_wave_nm'], row-1,
                                 np.ones_like(row), destination)
        residual.append(display)
    residual = np.asarray(residual)
    # Insert a masked plotting column at each detector gap. The standard
    # routine otherwise paints a cell across the missing physical interval.
    gaps = np.flatnonzero(np.diff(wave)>5*np.median(np.diff(wave)))
    noiseless, noisy = arrays['expected_flux'].copy(), raw_flux.copy()
    for gap in gaps[::-1]:
        wave = np.insert(wave, gap+1, (wave[gap]+wave[gap+1])/2)
        noiseless = np.insert(noiseless, gap+1, np.nan, axis=1)
        noisy = np.insert(noisy, gap+1, np.nan, axis=1)
        residual = np.insert(residual, gap+1, np.nan, axis=1)
    with_signal = np.flatnonzero((phase>=science.optical_contact_phases[0])&
                                (phase<=science.optical_contact_phases[-1]))
    without = np.asarray(science._reference_indices, dtype=int)
    label = 'Corrected (stellar-reference division; Earth-frame display)'
    window = plotting.get('pipeline_steps_xlim_um')
    with plt.rc_context():
        plot_pipeline_steps(run_name, str(directory)+'/', wave/1000, phase,
            with_signal, without, np.isfinite(residual).all(axis=0),
            noiseless, noisy, residual,
            spec_idx=int(np.argmin(abs(phase))), order_label=science.order_segment,
            xlim_1d=tuple(window) if window else None,
            sysrem_stages={label:residual}, sysrem_iters=[label],
            use_real_data=False, save_plot=True, show_plot=False)
    for planetary, key in ((False, 'stellar'), (True, 'planet')):
        model = []
        for row in truth[f'{key}_transmission']:
            y, _ = gap_resample(truth[f'{key}_wave_nm'], row,
                               np.ones_like(row), recovered[f'{key}_wave_nm'])
            model.append(y)
        plot_transmission_maps(directory, recovered[f'{key}_wave_nm'],
            recovered[f'{key}_transmission'], np.asarray(model), phase, rv_kms,
            berv, science, planetary=planetary)
    with (directory/'plot_provenance.json').open('x') as stream:
        json.dump(dict(steps_renderer='exoplore.plotting.steps.plot_pipeline_steps',
            steps_layout='Existing stacked grayscale raw/corrected layout',
            steps_frame='Earth-frame display; reference division calculated in stellar frame',
            existing_renderer_top_panel_noise_display_factor=1.4,
            map_smoothing='None', residual_model='Prepared noiseless injected model; no empirical fit',
            oh='Only explicitly configured markers; simulation contains no OH emission',
            output_run_name=run_name), stream, indent=2)
    return directory


def write_combined_helium_maps(output: Path, sources: list[Path], science,
                             *, directory: Path | None = None) -> None:
    """Save the same stellar-frame map/residual layout for a night combination."""
    from exoplore.pipelines.carmenes_helium import prepare_carmenes_transmission
    output = Path(output)
    directory = output/'plots' if directory is None else Path(directory)
    directory.mkdir(exist_ok=False)
    combined = dict(np.load(output/'combined_transmission.npz'))
    model_maps = []
    for index,path in enumerate(sources,1):
        obs = dict(np.load(path/'observations.npz'))
        recovered=dict(np.load(path/'transmission.npz'))
        from exoplore.pipelines.czesla2024 import timed_exposure_selection
        source_config=json.loads((path/'run_config.json').read_text())
        science,_,_,_=timed_exposure_selection(science,obs['bjd_tdb'],
            source_config['observation']['exposure_time_seconds'])
        good = obs['telluric_transmission']>=science.telluric_min_transmission
        truth = prepare_carmenes_transmission(obs['wave_nm'],
            np.where(good,obs['expected_flux']/obs['telluric_transmission'],np.nan),
            np.where(good,obs['raw_error']/obs['telluric_transmission'],np.nan),
            obs['berv_kms'],obs['planet_rv_kms'],science)
        model = [gap_resample(truth['stellar_wave_nm'],row,np.ones_like(row),
                              combined['stellar_wave_nm'])[0]
                 for row in truth['stellar_transmission']]
        model_maps.append(model)
        # Completed input nights remain read-only. Regenerate their standard
        # diagnostics in this new combination, so reuse still returns steps.
        night_output=directory/f'night_{index}'
        night_output.mkdir(exist_ok=False)
        source_config=json.loads((path/'run_config.json').read_text())
        write_helium_run_plots(night_output,obs,recovered,truth,obs['phase'],
                              obs['planet_rv_kms'],science,config=source_config)
    plot_transmission_maps(directory,combined['stellar_wave_nm'],
        combined['combined_stellar_map'],np.mean(model_maps,axis=0),combined['phase'],
        science.kp_kms*np.sin(2*np.pi*combined['phase']),obs['berv_kms'],science)


def plot_helium_summary(path: Path, result: dict, phase: np.ndarray,
                        rv_kms: np.ndarray, berv_kms: np.ndarray, science,
                        *, night_spectra=None, night_lightcurves=None,
                        night_lightcurve_errors=None, significance=None,
                        synthetic: bool = False) -> None:
    """Save the common three-panel result for real and simulated helium data.

    Show the stellar transmission map, planet-frame spectrum and planetary
    fixed-band light curve. Lower panels use transmission excess in percent,
    vacuum wavelengths in micrometres and shaded supplied uncertainties.
    Optional night arrays add individual curves to their combined result.
    No selections, binning, fits or uncertainty estimates are made here.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    phase = np.asarray(phase)
    hours = phase*science.period_days*24
    wave = np.asarray(result['stellar_wave_nm'])/1000
    planet_wave = np.asarray(result['planet_wave_nm'])/1000
    map_limits = np.asarray(science.plot_stellar_window_nm)/1000
    spectrum_limits = np.asarray(science.equivalent_width_window_nm)/1000
    band=(planet_wave>=spectrum_limits[0])&(planet_wave<=spectrum_limits[1])
    coadd_error = np.asarray(result.get('planet_coadd_error', result.get('coadd_error')))
    curve_error = np.asarray(result.get('planet_lightcurve_error', result.get('lightcurve_error')))
    combined = night_spectra is not None
    count = len(night_spectra) if combined else 1
    label = f'Combined ({count} nights)' if combined else ('Simulated night' if synthetic else 'Observed night')
    if significance is not None:
        measured = significance[-1] if combined else significance
        label += f" ({measured['significance_sigma']:.1f}σ)"
        legend_title = f"Significance from Allart et al. 2023. {measured['config']['band_width_nm']*10:.2f}Å band."
    else:
        legend_title = None
    with plt.rc_context({'font.size':14, 'axes.labelsize':17, 'axes.titlesize':17,
                         'xtick.labelsize':14, 'ytick.labelsize':14,
                         'legend.fontsize':13, 'xtick.major.size':5.6,
                         'ytick.major.size':5.6}):
        fig, axes = plt.subplots(3,1,figsize=(13,13),constrained_layout=True)
        mesh = axes[0].pcolormesh(wave,hours,result['stellar_transmission'],
                                  cmap='RdBu',shading='auto',vmin=.975,vmax=1.025)
        for index,line in enumerate(science.helium_vacuum_lines_nm):
            axes[0].axvline(line/1000,color='magenta',ls=':',lw=1.2,
                           label='He I rest wavelengths' if index==0 else None)
            axes[0].plot(line*(1+rv_kms/C_KMS)/1000,hours,color='red',ls='--',lw=1,
                         label='Planet velocity track' if index==0 else None)
            axes[1].axvline(line/1000,color='.65',ls=':',lw=1.2)
        for index,line in enumerate(science.oh_topocentric_lines_nm):
            axes[0].plot(stellar_wavelength(line,berv_kms,science.gamma_kms)/1000,
                         hours,color='gold',ls=':',lw=1,
                         label='OH sky wavelengths' if index==0 else None)
        axes[0].set(xlim=map_limits,xlabel='Wavelength (μm)',ylabel='Time from mid-transit (h)')
        axes[0].legend(loc='upper right',fontsize=12)
        fig.colorbar(mesh,ax=axes[0],label='Transmission')
        if combined:
            colors=plt.get_cmap('tab10')
            for index,(spectrum,curve) in enumerate(zip(night_spectra,night_lightcurves)):
                color=colors(index%10)
                night_label=f'Night {index+1}'
                if significance is not None:
                    night_label += f" ({significance[index]['significance_sigma']:.1f}σ)"
                axes[1].plot(planet_wave[band],100*(spectrum[band]-1),color=color,lw=1,alpha=.7,label=night_label)
                axes[2].plot(phase,100*(curve-1),'.-',color=color,lw=1,alpha=.7,label=night_label)
                if night_lightcurve_errors is not None:
                    error=night_lightcurve_errors[index]
                    axes[2].fill_between(phase,100*(curve-1-error),100*(curve-1+error),
                                         color=color,alpha=.10)
        spectrum=np.asarray(result['planet_coadd'])
        curve=np.asarray(result['planet_lightcurve'])
        axes[1].plot(planet_wave[band],100*(spectrum[band]-1),'k-',lw=2.2,label=label)
        axes[1].fill_between(planet_wave[band],100*(spectrum[band]-1-coadd_error[band]),
                             100*(spectrum[band]-1+coadd_error[band]),color='black',alpha=.15)
        axes[2].plot(phase,100*(curve-1),'k.-',lw=2.2,label=label)
        axes[2].fill_between(phase,100*(curve-1-curve_error),
                             100*(curve-1+curve_error),color='black',alpha=.15)
        axes[1].set(xlim=spectrum_limits,xlabel='Wavelength (μm)',ylabel='Transmission excess (%)',
                    title='Individual nights and combined He I transmission' if combined else 'Planet-frame He I transmission')
        lo,hi=np.asarray(science.planet_lightcurve_window_nm)/1000
        axes[2].set(xlabel='Orbital phase',ylabel='Band-averaged transmission excess (%)',
                    title=f'Planet-frame helium light curve ({lo:.6f}–{hi:.6f} μm)')
        for index,contact in enumerate(science.optical_contact_phases,1):
            hour=contact*science.period_days*24
            axes[0].axhline(hour,color='black',ls='--',lw=.9)
            axes[0].text(.01,hour,f'T{index}',transform=axes[0].get_yaxis_transform(),va='bottom',fontsize=12)
            axes[2].axvline(contact,color='.5',ls='--',lw=1)
            axes[2].text(contact,.98,f'T{index}',transform=axes[2].get_xaxis_transform(),
                         ha='right' if index in (1,3) else 'left',va='top',fontsize=12,
                         bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
        for axis in axes[1:]:
            axis.axhline(0,color='.6',lw=.8)
            axis.legend(title=legend_title,title_fontsize=13,loc='lower left')
        for axis in axes[:2]:
            axis.ticklabel_format(axis='x',style='plain',useOffset=False)
        with path.open('xb') as stream:
            fig.savefig(stream,format=path.suffix.lstrip('.'),dpi=180,bbox_inches='tight')
        plt.close(fig)
