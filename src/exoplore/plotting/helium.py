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
    without = np.asarray(science.reference_running_numbers)-1
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
