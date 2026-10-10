"""Independent synthetic CARMENES nights and direct helium transmission coadds.

The first implementation repeats the explicitly supplied cadence, BERV and
atmospheric conditions. Each night has its own Gaussian observation noise,
molecfit fit and out-of-transit reference. It does not sample uncertainties or
alter the molecular cross-correlation route. Existing night directories remain
read-only. The cross-night combination is an arithmetic mean, with propagated
shared-reference and interpolation covariance conditional on fitted continua
and telluric parameters.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from exoplore.pipelines.helium_math import window_mask
from exoplore.pipelines.helium_noise_propagation import (
    carmenes_coadd_covariance, carmenes_lightcurve_covariance, interpolation_operator,
)


def _physical_config(config: dict) -> dict:
    """Exclude only output/reuse controls and noise seed from physical identity."""
    result = json.loads(json.dumps(config))
    result['paths']['output_root'] = ''
    result['observation']['n_nights'] = 1
    for recipe in ('czesla2024','carmenes_helium'):
        choices=result.get('pipeline',{}).get(recipe)
        if choices:
            choices.pop('reference_running_numbers',None)
            choices.pop('in_transit_running_numbers',None)
    h = result['atmosphere']['helium']
    # A significance estimator changes reporting, not the saved observations.
    h.pop('allart2023_significance', None)
    if h.get('sunbather') is None:
        h.pop('sunbather', None)
    for key in ('completed_night_paths', 'resume_simulation_path', 'noise_seed'):
        h.pop(key, None)
    return result


def _verify_completed(source: Path, config) -> None:
    """Check a completed night against requested physics, inputs and fit gates."""
    required = ('run_config.json', 'provenance.json', 'observations.npz',
                'transmission.npz', 'molecfit_calibration.npz',
                'simulated_observations.npz', 'outflow.npz')
    for name in required:
        if not (source/name).is_file():
            raise ValueError(f'Incomplete saved helium night: {source/name}')
    old = json.loads((source/'run_config.json').read_text())
    if _physical_config(old) != _physical_config(asdict(config)):
        raise ValueError('Completed night differs in physical, ETC or preparation choices')
    provenance = json.loads((source/'provenance.json').read_text())
    for name, digest in provenance['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed input of completed night: {name}')
    reports = list((source/'molecfit').glob('exposure_*/detector_*/report.json'))
    with np.load(source/'simulated_observations.npz') as data:
        wave = data['wave_nm'][0]
        sections = 1 + np.count_nonzero(np.diff(wave) > 5*np.median(np.diff(wave)))
        expected_reports = len(data['phase'])*sections
    if len(reports) != expected_reports or any(
            not json.loads(p.read_text())['accepted'] for p in reports):
        raise ValueError('Completed night lacks accepted corrections for every section')


def read_saved_expectation(source: Path, config):
    """Read noiseless arrays and ETC error amplitudes for a new observation night."""
    _verify_completed(source, config)
    with np.load(source/'simulated_observations.npz', allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in (
            'wave_nm', 'expected_flux', 'no_helium_flux',
            'telluric_transmission', 'opaque_continuum')}
        error = data['raw_error'].copy()
    with np.load(source/'outflow.npz', allow_pickle=False) as data:
        profile = {k: data[k].copy() for k in data.files}
    # NPZ stores version/path strings as zero-dimensional arrays. Restore
    # Python strings for JSON provenance without changing physical arrays.
    for key, value in profile.items():
        if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in 'US':
            profile[key] = value.item()
    metadata = source/'outflow_metadata.json'
    if metadata.exists():
        profile.update(json.loads(metadata.read_text()))
    else:
        provenance = json.loads((source/'provenance.json').read_text())
        profile.update(p_winds_version=provenance['p_winds_version'],
                       warnings=provenance['solver_warnings'])
    return arrays, profile, error


def _ew_summary(wave, spectrum, covariance, window):
    """Integrate the same fixed vacuum band and its analytical noise covariance."""
    band = window_mask(wave, [window])
    if not np.all(np.isfinite(spectrum[band])):
        return dict(equivalent_width_mA=None, noise_error_mA=None)
    indices = np.flatnonzero(band)
    if len(indices) < 2:
        raise ValueError('Equivalent-width interval needs at least two common pixels')
    weights = np.zeros(len(wave))
    dx = np.diff(wave[indices])
    weights[indices[:-1]] += dx/2
    weights[indices[1:]] += dx/2
    value = np.sum(weights[band]*(1-spectrum[band]))*1e4
    error = np.sqrt(weights@(covariance@weights))*1e4
    return dict(equivalent_width_mA=float(value), noise_error_mA=float(error))


def combine_helium_nights(output: Path, sources: list[Path], science, settings):
    """Create individual/common-grid spectra, a coadd, light curves and maps."""
    from exoplore.core.helium_simulation import _write_npz
    nights = []
    common = stellar_grid = phases = None
    for source in sources:
        with np.load(source/'observations.npz') as data:
            obs = {k: data[k].copy() for k in data.files}
        with np.load(source/'transmission.npz') as data:
            result = {k: data[k].copy() for k in data.files}
        with np.load(source/'molecfit_calibration.npz') as data:
            calibration = {k: data[k].copy() for k in data.files}
        from exoplore.pipelines.czesla2024 import timed_exposure_selection
        config=json.loads((source/'run_config.json').read_text())
        science,_,_,_=timed_exposure_selection(science,obs['bjd_tdb'],
            config['observation']['exposure_time_seconds'])
        if common is None:
            common = result['planet_wave_nm'].copy()
            stellar_grid = result['stellar_wave_nm'].copy()
            phases = obs['phase'].copy()
        if not np.array_equal(phases, obs['phase']):
            raise ValueError('Phase-map coaddition requires matching exposure phases')
        covariance = carmenes_coadd_covariance(
            calibration['refined_wave_nm'], obs['corrected_flux'],
            obs['corrected_error'], obs['berv_kms'], obs['planet_rv_kms'],
            science, result)
        curve_covariance = carmenes_lightcurve_covariance(
            calibration['refined_wave_nm'], obs['corrected_flux'],
            obs['corrected_error'], obs['berv_kms'], obs['planet_rv_kms'],
            science, result)
        valid = np.isfinite(result['planet_coadd'])
        operator = interpolation_operator(result['planet_wave_nm'], common, valid)
        coverage = np.asarray(operator.sum(axis=1)).ravel() > .999999
        spectrum = operator@np.where(valid, result['planet_coadd'], 0)
        spectrum[~coverage] = np.nan
        covariance = operator@covariance@operator.T
        maps = []
        for row in result['stellar_transmission']:
            good = np.isfinite(row)
            op = interpolation_operator(result['stellar_wave_nm'], stellar_grid, good)
            covered = np.asarray(op.sum(axis=1)).ravel() > .999999
            value = op@np.where(good, row, 0)
            value[~covered] = np.nan
            maps.append(value)
        nights.append(dict(spectrum=spectrum, covariance=covariance,
                           lightcurve_covariance=curve_covariance,
                           map=np.asarray(maps), lightcurve=result['planet_lightcurve'],
                           ew=_ew_summary(common, spectrum, covariance,
                                          science.equivalent_width_window_nm)))
    spectra = np.asarray([n['spectrum'] for n in nights])
    complete = np.isfinite(spectra).all(axis=0)
    combined = np.mean(spectra, axis=0)
    combined[~complete] = np.nan
    count = len(nights)
    covariance = sum((n['covariance'] for n in nights),
                     csr_matrix((len(common), len(common))))/count**2
    error = np.sqrt(covariance.diagonal())
    error[~complete] = np.nan
    lightcurves = np.asarray([n['lightcurve'] for n in nights])
    curve_covariances = np.asarray([n['lightcurve_covariance'] for n in nights])
    curve_errors = np.sqrt(np.diagonal(curve_covariances, axis1=1, axis2=2))
    combined_curve_covariance = np.sum(curve_covariances, axis=0)/count**2
    combined_curve_error = np.sqrt(np.diag(combined_curve_covariance))
    maps = np.asarray([n['map'] for n in nights])
    _write_npz(output/'combined_transmission.npz', dict(
        planet_wave_nm=common, night_spectra=spectra, combined_spectrum=combined,
        combined_noise_error=error, coverage_count=np.isfinite(spectra).sum(axis=0),
        phase=phases, night_lightcurves=lightcurves,
        night_lightcurve_errors=curve_errors,
        combined_lightcurve_error=combined_curve_error,
        night_lightcurve_covariances=curve_covariances,
        combined_lightcurve_covariance=combined_curve_covariance,
        combined_lightcurve=np.mean(lightcurves, axis=0),
        stellar_wave_nm=stellar_grid, night_stellar_maps=maps,
        combined_stellar_map=np.mean(maps, axis=0)))
    save_npz(output/'combined_conditional_covariance.npz', covariance)
    for index, night in enumerate(nights, 1):
        save_npz(output/f'night_{index}_conditional_covariance.npz', night['covariance'])
    center = float(np.mean(science.helium_vacuum_lines_nm[1:]))
    band = abs(common-center) <= center/settings.instrumental_resolving_power/2
    weights = band.astype(float)/band.sum()
    summary = dict(
        combination='Arithmetic mean of independently prepared planetary spectra',
        grid='First night planet-frame vacuum nm; interpolation preserves masks/gaps',
        coverage='Combined pixels require all nights; coverage counts are retained',
        night_measurements=[n['ew'] for n in nights],
        combined_measurement=_ew_summary(common, combined, covariance,
                                        science.equivalent_width_window_nm),
        resolution_element_uncertainty_ppm=float(np.sqrt(weights@(covariance@weights))*1e6),
        uncertainty_scope='Analytical noise propagation including each independent night reference and interpolation covariance. Continuum/telluric fit parameters fixed; shared physical systematics excluded. No uncertainty resampling.')
    significance = None
    if settings.allart2023_significance is not None:
        from exoplore.pipelines.helium_significance import Allart2023SignificanceConfig, allart2023_significance
        choices = Allart2023SignificanceConfig(**settings.allart2023_significance)
        significance = [allart2023_significance(common, spectrum, choices) for spectrum in spectra]
        significance.append(allart2023_significance(common, combined, choices))
        summary['allart2023_significance'] = dict(nights=significance[:-1], combined=significance[-1])
    with (output/'combined_measurements.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    plot_helium_nights(output, common, spectra, combined, error, phases,
                      lightcurves, stellar_grid, maps, science, significance=significance,
                      lightcurve_errors=curve_errors,
                      combined_lightcurve_error=combined_curve_error)
    from exoplore.plotting.helium import write_combined_helium_maps
    write_combined_helium_maps(output, sources, science)
    if significance is not None:
        from exoplore.pipelines.helium_significance import plot_allan_noise
        plot_allan_noise(output/'plots', significance,
                         [f'Night {i+1}' for i in range(count)]+['Combined'])
    return summary


def plot_helium_nights(output, wave, spectra, combined, error, phases,
                      lightcurves, stellar_wave, maps, science, *, significance=None,
                      lightcurve_errors=None, combined_lightcurve_error=None):
    """Show every night and the combined spectrum/curve; four maps for three nights."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.get_cmap('tab10')
    if any((output/f'{stem}.{extension}').exists() for stem in ('helium_nights_comparison','helium_nights_maps') for extension in ('png','pdf')):
        raise FileExistsError('Preserve existing helium comparison figures; choose a new output')
    from exoplore.plotting.helium import plot_helium_summary
    from matplotlib.ticker import FuncFormatter
    display=dict(stellar_wave_nm=stellar_wave,stellar_transmission=np.mean(maps,axis=0),
                 planet_wave_nm=wave,planet_coadd=combined,planet_coadd_error=error,
                 planet_lightcurve=np.mean(lightcurves,axis=0),
                 planet_lightcurve_error=combined_lightcurve_error if combined_lightcurve_error is not None else np.full(len(phases),np.nan))
    config=json.loads((output/'run_config.json').read_text())
    berv=np.asarray(config['atmosphere']['helium']['berv_kms'])
    velocity=science.kp_kms*np.sin(2*np.pi*phases)
    for suffix in ('png','pdf'):
        plot_helium_summary(output/f'helium_nights_comparison.{suffix}',display,
                            phases,velocity,berv,science,night_spectra=spectra,
                            night_lightcurves=lightcurves,night_lightcurve_errors=lightcurve_errors,
                            significance=significance,synthetic=True)
    n = len(maps)+1
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4*n), constrained_layout=True,
                             sharex=True, sharey=True, squeeze=False)
    band = window_mask(stellar_wave, [science.plot_stellar_window_nm])
    data = np.concatenate([maps, np.mean(maps, axis=0)[None]], axis=0)
    limit = float(np.nanpercentile(abs(100*(maps[:, :, band]-1)), 98))
    for index, (axis, image) in enumerate(zip(axes[:, 0], data)):
        mesh = axis.pcolormesh(stellar_wave[band], phases, 100*(image[:, band]-1),
                               shading='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        velocity = science.kp_kms*np.sin(2*np.pi*phases)
        for line in science.helium_vacuum_lines_nm:
            axis.plot(line*(1+velocity/299792.458), phases, 'k--', lw=.7)
        axis.set_title(f'Night {index+1}' if index < len(maps) else f'Combined ({len(maps)} nights)')
        axis.set_ylabel('Orbital phase')
    axes[-1, 0].set_xlabel('Wavelength (μm)')
    axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value/1000:.4f}'))
    fig.colorbar(mesh, ax=axes[:, 0].tolist(), label='Transmission excess (%)')
    fig.savefig(output/'helium_nights_maps.png', dpi=180)
    fig.savefig(output/'helium_nights_maps.pdf')
    plt.close(fig)


def run_carmenes_multi_night(config):
    """Run n_nights with fixed conditions, optionally retaining completed nights."""
    from exoplore.core.carmenes_helium_simulation import run_carmenes_helium
    h = config.atmosphere.helium
    paths = [Path(p).resolve() for p in (h.completed_night_paths or [])]
    count = config.observation.n_nights
    if count < 2 or len(paths) > count or len(set(paths)) != len(paths):
        raise ValueError('Choose multiple nights and distinct completed-night inputs')
    if h.resume_simulation_path:
        raise ValueError('Use completed_night_paths for multi-night reuse, not resume_simulation_path')
    single = replace(config, observation=replace(config.observation, n_nights=1),
                     atmosphere=replace(config.atmosphere,
                                        helium=replace(h, completed_night_paths=None)))
    for path in paths:
        _verify_completed(path, single)
    seeds = []
    noise_hashes = []
    for path in paths:
        seeds.append(json.loads((path/'run_config.json').read_text())['atmosphere']['helium']['noise_seed'])
        with np.load(path/'simulated_observations.npz') as data:
            noise_hashes.append(hashlib.sha256((data['raw_flux']-data['expected_flux']).tobytes()).hexdigest())
    output = Path(config.paths.output_root)/config.planet.name/('helium_sunbather' if h.backend == 'sunbather' else 'helium_pwinds')
    output.mkdir(parents=True, exist_ok=False)
    with (output/'run_config.json').open('x') as stream:
        json.dump(asdict(config), stream, indent=2)
    while len(paths) < count:
        index = len(paths)
        seed = h.noise_seed if index == 0 else int(np.random.SeedSequence([h.noise_seed, index]).generate_state(1)[0])
        if seed in seeds:
            raise ValueError('Independent nights require distinct observation-noise seeds')
        seed_config = replace(single,
            paths=replace(single.paths, output_root=str(output/f'night_{index+1}')),
            atmosphere=replace(single.atmosphere, helium=replace(single.atmosphere.helium, noise_seed=seed)))
        print(f'  Helium night {index+1}/{count}: independent seed {seed}', flush=True)
        source = paths[0] if paths else None
        path = run_carmenes_helium(seed_config, expectation_source=source)
        paths.append(path.resolve())
        seeds.append(seed)
        with np.load(path/'simulated_observations.npz') as data:
            noise_hashes.append(hashlib.sha256((data['raw_flux']-data['expected_flux']).tobytes()).hexdigest())
    if h.noise_mode == 'gaussian' and len(set(noise_hashes)) != count:
        raise ValueError('Repeated observation-noise matrix cannot count as an independent night')
    manifest = dict(n_nights=count, night_paths=[str(p) for p in paths], noise_seeds=seeds,
                    observation_noise_sha256=noise_hashes, uncertainty_resamples=0,
                    observing_conditions='Identical supplied BERV, cadence, airmass, PWV and ETC prediction; idealized independent repetitions, not separate calendar epochs',
                    source_sha256={str(p/name):hashlib.sha256((p/name).read_bytes()).hexdigest()
                                   for p in paths for name in ('run_config.json','observations.npz','transmission.npz','provenance.json')})
    with (output/'night_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    summary = combine_helium_nights(output, paths, config.pipeline.carmenes_helium, h)
    print('  Completed helium multi-night comparison: '+str(output), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return output
