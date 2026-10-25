"""Notebook utilities for input paths, displays and run records.

These functions find local files, prepare a resolved configuration, display
saved products and record the settings and software used by an experiment."""
from __future__ import annotations

import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def show_details(title, text):
    """Display the complete diagnostic text in a collapsible section."""
    from IPython.display import HTML, display
    display(HTML(f"<details><summary>{html.escape(title)}</summary>"
                 f"<pre>{html.escape(text)}</pre></details>"))


def show_table(headers, rows):
    from IPython.display import HTML, display
    head = ''.join(f'<th>{html.escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>'
                                 for value in row) + '</tr>' for row in rows)
    display(HTML(f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'))


def resolve_inputs(repo, source_config, local_paths):
    """Copy the reference settings and resolve their local input-file paths."""
    required = ['numpy', 'scipy', 'astropy', 'matplotlib', 'petitRADTRANS', 'easychem', 'batman']
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Missing packages: {missing}. See docs/installation.md.')
    reference = json.loads(Path(source_config).read_text())
    resolved = copy.deepcopy(reference)
    external_keys = ['prt_input_data', 'phoenix_wave_file', 'phoenix_flux_file']
    env_keys = dict(zip(external_keys, ['pRT_input_data_path', 'EXOPLORE_PHOENIX_WAVE',
                                      'EXOPLORE_PHOENIX_FLUX']))
    local_configs = []
    for candidate in sorted((repo / 'configs').glob('*.json')):
        try:
            paths = json.loads(candidate.read_text()).get('paths', {})
        except (ValueError, OSError, AttributeError):
            continue
        if isinstance(paths, dict):
            local_configs.append((candidate, paths))
    path_sources = {}
    for key in external_keys:
        expected_name = Path(reference['paths'][key]).name
        candidates = [(local_paths.get(key), 'LOCAL_PATHS'),
                      (os.environ.get(env_keys[key]), env_keys[key])]
        candidates += [(paths.get(key), str(file.relative_to(repo)))
                       for file, paths in local_configs]
        for value, source in candidates:
            if not value:
                continue
            path = Path(value).expanduser().resolve()
            valid = (path.is_dir() if key == 'prt_input_data'
                     else path.is_file() and path.name == expected_name)
            if valid:
                resolved['paths'][key] = str(path)
                path_sources[key] = source
                break
        else:
            raise FileNotFoundError(f"Set LOCAL_PATHS['{key}'] or {env_keys[key]}.")
    for relative in [reference['planet']['parameter_file'],
                     reference['tellurics']['reference_telluric_file'],
                     'inputs/ANDES/HD189733b/ANDES_ETC_WAVE_SNR_YJHK_HD189733b.fits']:
        if not (repo / relative).is_file():
            raise FileNotFoundError(relative)
    return reference, resolved, path_sources


def differences(before, after, prefix=''):
    """Return the differences between two JSON configuration dictionaries."""
    result = {}
    for key in sorted(before.keys() | after.keys()):
        name = f'{prefix}.{key}' if prefix else key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            result.update(differences(old, new, name))
        elif old != new:
            result[name] = {'reference': old, 'run': new}
    return result


def configure_outputs(resolved, save_matrices, noise_seed_override=None):
    """Set notebook output flags and the optional noise-seed override."""
    from dataclasses import fields
    from exoplore.config.models import OutputConfig
    result = copy.deepcopy(resolved)
    result['output'] = {field.name: bool(save_matrices) for field in fields(OutputConfig)}
    if noise_seed_override is not None:
        if (isinstance(noise_seed_override, bool) or not isinstance(noise_seed_override, int)
                or not 0 <= noise_seed_override < 2**32):
            raise ValueError('NOISE_SEED_OVERRIDE must be None or an integer from 0 to 2**32-1.')
        result['noise']['noise_seed'] = noise_seed_override
    return result


def create_run(repo, reference, resolved, save_matrices, noise_seed_override=None):
    """Create a run directory and write its resolved configuration copy."""
    result = configure_outputs(resolved, save_matrices, noise_seed_override)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_root = repo / 'outputs/notebooks/01_first_ANDES_simulation' / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    result['paths']['output_root'] = str(run_root)
    overrides = differences(reference, result)
    config_path = run_root / 'resolved_config.json'
    config_path.write_text(json.dumps(result, indent=2) + '\n')
    (run_root / 'overrides.json').write_text(json.dumps(overrides, indent=2) + '\n')
    return result, run_root, config_path, overrides


def show_atmosphere(reference):
    """Display the configured atmospheric regions and search template in a table."""
    atmosphere = reference['atmosphere']
    print('Injected species:', ', '.join(atmosphere['planet_model']['species']))
    print('Search species:', ', '.join(atmosphere['ccf_template']['species']))
    print('C/O:', atmosphere['planet_model']['carbon_to_oxygen_ratio'])
    print('Metallicity [dex]:', atmosphere['planet_model']['metallicity_wrt_solar'])
    region_names = ['morning_day', 'morning_night', 'evening_day', 'evening_night']
    show_table(['Configured region', 'Equilibrium temperature [K]', 'Wind [km/s]', 'Chemistry'],
               [[name, atmosphere[name]['equilibrium_temperature_K'],
                 atmosphere[name]['wind_velocity_kms'],
                 'EasyChem' if atmosphere[name]['use_easychem'] else 'Configured abundances']
                for name in region_names])


def preview_simulation(simulator, config):
    """Display the simulator summary, wavelength grid and exposure sequence.

The exposure preview uses the public timing function called by the simulator."""
    from exoplore.instruments import load_instrument, get_WaveGrid
    from exoplore.planets import load_planet
    from exoplore.observation import get_event
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        print(simulator.summary())
        instrument = load_instrument(config)
        grid = get_WaveGrid(config.instrument, config.observation, instrument, instrument.n_orders_total)
        orders = config.instrument.order_indices or list(range(instrument.n_orders_total))
        print('Selected zero-based orders:', orders)
        for order in orders:
            wave = grid[0][order]
            print(f'  {order}: {wave.min():.4f}–{wave.max():.4f} micrometres')
        planet = load_planet(config.planet.parameter_file)
        exposure_count = len(get_event(config.observation, planet, grid[4])[0])
    show_table(['Experiment', 'Configured value'], [
        ['Target', config.planet.name], ['Instrument', config.instrument.name],
        ['Orders', len(orders)], ['Exposures (EXoPLORE timing preview)', exposure_count],
        ['Template', ', '.join(config.atmosphere.ccf_template.species)],
        ['Pipeline', config.pipeline.name], ['Random seed', config.noise.noise_seed],
        ['Output directory', simulator.output_dirs(create=False)['root']]])
    show_details('Complete EXoPLORE summary and wavelength-grid preview', text.getvalue())
    return orders, exposure_count


@contextlib.contextmanager
def capture_run_log(log_path):
    """Write Python stdout, stderr and warnings to the run log."""
    with Path(log_path).open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield


def show_run_completion(log_path, seconds, output_dir):
    print(f'Simulation completed in {seconds:.1f} s ({seconds / 60:.2f} min).')
    print('Log:', log_path)
    lines = Path(log_path).read_text(errors='replace').splitlines()
    notices = [line for line in lines if any(word in line for word in
               ['WARNING', 'RuntimeWarning', '[mask]', 'NOTE:'])]
    print(f'Warning/notice lines: {len(notices)}. Inspect run.log before changing scientific parameters.')
    for line in notices[:8]:
        print(line[:200])
    print('Output:', output_dir)


def load_product(matrix_dir, run_name, prefix):
    import numpy as np
    with np.load(Path(matrix_dir) / f'{prefix}_{run_name}.npz', allow_pickle=False) as archive:
        return archive['a'].copy()


def show_saved_pdf(path, notebook_dir):
    """Render a saved PDF inline at scale 1.2."""
    from IPython.display import Image, Markdown, display
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Expected diagnostic missing: {path}. Inspect run.log.')
    display(Markdown(f'[Open original PDF]({os.path.relpath(path, notebook_dir)})'))
    try:
        import fitz
    except ImportError:
        print('Open the linked PDF; inline preview requires the optional PyMuPDF package.')
    else:
        with fitz.open(path) as document:
            display(Image(data=document[0].get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes('png')))


def load_detection_map(path):
    """Read the saved map and velocity axes and check their dimensions."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        snr_map = archive['ccf_tot_sn'].copy()
        v_rest = archive['v_rest'].copy()
        kp_grid = archive['kp_range'].copy()
        convention = str(archive['v_rest_convention'])
    assert snr_map.shape == (v_rest.size, kp_grid.size)
    assert np.isfinite(snr_map).all()
    vi, ki = np.unravel_index(np.argmax(snr_map), snr_map.shape)
    return SimpleNamespace(snr_map=snr_map, v_rest=v_rest, kp_grid=kp_grid,
                           convention=convention, kp_index=ki,
                           peak_snr=float(snr_map[vi, ki]), peak_kp=float(kp_grid[ki]),
                           peak_vrest=float(v_rest[vi]))


def plot_detection_map(result, injected_kp, orders=None):
    """Plot the saved detection map with its selected orders and orbital reference."""
    from exoplore.plotting.kpvsys import plot_kp_vsys_map
    fig = plot_kp_vsys_map(result.kp_grid, result.v_rest, result.snr_map.T,
                          kp_true=injected_kp, vsys_true=0.0,
                          xlabel=r'$V_{\rm rest}$ offset (km/s)',
                          title=('H₂O recovery: full 76-order reference experiment'
                                 if orders is None or list(orders) == list(range(76))
                                 else f'H₂O recovery: {len(orders)} selected orders'))
    # Use black for the nominal orbital reference crosshair in this tutorial.
    for reference_line in fig.axes[0].lines:
        reference_line.set_color('black')
    fig.axes[1].set_ylabel('Map S/N (off-peak normalisation)')
    fig.axes[0].set_xlim(-70, 70)
    fig.axes[0].set_ylim(0, 320)
    return fig


def plot_velocity_slices(v_rest, peak_slice, nominal_slice, peak_kp):
    """Plot velocity slices through the peak and configured Kp rows."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(v_rest, peak_slice, label=f'Peak Kp row ({peak_kp:.0f} km/s)')
    ax.plot(v_rest, nominal_slice, label='Nearest configured Kp row', alpha=0.8)
    ax.axvline(0, color='0.5', ls='--')
    ax.set(xlim=(-50, 50), xlabel='Vrest offset (km/s)', ylabel='Map S/N')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def write_reproducibility(repo, source_config, reference, config_path, resolved,
                          run_root, overrides, path_sources, seconds, orders,
                          save_matrices, result):
    """Record the configuration, input hashes, software versions and run summary."""
    from exoplore import __version__
    # Compare the parsed source configuration with the recorded reference.
    assert json.loads(Path(source_config).read_text()) == reference
    sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    versions = {'exoplore': __version__}
    for package in ['numpy', 'scipy', 'astropy', 'matplotlib', 'petitRADTRANS',
                    'easychem', 'batman-package', 'nbformat', 'nbclient']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'distribution metadata unavailable'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], text=True)
    record = {
        'utc_run_id': run_root.name, 'full_reference_orders': True, 'orders': orders,
        'save_matrices': save_matrices, 'noise_seed': resolved['noise']['noise_seed'],
        'throughput_seed': resolved['observation'].get('throughput_seed'),
        'randomness_note': ('With both the Gaussian-noise and throughput seeds fixed, '
                            'the stochastic realisation is reproducible. End-to-end '
                            'floating-point results may still differ at machine-precision '
                            'level, so bitwise-identical output is not guaranteed.'),
        'exoplore_version': __version__, 'simulation_seconds': seconds,
        'python': sys.version, 'git_revision': revision, 'tracked_status': tracked,
        'versions': versions, 'source_config': str(source_config.relative_to(repo)),
        'resolved_config': str(config_path), 'output_directory': str(run_root),
        'source_config_integrity_check': 'parsed_JSON_equality_against_initial_reference',
        'source_config_sha256': sha256(source_config),
        'resolved_config_sha256': sha256(config_path),
        'planet_parameters_sha256': sha256(repo / reference['planet']['parameter_file']),
        'notebook_helper_sha256': sha256(Path(__file__)),
        'external_path_sources': path_sources, 'overrides': overrides,
        'peak': {'map_snr': result.peak_snr, 'kp_kms': result.peak_kp,
                 'vrest_kms': result.peak_vrest},
    }
    record_path = run_root / 'reproducibility.json'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    files = [p for p in run_root.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    show_table(['Reproducibility', 'Recorded value'], [
        ['EXoPLORE version', __version__], ['Git revision', revision],
        ['Gaussian / throughput seeds', f"{record['noise_seed']} / {record['throughput_seed']}"],
        ['Source config', source_config],
        ['Resolved config', config_path], ['Runtime', f'{seconds:.1f} s ({seconds / 60:.2f} min)'],
        ['Output directory', run_root], ['SAVE_MATRICES', save_matrices],
        ['Output files / size', f'{len(files)} files; {size / 1e6:.2f} MB ({size / 1024**2:.2f} MiB)'],
        ['Reference integrity check', 'Parsed JSON equals the initially loaded reference'],
        ['JSON record', record_path]])
    show_details('Complete reproducibility record', json.dumps(record, indent=2))
    return record


def export_combined_ccf(simulator, path):
    """Export the completed CCF, observing axes and map information for the animation.

This function reads the simulator state into a compact NPZ. Other animation
utilities read that exported bundle."""
    import numpy as np
    state = simulator._state
    config = simulator.config
    orders = config.instrument.order_indices or list(range(len(state['per_order_results'])))
    assert len(state['per_order_results']) == len(orders)
    ccf = np.asarray(state['ccf_complete'])
    phase = np.asarray(state['phase']).squeeze()
    velocity = np.asarray(state['v_ccf'])
    selected = np.asarray(state['with_signal'], dtype=int)
    assert ccf.shape == (velocity.size, phase.size)
    assert config.observation.n_nights == 1 and not config.observation.significant_eccentricity
    assert selected.ndim == 1 and selected.size > 0
    assert np.isfinite(ccf).all()
    metadata = {
        'axis_order': ['earth_frame_velocity', 'exposure'],
        'velocity_units': 'km/s', 'phase_units': 'orbital cycles; zero at mid-transit',
        'track_equation': 'Kp*sin(2*pi*phase) + Vsys - BERV + Vrest',
        'systemic_velocity_kms': float(state['planet'].systemic_velocity_kms),
        'significant_eccentricity': False, 'order_indices': list(map(int, orders)),
        'gaussian_noise_seed': config.noise.noise_seed,
        'throughput_base_seed': config.observation.throughput_seed,
        'normalisation': 'per-Kp off-peak mean/std; exclusion centred on row maximum',
        'snr_exclude_kms': config.cross_correlation.snr_exclude_kms,
        'source': 'completed ExoploreSimulator.run(); combined CCF, no synthetic replacement',
    }
    path = Path(path)
    np.savez_compressed(path, ccf=ccf, phase=phase, velocity=velocity,
                        with_signal=selected, berv=np.asarray(state['berv']),
                        v_rest=np.asarray(state['v_rest']), kp_grid=np.asarray(state['kp_range']),
                        metadata_json=json.dumps(metadata))
    return path


def load_ccf_bundle(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive if key != 'metadata_json'}
        data['metadata'] = json.loads(str(archive['metadata_json']))
    return SimpleNamespace(**data)


def prepare_animation_data(diagnostics, result, nominal_kp):
    """Sample the CCF with EXoPLORE's velocity-shifting and map-normalization functions."""
    import numpy as np
    from exoplore.ccf.compute import get_shifted_ccf_matrix, get_max_CCF_peak
    from exoplore.observation.velocity import get_V
    d = diagnostics
    np.testing.assert_array_equal(d.v_rest, result.v_rest)
    np.testing.assert_array_equal(d.kp_grid, result.kp_grid)
    # A small tour samples wrong slopes, wrong offsets and the recovered region.
    requested = [(-150, 25), (0, 25), (300, 25),
                 (result.peak_kp, 25), (result.peak_kp, -25),
                 (result.peak_kp, 0), (nominal_kp, result.peak_vrest),
                 (result.peak_kp, result.peak_vrest)]
    trials = []
    for kp, offset in requested:
        ki = int(np.argmin(abs(result.kp_grid - kp)))
        vi = int(np.argmin(abs(result.v_rest - offset)))
        if (ki, vi) not in trials:
            trials.append((ki, vi))
    kp_indices = sorted({ki for ki, vi in trials})
    options = {'significant_eccentricity': False,
               'CCF_SNR_exclude': d.metadata['snr_exclude_kms']}
    step = float(np.median(np.diff(d.velocity)))
    half_width = (len(d.v_rest) - 1) // 2
    shifted = get_shifted_ccf_matrix(
        options, d.with_signal, d.v_rest, d.velocity, result.kp_grid[kp_indices],
        d.phase, d.metadata['systemic_velocity_kms'], d.berv,
        half_width, step, d.ccf)
    coadded = np.nansum(shifted, axis=1)
    normalised = get_max_CCF_peak(options, coadded, d.v_rest,
                                  result.kp_grid[kp_indices])[0]
    expected = result.snr_map[:, kp_indices]
    np.testing.assert_allclose(normalised, expected, rtol=1e-9, atol=1e-9)
    error = float(np.max(abs(normalised - expected)))
    samples, tracks = [], []
    for ki, vi in trials:
        local_index = kp_indices.index(ki)
        samples.append(shifted[vi, :, local_index].copy())
        tracks.append(get_V(result.kp_grid[ki], d.phase, d.berv,
                            d.metadata['systemic_velocity_kms'], result.v_rest[vi])[d.with_signal])
    return SimpleNamespace(trials=trials, samples=samples, tracks=tracks,
                           normalisation_max_abs_error=error)


def animate_ccf_search(diagnostics, result, nominal_kp, directory):
    """Render a compact GIF from this run's CCF and saved normalised map.

    Partial sums use raw CCF units. A map S/N is displayed only after all
    selected exposures have been combined. The complete map appears last.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    tour = prepare_animation_data(diagnostics, result, nominal_kp)
    d = diagnostics
    phase = d.phase[d.with_signal]
    n = phase.size
    frames = [(i, max(1, int(round(fraction * n))), False)
              for i in range(len(tour.trials)) for fraction in [0.25, 0.60, 1.0]]
    frames += [(len(tour.trials) - 1, n, True)] * 4
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=90)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.24, top=0.82, wspace=0.60)
    visible = abs(d.velocity) <= 70
    data = d.ccf[visible][:, d.with_signal].T
    lo, hi = np.nanpercentile(data, [1, 99.5])
    image = left.pcolormesh(d.velocity[visible], phase, data, shading='auto',
                           cmap='viridis', vmin=lo, vmax=hi, rasterized=True)
    left.set(xlabel='Earth-frame velocity (km/s)', ylabel='Orbital phase',
             title='Sample the real CCF trail', xlim=(-70, 70))
    fig.colorbar(image, ax=left, fraction=0.047, pad=0.02, label='CCF')
    track, = left.plot([], [], color='white', lw=1.3, ls='--')
    visited, = left.plot([], [], color='#ffb000', lw=2.4)
    limit = max(abs(np.nanmin(result.snr_map)), abs(np.nanmax(result.snr_map)))
    norm = Normalize(-limit, limit)
    blank = np.full(result.snr_map.T.shape, np.nan)
    heatmap = right.imshow(blank, origin='lower', aspect='auto', cmap='RdBu_r', norm=norm,
                           extent=[result.v_rest[0]-.5, result.v_rest[-1]+.5,
                                   result.kp_grid[0]-.5, result.kp_grid[-1]+.5])
    right.set(xlabel='Vrest offset (km/s)', ylabel='Kp (km/s)',
              title='Completed trial values', xlim=(-45, 45), ylim=(-200, 320))
    fig.colorbar(heatmap, ax=right, fraction=0.047, pad=0.02, label='Map S/N')
    points = right.scatter([], [], c=[], cmap='RdBu_r', norm=norm, marker='s', s=85,
                           edgecolors='0.2', linewidths=.4)
    cursor, = right.plot([], [], 'o', ms=12, mfc='none', mec='#ffb000', mew=2)
    title = fig.suptitle('', fontsize=12)
    score = fig.text(.5, .115, '', ha='center', fontsize=10)
    note = fig.text(.5, .055, 'Orange: sampled exposures. Completed points use the saved off-peak normalisation.',
                    ha='center', fontsize=8)

    def update(frame):
        index, count, final = frame
        ki, vi = tour.trials[index]
        kp, offset = result.kp_grid[ki], result.v_rest[vi]
        track.set_data(tour.tracks[index], phase)
        visited.set_data(tour.tracks[index][:count], phase[:count])
        completed = list(range(index)) + ([index] if count == n else [])
        coords = [[result.v_rest[tour.trials[j][1]], result.kp_grid[tour.trials[j][0]]]
                  for j in completed]
        values = [result.snr_map[tour.trials[j][1], tour.trials[j][0]] for j in completed]
        points.set_offsets(np.asarray(coords).reshape(-1, 2))
        points.set_array(np.asarray(values))
        cursor.set_data([offset], [kp])
        raw_sum = float(np.nansum(tour.samples[index][:count]))
        title.set_text(f'Trial Kp = {kp:.0f} km/s; Vrest = {offset:+.0f} km/s')
        text = f'{count}/{n} exposures: accumulated CCF sum = {raw_sum:.3f}'
        if count == n:
            text += f'     |     map S/N = {result.snr_map[vi, ki]:.2f}'
        score.set_text(text)
        heatmap.set_data(result.snr_map.T if final else blank)
        right.set_title('Full saved detection map' if final else 'Completed trial values')
        return track, visited, points, cursor, heatmap, title, score

    animation = FuncAnimation(fig, update, frames=frames, interval=1000/3,
                              blit=False, repeat=True)
    gif = directory / '01_ccf_to_kp_vrest.gif'
    animation.save(gif, writer=PillowWriter(fps=3), dpi=90)
    update(frames[-1])
    poster = directory / '01_ccf_to_kp_vrest_static.png'
    fig.savefig(poster, dpi=120)
    plt.close(fig)
    with Image.open(gif) as saved:
        actual_frames = saved.n_frames
        duration_ms = sum((saved.seek(i) or saved.info.get('duration', 0))
                          for i in range(saved.n_frames))
    record = {'gif_bytes': gif.stat().st_size, 'frame_count': actual_frames,
              'duration_ms': duration_ms, 'trial_count': len(tour.trials),
              'normalisation_max_abs_error': tour.normalisation_max_abs_error,
              'ccf_source': 'combined_ccf.npz from this execution',
              'peak': {'snr': result.peak_snr, 'kp': result.peak_kp, 'vrest': result.peak_vrest}}
    (directory / 'animation_record.json').write_text(json.dumps(record, indent=2)+'\n')
    return SimpleNamespace(gif=gif, poster=poster, record=record)


def prepare_tutorial(repo, save_matrices=False,
                     source_config="configs/hd189733b_andes_transit_clean.json"):
    """Resolve local scientific inputs and create a fresh output directory.

The run configuration contains the resolved input paths, output location and
optional saving settings. Missing inputs raise an error linking to the guide."""
    repo = Path(repo).resolve()
    os.chdir(repo)  # Reference input paths are relative to the repository root.
    sys.path.insert(0, str(repo / 'src'))
    os.environ.setdefault('MPLCONFIGDIR', str(repo / 'outputs/notebook_cache/matplotlib'))
    source_config = repo / Path(source_config)
    reference, resolved, path_sources = resolve_inputs(repo, source_config, {})
    resolved, run_root, config_path, overrides = create_run(
        repo, reference, resolved, save_matrices)
    return SimpleNamespace(source_config=source_config, reference=reference,
                           resolved=resolved, run_root=run_root,
                           config_path=config_path, overrides=overrides,
                           path_sources=path_sources)


def show_animation(gif_path):
    """Embed the existing GIF bytes, including a fallback for HTML exporters."""
    import base64
    from IPython.display import display
    encoded = base64.b64encode(Path(gif_path).read_bytes()).decode('ascii')
    display({'image/gif': encoded,
             'text/html': f'<img src="data:image/gif;base64,{encoded}" '
                          'alt="CCF tracks and Kp–Vrest map" />'}, raw=True)


def save_working_config(experiment):
    """Save the notebook settings to the run configuration copy."""
    from exoplore.config import SimulationConfig
    # Validate through EXoPLORE before writing the edited working configuration.
    SimulationConfig.from_dict(experiment.resolved)
    experiment.overrides = differences(experiment.reference, experiment.resolved)
    experiment.config_path.write_text(json.dumps(experiment.resolved, indent=2) + '\n')
    (experiment.run_root / 'overrides.json').write_text(
        json.dumps(experiment.overrides, indent=2) + '\n')


def prepare_injection_demo(repo, experiment, factor=10.0, order_index=0):
    """Create settings for a separate observation.scale_injection illustration.

Copy the resolved experiment, select the displayed order and set its injection
factor, output directory and optional-array saving flags."""
    from exoplore.config import SimulationConfig
    resolved = copy.deepcopy(experiment.resolved)
    resolved['instrument']['order_indices'] = [order_index]
    resolved['observation']['scale_injection'] = factor
    SimulationConfig.from_dict(resolved)
    resolved, run_root, config_path, overrides = create_run(
        Path(repo), experiment.reference, resolved, save_matrices=False)
    return SimpleNamespace(source_config=experiment.source_config,
                           reference=experiment.reference, resolved=resolved,
                           run_root=run_root, config_path=config_path,
                           overrides=overrides, path_sources=experiment.path_sources)
