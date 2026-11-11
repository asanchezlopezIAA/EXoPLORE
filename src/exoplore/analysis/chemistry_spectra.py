"""Transmission spectra from the completed Tutorial 15 chemistry columns."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from exoplore.analysis.chemistry import comparison_settings, equilibrium_reference
from exoplore.atmosphere.pressure import create_log_pressure_grid
from exoplore.atmosphere.temperature import calculate_temperature_structure
from exoplore.atmosphere.prt import compute_transmission_spectrum, convolve
from exoplore.atmosphere.vulcan import regrid_composition
from exoplore.planets import load_planet

LINE_SPECIES = ('H2O', 'CO', 'CO2', 'CH4', 'NH3', 'HCN', 'OH')
CASES = ('easychem', 'thermal_control', 'mixing', 'photochemistry')


def _read_column(config, region_name, output):
    """Check that the saved chemistry belongs to the requested atmosphere."""
    output = Path(output)
    report = json.loads((output/'comparison.json').read_text())
    with np.load(output/'comparison.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    region = getattr(config.atmosphere, region_name)
    if region is None or region.chemistry_backend != 'vulcan':
        raise ValueError('Select the VULCAN region used for Step 2')
    settings = comparison_settings(region.chemistry.vulcan)
    planet_file = config.planet.parameter_file or str(
        Path(config.paths.planet_parameter_dir)/(config.planet.name+'.json'))
    planet = load_planet(planet_file)
    p = create_log_pressure_grid(region.pressure_min_bar, region.pressure_max_bar,
                                region.pressure_grid_size)
    gravity = 6.67430e-8*planet.M_pl/planet.R_pl**2
    t = calculate_temperature_structure(planet.t_int_K, p, gravity, region.isothermal,
        region.isothermal_temperature_K, region.equilibrium_temperature_K, region.two_point,
        region.two_point_pressures_bar, region.two_point_temperatures_K,
        region.kappa_ir, region.gamma_guillot, None)
    if (report['planet'] != planet.name or report['region'] != region_name
            or not np.array_equal(p, arrays['pressure_bar'])
            or not np.array_equal(t, arrays['temperature_K'])
            or report['elemental_abundances'] != region.chemistry.vulcan.elemental_abundances):
        raise ValueError('The config differs from the saved abundance comparison. Repeat Step 2 first.')
    return region, settings, planet, p, t, gravity, arrays, report


def prepare_spectral_gases(config, region_name, output):
    """Load accepted VULCAN columns and rebuild the matched equilibrium gas.

    All gases contribute to mean molecular weight and the mass denominator.
    The opacity species keep their fractions of that complete mixture.
    This function reads the converged columns without launching VULCAN.
    """
    region, settings, planet, p, t, gravity, arrays, report = _read_column(
        config, region_name, output)
    requested = ['H2', 'He', *LINE_SPECIES]
    equilibrium, _, gas = equilibrium_reference(p, t,
        region.chemistry.vulcan.elemental_abundances,
        Path(config.paths.vulcan_directory)/'thermo/all_compose.txt',
        arrays['species'].tolist(), return_gas=True)
    if not np.allclose(equilibrium, arrays['easychem'], rtol=1e-10, atol=0):
        raise ValueError('The equilibrium solution differs from Step 2. Repeat the abundance comparison.')
    names = gas['species'].tolist()
    mu = gas['mole_fractions'] @ gas['molecular_masses']
    mass = gas['mole_fractions']*gas['molecular_masses'][None, :]/mu[:, None]
    columns = {'easychem': ({name: mass[:, names.index(name)] for name in requested}, mu)}
    cache = Path(config.paths.vulcan_cache_dir or Path(config.paths.output_root)/'chemistry/vulcan')
    for label, case in settings.items():
        work = cache/report['cases'][label]['cache_keys'][-1]
        manifest = json.loads((work/'manifest.json').read_text())
        if manifest['end_case'] != 1 or any(abs(v) > case.max_atom_loss
                                            for v in manifest['atom_loss'].values()):
            raise ValueError(f'The {label} column lacks accepted convergence and elemental checks')
        data_path = work/'gas_profile.npz'
        if (manifest['request']['settings'] != asdict(case)
                or hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest['gas_profile_sha256']):
            raise ValueError(f'The saved {label} chemistry inputs or gas profile have changed')
        # The radius record connects VULCAN's hydrostatic reference to the
        # planet radius used by pRT. Check it before reusing the gas profile.
        alignments = [json.loads(path.read_text()) for path in work.glob('radius_alignment_*.json')]
        if not any(a['reference_pressure_bar'] == region.reference_pressure_bar
                   and a['reference_radius_cm'] == planet.R_pl
                   and a['planet_mass_g'] == planet.M_pl for a in alignments):
            raise ValueError('The saved pressure-radius alignment differs from this atmospheric region')
        if manifest['request']['physical']['stellar_radius_cm'] != planet.R_star:
            raise ValueError('The host-star radius differs from the saved chemistry calculation')
        with np.load(data_path, allow_pickle=False) as native:
            columns[label] = regrid_composition(p, t, native['pressure_bar'],
                native['temperature_K'], native['species'].tolist(), native['mole_fractions'],
                native['molecular_masses'], requested)
        # Confirm that the plotted abundances and the gases entering pRT
        # describe the same accepted column.
        mf, mmw = columns[label]
        with np.load(data_path, allow_pickle=False) as native:
            masses = dict(zip(native['species'].tolist(), native['molecular_masses']))
        for name in LINE_SPECIES:
            vmr = mf[name]*mmw/masses[name]
            if not np.allclose(vmr, arrays[label][arrays['species'].tolist().index(name)],
                               rtol=1e-12, atol=0):
                raise ValueError(f'The {label} {name} profile differs from Step 2')
    return region, planet, p, t, gravity, columns


def wavelength_chunks(low, high, resolving_power, log_width=.025):
    """Partition the wavelength range, including eight Gaussian sigma at each edge."""
    if not 0 < low < high or not np.isfinite([low, high, resolving_power, log_width]).all():
        raise ValueError('Supply finite positive wavelength boundaries and resolving power')
    if resolving_power <= 0 or log_width <= 0:
        raise ValueError('Resolving power and chunk width must be positive')
    count = max(1, int(np.ceil(np.log(high/low)/log_width)))
    edges = np.geomspace(low, high, count+1)
    margin = 8/(2*np.sqrt(2*np.log(2))*resolving_power)
    return [(a, b, a*np.exp(-margin), b*np.exp(margin))
            for a, b in zip(edges[:-1], edges[1:])]


def _radtrans(pressures, low, high, input_data):
    """Load native line-by-line opacities for one padded wavelength interval."""
    from petitRADTRANS.radtrans import Radtrans
    return Radtrans(pressures=pressures, line_species=list(LINE_SPECIES),
        rayleigh_species=['H2', 'He'], gas_continuum_contributors=['H2--H2', 'H2--He'],
        wavelength_boundaries=[low, high], line_opacity_mode='lbl',
        path_input_data=str(input_data))


def compare_transmission_spectra(config, region_name, output, *, input_data=None,
                                 low=.6, high=5., resolving_power=3500., log_width=.025):
    """Calculate native line-by-line transit depths and broaden to the requested R.

    Chunks limit opacity memory. The convolution uses the padded interval,
    then retains its central part. Output samples are separated by one third
    of the Gaussian FWHM in log wavelength. Completed chunks can be reused.
    """
    started = time.perf_counter()
    region, planet, p, t, gravity, columns = prepare_spectral_gases(config, region_name, output)
    input_data = Path(input_data or config.paths.prt_input_data).expanduser().resolve()
    if not input_data.is_dir():
        raise ValueError('Set paths.prt_input_data or --prt-input-data to the installed pRT opacity directory')
    chunks = wavelength_chunks(low, high, resolving_power, log_width)
    wave = np.geomspace(low, high, int(np.ceil(np.log(high/low)*3*resolving_power))+1)
    result = {name: np.full(len(wave), np.nan) for name in CASES}
    # Identify the opacity files as well as the gas profiles when reusing
    # a chunk. Modification times and sizes detect local table replacements.
    files = sorted((input_data/'opacities').rglob('*.h5'))
    opacity_files = [(str(f.relative_to(input_data)), f.stat().st_size, f.stat().st_mtime_ns)
                     for f in files if f.parts[-3] in LINE_SPECIES
                     or 'collision_induced_absorptions' in f.parts]
    identity = dict(low=low, high=high, resolving_power=resolving_power, log_width=log_width,
        reference_pressure_bar=region.reference_pressure_bar, radius_cm=planet.R_pl,
        stellar_radius_cm=planet.R_star, gravity_cgs=gravity,
        cloud_pressure_bar=region.cloud_pressure_bar, opacity_files=opacity_files,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for vector in (p, t, *[a for mf, mu in columns.values() for a in (*mf.values(), mu)]):
        digest.update(np.asarray(vector, dtype='<f8').tobytes())
    key = digest.hexdigest()
    chunk_root = Path(output)/'transmission_chunks'/key
    chunk_root.mkdir(parents=True, exist_ok=True)
    fresh = 0
    for index, (left, right, padded_left, padded_right) in enumerate(chunks):
        selected = (wave >= left) & ((wave <= right) if index == len(chunks)-1 else (wave < right))
        path = chunk_root/f'{index:03d}.npz'
        if path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                for name in CASES:
                    result[name][selected] = cached[name]
            continue
        rt = _radtrans(p, padded_left, padded_right, input_data)
        saved = {}
        for name, (mf, mu) in columns.items():
            native_wave, depth = compute_transmission_spectrum(rt, t, mf, mu,
                gravity, planet.R_pl, planet.R_star, region.reference_pressure_bar,
                cloud_pressure_bar=region.cloud_pressure_bar)
            if not np.isfinite(depth).all() or np.any(depth <= 0) or np.any(np.diff(native_wave) <= 0):
                raise ValueError('pRT returned invalid transit depths or wavelength ordering')
            if native_wave.min() > left or native_wave.max() < right:
                raise ValueError('pRT wavelength coverage does not span the requested chunk')
            broadened = convolve(native_wave, depth, resolving_power)
            saved[name] = np.interp(wave[selected], native_wave, broadened)
            result[name][selected] = saved[name]
        np.savez_compressed(path, **saved)
        fresh += 1
        del rt
        gc.collect()
        print(f'Transmission chunk {index+1}/{len(chunks)} complete', flush=True)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('The transmission comparison has incomplete wavelength coverage')
    report = dict(identity, planet=planet.name, region=region_name, chunks=len(chunks),
        fresh_chunks=fresh, reused_chunks=len(chunks)-fresh, wall_seconds=time.perf_counter()-started,
        samples=len(wave), cache_key=key, line_species=list(LINE_SPECIES),
        thermal_control_max_abs_ppm=float(np.max(abs(result['thermal_control']-result['easychem']))*1e6),
        peak_abs_difference_ppm={name:float(np.max(abs(result[name]-result['easychem']))*1e6)
                                 for name in ('mixing','photochemistry')})
    arrays = dict(wavelength_um=wave, **result)
    report_path = Path(output)/'transmission_spectra.json'
    if fresh == len(chunks):
        report['fresh_wall_seconds'] = report['wall_seconds']
    elif report_path.is_file():
        previous = json.loads(report_path.read_text())
        if previous.get('cache_key') == key:
            report['fresh_wall_seconds'] = previous.get('fresh_wall_seconds')
    np.savez_compressed(Path(output)/'transmission_spectra.npz', **arrays)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return arrays, report
