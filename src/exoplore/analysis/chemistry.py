"""Compare gas-phase equilibrium, vertical mixing and photochemistry.

The observing simulator keeps its chemistry prescriptions. This analysis uses
one explicit elemental inventory for both packages and excludes condensates
and ions from the equilibrium reference to match the neutral VULCAN experiment.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from exoplore.atmosphere.models import RadiativeTransferInputs
from exoplore.atmosphere.pressure import create_log_pressure_grid
from exoplore.atmosphere.temperature import calculate_temperature_structure
from exoplore.atmosphere.vulcan import run_vulcan
from exoplore.config.chemistry import EddyDiffusionConfig
from exoplore.planets import load_planet

SPECIES = ('H2', 'H', 'H2O', 'CO', 'CO2', 'CH4', 'NH3', 'HCN', 'OH')


def equilibrium_reference(pressures, temperatures, elemental_abundances, composition_file,
                          species=SPECIES):
    """Calculate EasyChem gas VMRs using the supplied element/H number ratios.

    Every selected neutral gas enters the VMR denominator. The plotted species
    retain their fractions of that complete gas mixture. EasyChem's usual
    observing calculation is independent of this comparison reference.
    """
    import easychem.easychem as ec

    table = np.genfromtxt(composition_file, names=True, dtype=None, encoding='utf-8')
    rows = {str(row['species']): row for row in np.atleast_1d(table)}
    atoms = list(elemental_abundances)
    allowed = set(atoms)
    elements = set(table.dtype.names) - {'species', 'mass', 'e'}
    reactants, canonical = [], []
    for label in ec.ExoAtmos().reactants:
        # EasyChem distinguishes isomers with comma suffixes. The supplied
        # acetylene entry corresponds to C2H2 in the neutral VULCAN network.
        name = label.split(',')[0]
        if '(' in label or '+' in label or '-' in label or name not in rows:
            continue
        row = rows[name]
        if row['e'] != 0 or any(row[element] != 0 for element in elements - allowed):
            continue
        reactants.append(label)
        canonical.append(name)
    if len(canonical) != len(set(canonical)):
        raise ValueError('Equilibrium gas selection contains duplicate chemical names')
    if set(species) - set(canonical):
        raise ValueError('The EasyChem gas set lacks a requested comparison species')
    abundances = np.array([elemental_abundances[atom] for atom in atoms], float)
    model = ec.ExoAtmos(atoms=atoms, reactants=reactants, atomAbunds=abundances)
    model.solve(pressures, temperatures)
    fractions = model.result_mol()
    mole = np.column_stack([fractions[label] for label in reactants])
    if not np.isfinite(mole).all() or np.any(mole < 0) or np.any(mole.sum(axis=1) <= 0):
        raise ValueError('EasyChem returned invalid gas mole fractions')
    mole /= mole.sum(axis=1)[:, None]
    nuclei = mole @ np.array([[rows[name][atom] for atom in atoms] for name in canonical], float)
    ratios = nuclei / nuclei[:, atoms.index('H'), None]
    expected = abundances / abundances[atoms.index('H')]
    error = float(np.max(abs(ratios / expected - 1)))
    # Check the native equilibrium solution independently. The installed
    # EasyChem solver leaves sub-per-mille elemental residuals in this column.
    # Keep the measured residual in the report alongside the profiles.
    if error > 1e-3:
        raise ValueError(f'Equilibrium elemental ratios differ from the supplied inventory by {error:g}')
    selected = np.array([mole[:, canonical.index(name)] for name in species])
    return selected, {'gas_species': canonical, 'max_relative_element_error': error,
                      'thermodata_sha256': hashlib.sha256(Path(model.thermofpath).read_bytes()).hexdigest()}


def comparison_settings(settings):
    """Keep the network and bulk composition fixed while selecting processes."""
    if not settings.photochemistry or not settings.stellar_flux_file:
        raise ValueError('Supply the target stellar UV spectrum and enable photochemistry before comparing profiles')
    if settings.condensation.enabled:
        raise ValueError('This gas-phase comparison requires condensation disabled')
    if settings.eddy_diffusion.mode == 'off':
        raise ValueError('Configure vertical mixing for the disequilibrium comparison')
    return {
        'thermal_control': replace(settings, photochemistry=False,
            eddy_diffusion=EddyDiffusionConfig(mode='off'), molecular_diffusion=False,
            vertical_velocity_cm_s=0.),
        'mixing': replace(settings, photochemistry=False),
        'photochemistry': settings,
    }


def _vmr_from_profile(profile, species):
    """Recover selected VMRs from gas mass fractions using native molecular masses."""
    with np.load(profile.provenance_file.parent/'gas_profile.npz', allow_pickle=False) as data:
        masses = dict(zip(data['species'].tolist(), data['molecular_masses']))
    return np.array([profile.mass_fractions[name] * profile.mean_molecular_weight / masses[name]
                     for name in species])


def compare_profiles(config, region_name, output, species=SPECIES):
    """Run the matched comparison from an ordinary EXoPLORE simulation config.

    Each accepted case is saved as it finishes. Only a complete comparison is
    plotted. Cache reuse and fresh calculation times are recorded separately.
    """
    region = getattr(config.atmosphere, region_name)
    if region is None or region.chemistry_backend != 'vulcan':
        raise ValueError('Select VULCAN chemistry in the atmospheric region being compared')
    cases = comparison_settings(region.chemistry.vulcan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    planet_file = config.planet.parameter_file or str(Path(config.paths.planet_parameter_dir)/(config.planet.name+'.json'))
    planet = load_planet(planet_file)
    p = create_log_pressure_grid(region.pressure_min_bar, region.pressure_max_bar, region.pressure_grid_size)
    gravity = 6.67430e-8 * planet.M_pl / planet.R_pl**2
    t = calculate_temperature_structure(planet.t_int_K, p, gravity, region.isothermal,
        region.isothermal_temperature_K, region.equilibrium_temperature_K, region.two_point,
        region.two_point_pressures_bar, region.two_point_temperatures_K,
        region.kappa_ir, region.gamma_guillot, None)
    physical = RadiativeTransferInputs('transit', planet.M_pl, planet.R_pl, planet.R_star,
        planet.stellar_teff_K, 1., planet.t_int_K, None, False)
    cache = Path(config.paths.vulcan_cache_dir or Path(config.paths.output_root)/'chemistry/vulcan').resolve()
    input_hash = hashlib.sha256(json.dumps({'p':p.tolist(), 't':t.tolist(),
        'atoms':region.chemistry.vulcan.elemental_abundances}, sort_keys=True).encode()).hexdigest()
    report = {'planet':planet.name, 'region':region_name, 'species':list(species),
        'layers':len(p), 'pressure_bar':[float(p.min()),float(p.max())],
        'temperature_K':[float(t.min()),float(t.max())],
        'elemental_abundances':region.chemistry.vulcan.elemental_abundances,
        'input_hash':input_hash, 'platform':platform.platform(), 'architecture':platform.machine(),
        'cases':{}}
    before = time.perf_counter()
    equilibrium, info = equilibrium_reference(p,t,region.chemistry.vulcan.elemental_abundances,
        Path(config.paths.vulcan_directory)/'thermo/all_compose.txt', species)
    report['cases']['easychem'] = dict(info, wall_seconds=time.perf_counter()-before)
    arrays = {'pressure_bar':p,'temperature_K':t,'species':np.array(species),'easychem':equilibrium}
    for label, settings in cases.items():
        print(f'Calculating the {label} column. Native logs are under {cache}.', flush=True)
        before_files = set(cache.glob('*/manifest.json'))
        start = time.perf_counter()
        profile = run_vulcan(settings,p,t,list(species),physical,config.paths,
                             reference_pressure_bar=region.reference_pressure_bar)
        wall = time.perf_counter()-start
        arrays[label] = _vmr_from_profile(profile,species)
        alignment = json.loads(profile.radius_alignment_file.read_text())
        manifests = list(dict.fromkeys(Path(step['chemistry_manifest']) for step in alignment['iterations']))
        native = [json.loads(path.read_text()) for path in manifests]
        record = {'wall_seconds':wall,'cache_hit':all(path in before_files for path in manifests),
            'native_solve_seconds':sum(item['runtime_seconds'] for item in native),
            'radius_iterations':len(manifests),'cache_keys':[item['cache_key'] for item in native],
            'max_native_atom_loss':max(abs(value) for item in native for value in item['atom_loss'].values())}
        # Retain the first fresh benchmark when a later invocation loads the
        # same physical columns from cache and redraws the figure.
        previous_path = output/(label+'.json')
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text())
            if previous.get('cache_keys') == record['cache_keys']:
                record['fresh_wall_seconds'] = previous.get('fresh_wall_seconds')
        record['new_columns'] = sum(path not in before_files for path in manifests)
        record['reused_columns'] = len(manifests) - record['new_columns']
        if record['reused_columns'] == 0:
            record['fresh_wall_seconds'] = wall
        report['cases'][label] = record
        previous_path.write_text(json.dumps(record,indent=2))
        np.savez_compressed(output/(label+'.npz'),pressure_bar=p,temperature_K=t,
                            species=np.array(species),vmr=arrays[label])
        print(f'{label}: {wall:.2f} s, cache_hit={record["cache_hit"]}',flush=True)
    mask = (equilibrium>1e-12)&(arrays['thermal_control']>1e-12)
    delta = np.zeros_like(equilibrium)
    delta[mask] = np.log10(arrays['thermal_control'][mask]/equilibrium[mask])
    report['thermal_control_max_abs_log10_ratio'] = {
        name:float(np.max(abs(delta[i, mask[i]]))) if mask[i].any() else None
        for i,name in enumerate(species)}
    report['thermal_control_comparison_vmr_floor'] = 1e-12
    np.savez_compressed(output/'comparison.npz',**arrays)
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return arrays, report
