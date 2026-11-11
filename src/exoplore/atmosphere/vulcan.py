"""Run an external VULCAN checkout and return gas abundances for pRT.

Each distinct physical column has its own working directory. VULCAN generates
Python chemistry functions and writes FastChem input files there. Completed
columns are reused across spectral orders through a content-addressed cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import time

import numpy as np

from exoplore.config.chemistry import VulcanConfig


@dataclass(frozen=True)
class ChemistryProfile:
    """Gas mass fractions and gas MMW for pRT, with native diagnostics.

    Every gas enters the molecular weight and mass denominator. The selected
    opacity species retain their share of that complete gas mixture. Native
    condensate loading is recorded separately. Mean masses use atomic units.
    """
    mass_fractions: dict[str, np.ndarray]
    mean_molecular_weight: np.ndarray
    provenance_file: Path
    radius_alignment_file: Path | None = None
    gas_mean_molecular_weight: np.ndarray | None = None
    condensate_to_gas_mass_ratio: np.ndarray | None = None
    native_hydrostatic_mean_mass: np.ndarray | None = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _column(pressures, temperatures):
    p, t = np.asarray(pressures, float), np.asarray(temperatures, float)
    if p.ndim != 1 or p.size < 3 or t.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("VULCAN requires matching finite pressure and temperature arrays with >=3 layers")
    if np.any(p <= 0) or np.any(t <= 0) or not (np.all(np.diff(p) > 0) or np.all(np.diff(p) < 0)):
        raise ValueError("VULCAN pressures must be positive and strictly monotonic, with positive temperatures")
    return p, t


def _table(path, label):
    # User transport tables have one commented header and two numerical columns.
    a = np.loadtxt(path, ndmin=2)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError(f"{label} requires two finite columns and at least two rows")
    if np.any(a[:, 0] <= 0) or not np.all(np.diff(a[:, 0]) > 0) or np.any(a[:, 1] < 0):
        raise ValueError(f"{label} requires ascending positive coordinates and non-negative values")
    return a


def gas_composition(data, masses, non_gas_species=(), *, max_atom_loss=0.01):
    """Convert native VULCAN number densities into gas-only mole fractions.

    All gaseous species enter the denominator, including gases without pRT
    opacity tables. Condensates remain recorded in the native output file.
    """
    if data.get('parameter', {}).get('end_case') != 1:
        raise ValueError("VULCAN did not reach steady state (parameter.end_case must be 1)")
    variable, atm = data['variable'], data['atm']
    species = list(variable['species'])
    if len(species) != len(set(species)):
        raise ValueError("VULCAN output contains duplicate species")
    y = np.asarray(variable['y'], float)
    p = np.asarray(atm['pco'], float) / 1.e6
    t = np.asarray(atm['Tco'], float)
    _column(p, t)
    if y.shape != (len(p), len(species)) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("VULCAN number densities must be finite, non-negative and match layer/species dimensions")
    losses = variable.get('atom_loss')
    if not isinstance(losses, dict) or not losses:
        raise ValueError("VULCAN output lacks elemental-conservation diagnostics")
    if any(not np.isfinite(v) or abs(v) > max_atom_loss for v in losses.values()):
        raise ValueError("VULCAN elemental conservation exceeds max_atom_loss")
    gas = [i for i, name in enumerate(species) if name not in non_gas_species]
    if not gas:
        raise ValueError("VULCAN output contains no gas species")
    names = [species[i] for i in gas]
    missing = set(names) - masses.keys()
    if missing:
        raise ValueError(f"Molecular masses missing for {sorted(missing)}")
    molecular_mass = np.array([masses[name] for name in names], float)
    if not np.isfinite(molecular_mass).all() or np.any(molecular_mass <= 0):
        raise ValueError("Gas molecular masses must be finite and positive")
    total = y[:, gas].sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("VULCAN output has an empty gas layer")
    mole = y[:, gas] / total[:, None]
    if non_gas_species:
        # Native ymix is normalised by gas number density. Including condensed
        # mass in that sum represents the weight carried by each gas particle.
        loading = condensate_mass_per_gas_particle(data, masses, non_gas_species)
        expected_mu = mole @ molecular_mass + loading
        native_mu = np.asarray(atm.get('mu', []), float)
        if native_mu.shape != expected_mu.shape or not np.allclose(native_mu, expected_mu, rtol=1e-6, atol=1e-8):
            raise ValueError("VULCAN hydrostatic mass differs from the gas plus condensate mass per gas particle")
    # Work in ascending pressure for interpolation into EXoPLORE's column.
    order = np.argsort(p)
    return p[order], t[order], names, mole[order], molecular_mass


def condensate_mass_per_gas_particle(data, masses, non_gas_species):
    """Return condensed mass per gas particle, in atomic mass units.

    VULCAN's condensed number densities count chemical formula units stored
    in particles. Particle radius and bulk density describe the grains used
    for growth and settling. The gas number density supplies the denominator.
    """
    names = list(data['variable']['species'])
    y = np.asarray(data['variable']['y'], float)
    gas = [i for i, name in enumerate(names) if name not in non_gas_species]
    total_gas = y[:, gas].sum(axis=1)
    if np.any(total_gas <= 0):
        raise ValueError("VULCAN output has an empty gas layer")
    loading = np.zeros(len(total_gas))
    for name in non_gas_species:
        if name not in names or name not in masses:
            raise ValueError(f"Missing condensate species or molecular mass: {name}")
        mass = masses[name]
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("Condensate molecular masses must be finite and positive")
        loading += y[:, names.index(name)] * mass / total_gas
    if not np.isfinite(loading).all() or np.any(loading < 0):
        raise ValueError("Condensate loading must be finite and non-negative")
    return loading


def regrid_composition(p, temperatures, source_p, source_t, names, mole, masses, requested, species_map=None,
                       condensed_mass_per_gas_particle=None):
    """Interpolate gas mole fractions, then calculate gas mass fractions and MMW.

    Condensate loading can be validated alongside the column. The pRT outputs
    use gas mass as their denominator, including every gas in the network.
    """
    from exoplore.atmosphere.prt import easychem_molecule
    p, temperatures = _column(p, temperatures)
    eps = 64 * np.finfo(float).eps
    if p.min() < source_p.min() * (1-eps) or p.max() > source_p.max() * (1+eps):
        raise ValueError("VULCAN pressure coverage does not span the pRT column")
    # The native solver reads T(P) by linear interpolation in pressure.
    # Compare on its returned grid before interpolating composition to pRT.
    order = np.argsort(p)
    expected_t = np.interp(source_p, p[order], temperatures[order])
    if not np.allclose(source_t, expected_t, rtol=1e-8, atol=1e-6):
        raise ValueError("VULCAN returned a different temperature profile")
    out = np.column_stack([np.interp(np.log(p), np.log(source_p), mole[:, i]) for i in range(len(names))])
    if not np.allclose(out.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("Interpolated gas mole fractions do not sum to one")
    mu = out @ masses
    if condensed_mass_per_gas_particle is not None:
        loading = np.asarray(condensed_mass_per_gas_particle, float)
        if loading.shape != source_p.shape or not np.isfinite(loading).all() or np.any(loading < 0):
            raise ValueError("Condensate loading must match the source pressure grid and be finite and non-negative")
    fractions = out * masses[None, :] / mu[:, None]
    mapping = species_map or {}
    selected = {}
    for opacity in requested:
        name = mapping.get(opacity, easychem_molecule(opacity))
        if name not in names:
            raise ValueError(f"Requested opacity {opacity!r} maps to {name!r}, absent from the VULCAN gas network")
        selected[opacity] = fractions[:, names.index(name)].copy()
    return selected, mu


# Working outputs and compiled intermediate files change during a native run.
# Source files, input data and the compiled FastChem executable identify a run.
_IGNORED = {'.git', '__pycache__', 'output', 'plot', 'movie', 'obj', '.DS_Store'}


def source_digest(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if path.is_file() and not any(part in _IGNORED for part in rel.parts):
            digest.update(str(rel).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _masses(path):
    table = np.genfromtxt(path, names=True, dtype=None, encoding='utf-8')
    return {str(row['species']): float(row['mass']) for row in np.atleast_1d(table)}


def _inputs(settings, source, p):
    """Resolve external inputs before launching chemistry or accepting a cache."""
    network = Path(settings.network_file).expanduser()
    if not network.is_absolute():
        network = source / network
    files = {'network': network.resolve()}
    if settings.photochemistry:
        files['uv'] = Path(settings.stellar_flux_file).expanduser().resolve()
        uv = _table(files['uv'], 'Stellar UV spectrum')
        if uv[0, 0] > 2 or uv[-1, 0] < 700:
            raise ValueError("Supply stellar-surface UV flux covering 2–700 nm for the native VULCAN photochemical bins")
    if settings.eddy_diffusion.mode == 'file':
        files['kzz'] = Path(settings.eddy_diffusion.profile_file).expanduser().resolve()
        table = _table(files['kzz'], 'Kzz profile [bar, cm2/s]')
        if table[0, 0] > p.min() or table[-1, 0] < p.max():
            raise ValueError("Kzz profile must span the complete pressure column")
    for side in ('top', 'bottom'):
        boundary = getattr(settings, side+'_boundary')
        if boundary.type == 'flux_file':
            files[side] = Path(boundary.file).expanduser().resolve()
    for label, path in files.items():
        if not path.is_file():
            raise ValueError(f"Missing VULCAN {label} input: {path}")
    return files


def _repair_ammonia_condensation(work, settings):
    """Apply the two validated ammonia corrections in the private work copy.

    The inspected upstream table gives NH3_l_s the mass of NH2. Its direct
    condensation dispatcher also uses NH3_l, while the network uses NH3_l_s.
    Record each changed file so the calculation identifies the executed source.
    Already corrected checkouts pass through unchanged.
    """
    if not settings.condensation.enabled or 'NH3_l_s' not in settings.condensation.condensed_species:
        return []
    corrections = []
    composition = work/'thermo/all_compose.txt'
    rows = composition.read_text().splitlines(keepends=True)
    masses = _masses(composition)
    if 'NH3' not in masses or 'NH3_l_s' not in masses:
        raise ValueError("Ammonia condensation requires NH3 and NH3_l_s in the native mass table")
    if masses['NH3_l_s'] != masses['NH3']:
        if masses['NH3_l_s'] != 16.023 or masses['NH3'] != 17.031:
            raise ValueError("NH3 and NH3_l_s formula-unit masses disagree in the native table")
        before = _sha(composition)
        for i, row in enumerate(rows):
            fields = row.split()
            if fields and fields[0] == 'NH3_l_s':
                fields[-1] = '17.031'
                rows[i] = '\t'.join(fields)+'\n'
        composition.write_text(''.join(rows))
        corrections.append(dict(file='thermo/all_compose.txt', correction='NH3_l_s formula-unit mass 17.031 amu',
                                before_sha256=before, after_sha256=_sha(composition)))
    operator = work/'op.py'
    text = operator.read_text()
    old = "var.Rf[re] == 'NH3 -> NH3_l'"
    if old in text:
        if text.count(old) != 1:
            raise ValueError("Unexpected repeated ammonia condensation dispatcher in native op.py")
        before = _sha(operator)
        operator.write_text(text.replace(old, "var.Rf[re] == 'NH3 -> NH3_l_s'"))
        corrections.append(dict(file='op.py', correction='Match NH3_l_s condensation reaction name',
                                before_sha256=before, after_sha256=_sha(operator)))
    return corrections


def prepare_workdir(work, source, settings, p, t, physical, files):
    """Copy the checkout and write one native cfg from the master JSON."""
    shutil.copytree(source, work, ignore=shutil.ignore_patterns(*_IGNORED), dirs_exist_ok=True)
    corrections = _repair_ammonia_condensation(work, settings)
    (work/'source_corrections.json').write_text(json.dumps(corrections, indent=2))
    if (work/'build_atm.py').is_file():
        shutil.copy2(Path(__file__).with_name('vulcan_initialisation.py'), work/'exoplore_initialisation.py')
        (work/'exoplore_initialisation_inputs.json').write_text(json.dumps({
            'elemental_abundances': settings.elemental_abundances,
            'non_gas_species': settings.condensation.condensed_species,
            'tolerance': settings.initial_element_tolerance}, indent=2))
    for directory in ('output', 'plot', 'fastchem_vulcan/output', 'exoplore_inputs'):
        (work/directory).mkdir(parents=True, exist_ok=True)
    local = {}
    for label, path in files.items():
        target = work/'exoplore_inputs'/label
        shutil.copy2(path, target)
        local[label] = str(target.relative_to(work))
    # VULCAN's layer centres run from the bottom to the top of the atmosphere.
    order = np.argsort(p)[::-1]
    grid = p[order]
    kzz = settings.eddy_diffusion
    if kzz.mode == 'file':
        a = _table(files['kzz'], 'Kzz profile [bar, cm2/s]')
        within = a[(a[:, 0] >= p.min()) & (a[:, 0] <= p.max()), 0]
        grid = np.unique(np.concatenate([p, within]))[::-1]
    temperature = np.interp(grid, p[order][::-1], t[order][::-1])
    columns = [grid*1e6, temperature]
    header = '# Pressure[dyn/cm2] Temp[K]\nPressure Temp'
    if kzz.mode == 'file':
        columns.append(np.interp(grid, a[:, 0], a[:, 1]))
        header = '# Pressure[dyn/cm2] Temp[K] Kzz[cm2/s]\nPressure Temp Kzz'
    np.savetxt(work/'exoplore_inputs/atmosphere.txt', np.column_stack(columns), header=header, comments='')
    if not settings.photochemistry:
        # The upstream Variables constructor reads bin bounds even in a thermal
        # run. These unused zero-flux bounds satisfy that file interface.
        np.savetxt(work/'exoplore_inputs/uv', [[2., 0.], [700., 0.]], header='nm erg/cm2/s/nm')
        local['uv'] = 'exoplore_inputs/uv'
    c = settings.condensation
    overrides = dict(
        network=local['network'], com_file='thermo/all_compose.txt',
        gibbs_text='thermo/gibbs_text.txt', cross_folder='thermo/photo_cross/', atm_type='file', atm_file='exoplore_inputs/atmosphere.txt',
        nz=len(p), P_b=float(p.max()*1e6), P_t=float(p.min()*1e6), high_temp_cut=False,
        use_solar=False, atom_list=list(settings.elemental_abundances),
        ini_mix='EQ' if settings.initialization == 'equilibrium' else 'const_mix',
        use_other_ele=False, use_photo=settings.photochemistry, use_ion=False,
        const_mix=settings.initial_mixing_ratios,
        sflux_file=local['uv'], r_star=physical['stellar_radius_cm']/6.957e10,
        Rp=physical['planet_radius_cm'], Mp=physical['planet_mass_g'], rocky=False,
        orbit_radius=settings.orbit_radius_au or 1.,
        sl_angle=float(np.deg2rad(settings.zenith_angle_deg or 0.)),
        f_diurnal=settings.diurnal_factor if settings.photochemistry else 1.,
        scat_sp=['H2', 'He'] if settings.background_gas == 'H2' else [settings.background_gas],
        T_cross_sp=[], use_Kzz=kzz.mode != 'off', Kzz_prof='file' if kzz.mode == 'file' else 'const',
        const_Kzz=kzz.value_cm2_s or 0., use_moldiff=settings.molecular_diffusion,
        use_vm_mol=settings.molecular_diffusion_scheme in ('upwind', 'hybrid'),
        use_hybrid_vm_mol=settings.molecular_diffusion_scheme == 'hybrid',
        atm_base=settings.background_gas, use_vz=settings.vertical_velocity_cm_s != 0,
        vz_prof='const', const_vz=settings.vertical_velocity_cm_s,
        use_topflux=settings.top_boundary.type == 'flux_file',
        use_botflux=settings.bottom_boundary.type == 'flux_file',
        use_fix_sp_bot=settings.bottom_boundary.mixing_ratios,
        diff_esc=[], remove_list=[], use_condense=c.enabled, use_settling=c.settling,
        condense_sp=c.gas_species, non_gas_sp=c.condensed_species,
        r_p=c.particle_radius_cm, rho_p=c.particle_density_g_cm3,
        use_ini_cold_trap=False, use_sat_surfaceH2O=False, use_relax=[], fix_species=[], fix_species_time=0,
        start_conden_time=0, stop_conden_time=float('inf'),
        conver_ignore=[], use_live_plot=False, use_live_flux=False, use_plot_end=False,
        use_plot_evo=False, use_save_movie=False, use_flux_movie=False, use_shark=False,
        save_evolution=False, output_humanread=False, output_dir='output/', out_name='column.vul',
    )
    # Elemental ratios are explicit number ratios to H. H itself is the reference.
    overrides.update({element+'_H': value for element, value in settings.elemental_abundances.items() if element != 'H'})
    for side, native in (('top', 'top_BC_flux_file'), ('bottom', 'bot_BC_flux_file')):
        if side in local:
            overrides[native] = local[side]
    overrides.update(settings.solver_options)
    text = (source/'vulcan_cfg.py').read_text()
    text += '\n# EXoPLORE column inputs generated from the simulation JSON.\n'
    for key, value in overrides.items():
        # Python repr has no literal for infinity. VULCAN needs condensation
        # to remain active until convergence for the requested depletion model.
        literal = "float('inf')" if isinstance(value, float) and np.isinf(value) else repr(value)
        text += f'{key} = {literal}\n'
    (work/'vulcan_cfg.py').write_text(text)
    return overrides



def _execute_vulcan(python, work, timeout):
    """Keep the native solver and its network/FastChem children in one job."""
    entry = 'exoplore_initialisation.py' if (work/'exoplore_initialisation.py').is_file() else 'vulcan.py'
    command = [str(python), '-u', entry]
    env = {**os.environ, 'MPLBACKEND':'Agg', 'MPLCONFIGDIR':str(work/'matplotlib_cache')}
    with (work/'run.log').open('w') as log:
        with subprocess.Popen(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                              env=env, start_new_session=os.name == 'posix') as process:
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                # A timeout or interrupted parent must also stop code generation
                # and FastChem. They inherit this process group on macOS/Linux.
                if os.name == 'posix':
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)

def _run_native_column(settings: VulcanConfig, pressures, temperatures, species, model_inputs, paths):
    """Run or reuse a converged column and map its gases to the selected opacities."""
    p, t = _column(pressures, temperatures)
    if not paths.vulcan_directory or not paths.vulcan_python:
        raise ValueError('Set paths.vulcan_directory and paths.vulcan_python')
    source = Path(paths.vulcan_directory).expanduser().resolve()
    # Preserve the environment's symlink path. Resolving it would invoke
    # the base interpreter and bypass the selected virtual environment.
    python = Path(os.path.abspath(Path(paths.vulcan_python).expanduser()))
    for path in (source/'vulcan.py', source/'vulcan_cfg.py', source/'thermo/all_compose.txt', python):
        if not path.is_file():
            raise ValueError(f"Missing VULCAN installation input: {path}")
    if not (source/'fastchem_vulcan/fastchem').is_file():
        raise ValueError("Compile VULCAN's bundled FastChem with make in fastchem_vulcan first")
    files = _inputs(settings, source, p)
    mass = model_inputs.planet_mass_g
    if model_inputs.gravity_cgs is not None:
        mass = model_inputs.gravity_cgs * model_inputs.planet_radius_cm**2 / 6.67430e-8
    physical = dict(planet_radius_cm=float(model_inputs.planet_radius_cm), planet_mass_g=float(mass),
                    stellar_radius_cm=float(model_inputs.stellar_radius_cm))
    if any(not np.isfinite(v) or v <= 0 for v in physical.values()):
        raise ValueError("VULCAN requires positive planet mass/radius and stellar radius")
    # Library versions can alter numerical integration even with the same source.
    runtime_code = "import json,sys,importlib.metadata as m; print(json.dumps({'python':sys.version,'prefix':sys.prefix,'packages':{n:m.version(n) for n in ['numpy','scipy','sympy','matplotlib','astropy']}}))"
    runtime = json.loads(subprocess.check_output([str(python), '-c', runtime_code], text=True, timeout=30))
    try:
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = None
    request = dict(adapter_version=1, runtime=runtime, source_revision=revision, settings=asdict(settings), pressure_bar=p.tolist(),
                   temperature_K=t.tolist(), physical=physical, source_sha256=source_digest(source), adapter_sha256=_sha(__file__), initialisation_validator_sha256=_sha(Path(__file__).with_name('vulcan_initialisation.py')),
                   input_sha256={key:_sha(value) for key,value in files.items()},
                   python_sha256=_sha(python))
    key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cache = Path(paths.vulcan_cache_dir or Path(paths.output_root)/'chemistry/vulcan').expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    work = cache/key
    manifest = work/'manifest.json'
    lock = cache/(key+'.lock')
    # Independent processes using this column share one writer. Other columns
    # have separate workspaces and can generate different networks safely.
    started = time.monotonic()
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic()-started > settings.timeout_seconds:
                raise TimeoutError(f"VULCAN cache lock timed out: {lock}")
            time.sleep(.2)
    try:
        if not manifest.is_file():
            if work.exists():
                # Keep failed runs for inspection, then start a fresh attempt.
                work.rename(cache/(key+'.failed-'+str(time.time_ns())))
            prepare_workdir(work, source, settings, p, t, physical, files)
            before = time.monotonic()
            _execute_vulcan(python, work, settings.timeout_seconds)
            # This pickle was produced by the configured external executable in
            # this workspace. Public profile interchange uses the NPZ below.
            with (work/'output/column.vul').open('rb') as handle:
                data = pickle.load(handle)
            sp, st, names, mole, masses = gas_composition(data, _masses(work/'thermo/all_compose.txt'),
                settings.condensation.condensed_species, max_atom_loss=settings.max_atom_loss)
            native_masses = _masses(work/'thermo/all_compose.txt')
            loading = condensate_mass_per_gas_particle(data, native_masses, settings.condensation.condensed_species)
            order = np.argsort(np.asarray(data['atm']['pco'], float))
            loading = loading[order]
            regrid_composition(p, t, sp, st, names, mole, masses, species, settings.species_map, loading)
            np.savez_compressed(work/'gas_profile.npz', pressure_bar=sp, temperature_K=st,
                                species=np.array(names), mole_fractions=mole, molecular_masses=masses,
                                condensed_mass_per_gas_particle=loading)
            record = dict(request=request, cache_key=key, runtime_seconds=time.monotonic()-before,
                          source_corrections=json.loads((work/'source_corrections.json').read_text()),
                          prt_mass_fraction_basis='gas',
                          prt_mean_molecular_weight='gas_mass_per_gas_particle_in_amu',
                          native_hydrostatic_mean_mass='total_mass_per_gas_particle_in_amu',
                          max_condensate_to_gas_mass_ratio=float(np.max(loading/(mole @ masses))),
                          end_case=data['parameter']['end_case'], atom_loss=data['variable']['atom_loss'],
                          convergence={name:data['variable'].get(name) for name in ('longdy','longdydt','aflux_change','t')},
                          native_radius_reference_pressure_bar=float(data['atm']['pico'][data['atm']['pref_indx']]/1e6) if 'pref_indx' in data['atm'] else None,
                          gas_profile_sha256=_sha(work/'gas_profile.npz'), native_output_sha256=_sha(work/'output/column.vul'))
            manifest.write_text(json.dumps(record, indent=2, allow_nan=False))
        record = json.loads(manifest.read_text())
        if record['request'] != request or record['gas_profile_sha256'] != _sha(work/'gas_profile.npz'):
            raise ValueError("VULCAN cache integrity check failed")
        with np.load(work/'gas_profile.npz', allow_pickle=False) as a:
            mf, mu = regrid_composition(p, t, a['pressure_bar'], a['temperature_K'], a['species'].tolist(),
                                       a['mole_fractions'], a['molecular_masses'], species, settings.species_map,
                                       a['condensed_mass_per_gas_particle'])
            loading = np.interp(np.log(p), np.log(a['pressure_bar']),
                                a['condensed_mass_per_gas_particle'])
        return ChemistryProfile(mf, mu, manifest, gas_mean_molecular_weight=mu.copy(),
                                condensate_to_gas_mass_ratio=loading/mu,
                                native_hydrostatic_mean_mass=mu+loading)
    finally:
        lock.rmdir()


def radius_at_pressure(pressures, temperatures, mu, radius_cm, gravity_cgs,
                       reference_pressure_bar, destination_pressure_bar):
    """Translate a pressure-radius reference using pRT's hydrostatic equation.

    Refinement affects this auxiliary integral only. The chemistry and opacity
    grids retain their configured sampling. Temperature follows VULCAN's linear
    interpolation in pressure, while molecular weight is interpolated in log P.
    """
    from petitRADTRANS.radtrans import Radtrans
    p, t = _column(pressures, temperatures)
    order = np.argsort(p)
    mu = np.asarray(mu, float)
    if mu.shape != p.shape or not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("Radius conversion requires positive finite molecular weights on the pressure grid")
    p, t, mu = p[order], t[order], mu[order]
    refs = np.array([reference_pressure_bar, destination_pressure_bar], float)
    if not np.isfinite(refs).all() or np.any(refs < p[0]) or np.any(refs > p[-1]):
        raise ValueError("Both radius reference pressures must lie inside the supplied atmospheric column")
    if not np.isfinite([radius_cm, gravity_cgs]).all() or min(radius_cm, gravity_cgs) <= 0:
        raise ValueError("Radius conversion requires positive finite radius and gravity")
    if reference_pressure_bar == destination_pressure_bar:
        return float(radius_cm)
    grid = np.unique(np.r_[p, refs])
    previous, stable = None, 0
    for refinement in range(11):
        radii = Radtrans._compute_radius_hydrostatic_equilibrium(
            grid, np.interp(grid, p, t), np.interp(np.log(grid), np.log(p), mu),
            gravity_cgs, reference_pressure_bar, radius_cm, variable_gravity=True)
        if not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("Radius conversion produced invalid hydrostatic radii")
        value = float(radii[np.flatnonzero(grid == destination_pressure_bar)[0]])
        relative_step = abs(value-previous)/radius_cm if previous is not None else np.inf
        stable = stable+1 if relative_step <= 1e-8 else 0
        if stable >= 2:
            return value
        previous = value
        grid = np.sort(np.r_[grid, np.sqrt(grid[:-1]*grid[1:])])
    raise ValueError("Hydrostatic radius conversion did not converge")


def run_vulcan(settings: VulcanConfig, pressures, temperatures, species, model_inputs,
               paths, *, reference_pressure_bar=None):
    """Align VULCAN's radius reference with the atmospheric region used by pRT.

    Each accepted chemistry column supplies the molecular weights for the next
    hydrostatic conversion. Planet mass stays fixed throughout this iteration.
    Native columns are cached individually, including the initial calculation.
    """
    if reference_pressure_bar is None:
        raise ValueError("VULCAN requires the atmospheric region's reference_pressure_bar")
    p, t = _column(pressures, temperatures)
    if not np.isfinite(reference_pressure_bar) or not p.min() <= reference_pressure_bar <= p.max():
        raise ValueError("The radius reference pressure must lie inside the atmospheric column")
    radius = float(model_inputs.planet_radius_cm)
    gravity = model_inputs.gravity_cgs
    mass = model_inputs.planet_mass_g
    if gravity is None:
        gravity = 6.67430e-8 * mass / radius**2
    else:
        # Preserve the mass implied by the user's gravity at the pRT reference.
        mass = gravity * radius**2 / 6.67430e-8
    native_radius = radius
    history = []
    for iteration in range(8):
        native_inputs = replace(model_inputs, planet_radius_cm=native_radius,
                                planet_mass_g=mass, gravity_cgs=None)
        profile = _run_native_column(settings, p, t, species, native_inputs, paths)
        record = json.loads(profile.provenance_file.read_text())
        pref = record['native_radius_reference_pressure_bar']
        if pref is None:
            raise ValueError("VULCAN output lacks its hydrostatic radius reference interface")
        # Use the native gas grid to retain any structure between pRT layers.
        with np.load(profile.provenance_file.parent/'gas_profile.npz', allow_pickle=False) as gas:
            native_p = gas['pressure_bar']
            native_t = gas['temperature_K']
            gas_mu = gas['mole_fractions'] @ gas['molecular_masses']
            native_mu = gas_mu + gas['condensed_mass_per_gas_particle']
            max_loading = float(np.max(gas['condensed_mass_per_gas_particle']/gas_mu))
        converted = radius_at_pressure(native_p, native_t, native_mu, radius,
                                       gravity, reference_pressure_bar, pref)
        gas_reference_radius = converted if max_loading == 0 else radius_at_pressure(
            native_p, native_t, gas_mu, radius, gravity, reference_pressure_bar, pref)
        history.append(dict(native_radius_cm=native_radius,
                            gas_radius_at_native_reference_cm=gas_reference_radius,
                            loaded_minus_gas_reference_radius_cm=converted-gas_reference_radius,
                            max_condensate_to_gas_mass_ratio=max_loading,
                            converted_radius_cm=converted,
                            native_reference_pressure_bar=pref,
                            chemistry_manifest=str(profile.provenance_file)))
        if abs(converted-native_radius)/radius <= 1e-8:
            alignment = dict(reference_pressure_bar=reference_pressure_bar,
                             reference_radius_cm=radius, planet_mass_g=mass,
                             reference_gravity_cgs=gravity, iterations=history)
            # The record identifies the pRT reference separately from the native
            # column cache, which can be shared by different requested opacities.
            identifier = hashlib.sha256(json.dumps(alignment, sort_keys=True).encode()).hexdigest()
            target = profile.provenance_file.parent/('radius_alignment_'+identifier+'.json')
            target.write_text(json.dumps(alignment, indent=2, allow_nan=False))
            return replace(profile, radius_alignment_file=target)
        native_radius = converted
    raise ValueError("VULCAN chemistry and pressure-radius conversion did not converge after eight columns")
