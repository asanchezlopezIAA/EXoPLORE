"""Sunbather atmosphere and phase-dependent vacuum helium spectrum provider.

EXoPLORE receives planet-frame wavelength and transmission arrays. Its existing
observing adapter performs Earth-frame shifts, flux multiplication and noise.
Cloudy iterations and their original energy files are retained in a new project.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from importlib.metadata import version
from pathlib import Path
import csv
import hashlib
import json
import os
import warnings

import numpy as np

from exoplore.config.sunbather import SunbatherConfig
from exoplore.atmosphere.helium_pwinds import geometry


@contextmanager
def _preserve_solver_files(tools, solveT):
    """Disable cleanup/compression and version Sunbather's iteration table."""
    import pandas as pd
    clean, energies = solveT.clean_converged_folder, tools.process_energies
    read, write = pd.read_csv, pd.DataFrame.to_csv

    def read_table(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name == 'iterations.txt':
            versions = list(Path(path).parent.glob('iterations_version_*.txt'))
            if versions:
                path = max(versions, key=lambda p: int(p.stem.split('_')[-1]))
        return read(path, *args, **kwargs)

    def write_table(frame, path=None, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name == 'iterations.txt':
            parent = Path(path).parent
            number = len(list(parent.glob('iterations_version_*.txt'))) + 1
            path = parent / f'iterations_version_{number}.txt'
            if path.exists():
                raise FileExistsError(path)
        return write(frame, path, *args, **kwargs)

    solveT.clean_converged_folder = lambda path: None
    tools.process_energies = partial(energies, rewrite=False)
    pd.read_csv, pd.DataFrame.to_csv = read_table, write_table
    try:
        yield
    finally:
        solveT.clean_converged_folder, tools.process_energies = clean, energies
        pd.read_csv, pd.DataFrame.to_csv = read, write


def solve_sunbather(settings, planet) -> dict:
    """Solve one nonisothermal H/He wind; reject absent or failed convergence.

Input F_lambda is at the orbital separation in erg/s/cm²/Angstrom. Cloudy
receives lambda*F_lambda at 1 au. Sunbather then dilutes to its illuminated
outer boundary (a-r_max), as prescribed by its native solver.
"""
    s = SunbatherConfig(**settings.sunbather)
    root = Path(s.project_path).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cloudy = Path(s.cloudy_path).resolve()
    if not (cloudy/'source/cloudy.exe').is_file():
        raise FileNotFoundError('Cloudy executable is missing')
    if settings.irradiation_mode != 'spectrum':
        raise ValueError('Sunbather requires a stellar spectrum')
    if settings.radial_max_rp != int(settings.radial_max_rp):
        raise ValueError('Sunbather requires an integer outer radius in Rp')
    os.environ['CLOUDY_PATH'] = str(cloudy)
    os.environ['CLOUDY_VERSION'] = '23.01'
    os.environ['SUNBATHER_PROJECT_PATH'] = str(root)
    from sunbather import tools, construct_parker, convergeT_parker, solveT
    data = np.loadtxt(settings.irradiation_spectrum_path)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all() or np.any(data <= 0) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Invalid irradiation spectrum')
    name = 'exoplore_' + hashlib.sha256(str(root).encode()).hexdigest()[:16] + '.spec'
    sed = cloudy/'data/SED'/name
    with sed.open('x') as f:
        f.write(f'{data[0,0]:.12e} {data[0,1]*data[0,0]*planet.semi_major_axis_au**2:.12e} units Angstrom nuFnu\n')
        np.savetxt(f, np.column_stack((data[1:,0], data[1:,1]*data[1:,0]*planet.semi_major_axis_au**2)), fmt='%.12e')
    _, bp, _ = geometry(0, planet)
    with (root/'planets.txt').open('x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name','full name','R [RJ]','Rstar [Rsun]','a [AU]','M [MJ]','Mstar [Msun]','transit impact parameter','SEDname'])
        w.writerow([planet.name, planet.name, planet.planet_radius_rjup, planet.stellar_radius_rsun, planet.semi_major_axis_au, planet.planet_mass_mjup, planet.stellar_mass_msun, bp, name])
    p = tools.Planet(planet.name)
    if p.a <= settings.radial_max_rp*p.R:
        raise ValueError('Atmospheric outer boundary reaches the stellar distance')
    spectrum = construct_parker.cloudy_spec_to_pwinds(str(sed), tools.AU, p.a-settings.radial_max_rp*p.R)
    mdot = float(np.log10(settings.mass_loss_rate_g_s))
    he_scale = (1-settings.hydrogen_number_fraction)/settings.hydrogen_number_fraction/0.1
    zdict = tools.get_zdict(z=s.metal_scale, zelem={'He':he_scale})
    with warnings.catch_warnings(record=True) as caught, _preserve_solver_files(tools, solveT):
        warnings.simplefilter('always')
        construct_parker.save_plain_parker_profile(p, mdot, settings.temperature_K, spectrum,
            h_fraction=settings.hydrogen_number_fraction, pdir='exoplore', overwrite=False,
            no_tidal=not s.stellar_tidal_gravity, altmax=int(settings.radial_max_rp))
        convergeT_parker.run_s(planet.name, mdot, int(settings.temperature_K), 1,
            s.heating_cooling_factor, 'exoplore', 'real', False,
            s.starting_temperature, 'exoplore', zdict=zdict,
            altmax=int(settings.radial_max_rp), save_sp=['He'],
            constantT=False, maxit=s.maximum_iterations)
        folder = root/'sims/1D'/planet.name/'exoplore'/f'parker_{int(settings.temperature_K)}_{mdot:.3f}'
        if not (folder/'converged.out').exists() or not (folder/'converged.txt').exists():
            raise RuntimeError(f'Sunbather did not converge; all iterations retained at {folder}')
        sim = tools.Sim(str(folder/'converged'))
        nist = __import__('sunbather.RT', fromlist=['read_NIST_lines']).read_NIST_lines('He', 10830, 10835)
        cols = [tools.find_line_lowerstate_in_en_df('He', row, sim.en)[0] for _,row in nist.iterrows()]
        if len(cols) != 3 or any(c is None for c in cols) or len(set(cols)) != 1:
            raise RuntimeError('Cannot identify the metastable helium lower level')
        o = sim.ovr
        result = dict(radius_m=o.alt.values[::-1]/100, radius_rp=o.alt.values[::-1]/p.R,
            temperature_K=o.Te.values[::-1], velocity_m_s=o.v.values[::-1]/100,
            density_g_cm3=o.rho.values[::-1], helium_triplet_m3=sim.den[cols[0]].values[::-1]*1e6,
            mean_molecular_weight=float(np.mean(o.mu)), p_winds_version=version('p-winds'),
            sunbather_version=version('sunbather'), cloudy_simulation=str(folder/'converged'),
            warnings=[str(w.message) for w in caught])
    for key in ('temperature_K','radius_m','velocity_m_s','density_g_cm3','helium_triplet_m3'):
        if not np.isfinite(result[key]).all() or np.any(result[key] < 0):
            raise RuntimeError(f'Invalid Sunbather profile: {key}')
    with (root/'provider_provenance.json').open('x') as f:
        json.dump(dict(backend='sunbather',T0_K=settings.temperature_K,
            Tmax_K=float(np.max(result['temperature_K'])), irradiation_source=settings.irradiation_source,
            irradiation_sha256=hashlib.sha256(Path(settings.irradiation_spectrum_path).read_bytes()).hexdigest(),
            boundary='Native Sunbather: irradiation at a-r_max', cosmic_rays=s.cosmic_rays,
            stellar_tidal_gravity=s.stellar_tidal_gravity,
            helium_lower_level=cols[0], warnings=result['warnings']), f, indent=2)
    return result


class SunbatherTransit:
    """Cache Sunbather optical depth; return syn_spec(phase) in the planet frame."""
    def __init__(self, settings, planet, science, profile):
        from sunbather import RT, tools
        self.settings, self.planet = settings, planet
        self.wave_nm = np.linspace(*settings.model_wavelength_range_nm, settings.model_wavelength_points)
        self.radius_m = profile['radius_m']
        r = profile['radius_m']*100
        rp = planet.planet_radius_rjup*7149200000.0
        be, _, x, Te = RT.project_1D_to_2D(r, profile['temperature_K'], rp, numb=settings.line_of_sight_points)
        _, _, _, vx = RT.project_1D_to_2D(r, profile['velocity_m_s']*100, rp, numb=settings.line_of_sight_points, x_projection=True)
        _, _, _, nd = RT.project_1D_to_2D(r, profile['helium_triplet_m3']/1e6, rp, numb=settings.line_of_sight_points)
        table = RT.read_NIST_lines('He',10830,10835).sort_values('ritz_wl_vac(A)')
        if len(table) != 3:
            raise ValueError('Expected three helium transitions')
        self.edges = be
        self.tau = np.zeros((len(self.wave_nm),len(be)-1))
        # Use configured vacuum atomic data; Sunbather supplies natural widths.
        # Small frequency blocks keep resolved-ray Voigt integration in memory.
        nu0 = tools.c/(np.asarray(science.helium_vacuum_lines_nm)*1e-7)
        sigma0 = RT.sigt0*np.asarray(science.helium_oscillator_strengths)
        for start in range(0,len(self.wave_nm),4):
            sl=slice(start,start+4)
            self.tau[sl] = RT.calc_tau(x,nd,Te,vx,tools.c/(self.wave_nm[sl]*1e-7),
                nu0,tools.get_mass('He'),sigma0,table.lorgamma.values, v_turb=0)
        if not np.isfinite(self.tau).all() or np.any(self.tau < 0):
            raise RuntimeError('Invalid Sunbather optical depths')

    def spectrum(self, phase):
        """Return helium-only transmission and opaque continuum, without RV shift."""
        from sunbather import RT
        x,y,p = geometry(phase,self.planet)
        separation = np.hypot(x,y)
        rs = self.planet.stellar_radius_rsun*69570000000.0
        if np.cos(2*np.pi*phase)<=0 or separation >= 1+max(p,self.radius_m[-1]*100/rs):
            return np.ones_like(self.wave_nm),1.0
        law=self.settings.limb_darkening_law
        ab=np.array(self.settings.limb_darkening_coefficients if law=='quadratic' else [self.settings.limb_darkening_coefficients[0],0] if law=='linear' else [0,0])
        # Rotating a spherical atmosphere preserves exact projected separation.
        full=RT.tau_to_FinFout(self.edges,self.tau,rs,bp=separation,ab=ab)
        opaque=float(RT.tau_to_FinFout(self.edges,np.zeros((1,len(self.edges)-1)),rs,bp=separation,ab=ab)[0])
        helium=full/opaque
        if not np.isfinite(helium).all() or np.any(helium<0) or np.any(helium>1+1e-12):
            raise RuntimeError('Invalid Sunbather spectrum')
        if np.max(np.abs(1-helium[[0,-1]]))>self.settings.model_edge_absorption_tolerance:
            raise ValueError('Model wavelength range truncates absorption')
        return helium,opaque
