"""Metal-bearing Parker profiles using Sunbather's native mean-mass iteration.

Use the same native p-winds/Cloudy functions and convergence criterion as
save_cloudy_parker_profile, retaining each iteration at a unique path. No
temporary files are deleted or overwritten. Failure to converge is explicit.
"""
from __future__ import annotations

# Role in the calculation
# Calculate the composition and mean-mass iteration for metal-bearing Sunbather
# atmospheres. Convergence checks determine whether the solved atmosphere can be used.

from pathlib import Path
import json
import warnings
import numpy as np
from exoplore.config.sunbather import SunbatherConfig


def _select_metal_abundances(zdict: dict, elements: list[str] | None) -> dict:
    """Disable unselected metals without redistributing their abundances."""
    selected=dict(zdict)
    if elements is not None:
        for element in selected:
            if element not in ('H','He') and element not in elements:
                selected[element]=0.
    return selected


def save_metal_parker_profile(construct,tools,planet,logmass: float,temperature: int,
                             spectrum: dict,zdict: dict,settings: SunbatherConfig,
                             outer_radius: int,root: Path) -> Path:
    """Iterate the mean particle mass with fixed composition and Parker T0.

    Each isothermal Cloudy evaluation supplies mu(r) and the native weighted
    mean mass for the next Parker profile. After convergence, save the updated
    profile at the path expected by Sunbather's nonisothermal temperature solver.
    H/He is encoded in zdict, independently of the metal/hydrogen scale factor.
    """
    if not settings.metals_enabled:raise ValueError('Metal-bearing Parker helper requires enabled metals')
    destination=root/'parker_profiles'/planet.name/'exoplore'/f'pprof_{planet.name}_T={temperature}_M={logmass:.3f}.txt'
    if destination.exists() or (root/'parker_mu_convergence.json').exists():
        raise FileExistsError(f'Preserve existing Parker profile: {destination}')
    mu=None;structure=None
    if settings.parker_initialization=='neutral':
        mu=float(construct.calc_neutral_mu(zdict))
        structure=np.array([[1.,mu],[outer_radius,mu]])
    history=[]
    for iteration in range(1,settings.parker_mu_max_iterations+1):
        pdir=f'exoplore/mu_iterations/iteration_{iteration:03d}'
        iteration_directory=root/'parker_profiles'/planet.name/pdir
        if iteration_directory.exists() or (root/f'parker_mu_iteration_{iteration:03d}.json').exists():
            raise FileExistsError(f'Preserve existing Parker iteration: {iteration_directory}')
        filename,previous,launch=construct.save_temp_parker_profile(
            planet,logmass,temperature,spectrum,zdict,pdir,mu_bar=mu,mu_struc=structure,
            no_tidal=not settings.stellar_tidal_gravity,altmax=outer_radius,projectpath=str(root))
        stem=Path(filename).with_suffix('')
        if any(Path(str(stem)+suffix).exists() for suffix in ('.in','.out','.ovr')):
            raise FileExistsError(f'Preserve existing Parker/Cloudy iteration: {stem}')
        simulation,parker=construct.run_parker_with_cloudy(filename,temperature,planet,zdict)
        sim=tools.Sim(simulation,altmax=outer_radius,planet=planet)
        sim.addv(parker.alt,parker.v)
        mu=float(construct.calc_mu_bar(sim))
        if not np.isfinite(mu) or mu<=0 or not np.isfinite(previous) or previous<=0:
            raise RuntimeError('Invalid mean particle mass in metal-bearing Parker iteration')
        structure=np.column_stack((np.asarray(sim.ovr.alt)[::-1]/planet.R,
                                   np.asarray(sim.ovr.mu)[::-1]))
        change=abs(mu-previous)/previous
        row=dict(iteration=iteration,previous_mu=float(previous),cloudy_mu=mu,
                 relative_change=float(change),simulation=str(simulation))
        history.append(row)
        with (root/f'parker_mu_iteration_{iteration:03d}.json').open('x') as stream:
            json.dump(row,stream,indent=2)
        if change<settings.parker_mu_tolerance:break
    else:
        raise RuntimeError(f'Metal-bearing Parker mean particle mass did not converge after {settings.parker_mu_max_iterations} iterations; files retained at {root}')
    final_directory=root/'parker_profiles'/planet.name/'exoplore/mu_iterations/final'
    if final_directory.exists():
        raise FileExistsError(f'Preserve existing final Parker iteration: {final_directory}')
    filename,_,launch=construct.save_temp_parker_profile(
        planet,logmass,temperature,spectrum,zdict,'exoplore/mu_iterations/final',
        mu_bar=mu,mu_struc=structure,no_tidal=not settings.stellar_tidal_gravity,
        altmax=outer_radius,projectpath=str(root))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:stream.write(Path(filename).read_bytes())
    if launch>1:warnings.warn(f'This Parker wind profile is supersonic already at Rp: {destination}')
    with (root/'parker_mu_convergence.json').open('x') as stream:
        json.dump(dict(converged=True,tolerance=settings.parker_mu_tolerance,
            initialization=settings.parker_initialization,iterations=history,final_mu=mu,
            profile=str(destination)),stream,indent=2)
    return destination
