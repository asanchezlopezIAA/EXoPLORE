"""Explicit inputs for Sunbather temperature–mass-loss grid fitting.

The grid is an opt-in inference stage on an already prepared helium spectrum.
It does not simulate noise, run molecfit or decide whether a signal is detected.
"""
from __future__ import annotations

# Reading this file
# A grid fit starts with a measured, prepared helium spectrum and explores
# explicit model axes. These settings specify that inference stage, not a new
# noisy observation. axis_values expands the user's grid description before
# retrieval/helium_grid.py evaluates its points.


from dataclasses import dataclass
import math


def axis_values(specification, name: str, integer: bool = False) -> list[float]:
    """Expand an explicit list or a minimum/maximum/points grid specification."""
    if isinstance(specification, dict):
        if set(specification) != {'minimum','maximum','points'}:
            raise ValueError(f'{name}: supply minimum, maximum and points')
        lo,hi,n = (specification[k] for k in ('minimum','maximum','points'))
        if type(n) is not int or n<2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (lo,hi)) or lo>=hi:
            raise ValueError(f'{name}: require increasing finite limits and >=2 points')
        values=[lo+(hi-lo)*i/(n-1) for i in range(n)]
        if integer:
            values=[float(round(x)) for x in values]
    elif isinstance(specification,list) and specification:
        values=list(specification)
    else:
        raise ValueError(f'{name}: supply an explicit nonempty list or grid limits')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
        raise ValueError(f'{name}: all grid values must be finite numbers')
    if integer and any(x<=0 or x!=int(x) for x in values):
        raise ValueError(f'{name}: Sunbather requires positive integer Parker temperatures')
    if any(a>=b for a,b in zip(values,values[1:])):
        raise ValueError(f'{name}: grid values must be unique and increasing')
    return [float(x) for x in values]


@dataclass(frozen=True)
class HeliumGridConfig:
    """A user-selected physical grid and explicitly defined measured spectrum."""

    enabled: bool
    signal_reviewed: bool
    input: dict
    temperature_grid_K: list | dict
    log10_mass_loss_grid_g_s: list | dict
    atmosphere: dict
    resolving_power: float
    fit_window_nm: list[float]
    velocity_bounds_kms: list[float]
    velocity_grid_points: int
    exposure_quadrature_points: int
    instrumental_oversampling: int
    covariance_eigenvalue_rtol: float
    workers: int
    output_name: str
    phase_sampling: dict | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous data conventions and missing physical choices."""
        if any(type(getattr(self,name)) is not bool for name in ('enabled','signal_reviewed','resume')):
            raise ValueError('Grid enabled, signal_reviewed and resume must be booleans')
        if self.enabled and not self.signal_reviewed:
            raise ValueError('Assess the helium feature first, then set helium_grid.signal_reviewed=true to investigate it; a grid fit is not a detection test')
        axis_values(self.temperature_grid_K,'temperature_grid_K',integer=True)
        axis_values(self.log10_mass_loss_grid_g_s,'log10_mass_loss_grid_g_s')
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError('workers must be a positive integer: the number of independent Sunbather models calculated concurrently')
        if not math.isfinite(self.resolving_power) or self.resolving_power<=0:
            raise ValueError('Supply a positive instrumental resolving power')
        for name in ('fit_window_nm','velocity_bounds_kms'):
            pair=getattr(self,name)
            if len(pair)!=2 or not all(math.isfinite(x) for x in pair) or pair[0]>=pair[1]:
                raise ValueError(f'{name} must contain two increasing finite values')
        if self.fit_window_nm[0]<=0:
            raise ValueError('Fit wavelengths must be positive vacuum nm')
        if type(self.velocity_grid_points) is not int or self.velocity_grid_points<3:
            raise ValueError('Use at least three coarse velocity samples before refinement')
        if type(self.exposure_quadrature_points) is not int or self.exposure_quadrature_points<1:
            raise ValueError('Exposure quadrature count must be a positive integer')
        if type(self.instrumental_oversampling) is not int or self.instrumental_oversampling<2:
            raise ValueError('Supply instrumental oversampling >=2 for native pixel integration')
        if not 0<self.covariance_eigenvalue_rtol<1:
            raise ValueError('Covariance eigenvalue tolerance must lie between zero and one')
        from pathlib import Path
        if not self.output_name or self.output_name in ('.','..') or Path(self.output_name).name!=self.output_name:
            raise ValueError('Supply a single new grid output directory name')
        data=self.input
        if data.get('kind') not in ('exoplore','ascii','fits') or not data.get('path'):
            raise ValueError('Supply helium_grid.input.kind and path: exoplore folder, ascii or fits spectrum')
        if data.get('uncertainty') not in ('saved_covariance','diagonal','external_covariance'):
            raise ValueError('Choose saved_covariance, diagonal or external_covariance explicitly')
        if data['uncertainty']=='external_covariance' and not data.get('covariance_path'):
            raise ValueError('External covariance requires covariance_path')
        if data['kind']!='exoplore':
            if data.get('sampling') not in ('point','pixel_average'):
                raise ValueError('Declare whether external values are point samples or pixel averages')
            if data.get('wavelength_unit') not in ('nm','um','angstrom'):
                raise ValueError('Declare external wavelength_unit: nm, um or angstrom')
            if data.get('wavelength_frame')!='planet_rest_vacuum':
                raise ValueError('External fit spectra must be in the planetary rest frame and vacuum')
            if data.get('value_kind') not in ('transmission','absorption_fraction','absorption_percent','transmission_excess_percent'):
                raise ValueError('Declare the spectrum convention and corresponding uncertainty units')
            if not isinstance(data.get('columns'),dict) or set(data['columns'])!={'wavelength','value','uncertainty'}:
                raise ValueError('Map wavelength, value and uncertainty columns explicitly')
            phase=self.phase_sampling
            if not phase or phase.get('mode') not in ('mid_transit','explicit'):
                raise ValueError('External coadds require explicit transit averaging or a declared mid_transit approximation')
            if phase['mode']=='explicit':
                for key in ('phase_midpoints','weights','exposure_seconds'):
                    if key not in phase:
                        raise ValueError('Explicit averaging requires phase_midpoints, weights and exposure_seconds')
                n=len(phase['phase_midpoints'])
                if not n or len(phase['weights'])!=n or len(phase['exposure_seconds'])!=n:
                    raise ValueError('External phase, weight and exposure-time arrays must match')
                if not all(math.isfinite(x) and abs(x)<.25 for x in phase['phase_midpoints']):
                    raise ValueError('Provide finite phases near primary transit')
                if not all(math.isfinite(x) and x>=0 for x in phase['weights']+phase['exposure_seconds']) or sum(phase['weights'])<=0:
                    raise ValueError('Averaging weights must sum positively and durations must be nonnegative')
        a=self.atmosphere
        required=('hydrogen_number_fraction','radial_max_rp','irradiation_spectrum_path','irradiation_source',
                  'model_wavelength_range_nm','model_wavelength_points','line_of_sight_points',
                  'limb_darkening_law','limb_darkening_coefficients','model_edge_absorption_tolerance',
                  'helium_vacuum_lines_nm','helium_oscillator_strengths','helium_atomic_source','sunbather')
        missing=[k for k in required if k not in a]
        if missing:
            raise ValueError('Missing grid atmosphere settings: '+', '.join(missing))
        if not 0<a['hydrogen_number_fraction']<1 or a['radial_max_rp']<=1 or a['radial_max_rp']!=int(a['radial_max_rp']):
            raise ValueError('Supply a mixed H/He composition and integer outer radius >Rp')
        if type(a['model_wavelength_points']) is not int or a['model_wavelength_points']<100 or type(a['line_of_sight_points']) is not int or a['line_of_sight_points']<20:
            raise ValueError('Supply sufficient wavelength and line-of-sight sampling')
        bounds=a['model_wavelength_range_nm']
        if len(bounds)!=2 or not all(math.isfinite(x) and x>0 for x in bounds) or bounds[0]>=bounds[1]:
            raise ValueError('Supply increasing physical model wavelength bounds in vacuum nm')
        if not bounds[0]<self.fit_window_nm[0]<self.fit_window_nm[1]<bounds[1]:
            raise ValueError('The physical model grid must extend beyond the fit window')
        if len(a['helium_vacuum_lines_nm'])!=3 or len(a['helium_oscillator_strengths'])!=3 or not a['helium_atomic_source']:
            raise ValueError('Supply the three vacuum wavelengths, oscillator strengths and their source')
        if not all(math.isfinite(x) and x>0 for x in a['helium_vacuum_lines_nm']+a['helium_oscillator_strengths']):
            raise ValueError('Helium atomic data must be finite and positive')
        law=a['limb_darkening_law']
        if law not in ('none','linear','quadratic') or len(a['limb_darkening_coefficients'])!={'none':0,'linear':1,'quadratic':2}.get(law):
            raise ValueError('Supply a supported limb-darkening law and matching coefficients')
        from exoplore.config.sunbather import SunbatherConfig
        SunbatherConfig(**dict(a['sunbather'],project_path='assigned inside each new grid model'))
