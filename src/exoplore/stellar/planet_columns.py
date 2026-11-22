"""Integrate two planetary sectors against the local stellar illumination.

Each sector keeps its native pRT impact parameters and opaque lower radius.
Angular intervals are split at the stellar limb, planetary sector boundary
and stellar component boundaries before evaluating their contributions.
"""
import numpy as np
from numba import njit
from .quadrature import ComponentRays
from .regions import ring_boundaries


def sector_rays(model, radii, px, py, time, leading):
    """Return illuminated samples in one planet-centred semicircle."""
    nodes, weights = model._angle_nodes, model._angle_weights
    distance = np.hypot(px, py)
    angles, ids, areas = [], [], []
    for index, radius in enumerate(radii):
        if distance == 0:
            start, width = 0., 2*np.pi if radius < 1 else 0.
        else:
            cut = np.arccos(np.clip((1-distance**2-radius**2)/(2*distance*radius),-1,1))
            start, width = np.arctan2(py,px)+cut, 2*(np.pi-cut)
        if width <= 0:
            continue
        breaks = [start,start+width]
        boundaries = [np.pi/2,3*np.pi/2]
        if model.field is not None:
            for region in model.field.settings.regions:
                boundaries.extend(ring_boundaries(radius,px,py,region,
                    model.field.settings,model.field.obliquity,time))
        for angle in boundaries:
            mapped = start+np.mod(angle-start,2*np.pi)
            if start < mapped < start+width:
                breaks.append(mapped)
        breaks = np.unique(breaks)
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            if (np.cos((lo+hi)/2) > 0) != leading:
                continue
            angles.extend(lo+(nodes+1)*(hi-lo)/2)
            ids.extend([index]*len(nodes))
            areas.extend(weights*(hi-lo)/2)
    if not angles:
        return None
    ids, angles = np.asarray(ids), np.asarray(angles)
    x, y = px+radii[ids]*np.cos(angles), py+radii[ids]*np.sin(angles)
    mu = np.sqrt(np.maximum(1-x*x-y*y,0))
    velocity = model.vsini_kms*(x*np.cos(model.obliquity)-y*np.sin(model.obliquity))
    labels = (np.full(len(x),'quiet') if model.field is None else model.field.labels(x,y,time))
    return ComponentRays(x,y,mu,velocity,np.asarray(areas),ids,labels)


def local_spectra(model, wave, rays):
    """Keep one stellar spectrum per ray until planetary absorption is applied."""
    if model.field is not None:
        return model.field.sample(wave,rays.mu,rays.velocity,rays.labels)
    from .coupled import _integrate_local_intensities
    local = model.surface.local
    if rays.mu.min() < local.mu[0] or rays.mu.max() > local.mu[-1]:
        raise ValueError('Local stellar intensities do not cover the two-column ray angles')
    if (wave[0]/(1+rays.velocity.max()/299792.458) < local.wavelength_um[0]
            or wave[-1]/(1+rays.velocity.min()/299792.458) > local.wavelength_um[-1]):
        raise ValueError('Local stellar wavelength coverage is too narrow for two-column rays')
    upper = np.clip(np.searchsorted(local.mu,rays.mu,side='right'),1,len(local.mu)-1)
    fraction = (rays.mu-local.mu[upper-1])/(local.mu[upper]-local.mu[upper-1])
    count = len(rays.mu)
    return _integrate_local_intensities(wave,np.arange(count),upper,fraction,rays.velocity,
        np.ones((count,1)),local.wavelength_um,local.intensity,local.continuum,count)


@njit(cache=True)
def absorbed_rays(wave, velocities, rings, native_wave, transmission):
    """Interpolate each radius-resolved transmission at its local Doppler shift."""
    values = np.empty((len(rings),len(wave)))
    for ray in range(len(rings)):
        rest = wave/(1+velocities[ray]/299792.458)
        values[ray] = 1-np.interp(rest,native_wave,transmission[:,rings[ray]])
    return values


def column_blocking(model, tables, star_wave, observed_wave, px, py, time, planet_velocity):
    """Return opaque line/continuum flux and atmospheric absorbed flux."""
    from .coupled import _occulting_disk_rings
    from .quadrature import radial_rule
    blocked, continuum, atmospheric = (np.zeros(len(star_wave)) for _ in range(3))
    for column,leading,wind in zip(tables.columns,(True,False),
                                 (tables.morning_wind_kms,tables.evening_wind_kms)):
        base = column.impact_parameter_cm[-1]/model.planet.R_star
        body_radii, body_weights = _occulting_disk_rings(base,
            model.choices.planet_radial_points,np.hypot(px,py))
        if model.field is not None:
            body_radii, body_weights = radial_rule(base,model.choices.planet_radial_points,
                                                   model.field,px,py,time)
        body = sector_rays(model,body_radii,px,py,time,leading)
        if body is not None:
            intensity, local_continuum = local_spectra(model,star_wave,body)
            weights = body.weights*body_weights[body.rings]
            blocked += weights@intensity
            continuum += weights@local_continuum
        radii = column.impact_parameter_cm/model.planet.R_star
        rays = sector_rays(model,radii,px,py,time,leading)
        if rays is None:
            continue
        intensity, _ = local_spectra(model,star_wave,rays)
        # Rigid rotation projected onto the line of sight is constant along
        # a parallel ray. Its x coordinate gives zero rotation at the poles.
        velocity = (planet_velocity+wind+tables.angular_speed_s*
            (rays.x-px)*model.planet.R_star/1e5*np.sin(tables.spin_inclination_rad))
        low = observed_wave[0]/(1+velocity.max()/299792.458)
        high = observed_wave[-1]/(1+velocity.min()/299792.458)
        if low < column.wavelength_um[0] or high > column.wavelength_um[-1]:
            raise ValueError('Planetary column needs wider wavelength coverage for local velocities')
        absorption = absorbed_rays(observed_wave,velocity,rays.rings,
                                   column.wavelength_um,column.transmission)
        ring_flux = np.zeros((len(radii),len(star_wave)))
        np.add.at(ring_flux,rays.rings,intensity*absorption*rays.weights[:,None])
        integrand = ring_flux*radii[:,None]
        atmospheric += np.sum((integrand[1:]+integrand[:-1])*
                               (-np.diff(radii))[:,None]/2,axis=0)
    return blocked, continuum, atmospheric
