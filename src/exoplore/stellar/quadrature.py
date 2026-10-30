"""Region-aware quadrature using the existing projected stellar geometry.

Angular intervals are split at component boundaries before integration.
The planetary atmosphere keeps every supplied pRT impact parameter.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from .regions import ring_boundaries, cap_centre, cap_radial_boundaries


@dataclass(frozen=True)
class ComponentRays:
    x: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    velocity: np.ndarray
    weights: np.ndarray
    rings: np.ndarray
    labels: np.ndarray


def radial_rule(radius, count, field, px=0., py=0., time=None):
    """Split squared-radius integration near stellar and region intersections."""
    breaks = [0., radius, abs(1-np.hypot(px, py))]
    for region in field.settings.regions:
        if region.geometry == 'projected':
            x, y, extent = region.x, region.y, region.radius
        else:
            centre = cap_centre(region, field.settings, field.obliquity, time)
            breaks.extend(cap_radial_boundaries(centre, region.angular_radius_deg, px, py))
            continue
        distance = np.hypot(x-px, y-py)
        breaks.extend([max(0., distance-extent), distance+extent])
    breaks = np.unique([v for v in breaks if 0 <= v <= radius])
    nodes, weights = leggauss(count)
    radii, areas = [], []
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        radii.extend(np.sqrt(lower**2+(nodes+1)/2*(upper**2-lower**2)))
        areas.extend((upper**2-lower**2)*weights/4)
    return np.asarray(radii), np.asarray(areas)


def component_rays(model, radii, px, py, time):
    """Locate local stellar samples along illuminated arcs of each radius."""
    distance = np.hypot(px, py)
    theta, radial_ids, weights = [], [], []
    for index, radius in enumerate(radii):
        if distance == 0:
            start, width = 0., 2*np.pi if radius < 1 else 0.
        else:
            cut = np.arccos(np.clip((1-distance**2-radius**2)/(2*distance*radius), -1, 1))
            start, width = np.arctan2(py, px)+cut, 2*(np.pi-cut)
        if width <= 0:
            continue
        breaks = [start, start+width]
        for region in model.field.settings.regions:
            for angle in ring_boundaries(radius, px, py, region,
                    model.field.settings, model.field.obliquity, time):
                mapped = start+np.mod(angle-start, 2*np.pi)
                if start < mapped < start+width:
                    breaks.append(mapped)
        breaks = np.unique(breaks)
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            angles = lo+(model._angle_nodes+1)/2*(hi-lo)
            theta.extend(angles)
            radial_ids.extend([index]*len(angles))
            weights.extend(model._angle_weights*(hi-lo)/2)
    if not theta:
        return None
    ids, theta = np.asarray(radial_ids), np.asarray(theta)
    x, y = px+radii[ids]*np.cos(theta), py+radii[ids]*np.sin(theta)
    mu = np.sqrt(np.maximum(1-x*x-y*y, 0))
    velocity = model.vsini_kms*(x*np.cos(model.obliquity)-y*np.sin(model.obliquity))
    return ComponentRays(x, y, mu, velocity, np.asarray(weights), ids,
                         model.field.labels(x, y, time))
