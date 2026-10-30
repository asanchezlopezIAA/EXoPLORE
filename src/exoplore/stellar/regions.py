"""Project surface regions and select components at stellar sample positions.

The observer looks along +z. x and y follow the existing transit sky frame.
Increasing stellar longitude moves an equatorial region towards positive
x*cos(obliquity)-y*sin(obliquity), the receding rotation direction.
"""
import numpy as np


def cap_centre(region, surface, obliquity_deg, time_bjd):
    """Return a rotating spherical-cap centre in observer-frame unit coordinates."""
    if time_bjd is None or not np.isfinite(time_bjd):
        raise ValueError('Physical stellar regions require exposure times in BJD_TDB')
    inc, angle = np.deg2rad([surface.stellar_inclination_deg, obliquity_deg])
    # The pole and the central meridian form an orthonormal stellar basis.
    pole = np.array([np.sin(angle)*np.sin(inc), np.cos(angle)*np.sin(inc), np.cos(inc)])
    meridian = np.array([-np.sin(angle)*np.cos(inc), -np.cos(angle)*np.cos(inc), np.sin(inc)])
    east = np.array([np.cos(angle), -np.sin(angle), 0.])
    lat = np.deg2rad(region.latitude_deg)
    lon = np.deg2rad(region.longitude_deg + surface.reference_phase_deg
                    + 360*(time_bjd-surface.reference_bjd)/surface.rotation_period_days)
    return np.sin(lat)*pole + np.cos(lat)*(np.cos(lon)*meridian+np.sin(lon)*east)


def component_map(x, y, surface, obliquity_deg=0., time_bjd=None):
    """Return component names with deterministic priority and visible-disk clipping."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Surface coordinates must be finite')
    visible = x*x+y*y <= 1.
    z = np.sqrt(np.maximum(1-x*x-y*y, 0.))
    labels = np.full(x.shape, surface.default_component, dtype=object)
    # Stable sorting lets a later list entry win when priorities are equal.
    for region in sorted(surface.regions, key=lambda r: r.priority):
        if region.geometry == 'projected':
            inside = (x-region.x)**2+(y-region.y)**2 <= region.radius**2
        else:
            centre = cap_centre(region, surface, obliquity_deg, time_bjd)
            inside = x*centre[0]+y*centre[1]+z*centre[2] >= np.cos(np.deg2rad(region.angular_radius_deg))
        labels[inside & visible] = region.component
    return labels


def ring_boundaries(radius, px, py, region, surface, obliquity, time):
    """Return angular crossings of a sampling ring with a region boundary."""
    if region.geometry == 'projected':
        dx, dy = region.x-px, region.y-py
        distance = np.hypot(dx, dy)
        if distance == 0:
            return []
        value = (radius**2+distance**2-region.radius**2)/(2*radius*distance)
        if abs(value) >= 1:
            return []
        angle, width = np.arctan2(dy, dx), np.arccos(value)
        return [angle-width, angle+width]
    # Intersect a projected ring with the spherical-cap boundary. With the
    # tangent-half-angle substitution, the squared equation is a quartic.
    from numpy.polynomial import polynomial as poly
    centre = cap_centre(region, surface, obliquity, time)
    cx, cy, cz = centre
    q = np.cos(np.deg2rad(region.angular_radius_deg))-cx*px-cy*py
    if abs(cz) < 1e-12:
        amplitude = radius*np.hypot(cx, cy)
        if amplitude == 0 or abs(q) >= amplitude:
            return []
        angle, width = np.arctan2(cy, cx), np.arccos(q/amplitude)
        return [angle-width, angle+width]
    den, cosine, sine = np.array([1., 0., 1.]), np.array([1., 0., -1.]), np.array([0., 2., 0.])
    rhs = q*den-radius*(cx*cosine+cy*sine)
    height = poly.polysub((1-px*px-py*py-radius**2)*poly.polymul(den, den),
                         2*radius*poly.polymul(px*cosine+py*sine, den))
    equation = poly.polysub(poly.polymul(rhs, rhs), cz*cz*height)
    roots = poly.polyroots(poly.polytrim(equation, tol=1e-14))
    angles = []
    # theta=pi is the infinite tangent-half-angle root and is checked explicitly.
    candidates = [np.pi]+[2*np.arctan(root.real) for root in roots if abs(root.imag) <= 1e-6]
    for angle in candidates:
        x, y = px+radius*np.cos(angle), py+radius*np.sin(angle)
        if x*x+y*y <= 1+1e-12:
            z = np.sqrt(max(0., 1-x*x-y*y))
            if abs(cx*x+cy*y+cz*z-np.cos(np.deg2rad(region.angular_radius_deg))) < 1e-7:
                angles.append(angle)
    return angles


def cap_radial_boundaries(centre, angular_radius_deg, px, py):
    """Find radial tangencies and limb endpoints of a projected spherical cap.

    Its boundary projects to an ellipse. Splitting radial quadrature at the
    visible ellipse's distance extrema resolves where a ring enters the cap.
    """
    from numpy.polynomial import polynomial as poly
    alpha = np.deg2rad(angular_radius_deg)
    rho = np.hypot(*centre[:2])
    direction = centre[:2]/rho if rho > 1e-14 else np.array([1., 0.])
    perpendicular = np.array([-direction[1], direction[0]])
    a = np.sin(alpha)*centre[2]*direction
    b = np.sin(alpha)*perpendicular
    offset = centre[:2]*np.cos(alpha)-np.array([px, py])
    # Differentiate squared projected distance, using tan(theta/2) for the ellipse angle.
    den, sine, cosine = np.array([1., 0., 1.]), np.array([0., 2., 0.]), np.array([1., 0., -1.])
    equation = poly.polyadd(-np.dot(offset, a)*poly.polymul(sine, den),
                            np.dot(offset, b)*poly.polymul(cosine, den))
    equation = poly.polyadd(equation, (np.dot(b,b)-np.dot(a,a))*poly.polymul(sine, cosine))
    roots = poly.polyroots(poly.polytrim(equation, tol=1e-14))
    angles = [0., np.pi]+[2*np.arctan(root.real) for root in roots if abs(root.imag) < 1e-7]
    height_amplitude = np.sin(alpha)*rho
    if height_amplitude > 1e-14:
        limit = centre[2]*np.cos(alpha)/height_amplitude
        if abs(limit) <= 1:
            angles.extend([np.arccos(limit), -np.arccos(limit)])
    boundaries = []
    for angle in angles:
        z = centre[2]*np.cos(alpha)-height_amplitude*np.cos(angle)
        if z >= -1e-12:
            boundaries.append(float(np.linalg.norm(offset+a*np.cos(angle)+b*np.sin(angle))))
    return boundaries
