"""Compatibility imports for :mod:`exoplore.observation.velocity`.

Orbital velocity calculations belong to the observation layer, where phase and
reference-frame conventions are defined. The original top-level import remains
valid for existing scripts.

Maintain the implementation in ``exoplore.observation.velocity``; this module
only preserves the older public entry points.
"""

# Import the original objects directly rather than wrapping them: signatures,
# defaults and numerical calculations then stay identical through either path.
from exoplore.observation.velocity import (
    planet_radial_velocity,
    planet_radial_velocity_eccentric,
    stellar_radial_velocity,
)

__all__ = ['planet_radial_velocity', 'planet_radial_velocity_eccentric', 'stellar_radial_velocity']
