"""
exoplore.planets
================

Planet and stellar system parameter objects.

The primary entry point is :func:`load_planet`, which reads a JSON file
and returns a :class:`PlanetParameters` dataclass.  The JSON files live
in ``planet_params/`` at the repository root, one file per target.
They use human-readable field names with explicit units
(``orbital_period_days``, ``planet_radius_rjup``, etc.) so they are
easy to read and edit.

Quick start
-----------
>>> from exoplore.planets import load_planet
>>> p = load_planet("planet_params/HD189733b.json")
>>> p.name
'HD189733b'
>>> round(p.kp_kms, 1)
152.8
>>> round(p.transit_duration_hours, 2)
1.83

inp_dat conversion
------------------
>>> inp_dat = p.to_inp_dat()
>>> inp_dat["K_p"]   # km/s, ready to pass to the simulator
152.8...
"""

from exoplore.planets.models import PlanetParameters
from exoplore.planets.catalog import load_planet

__all__ = [
    "PlanetParameters",
    "load_planet",
]
