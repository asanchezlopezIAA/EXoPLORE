"""
exoplore.atmosphere.pressure
=============================

Pressure grid construction for petitRADTRANS (pRT) calculations.

pRT works with a 1-D pressure grid in bar.  Atmospheres are typically
sampled logarithmically from a very low pressure at the top (e.g. 1e-6 bar)
to a high pressure at the bottom (e.g. 100 bar = 1e2 bar).

All functions return numpy arrays in bar.

Examples
--------
>>> from exoplore.atmosphere import create_log_pressure_grid
>>> p = create_log_pressure_grid(1e-6, 1e2, 100)
>>> p.shape
(100,)
>>> p[0], p[-1]
(1e-06, 100.0)
"""

from __future__ import annotations

import numpy as np


def create_log_pressure_grid(
    pressure_min_bar: float = 1e-6,
    pressure_max_bar: float = 1e2,
    size: int = 100,
) -> np.ndarray:
    """Create a logarithmically spaced pressure grid.

    Parameters
    ----------
    pressure_min_bar:
        Pressure at the top of the atmosphere in bar (e.g. 1e-6).
    pressure_max_bar:
        Pressure at the bottom of the atmosphere in bar (e.g. 1e2 = 100 bar).
    size:
        Number of pressure levels.

    Returns
    -------
    numpy.ndarray, shape (size,)
        Pressure grid in bar, from ``pressure_min_bar`` to
        ``pressure_max_bar``, logarithmically spaced.

    Examples
    --------
    >>> p = create_log_pressure_grid(1e-6, 1e2, 100)
    >>> len(p)
    100
    """
    if pressure_min_bar <= 0:
        raise ValueError("pressure_min_bar must be positive.")
    if pressure_max_bar <= pressure_min_bar:
        raise ValueError("pressure_max_bar must be greater than pressure_min_bar.")
    if size < 2:
        raise ValueError("size must be at least 2.")
    return np.logspace(
        np.log10(pressure_min_bar),
        np.log10(pressure_max_bar),
        size,
    )
