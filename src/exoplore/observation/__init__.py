"""
exoplore.observation
====================

Observation geometry and timing for high-resolution exoplanet spectroscopy.

Modules
-------
timing
    Orbital phase, in-transit indices, Julian date grids.
velocity
    Planet and stellar radial velocities (circular + eccentric orbits).
airmass
    Synthetic airmass curves for ETC-based simulations.
noise
    Shot-noise and read-noise models, SNR estimation.
"""

from exoplore.observation.timing import (
    orbital_phase,
    in_transit_indices,
    observation_julian_dates,
    transit_contact_times,
    # public API
    get_event,
    get_event_v2,
    find_nights_with_extrema,
    get_transit,
    dayside_fraction,
    UTC_to_TDB_CARMENES,
    block_parameter,
)
from exoplore.observation.velocity import (
    planet_radial_velocity,
    planet_radial_velocity_eccentric,
    stellar_radial_velocity,
    # public API
    get_V,
    get_V_eccentric,
    compute_Kp,
    get_phase,
)
from exoplore.observation.airmass import (
    synthetic_airmass,
    # public API
    get_airmass,
    skycalc_model,
    pwv_gen_skycalc,
    PWV_handling,
)
from exoplore.observation.noise import (
    photon_noise,
    total_noise,
    snr_per_pixel,
    compute_global_exposure_limit,
    add_throughput,
)

__all__ = [
    # timing (clean API)
    "orbital_phase",
    "in_transit_indices",
    "observation_julian_dates",
    "transit_contact_times",
    # timing
    "get_event",
    "find_nights_with_extrema",
    "get_transit",
    "dayside_fraction",
    "UTC_to_TDB_CARMENES",
    "block_parameter",
    # velocity (clean API)
    "planet_radial_velocity",
    "planet_radial_velocity_eccentric",
    "stellar_radial_velocity",
    # velocity
    "get_V",
    "get_V_eccentric",
    "compute_Kp",
    "get_phase",
    # airmass (clean API)
    "synthetic_airmass",
    # airmass
    "get_airmass",
    "skycalc_model",
    "pwv_gen_skycalc",
    "PWV_handling",
    # noise
    "photon_noise",
    "total_noise",
    "snr_per_pixel",
    "compute_global_exposure_limit",
    "add_throughput",
]
