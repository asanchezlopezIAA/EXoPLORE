"""exoplore.atmosphere, atmospheric forward model building blocks."""

from exoplore.atmosphere.pressure import create_log_pressure_grid
from exoplore.atmosphere.temperature import (
    isothermal_profile,
    two_point_profile,
    guillot_profile,
    # public aliases
    create_temperature_profile,
    calculate_temperature_structure,
    calculate_temperature_structure_limbs,
    create_pressure_temperature_structure,
    create_pressure_temperature_structure2,
)
from exoplore.atmosphere.chemistry import (
    compute_equilibrium_chemistry,
    manual_mass_fractions,
    # public API
    mass_frac_to_vmr,
)
from exoplore.atmosphere.prt import (
    create_radtrans,
    compute_transmission_spectrum,
    compute_emission_spectrum,
    compute_limb_averaged_transit_spectrum,
    surface_gravity_cgs,
    # public API
    convolve,
    convolve_velocity,
    call_easyCHEM,
    call_pRT,
    call_pRT_limbs,
    pRT_LRES_stellar_model,
)
from exoplore.atmosphere.winds import (
    planet_rot_vel,
    rotation_angle_during_transit,
    atmospheric_scale_height,
    create_velocity_grid,
    rotation_kernel_Maguire24,
    wind_broadening_triangular_kernel,
    convolve_spectrum_with_kernel,
    get_sflimbs,
)

__all__ = [
    # pressure
    "create_log_pressure_grid",
    # temperature (clean API)
    "isothermal_profile",
    "two_point_profile",
    "guillot_profile",
    # temperature
    "create_temperature_profile",
    "calculate_temperature_structure",
    "calculate_temperature_structure_limbs",
    "create_pressure_temperature_structure",
    "create_pressure_temperature_structure2",
    # chemistry
    "compute_equilibrium_chemistry",
    "manual_mass_fractions",
    "mass_frac_to_vmr",
    # pRT (clean API)
    "create_radtrans",
    "compute_transmission_spectrum",
    "compute_emission_spectrum",
    "compute_limb_averaged_transit_spectrum",
    "surface_gravity_cgs",
    # pRT
    "convolve",
    "convolve_velocity",
    "call_easyCHEM",
    "call_pRT",
    "call_pRT_limbs",
    "pRT_LRES_stellar_model",
    # winds / rotation
    "planet_rot_vel",
    "rotation_angle_during_transit",
    "atmospheric_scale_height",
    "create_velocity_grid",
    "rotation_kernel_Maguire24",
    "wind_broadening_triangular_kernel",
    "convolve_spectrum_with_kernel",
    "get_sflimbs",
]
