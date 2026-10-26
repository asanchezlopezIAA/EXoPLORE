"""Physical inputs shared by atmospheric spectrum calculations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RadiativeTransferInputs:
    """Planet, star and instrument quantities used by radiative transfer.

    Masses and radii use CGS units, temperatures use kelvin and resolving
    power is dimensionless. gravity_cgs supplies an explicit gravity when
    requested. With gravity_cgs=None the calculation uses the supplied mass
    and radius. A transit-only model can use stellar_teff_K=None.

    The isothermal profile uses its specified temperature. Its inputs can
    therefore carry internal_temperature_K=None when that quantity is unused.
    """

    event_type: str
    planet_mass_g: float | None
    planet_radius_cm: float
    stellar_radius_cm: float
    stellar_teff_K: float | None
    resolving_power: float
    internal_temperature_K: float | None
    gravity_cgs: float | None
    convolve_to_resolution: bool | None
