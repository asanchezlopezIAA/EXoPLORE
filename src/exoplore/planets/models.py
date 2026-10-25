"""
exoplore.planets.models
=======================

Typed representation of a planet + host star system.

All quantities use human-friendly astronomical units and the field name
always includes the unit suffix (e.g. ``orbital_period_days``,
``planet_radius_rjup``).  This makes the JSON file self-documenting, 
a scientist reading the file knows exactly what every number means
without digging into the code.

Supply the physical system parameters explicitly. The class checks that
these values are finite. Derived quantities (Kp, stellar RV amplitude and
transit duration) are calculated in ``__post_init__`` when omitted.

Use :func:`exoplore.planets.catalog.load_planet` to create an instance
from a JSON file. The simulator reads its named physical fields and
unit-conversion properties.
"""

from __future__ import annotations

# Role in the calculation
# Store planetary-system parameters and calculate derived orbital quantities. Unit-
# labelled fields supply the atmosphere and timing code. Properties
# convert them to the conventions used by their callers.


import math
from numbers import Real
from dataclasses import dataclass, field, fields, MISSING
from typing import List, Optional


@dataclass
class PlanetParameters:
    """Physical and orbital parameters for one planet + host star.

    Parameters
    ----------
    name : str
        Planet name, e.g. ``"HD189733b"``.

    Orbital
    -------
    orbital_period_days : float
        Orbital period in days.
    transit_epoch_bjd : float
        Reference mid-transit time in BJD (TDB).
    semi_major_axis_au : float
        Semi-major axis in AU.
    inclination_deg : float
        Orbital inclination in degrees (90 = edge-on).
    eccentricity : float
        Orbital eccentricity (0 = circular).
    argument_of_periastron_deg : float
        Argument of periastron ω in degrees.

    Planet
    ------
    planet_radius_rjup : float
        Planet radius in Jupiter radii.
    planet_mass_mjup : float
        Planet mass in Jupiter masses.

    Star
    ----
    stellar_radius_rsun : float
        Stellar radius in solar radii.
    stellar_mass_msun : float
        Stellar mass in solar masses.
    stellar_teff_K : float
        Stellar effective temperature in K.
    stellar_logg : float
        Stellar surface gravity log g (cgs).
    stellar_metallicity : float
        Stellar [Fe/H].
    v_rotsini_kms : float
        Projected stellar rotational velocity v sin i in km/s.

    Velocities
    ----------
    systemic_velocity_kms : float
        Systemic (barycentric) RV of the system in km/s.
    kp_kms : float or None
        Planet RV semi-amplitude Kp in km/s.  Computed automatically
        from orbital parameters if not provided.
    stellar_rv_semiamplitude_kms : float or None
        Stellar RV semi-amplitude K_s in km/s.  Computed automatically
        from masses and orbital parameters if not provided.

    Atmospheric / thermal
    ---------------------
    equilibrium_temperature_K : float
        Planet equilibrium temperature in K.
    t_int_K : float
        Planet internal temperature in K (Guillot T-P profile).
    kappa_ir : float
        Infrared opacity κ_IR (Guillot profile).
    gamma_guillot : float
        Ratio of optical to IR opacity γ (Guillot profile).

    Limb darkening
    --------------
    limb_darkening_coeffs : list of float
        Quadratic limb-darkening coefficients [u1, u2] for the batman
        transit model.

    Sky coordinates
    ---------------
    ra_deg : float or None
        Right ascension of the host star in degrees.
    dec_deg : float or None
        Declination of the host star in degrees.

    Derived (auto-computed)
    -----------------------
    transit_duration_hours : float or None
        Full transit duration T₁₄ in hours.  Computed from geometry
        if not supplied.
    """

    name: str

    # ── Orbital ───────────────────────────────────────────────────────────
    orbital_period_days: float
    transit_epoch_bjd: float
    semi_major_axis_au: float
    inclination_deg: float
    eccentricity: float
    argument_of_periastron_deg: float

    # ── Planet ────────────────────────────────────────────────────────────
    planet_radius_rjup: float
    planet_mass_mjup: float

    # ── Star ──────────────────────────────────────────────────────────────
    stellar_radius_rsun: float
    stellar_mass_msun: float
    stellar_teff_K: float
    stellar_logg: float
    stellar_metallicity: float
    v_rotsini_kms: float

    # ── Velocities ────────────────────────────────────────────────────────
    systemic_velocity_kms: float

    # ── Atmospheric / thermal ─────────────────────────────────────────────
    equilibrium_temperature_K: float
    t_int_K: float
    kappa_ir: float
    gamma_guillot: float

    # ── Limb darkening ────────────────────────────────────────────────────
    limb_darkening_coeffs: List[float]

    # ── Sky coordinates ───────────────────────────────────────────────────
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None

    # ── Derived ───────────────────────────────────────────────────────────
    kp_kms: Optional[float] = None
    stellar_rv_semiamplitude_kms: Optional[float] = None
    transit_duration_hours: Optional[float] = None

    # ─────────────────────────────────────────────────────────────────────
    # Physical constants (class-level, not stored in JSON)
    # ─────────────────────────────────────────────────────────────────────
    _AU_km: float = field(default=1.496e8, init=False, repr=False, compare=False)
    _R_jup_km: float = field(default=71_492.0, init=False, repr=False, compare=False)
    _R_sun_km: float = field(default=695_700.0, init=False, repr=False, compare=False)
    _M_jup_kg: float = field(default=1.898e27, init=False, repr=False, compare=False)
    _M_sun_kg: float = field(default=1.989e30, init=False, repr=False, compare=False)
    _R_jup_m: float = field(default=7.1492e7, init=False, repr=False, compare=False)
    _R_sun_m: float = field(default=6.957e8, init=False, repr=False, compare=False)
    _AU_m: float = field(default=1.496e11, init=False, repr=False, compare=False)
    # CGS constants used to match the petitRADTRANS unit convention
    # (G = 6.674e-8 cm³ g⁻¹ s⁻² in CGS, so gravity must be in CGS)
    _R_jup_cm: float = field(default=7.1492e9, init=False, repr=False, compare=False)
    _R_sun_cm: float = field(default=6.957e10, init=False, repr=False, compare=False)
    _M_jup_g: float = field(default=1.898e30, init=False, repr=False, compare=False)
    _M_sun_g: float = field(default=1.989e33, init=False, repr=False, compare=False)

    # ─────────────────────────────────────────────────────────────────────
    # Derived quantity calculators
    # ─────────────────────────────────────────────────────────────────────

    def compute_kp(self) -> float:
        """Compute the planet radial-velocity semi-amplitude Kp in km/s.

        Kp is the projected orbital speed of the planet along the line of
        sight.  It is the primary quantity used as the y-axis of the
        Kp-Vsys (Kp-Vrest) detection map: a genuine atmospheric signal
        appears as a peak at the true Kp value.

        The formula applied is:

        .. math::

            K_p = \\frac{2\\pi\\, a}{P\\,\\sqrt{1-e^2}} \\sin i

        where :math:`a` is the semi-major axis in km, :math:`P` is the
        orbital period in seconds, :math:`e` is the eccentricity, and
        :math:`i` is the inclination.

        Parameters
        ----------
        (no arguments, reads from dataclass fields)
            ``semi_major_axis_au`` (AU), ``orbital_period_days`` (days),
            ``inclination_deg`` (degrees), ``eccentricity`` (dimensionless).

        Returns
        -------
        float
            Planet RV semi-amplitude Kp in km/s.

        Notes
        -----
        For a circular edge-on orbit of HD 189733 b
        (a ≈ 0.031 AU, P ≈ 2.22 d, i ≈ 85.7°) this gives Kp ≈ 152 km/s,
        consistent with the literature value.
        """
        period_s = self.orbital_period_days * 24.0 * 3600.0
        a_km = self.semi_major_axis_au * self._AU_km
        inc_rad = math.radians(self.inclination_deg)
        return (
            (2.0 * math.pi * a_km)
            / (period_s * math.sqrt(1.0 - self.eccentricity ** 2))
            * math.sin(inc_rad)
        )

    def compute_stellar_rv_semiamplitude(self) -> float:
        """Compute the stellar radial-velocity semi-amplitude K_s in km/s.

        K_s is the reflex velocity of the host star induced by the orbiting
        planet.  Although typically only a few hundred m/s and unresolved in
        individual high-resolution exposures, it is used in the simulator to
        construct the stellar template in the system barycentre frame and to
        apply the stellar Doppler shift to the injected starlight.

        The formula applied is:

        .. math::

            K_s = \\frac{2\\pi\\,a\\,M_p\\sin i}
                        {P\\,(M_p+M_s)\\,\\sqrt{1-e^2}}

        where masses are in kg, semi-major axis in m, period in seconds.

        Parameters
        ----------
        (no arguments, reads from dataclass fields)
            ``planet_mass_mjup`` (M_Jup), ``stellar_mass_msun`` (M_sun),
            ``semi_major_axis_au`` (AU), ``orbital_period_days`` (days),
            ``inclination_deg`` (degrees), ``eccentricity`` (dimensionless).

        Returns
        -------
        float
            Stellar RV semi-amplitude K_s in km/s.  Returns 0.0 if either
            mass is zero (placeholder / missing parameter).
        """
        if self.planet_mass_mjup == 0.0 or self.stellar_mass_msun == 0.0:
            return 0.0
        M_p_kg = self.planet_mass_mjup * self._M_jup_kg
        M_s_kg = self.stellar_mass_msun * self._M_sun_kg
        period_s = self.orbital_period_days * 24.0 * 3600.0
        inc_rad = math.radians(self.inclination_deg)
        # K_s = (2π/P) * (M_p sin i) / ((M_p + M_s) * sqrt(1-e²)) * a
        # Simplified using a/P = (G(M+m)/4π²)^(1/3) / P^(2/3)
        G = 6.674e-11  # m³ kg⁻¹ s⁻²
        a_m = self.semi_major_axis_au * self._AU_m
        K_s_ms = (
            (2.0 * math.pi * a_m * M_p_kg * math.sin(inc_rad))
            / (period_s * (M_p_kg + M_s_kg) * math.sqrt(1.0 - self.eccentricity ** 2))
        )
        return K_s_ms / 1000.0  # m/s → km/s

    def compute_transit_duration_hours(self) -> float:
        """Compute the full transit duration T₁₄ in hours from orbital geometry.

        This is the time between first and fourth contact (T₁ to T₄), i.e.,
        the duration over which any part of the planet's disk overlaps the
        stellar disk.  It is used by the simulator to select in-transit
        exposures and to set the integration window for the CCF.

        The T₁₄ formula applied assumes a **circular orbit** and the
        **small-planet approximation** is NOT used: the chord length includes
        both stellar and planetary radii (R_s + R_p).  The impact parameter
        b = (a / R_s) cos i is computed explicitly.  This is the standard
        T₁₄ expression:

        .. math::

            T_{14} = \\frac{P}{\\pi}
                     \\arcsin\\!\\left(
                         \\frac{\\sqrt{(R_s+R_p)^2 - (b\\,R_s)^2}}
                              {a \\sin i}
                     \\right)

        Parameters
        ----------
        (no arguments, reads from dataclass fields)
            ``planet_radius_rjup`` (R_Jup), ``stellar_radius_rsun`` (R_sun),
            ``semi_major_axis_au`` (AU), ``inclination_deg`` (degrees),
            ``orbital_period_days`` (days).

        Returns
        -------
        float
            Full transit duration T₁₄ in hours.  Returns 0.0 if the geometry
            is non-transiting (negative chord) or if required parameters are
            zero.

        Notes
        -----
        This formula is valid for circular orbits.  For eccentric orbits an
        additional factor :math:`\\sqrt{1-e^2} / (1 \\pm e\\sin\\omega)` applies,
        which is not implemented here. For near-circular hot Jupiters the
        correction is typically < 1 %.
        """
        R_p_km = self.planet_radius_rjup * self._R_jup_km
        R_s_km = self.stellar_radius_rsun * self._R_sun_km
        a_km = self.semi_major_axis_au * self._AU_km
        inc_rad = math.radians(self.inclination_deg)
        if R_s_km == 0.0 or a_km == 0.0:
            return 0.0
        b = (a_km / R_s_km) * math.cos(inc_rad)
        chord_sq = (R_s_km + R_p_km) ** 2 - (b * R_s_km) ** 2
        if chord_sq <= 0.0:
            return 0.0
        num = math.sqrt(chord_sq)
        den = a_km * math.sin(inc_rad)
        if den == 0.0:
            return 0.0
        duration_days = (self.orbital_period_days / math.pi) * math.asin(num / den)
        return duration_days * 24.0

    def __post_init__(self) -> None:
        # Required fields describe the physical system supplied by the user.
        # Validate them before calculating velocities or transit geometry.
        for item in fields(self):
            if item.default is not MISSING or item.default_factory is not MISSING:
                continue
            value = getattr(self, item.name)
            if item.name == "name":
                continue
            values = value if item.name == "limb_darkening_coeffs" else [value]
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"Planet parameter {item.name!r} requires numeric values.")
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                   for v in values):
                raise ValueError(f"Planet parameter {item.name!r} must contain finite numbers.")
        if self.kp_kms is None and self.semi_major_axis_au > 0:
            self.kp_kms = self.compute_kp()
        if self.stellar_rv_semiamplitude_kms is None and self.semi_major_axis_au > 0:
            self.stellar_rv_semiamplitude_kms = self.compute_stellar_rv_semiamplitude()
        if self.transit_duration_hours is None and self.orbital_period_days > 0:
            self.transit_duration_hours = self.compute_transit_duration_hours()

    # ─────────────────────────────────────────────────────────────────────
    # CGS / radian convenience properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def M_pl(self) -> float:
        """Planet mass in grams (CGS)."""
        return self.planet_mass_mjup * self._M_jup_g

    @property
    def R_pl(self) -> float:
        """Planet radius in cm (CGS)."""
        return self.planet_radius_rjup * self._R_jup_cm

    @property
    def R_star(self) -> float:
        """Stellar radius in cm (CGS)."""
        return self.stellar_radius_rsun * self._R_sun_cm

    @property
    def M_star(self) -> float:
        """Stellar mass in grams (CGS)."""
        return self.stellar_mass_msun * self._M_sun_g

    @property
    def a(self) -> float:
        """Semi-major axis in km (AU × _AU_km)."""
        return self.semi_major_axis_au * self._AU_km

    @property
    def inc(self) -> float:
        """Orbital inclination in radians."""
        import math
        return math.radians(self.inclination_deg)
