"""
exoplore.atmosphere.temperature
================================

Temperature-pressure (T-P) profile functions for pRT calculations.

Three profile types are supported:

1. **Isothermal**, uniform temperature at all pressure levels.
2. **Two-point**, linear interpolation in log-pressure between two
   (pressure, temperature) anchor points.
3. **Guillot**, analytic Guillot (2010) irradiated atmosphere profile,
   implemented via petitRADTRANS's
   ``temperature_profile_function_guillot_global``.

All functions take a pressure grid in bar and return a temperature array
in Kelvin with the same shape.

Examples
--------
>>> import numpy as np
>>> from exoplore.atmosphere import create_log_pressure_grid, isothermal_profile
>>> p = create_log_pressure_grid(1e-6, 1e2, 100)
>>> T = isothermal_profile(p, 1200.0)
>>> T.shape, T[0]
((100,), 1200.0)
"""

from __future__ import annotations

import numpy as np


def isothermal_profile(
    pressure_grid_bar: np.ndarray,
    temperature_K: float,
) -> np.ndarray:
    """Return a uniform temperature profile.

    Parameters
    ----------
    pressure_grid_bar:
        Pressure grid in bar (any shape).
    temperature_K:
        Temperature in Kelvin at all levels.

    Returns
    -------
    numpy.ndarray
        Temperature array in K, same shape as ``pressure_grid_bar``.
    """
    return np.full_like(pressure_grid_bar, float(temperature_K))


def two_point_profile(
    pressure_grid_bar: np.ndarray,
    p_top_bar: float,
    t_top_K: float,
    p_bottom_bar: float,
    t_bottom_K: float,
) -> np.ndarray:
    """Return a temperature profile linearly interpolated in log-pressure.

    Temperature is interpolated between two anchor points
    ``(p_top, t_top)`` and ``(p_bottom, t_bottom)`` in log10-pressure
    space.  Pressures outside the anchor range are clamped to the
    nearest anchor temperature.

    Parameters
    ----------
    pressure_grid_bar:
        Pressure grid in bar, shape (n,).
    p_top_bar:
        Pressure of the upper anchor point in bar.
    t_top_K:
        Temperature at the upper anchor point in K.
    p_bottom_bar:
        Pressure of the lower anchor point in bar.
    t_bottom_K:
        Temperature at the lower anchor point in K.

    Returns
    -------
    numpy.ndarray
        Temperature array in K, shape (n,).

    Examples
    --------
    >>> from exoplore.atmosphere import create_log_pressure_grid, two_point_profile
    >>> p = create_log_pressure_grid(1e-6, 1e2, 100)
    >>> T = two_point_profile(p, 10**0.1, 1750.0, 10**-2.75, 520.0)
    """
    log_p = np.log10(pressure_grid_bar)
    log_p_top = np.log10(p_top_bar)
    log_p_bot = np.log10(p_bottom_bar)

    # Linear interpolation weight
    w = np.clip((log_p - log_p_top) / (log_p_bot - log_p_top), 0.0, 1.0)
    return t_top_K + w * (t_bottom_K - t_top_K)


def guillot_profile(
    pressure_grid_bar: np.ndarray,
    equilibrium_temperature_K: float,
    t_int_K: float,
    stellar_gravity_cgs: float,
    kappa_ir: float,
    gamma: float,
) -> np.ndarray:
    """Return a Guillot (2010) temperature-pressure profile.

    This is the same analytic profile used by petitRADTRANS.  It requires
    petitRADTRANS to be installed; if not available, a fallback isothermal
    profile at ``equilibrium_temperature_K`` is returned with a warning.

    Parameters
    ----------
    pressure_grid_bar:
        Pressure grid in bar, shape (n,).
    equilibrium_temperature_K:
        Planet equilibrium temperature in K.
    t_int_K:
        Planet internal temperature in K (default ~200 K for hot Jupiters).
    stellar_gravity_cgs:
        Stellar surface gravity in cm/s² (NOT log g).  For pRT this is
        the planetary surface gravity.
    kappa_ir:
        IR opacity in cm²/g.
    gamma:
        Ratio of optical to IR opacities (Guillot gamma parameter).

    Returns
    -------
    numpy.ndarray
        Temperature array in K, shape (n,).

    Notes
    -----
    The Guillot profile is defined in:
    Guillot, T. (2010), A&A 520, A27.
    """
    try:
        from petitRADTRANS.physics import (
            temperature_profile_function_guillot_global as guillot_func,
        )
        T = guillot_func(
            pressure_grid_bar,
            kappa_ir,
            gamma,
            stellar_gravity_cgs,
            t_int_K,
            equilibrium_temperature_K,
        )
        return T
    except ImportError:
        import warnings
        warnings.warn(
            "petitRADTRANS is not installed. Falling back to isothermal profile "
            f"at T_equ = {equilibrium_temperature_K} K.",
            ImportWarning,
            stacklevel=2,
        )
        return isothermal_profile(pressure_grid_bar, equilibrium_temperature_K)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_pressure_temperature_structure(p1, t1, p2, t2, pressures):
    """Linear-in-log10(P) two-point interpolation.

    Parameters
    ----------
    p1 : float  Pressure at the high-pressure anchor (bar).
    t1 : float  Temperature at *p1* (K).
    p2 : float  Pressure at the low-pressure anchor (bar).
    t2 : float  Temperature at *p2* (K).
    pressures : array-like  Full pressure grid (bar).

    Returns
    -------
    numpy.ndarray
    """
    t = np.zeros_like(pressures)
    slope = (t2 - t1) / (np.log10(p2) - np.log10(p1))
    intercept = t1 - slope * np.log10(p1)
    t[pressures > p1] = t1
    t[pressures < p2] = t2
    mask = (pressures <= p1) & (pressures >= p2)
    t[mask] = slope * np.log10(pressures[mask]) + intercept
    return t


def create_pressure_temperature_structure2(p1, t1, p2, t2, pressures):
    """Alternative two-point interpolation using natural log of pressure.
    """
    temperatures = np.zeros_like(pressures)
    temperatures[np.log(pressures) > np.log(p1)] = t1
    temperatures[np.log(pressures) < np.log(p2)] = t2
    idx = (np.log(pressures) >= np.log(p2)) & (np.log(pressures) <= np.log(p1))
    temperatures[idx] = (
        ((t1 - t2) / (np.log(p1) - np.log(p2)))
        * (np.log(pressures[idx]) - np.log(p2))
        + t2
    )
    return temperatures


def create_temperature_profile(
    inp_dat, gravity, isothermal, isothermal_value, T_equ,
    two_point_T, p_points, t_points, kappa, gamma, pressures
):
    """Choose and compute a temperature profile from simulation inputs.

    Parameters
    ----------
    inp_dat : dict  Must contain key ``"T_int"`` for the Guillot profile.
    gravity : float  Surface gravity (cm/s²).
    isothermal : bool  Use an isothermal profile.
    isothermal_value : float or None
        Fixed temperature (K).  If None and isothermal is True, *T_equ* is used.
    T_equ : float   Equilibrium temperature (K).
    two_point_T : bool  Use two-point approximation instead of Guillot.
    p_points : list[float, float]  [p_bottom_bar, p_top_bar].
    t_points : list[float, float]  [T_bottom_K, T_top_K].
    kappa : float  Guillot kappa_IR.
    gamma : float  Guillot gamma.
    pressures : array-like  Pressure grid (bar).

    Returns
    -------
    numpy.ndarray
    """
    from petitRADTRANS.physics import temperature_profile_function_guillot_global

    if isothermal:
        T_use = T_equ if isothermal_value is None else isothermal_value
        return np.full_like(pressures, T_use)
    elif not two_point_T:
        return temperature_profile_function_guillot_global(
            pressures, kappa, gamma, gravity, inp_dat["T_int"], T_equ
        )
    else:
        return create_pressure_temperature_structure(
            p_points[0], t_points[0], p_points[1], t_points[1], pressures
        )


def calculate_temperature_structure(
    inp_dat, pressures, gravity, isothermal, isothermal_value,
    T_equil, two_point_T, p_points, t_points, kappa, gamma, mode
):
    """Compute the temperature-pressure profile for a 1-D (non-limb) atmosphere.

    Selects among three profile families based on the boolean flags:

    * ``isothermal=True``  → uniform temperature at all pressure levels
      (``isothermal_value`` K, or ``T_equil`` if ``isothermal_value`` is None).
    * ``isothermal=False, two_point_T=False``  → analytic Guillot (2010)
      profile computed by petitRADTRANS, parameterised by ``kappa``,
      ``gamma``, ``gravity``, and ``inp_dat["T_int"]``.
    * ``isothermal=False, two_point_T=True``  → linear interpolation in
      log10(P) between two anchor points (``p_points``, ``t_points``).

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``"T_int"`` (internal
        temperature in K) when the Guillot profile is selected.
    pressures : numpy.ndarray
        Pressure grid in bar, shape ``(n_pressure_levels,)``.
    gravity : float
        Surface gravity in cm/s² (CGS, as used by petitRADTRANS).
    isothermal : bool
        If True, return a uniform temperature profile.
    isothermal_value : float or None
        Fixed temperature in K for the isothermal case.  If None,
        ``T_equil`` is used.
    T_equil : float
        Planet equilibrium temperature in K.  Used as the isothermal
        fallback and as the irradiation temperature in the Guillot profile.
    two_point_T : bool
        If True and ``isothermal=False``, use the two-point log-pressure
        interpolation instead of Guillot.
    p_points : list of float
        [p_bottom_bar, p_top_bar] anchor pressures for the two-point profile
        (bar).
    t_points : list of float
        [T_bottom_K, T_top_K] anchor temperatures for the two-point profile
        (K).
    kappa : float
        Guillot infrared opacity κ_IR (cm²/g).
    gamma : float
        Guillot ratio of optical to IR opacity γ (dimensionless).
    mode : str
        Accepted but unused (``'full'``, ``'morning'``, or ``'evening'``).
        Present for API compatibility with the limb variant.

    Returns
    -------
    numpy.ndarray
        Temperature array in K, shape ``(n_pressure_levels,)``.
    """
    return create_temperature_profile(
        inp_dat, gravity, isothermal, isothermal_value,
        T_equil, two_point_T, p_points, t_points, kappa, gamma, pressures
    )


def calculate_temperature_structure_limbs(
    inp_dat,
    pressures_morning_day, pressures_morning_night,
    pressures_evening_day, pressures_evening_night,
    gravity, mode
):
    """Compute separate T-P profiles for morning and evening limb components.

    Used in 3-D limb-asymmetric transit simulations, where the terminator
    is divided into a morning limb (planet rotation brings it toward the
    observer at ingress) and an evening limb (receding at egress).  Each
    limb is further split into a day-side and night-side sector when
    ``inp_dat["Limb_divisions"] == "quarters"``.

    The profile family for each sector is selected independently via the
    corresponding ``isothermal_*``, ``two_point_T_*``, ``T_equ_*``,
    ``p_points_*``, ``t_points_*``, ``Kappa_IR_*``, and ``Gamma_*`` keys
    in ``inp_dat``.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain:

        * ``"Limb_divisions"``, ``"quarters"`` or ``"gradual"``.
        * Per-sector profile flags and parameters (``isothermal_morning_day``,
          ``T_equ_morning_day``, ``p_points_morning_day``, …, and the
          corresponding ``_morning_night``, ``_evening_day``,
          ``_evening_night`` variants for the quarters case).
        * ``"T_int"``, internal temperature in K (Guillot profiles).

    pressures_morning_day : numpy.ndarray
        Pressure grid for the morning-day sector in bar,
        shape ``(n_pressure_levels,)``.
    pressures_morning_night : numpy.ndarray
        Pressure grid for the morning-night sector in bar.
        Unused when ``Limb_divisions == "gradual"``.
    pressures_evening_day : numpy.ndarray
        Pressure grid for the evening-day sector in bar.
    pressures_evening_night : numpy.ndarray
        Pressure grid for the evening-night sector in bar.
        Unused when ``Limb_divisions == "gradual"``.
    gravity : float
        Surface gravity in cm/s² (CGS).
    mode : str
        Which limb(s) to compute:

        * ``'full'``, compute both morning and evening.
        * ``'morning'``, compute morning only; evening arrays are None.
        * ``'evening'``, compute evening only; morning arrays are None.

    Returns
    -------
    tuple
        **If** ``inp_dat["Limb_divisions"] == "quarters"``:
            ``(t_morning_day, t_morning_night, t_evening_day, t_evening_night)``
, 4-tuple of numpy arrays, shape ``(n_pressure_levels,)`` each.
            Arrays not requested by ``mode`` are ``None``.

        **If** ``inp_dat["Limb_divisions"] == "gradual"``:
            ``(t_morning, None, t_evening, None)``
, 4-tuple where the night-side slots are always ``None``
            and only day-side pressure grids are used.
    """
    t_morning, t_evening = None, None

    if inp_dat["Limb_divisions"] == "quarters":
        t_morning_day = t_morning_night = t_evening_day = t_evening_night = None

        if mode in ["full", "morning"]:
            t_morning_day = create_temperature_profile(
                inp_dat, gravity,
                inp_dat['isothermal_morning_day'],
                inp_dat['isothermal_T_value_morning_day'],
                inp_dat["T_equ_morning_day"],
                inp_dat['two_point_T_morning_day'],
                inp_dat['p_points_morning_day'],
                inp_dat['t_points_morning_day'],
                inp_dat['Kappa_IR_morning_day'],
                inp_dat['Gamma_morning_day'],
                pressures_morning_day,
            )
            t_morning_night = create_temperature_profile(
                inp_dat, gravity,
                inp_dat['isothermal_morning_night'],
                inp_dat['isothermal_T_value_morning_night'],
                inp_dat["T_equ_morning_night"],
                inp_dat['two_point_T_morning_night'],
                inp_dat['p_points_morning_night'],
                inp_dat['t_points_morning_night'],
                inp_dat['Kappa_IR_morning_night'],
                inp_dat['Gamma_morning_night'],
                pressures_morning_night,
            )

        if mode in ["full", "evening"]:
            t_evening_day = create_temperature_profile(
                inp_dat, gravity,
                inp_dat['isothermal_evening_day'],
                inp_dat['isothermal_T_value_evening_day'],
                inp_dat["T_equ_evening_day"],
                inp_dat['two_point_T_evening_day'],
                inp_dat['p_points_evening_day'],
                inp_dat['t_points_evening_day'],
                inp_dat['Kappa_IR_evening_day'],
                inp_dat['Gamma_evening_day'],
                pressures_evening_day,
            )
            t_evening_night = create_temperature_profile(
                inp_dat, gravity,
                inp_dat['isothermal_evening_night'],
                inp_dat['isothermal_T_value_evening_night'],
                inp_dat["T_equ_evening_night"],
                inp_dat['two_point_T_evening_night'],
                inp_dat['p_points_evening_night'],
                inp_dat['t_points_evening_night'],
                inp_dat['Kappa_IR_evening_night'],
                inp_dat['Gamma_evening_night'],
                pressures_evening_night,
            )

        return t_morning_day, t_morning_night, t_evening_day, t_evening_night

    elif inp_dat["Limb_divisions"] in ("gradual", "asymmetric", "simplified_step"):
        if mode in ["full", "morning"]:
            t_morning = create_temperature_profile(
                inp_dat, gravity,
                inp_dat['isothermal_morning_day'],
                inp_dat['isothermal_T_value_morning_day'],
                inp_dat["T_equ_morning_day"],
                inp_dat['two_point_T_morning_day'],
                inp_dat['p_points_morning_day'],
                inp_dat['t_points_morning_day'],
                inp_dat['Kappa_IR_morning_day'],
                inp_dat['Gamma_morning_day'],
                pressures_morning_day,
            )
        if mode in ["full", "evening"]:
            t_evening = create_temperature_profile(
                inp_dat, gravity,
                inp_dat['isothermal_evening_day'],
                inp_dat['isothermal_T_value_evening_day'],
                inp_dat["T_equ_evening_day"],
                inp_dat['two_point_T_evening_day'],
                inp_dat['p_points_evening_day'],
                inp_dat['t_points_evening_day'],
                inp_dat['Kappa_IR_evening_day'],
                inp_dat['Gamma_evening_day'],
                pressures_evening_day,
            )
        return t_morning, None, t_evening, None
