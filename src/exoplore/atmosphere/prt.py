"""
exoplore.atmosphere.prt
=======================

Clean petitRADTRANS interface for computing transmission and emission
spectra of exoplanet atmospheres.

This module wraps the petitRADTRANS ``Radtrans`` object with clearly
named functions and units.  The raw ``Radtrans`` object is still
exposed so that advanced users can call petitRADTRANS directly.

Workflow
--------
1. Call :func:`create_radtrans` once per atmosphere region to build the
   ``Radtrans`` opacity table (expensive; cache this object).
2. Call :func:`compute_transmission_spectrum` or
   :func:`compute_emission_spectrum` to compute a spectrum from a
   temperature profile and mass fractions (cheap after step 1).
3. Call :func:`convolve` to broaden the spectrum to the
   instrument resolving power.

Limb-asymmetric transit spectra
--------------------------------
Use :func:`compute_limb_averaged_transit_spectrum` to combine morning and
evening (and optionally day/night) spectra with equal weighting.

Notes
-----
petitRADTRANS is an optional dependency.  Import errors are deferred to
call time with a clear message.

All wavelengths are in **microns** (μm) unless the function name says
otherwise.  Radii are in **cm**, pressures in **bar**.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Radtrans factory
# ---------------------------------------------------------------------------


def create_radtrans(
    pressures: np.ndarray,
    species: List[str],
    wavelength_min_um: float,
    wavelength_max_um: float,
    rayleigh_species: Optional[List[str]] = None,
    continuum_contributors: Optional[List[str]] = None,
    line_opacity_mode: str = "lbl",
) -> "Radtrans":
    """Create and return a petitRADTRANS ``Radtrans`` object.

    The pressure grid is passed at construction time (petitRADTRANS ≥ 3
    style).  Opacity tables are loaded immediately; this call is slow
    the first time for a given species set (minutes), but subsequent
    calls with the same cached object are fast.

    Parameters
    ----------
    pressures:
        Pressure grid in bar, shape ``(n_layers,)``.
    species:
        Line-absorbing species, e.g.
        ``["H2O_main_iso", "CO_all_iso"]``.
        Do **not** include ``"H2"`` or ``"He"`` here; pass them as
        ``rayleigh_species`` instead.
    wavelength_min_um:
        Minimum wavelength in μm.
    wavelength_max_um:
        Maximum wavelength in μm.
    rayleigh_species:
        Species that contribute Rayleigh scattering (default:
        ``["H2", "He"]``).
    continuum_contributors:
        CIA/continuum opacity contributors (default:
        ``["H2--H2", "H2--He"]``).
    line_opacity_mode:
        Opacity mode: ``"lbl"`` (line-by-line) or ``"c-k"``
        (correlated-k).  Default ``"lbl"`` for high-resolution work.

    Returns
    -------
    petitRADTRANS.radtrans.Radtrans
        Initialised ``Radtrans`` instance.

    Raises
    ------
    ImportError
        If ``petitRADTRANS`` is not installed.
    """
    try:
        from petitRADTRANS.radtrans import Radtrans
    except ImportError as exc:
        raise ImportError(
            "petitRADTRANS is required for create_radtrans().\n"
            "Install it with:  pip install petitRADTRANS\n"
            "or:               pip install exoplore[prt]"
        ) from exc

    if rayleigh_species is None:
        rayleigh_species = ["H2", "He"]
    if continuum_contributors is None:
        continuum_contributors = ["H2--H2", "H2--He"]

    rt = Radtrans(
        pressures=np.asarray(pressures, dtype=float),
        line_species=species,
        rayleigh_species=rayleigh_species,
        gas_continuum_contributors=continuum_contributors,
        wavelength_boundaries=[wavelength_min_um - 0.01, wavelength_max_um + 0.01],
        line_opacity_mode=line_opacity_mode,
    )
    return rt


# ---------------------------------------------------------------------------
# Transmission spectrum (transit geometry)
# ---------------------------------------------------------------------------


def compute_transmission_spectrum(
    radtrans: "Radtrans",
    temperatures: np.ndarray,
    mass_fractions: Dict[str, np.ndarray],
    mean_molecular_weight: np.ndarray,
    gravity_cgs: float,
    planet_radius_cm: float,
    stellar_radius_cm: float,
    reference_pressure_bar: float,
    cloud_pressure_bar: Optional[float] = None,
    haze_factor: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a transmission (transit) spectrum.

    The spectrum is the fractional area occulted by the planet+atmosphere
    disc, i.e. ``(R_p(λ) / R_*)²``.

    Parameters
    ----------
    radtrans:
        Initialised ``Radtrans`` object from :func:`create_radtrans`.
    temperatures:
        Temperature profile in K, shape ``(n_layers,)``.
    mass_fractions:
        Dict from species name → mass fraction array ``(n_layers,)``.
    mean_molecular_weight:
        MMW array in AMU, shape ``(n_layers,)``.
    gravity_cgs:
        Surface gravity in cm s⁻².
    planet_radius_cm:
        Planet radius in cm (used as reference radius for
        petitRADTRANS).
    stellar_radius_cm:
        Stellar radius in cm (used to normalise transit depth).
    reference_pressure_bar:
        Reference pressure (p₀) at the defined planet radius in bar.
    cloud_pressure_bar:
        If provided, an opaque grey cloud deck is placed at this
        pressure in bar.
    haze_factor:
        Haze enhancement factor (currently not passed to pRT; reserved
        for future use).

    Returns
    -------
    wavelengths_um:
        Wavelength grid in μm, shape ``(n_wavelengths,)``.
    spectrum:
        Transit depth ``(R_p(λ) / R_*)²``, dimensionless,
        shape ``(n_wavelengths,)``.
    """
    kwargs: dict = dict(
        temperatures=np.asarray(temperatures),
        mass_fractions=mass_fractions,
        mean_molar_masses=np.asarray(mean_molecular_weight),
        reference_gravity=gravity_cgs,
        planet_radius=planet_radius_cm,
        reference_pressure=reference_pressure_bar,
    )
    if cloud_pressure_bar is not None:
        kwargs["opaque_cloud_top_pressure"] = cloud_pressure_bar

    wavelengths_cm, transit_radii, _ = radtrans.calculate_transit_radii(**kwargs)
    wavelengths_um = wavelengths_cm * 1e4  # cm → μm
    spectrum = transit_radii ** 2 / stellar_radius_cm ** 2
    return wavelengths_um, spectrum


# ---------------------------------------------------------------------------
# Emission spectrum (dayside geometry)
# ---------------------------------------------------------------------------


def compute_emission_spectrum(
    radtrans: "Radtrans",
    temperatures: np.ndarray,
    mass_fractions: Dict[str, np.ndarray],
    mean_molecular_weight: np.ndarray,
    gravity_cgs: float,
    stellar_teff_K: float,
    planet_radius_cm: float,
    stellar_radius_cm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a thermal emission (dayside) spectrum.

    The planet spectrum is normalised by ``(R_p / R_*)²`` and a PHOENIX
    stellar spectrum is fetched at ``stellar_teff_K``.

    Parameters
    ----------
    radtrans:
        Initialised ``Radtrans`` object from :func:`create_radtrans`.
    temperatures:
        Temperature profile in K, shape ``(n_layers,)``.
    mass_fractions:
        Dict from species name → mass fraction array ``(n_layers,)``.
    mean_molecular_weight:
        MMW array in AMU, shape ``(n_layers,)``.
    gravity_cgs:
        Surface gravity in cm s⁻².
    stellar_teff_K:
        Stellar effective temperature in K (used to fetch PHOENIX
        spectrum for normalisation).
    planet_radius_cm:
        Planet radius in cm.
    stellar_radius_cm:
        Stellar radius in cm.

    Returns
    -------
    wavelengths_um:
        Wavelength grid in μm.
    planet_spectrum:
        Planet emission spectrum normalised by ``(R_p / R_*)²``.
    stellar_spectrum:
        Stellar flux interpolated onto ``wavelengths_um``.
    """
    try:
        from petitRADTRANS import physical_constants as cst
    except ImportError as exc:
        raise ImportError(
            "petitRADTRANS is required for compute_emission_spectrum().\n"
            "Install it with:  pip install exoplore[prt]"
        ) from exc

    radtrans.calc_flux(
        np.asarray(temperatures),
        mass_fractions,
        gravity_cgs,
        np.asarray(mean_molecular_weight),
    )
    wavelengths_um = cst.c / radtrans.freq / 1e-4  # convert Hz→μm
    planet_spectrum = radtrans.flux * (planet_radius_cm / stellar_radius_cm) ** 2

    stellar_data = cst.get_PHOENIX_spec(stellar_teff_K)
    wave_star_um = stellar_data[:, 0] / 1e-4  # Å → μm
    flux_star = stellar_data[:, 1]
    stellar_spectrum = np.interp(wavelengths_um, wave_star_um, flux_star)

    return wavelengths_um, planet_spectrum, stellar_spectrum


# ---------------------------------------------------------------------------
# Limb-averaged transit spectrum
# ---------------------------------------------------------------------------


def compute_limb_averaged_transit_spectrum(
    wavelengths_um: np.ndarray,
    spectra: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Average multiple limb-region spectra with optional weights.

    Used to combine morning/evening and day/night spectra into a single
    disc-averaged transit spectrum.

    Parameters
    ----------
    wavelengths_um:
        Common wavelength grid in μm.
    spectra:
        List of spectrum arrays, each shape ``(n_wavelengths,)``.
        All must have the same shape as ``wavelengths_um``.
    weights:
        Optional list of weights (same length as ``spectra``).
        If None, equal weights are used.

    Returns
    -------
    np.ndarray
        Weighted average spectrum, shape ``(n_wavelengths,)``.
    """
    if weights is None:
        weights = [1.0] * len(spectra)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    result = np.zeros_like(wavelengths_um, dtype=float)
    for w, spec in zip(weights, spectra):
        result += w * np.asarray(spec)
    return result


# ---------------------------------------------------------------------------
# Spectral convolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gravity helper
# ---------------------------------------------------------------------------


def surface_gravity_cgs(
    planet_mass_kg: float,
    planet_radius_cm: float,
) -> float:
    """Compute surface gravity in cm s⁻².

    Parameters
    ----------
    planet_mass_kg:
        Planet mass in kg.
    planet_radius_cm:
        Planet radius in cm.

    Returns
    -------
    float
        Surface gravity in cm s⁻².
    """
    G_cgs = 6.674e-8  # cm³ g⁻¹ s⁻²
    mass_g = planet_mass_kg * 1e3
    return G_cgs * mass_g / planet_radius_cm ** 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def convolve(wave, spec, res):
    """Gaussian-broaden a spectrum to a given resolving power.

    The single instrument-LSF convolution used throughout EXoPLORE.

    Parameters
    ----------
    wave : array  Wavelength array (μm).
    spec : array  Flux or depth array.
    res : float   Resolving power R = λ/Δλ.

    Returns
    -------
    numpy.ndarray  Smoothed spectrum.

    Notes
    -----
    The LSF FWHM is ``λ_mean / R`` and σ = FWHM / (2√(2 ln 2)).  To pass σ to
    ``gaussian_filter`` it must be expressed in pixels, i.e. divided by the
    wavelength per pixel = ``step * wave.mean()`` where ``step`` is the mean
    fractional pixel spacing ``Δλ/λ``.  An earlier version divided by
    ``step`` alone, implicitly assuming ``wave.mean() == 1`` and thereby
    over-broadening the LSF by a factor of the mean wavelength (≈1.6× at
    1.6 μm, ≈2.2× at 2.2 μm).  This is corrected here.
    """
    if len(wave) != len(spec):
        raise ValueError('Dimensions for wave and spec do not match.')
    fwhm = wave.mean() / res
    std_dev = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    step = np.mean(2.0 * np.diff(wave) / (wave[1:] + wave[:-1]))
    std_dev /= (step * wave.mean())
    return gaussian_filter(spec, sigma=std_dev, mode='nearest')


def convolve_velocity(delta_v, spec, res, central_wavelength):
    """Gaussian-broaden a spectrum in velocity space.

    Parameters
    ----------
    delta_v : array  Velocity array in km/s.
    spec : array     Flux array.
    res : float      Resolving power.
    central_wavelength : float  Central wavelength in microns.

    Returns
    -------
    numpy.ndarray  Smoothed spectrum.
    """
    if len(delta_v) != len(spec):
        raise ValueError('Dimensions for delta_v and spec do not match.')
    fwhm_wavelength = central_wavelength / res
    c = 3e5  # km/s
    fwhm_velocity = (fwhm_wavelength * 1e-6) * c / (central_wavelength * 1e-6)
    std_dev_velocity = fwhm_velocity / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    step = np.mean(np.diff(delta_v))
    std_dev_velocity_bins = std_dev_velocity / step
    return gaussian_filter(spec, sigma=std_dev_velocity_bins, mode='nearest')


# ExoMol/opacity isotopologue string -> easyCHEM chemical-species name.
_EASYCHEM_MOLECULE = {
    "H2": "H2", "He": "He",
    "1H2-32S": "H2S", "1H2-16O": "H2O", "12C-1H4": "CH4",
    "14N-1H3": "NH3", "12C-16O": "CO", "12C-16O2": "CO2", "32S-16O2": "SO2",
}


def easychem_molecule(species_name: str) -> str:
    """Map a line-opacity name (e.g. '1H2-32S__AYT2.R1e+06_0.3-28.0mu') to the
    easyCHEM chemical-species key (e.g. 'H2S').  Strips the opacity suffix after
    '__' or '.', then looks up the isotopologue; plain names pass through."""
    base = species_name.split("__")[0].split(".")[0]
    return _EASYCHEM_MOLECULE.get(base, _EASYCHEM_MOLECULE.get(species_name, base))


def call_easyCHEM(inp_dat, metallicity, C_to_O, pressures, t, species,
                  ret_CtoO=False):
    """Compute equilibrium chemistry with EasyChem.

    Parameters
    ----------
    inp_dat : dict  Simulation input dictionary (unused directly here;
        kept for backward compatibility).
    metallicity : float  [M/H] metallicity relative to solar.
    C_to_O : float  Carbon-to-oxygen number ratio.
    pressures : array  Pressure grid in bar.
    t : array  Temperature profile in K.
    species : list of str  Species to extract from the EasyChem solution.
    ret_CtoO : bool  If True, use the symmetric C/O modification scheme.

    Returns
    -------
    mass_fracs : dict  Species → mass fraction array (n_layers,).
    MMW_tot : array    Mean molecular weight at each pressure level.
    """
    try:
        import easychem.easychem as ec
    except ImportError as exc:
        raise ImportError(
            "easychem is required for call_easyCHEM().\n"
            "Install with:  pip install easychem"
        ) from exc

    if not ret_CtoO:
        exo = ec.ExoAtmos()
        exo.metallicity = metallicity
        exo.co = C_to_O
        exo.solve(pressures, t)
        mass_fractions = exo.result_mass()
        mass_fracs = {sp: mass_fractions[easychem_molecule(sp)] for sp in species}
        MMW_tot = exo.mmw
    else:
        exo = ec.ExoAtmos()
        atom_abundances_solar = exo._atomAbunds.copy()
        atom_names = exo.atoms
        modif_abundances = atom_abundances_solar.copy()

        for i, name in enumerate(atom_names):
            if name not in ['H', 'He']:
                modif_abundances[i] *= 10 ** metallicity

        iC = atom_names.index('C')
        iO = atom_names.index('O')
        C = modif_abundances[iC]
        O = modif_abundances[iO]
        current_CO = C / O
        target_CO = C_to_O
        factor = np.sqrt(target_CO / current_CO)
        modif_abundances[iC] = C * factor
        modif_abundances[iO] = O / factor
        modif_abundances /= np.sum(modif_abundances)

        exo.updateAtomAbunds(modif_abundances)
        exo.solve(pressures, t)
        mass_fractions = exo.result_mass()
        mass_fracs = {sp: mass_fractions[easychem_molecule(sp)] for sp in species}
        MMW_tot = exo.mmw

    return mass_fracs, MMW_tot


def call_pRT(inp_dat, pressures, prt_object, species, vmr, MMW, p0,
             isothermal, iso_T_value, two_point_T, p_points, t_points,
             kappa, gamma, T_equil, metallicity, C_to_O,
             use_easyCHEM=False, P_cloud=None,
             easychem_CtoO_ret=False, haze_fac=None):
    """Run petitRADTRANS for a single, symmetric atmosphere.

    Parameters
    ----------
    inp_dat : dict
        Full simulation input dictionary.
    pressures : array
        Pressure grid in bar.
    prt_object : Radtrans
        Initialised petitRADTRANS ``Radtrans`` object.
    species : list of str
        Line species.
    vmr : list of float
        Volume mixing ratios (one per species).  Ignored when
        ``use_easyCHEM=True``.
    MMW : float or None
        Mean molecular weight in AMU.  Required when
        ``use_easyCHEM=False``.
    p0 : float
        Reference pressure (bar) for the planet radius.
    isothermal, iso_T_value, two_point_T, p_points, t_points,
    kappa, gamma, T_equil : see
        :func:`exoplore.atmosphere.temperature.calculate_temperature_structure`.
    metallicity, C_to_O : float
        EasyChem metallicity and C/O ratio (used when
        ``use_easyCHEM=True``).
    use_easyCHEM : bool
        Use EasyChem equilibrium chemistry.
    P_cloud : float or None
        Opaque cloud deck pressure in bar.
    easychem_CtoO_ret : bool
        Use symmetric C/O modification in EasyChem.
    haze_fac : float or None
        Haze factor (reserved; currently not passed to pRT).

    Returns
    -------
    wave_pRT : array  Wavelength in μm (actually Å * 1e4 from pRT).
    spec_conv : array  Convolved spectrum.
    mass_fracs : dict  Mass fractions from chemistry.
    MMW_tot : array  Mean molecular weight profile.
    spec_star_conv : array  Convolved stellar spectrum (zeros for transit).
    t : array  Temperature profile used.
    """
    from petitRADTRANS import physical_constants as cst
    from exoplore.atmosphere.temperature import calculate_temperature_structure

    gravity = (inp_dat['Gravity']
               if inp_dat['Gravity'] is not None
               else cst.G * inp_dat['M_pl'] / inp_dat['R_pl'] ** 2)

    if MMW is None and not use_easyCHEM:
        raise ValueError("Mean molecular weight should be indicated for pRT!")
    if inp_dat['event'] == 'dayside' and inp_dat['T_star'] is None:
        raise ValueError("T_star should be indicated for dayside simulations!")

    t = calculate_temperature_structure(
        inp_dat, pressures, gravity, isothermal, iso_T_value,
        T_equil, two_point_T, p_points, t_points, kappa, gamma, None
    )

    def _calc_mass_fracs(vmr_list, species_list, temperatures, MMW_val):
        if len(vmr_list) != len(species_list):
            raise ValueError(
                "You did not supply the VMR of all compounds. "
                "len(vmr) != len(species)"
            )
        MMW_tot = MMW_val * np.ones_like(temperatures)
        mass_fracs = {
            sp: vmr_list[cont] * np.ones_like(temperatures)
            for cont, sp in enumerate(species_list)
        }
        return mass_fracs, MMW_tot

    if use_easyCHEM:
        # easyCHEM expects the metallicity in dex (it scales metal abundances
        # by 10**metallicity), which is exactly what our config's
        # metallicity_wrt_solar already carries, so it is passed through
        # unchanged.  Species may be given as full opacity names
        # (e.g. '1H2-32S__AYT2…'); map them to the easyCHEM chemical keys,
        # then re-key the returned mass fractions back onto the opacity names
        # pRT expects.  Plain names (e.g. 'H2O') pass through unchanged.
        chem = [easychem_molecule(s) for s in species]
        ec_fracs, MMW_tot = call_easyCHEM(
            inp_dat, metallicity, C_to_O, pressures, t, chem, easychem_CtoO_ret
        )
        mass_fracs = {opac: ec_fracs[chem[i]] for i, opac in enumerate(species)}
    else:
        mass_fracs, MMW_tot = _calc_mass_fracs(vmr, species, t, MMW)

    def _transit(t_profile, mf, mmw, prt_obj, p0_val, P_cl=None):
        kwargs = dict(
            temperatures=t_profile,
            mass_fractions=mf,
            mean_molar_masses=mmw,
            reference_gravity=gravity,
            planet_radius=inp_dat['R_pl'],
            reference_pressure=p0_val,
        )
        if P_cl is not None:
            kwargs["opaque_cloud_top_pressure"] = P_cl
        wavelengths, transit_radii, _ = prt_obj.calculate_transit_radii(**kwargs)
        spec = transit_radii ** 2 / inp_dat['R_star'] ** 2
        wave_pRT = wavelengths * 1e4
        return wave_pRT, spec

    def _dayside(t_profile, mf, mmw):
        prt_object.calc_flux(t_profile, mf, gravity, mmw)
        spec = prt_object.flux
        wave_pRT = cst.c / prt_object.freq / 1e-4
        stellar_spec = cst.get_PHOENIX_spec(inp_dat["T_star"])
        wave_star = stellar_spec[:, 0] / 1e-4
        spec_star = stellar_spec[:, 1]
        spec_star = np.interp(wave_pRT, wave_star, spec_star)
        spec *= (inp_dat['R_pl'] / inp_dat['R_star']) ** 2
        return wave_pRT, spec, spec_star

    if inp_dat['event'] == 'transit':
        wave_pRT, spec = _transit(t, mass_fracs, MMW_tot, prt_object, p0,
                                  P_cloud)
        spec_star = np.zeros_like(spec)
    elif inp_dat['event'] == 'dayside':
        wave_pRT, spec, spec_star = _dayside(t, mass_fracs, MMW_tot)

    if inp_dat['res'] is None:
        raise ValueError(
            "Please provide the resolving power. inp_dat['res'] cannot be None!"
        )
    return (wave_pRT,
            convolve(wave_pRT, spec, inp_dat['res']),
            mass_fracs,
            MMW_tot,
            convolve(wave_pRT, spec_star, inp_dat['res']),
            t)


def call_pRT_limbs(inp_dat,
                   pressures_morning_day, pressures_morning_night,
                   pressures_evening_day, pressures_evening_night,
                   prt_object_morning_day, prt_object_morning_night,
                   prt_object_evening_day, prt_object_evening_night,
                   mode="full", retrieval=False, inp_dat_ret=None,
                   easychem_CtoO_ret=False, haze_fac=None):
    """Run petitRADTRANS for limb-asymmetric atmospheres.

    Supports ``Limb_divisions`` modes ``"gradual"``, ``"asymmetric"``, and ``"simplified_step"``.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary (contains per-limb chemistry/T-P
        parameters).
    pressures_* : array
        Pressure grids for each limb component (morning/evening ×
        day/night).
    prt_object_* : Radtrans
        Initialised petitRADTRANS objects for each limb component.
    mode : str
        ``"full"``, ``"morning"``, or ``"evening"``.
    retrieval : bool
        If True, use ``inp_dat_ret`` for species and VMR.
    inp_dat_ret : dict or None
        Retrieval parameter dictionary (used when ``retrieval=True``).
    easychem_CtoO_ret : bool
        Symmetric C/O modification for EasyChem.
    haze_fac : float or None
        Reserved.

    Returns
    -------
    tuple
        (wave_pRT, spec_morning_day, spec_morning_night, spec_evening_day,
        spec_evening_night, mass_fracs_morning_day, mass_fracs_morning_night,
        MMW_morning_day, MMW_morning_night, mass_fracs_evening_day,
        mass_fracs_evening_night, MMW_evening_day, MMW_evening_night,
        syn_star, t_morning_day, t_morning_night, t_evening_day, t_evening_night)

        Night values are ``None`` in ``"gradual"`` mode.
    """
    from petitRADTRANS import physical_constants as cst
    from exoplore.atmosphere.temperature import (
        calculate_temperature_structure_limbs,
    )

    gravity = (inp_dat['Gravity']
               if inp_dat['Gravity'] is not None
               else cst.G * inp_dat['M_pl'] / inp_dat['R_pl'] ** 2)

    def _calc_mass_fracs(vmr_list, species_list, temperatures):
        if len(vmr_list) != len(species_list):
            raise ValueError(
                "You did not supply the VMR of all compounds. "
                "len(vmr) != len(species)"
            )
        MMW_tot = inp_dat['MMW'] * np.ones_like(temperatures)
        mass_fracs = {
            sp: vmr_list[cont] * np.ones_like(temperatures)
            for cont, sp in enumerate(species_list)
        }
        return mass_fracs, MMW_tot

    def _transit(t_profile, mf, mmw, prt_obj, p0_val):
        wavelengths, transit_radii, _ = prt_obj.calculate_transit_radii(
            temperatures=t_profile,
            mass_fractions=mf,
            mean_molar_masses=mmw,
            reference_gravity=gravity,
            planet_radius=inp_dat['R_pl'],
            reference_pressure=p0_val,
        )
        spec = transit_radii ** 2 / inp_dat['R_star'] ** 2
        wave_pRT = wavelengths * 1e4
        return wave_pRT, spec

    if inp_dat["Limb_divisions"] == "quarters":
        species_morning_day = inp_dat['species_morning_day']
        species_morning_night = inp_dat['species_morning_night']
        species_evening_day = inp_dat['species_evening_day']
        species_evening_night = inp_dat['species_evening_night']
        vmr_morning_day = inp_dat['vmr_morning_day']
        vmr_morning_night = inp_dat['vmr_morning_night']
        vmr_evening_day = inp_dat['vmr_evening_day']
        vmr_evening_night = inp_dat['vmr_evening_night']

        if inp_dat['MMW_morning_day'] is None and not inp_dat["use_easyCHEM_morning_day"]:
            raise ValueError("Mean molecular weight should be indicated for pRT!")
        if inp_dat['MMW_morning_night'] is None and not inp_dat["use_easyCHEM_morning_night"]:
            raise ValueError("Mean molecular weight should be indicated for pRT!")
        if inp_dat['MMW_evening_day'] is None and not inp_dat["use_easyCHEM_evening_day"]:
            raise ValueError("Mean molecular weight should be indicated for pRT!")
        if inp_dat['MMW_evening_night'] is None and not inp_dat["use_easyCHEM_evening_night"]:
            raise ValueError("Mean molecular weight should be indicated for pRT!")
        if inp_dat['event'] == 'dayside' and inp_dat['T_star'] is None:
            raise ValueError("T_star should be indicated for dayside simulations!")

        t_morning_day, t_morning_night, t_evening_day, t_evening_night = \
            calculate_temperature_structure_limbs(
                inp_dat,
                pressures_morning_day, pressures_morning_night,
                pressures_evening_day, pressures_evening_night,
                gravity, mode
            )

        if mode in ["full", "morning"]:
            if inp_dat["use_easyCHEM_morning_day"]:
                mass_fracs_morning_day, MMW_tot_morning_day = call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_morning_day"],
                    inp_dat["C_to_O_morning_day"], pressures_morning_day,
                    t_morning_day, inp_dat['species_morning_day']
                )
            else:
                mass_fracs_morning_day, MMW_tot_morning_day = _calc_mass_fracs(
                    vmr_morning_day, species_morning_day, t_morning_day)
            if inp_dat["use_easyCHEM_morning_night"]:
                mass_fracs_morning_night, MMW_tot_morning_night = call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_morning_night"],
                    inp_dat["C_to_O_morning_night"], pressures_morning_night,
                    t_morning_night, inp_dat['species_morning_night']
                )
            else:
                mass_fracs_morning_night, MMW_tot_morning_night = _calc_mass_fracs(
                    vmr_morning_night, species_morning_night, t_morning_night)

        if mode in ["full", "evening"]:
            if inp_dat["use_easyCHEM_evening_day"]:
                mass_fracs_evening_day, MMW_tot_evening_day = call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_evening_day"],
                    inp_dat["C_to_O_evening_day"], pressures_evening_day,
                    t_evening_day, inp_dat['species_evening_day']
                )
            else:
                mass_fracs_evening_day, MMW_tot_evening_day = _calc_mass_fracs(
                    vmr_evening_day, species_evening_day, t_evening_day)
            if inp_dat["use_easyCHEM_evening_night"]:
                mass_fracs_evening_night, MMW_tot_evening_night = call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_evening_night"],
                    inp_dat["C_to_O_evening_night"], pressures_evening_night,
                    t_evening_night, inp_dat['species_evening_night']
                )
            else:
                mass_fracs_evening_night, MMW_tot_evening_night = _calc_mass_fracs(
                    vmr_evening_night, species_evening_night, t_evening_night)

            if inp_dat['event'] == 'transit':
                syn_star = None
                if mode in ["full", "morning"]:
                    wave_pRT, spec_morning_day = _transit(
                        t_morning_day, mass_fracs_morning_day,
                        MMW_tot_morning_day, prt_object_morning_day,
                        inp_dat["p0_morning_day"])
                    wave_pRT, spec_morning_night = _transit(
                        t_morning_night, mass_fracs_morning_night,
                        MMW_tot_morning_night, prt_object_morning_night,
                        inp_dat["p0_morning_night"])
                if mode in ["full", "evening"]:
                    wave_pRT, spec_evening_day = _transit(
                        t_evening_day, mass_fracs_evening_day,
                        MMW_tot_evening_day, prt_object_evening_day,
                        inp_dat["p0_evening_day"])
                    wave_pRT, spec_evening_night = _transit(
                        t_evening_night, mass_fracs_evening_night,
                        MMW_tot_evening_night, prt_object_evening_night,
                        inp_dat["p0_evening_night"])

            return (wave_pRT,
                    spec_morning_day, spec_morning_night,
                    spec_evening_day, spec_evening_night,
                    mass_fracs_morning_day, mass_fracs_morning_night,
                    MMW_tot_morning_day, MMW_tot_morning_night,
                    mass_fracs_evening_day, mass_fracs_evening_night,
                    MMW_tot_evening_day, MMW_tot_evening_night,
                    syn_star,
                    t_morning_day, t_morning_night, t_evening_day, t_evening_night)

    elif inp_dat["Limb_divisions"] in ("gradual", "asymmetric", "simplified_step"):
        if not retrieval:
            species_morning = inp_dat['species_morning_day']
            species_evening = inp_dat['species_evening_day']
            vmr_morning = inp_dat['vmr_morning_day']
            vmr_evening = inp_dat['vmr_evening_day']
        else:
            species_morning = inp_dat_ret['species_ret_morning']
            species_evening = inp_dat_ret['species_ret_evening']
            vmr_morning = inp_dat_ret['vmr_morning_day']
            vmr_evening = inp_dat_ret['vmr_evening_day']

        inp_dat_t = inp_dat_ret if retrieval else inp_dat
        t_morning, _, t_evening, _ = calculate_temperature_structure_limbs(
            inp_dat_t, pressures_morning_day, None, pressures_evening_day, None,
            gravity, mode
        )
        del inp_dat_t

        if mode in ["full", "morning"]:
            if inp_dat["use_easyCHEM_morning_day"]:
                mass_fracs_morning, MMW_tot_morning = call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_morning_day"],
                    inp_dat["C_to_O_morning_day"], pressures_morning_day,
                    t_morning, inp_dat['species_morning_day'], easychem_CtoO_ret
                )
            else:
                mass_fracs_morning, MMW_tot_morning = _calc_mass_fracs(
                    vmr_morning, species_morning, t_morning)

        if mode in ["full", "evening"]:
            if inp_dat["use_easyCHEM_evening_day"]:
                mass_fracs_evening, MMW_tot_evening = call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_evening_day"],
                    inp_dat["C_to_O_evening_day"], pressures_evening_day,
                    t_evening, inp_dat['species_evening_day'], easychem_CtoO_ret
                )
            else:
                mass_fracs_evening, MMW_tot_evening = _calc_mass_fracs(
                    vmr_evening, species_evening, t_evening)

        if inp_dat['event'] == 'transit':
            syn_star = None
            if mode in ["full", "morning"]:
                wave_pRT, spec_morning = _transit(
                    t_morning, mass_fracs_morning, MMW_tot_morning,
                    prt_object_morning_day, inp_dat["p0_morning_day"])
            if mode in ["full", "evening"]:
                wave_pRT, spec_evening = _transit(
                    t_evening, mass_fracs_evening, MMW_tot_evening,
                    prt_object_evening_day, inp_dat["p0_evening_day"])

        return (wave_pRT,
                spec_morning, None, spec_evening, None,
                mass_fracs_morning, None, MMW_tot_morning, None,
                mass_fracs_evening, None, MMW_tot_evening, None,
                syn_star,
                t_morning, None, t_evening, None)


def pRT_LRES_stellar_model(inp_dat, prt_object):
    """Compute a low-resolution Planck stellar model on the pRT frequency grid.

    Builds a blackbody stellar spectrum using the petitRADTRANS Planck
    function ``cst.b`` evaluated at the stellar effective temperature,
    then optionally convolves it to the instrument resolution.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``'T_star'``
        (stellar effective temperature in K), ``'conv'`` (bool flag),
        and ``'res'`` (resolving power).
    prt_object : petitRADTRANS.Radtrans
        Initialised pRT object whose ``freq`` attribute provides the
        frequency grid.

    Returns
    -------
    wave_pRT : numpy.ndarray
        Wavelength array in microns corresponding to ``prt_object.freq``.
    spec_star : numpy.ndarray
        Stellar flux spectrum (convolved if ``inp_dat['conv']`` is True,
        raw Planck otherwise).
    """
    from petitRADTRANS import physical_constants as cst
    wave_pRT = cst.c / prt_object.freq / 1.e-4
    freq = cst.c / (wave_pRT * 1e-4)
    planck = cst.b(inp_dat['T_star'], freq)
    spec_star = np.pi * planck
    if inp_dat['conv']:
        return wave_pRT, convolve(wave_pRT, spec_star, inp_dat['res'])
    else:
        return spec_star
