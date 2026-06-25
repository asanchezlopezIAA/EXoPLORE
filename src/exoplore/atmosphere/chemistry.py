"""
exoplore.atmosphere.chemistry
=============================

Clean interface to equilibrium atmospheric chemistry via EasyChem.

This module wraps the EasyChem ``ExoAtmos`` object so that callers work
with clearly named parameters (``metallicity_wrt_solar``,
``carbon_to_oxygen_ratio``) rather than with the ``inp_dat``
dictionary.

Two computation modes are supported:

``adjust_co_symmetrically=False`` (default)
    Standard EasyChem path: set ``exo.metallicity`` and ``exo.co``
    directly, then call ``exo.solve()``.

``adjust_co_symmetrically=True``
    Retrieval-friendly path: manually modify atomic abundances so that
    the C/O is enforced by symmetric scaling of both C and O, then
    renormalise.  This avoids discontinuities when C/O is a free
    parameter during atmospheric retrieval.

If EasyChem is not installed, :func:`compute_equilibrium_chemistry`
raises an :class:`ImportError` with a helpful message.  All other
functions in this module work without EasyChem.

Examples
--------
>>> import numpy as np
>>> from exoplore.atmosphere.pressure import create_log_pressure_grid
>>> from exoplore.atmosphere.temperature import isothermal_profile
>>> from exoplore.atmosphere.chemistry import compute_equilibrium_chemistry
>>> pressures = create_log_pressure_grid(1e-6, 100.0, 100)
>>> temperatures = isothermal_profile(1200.0, pressures)
>>> species = ["H2O_main_iso", "CO_all_iso", "CH4_main_iso"]
>>> mass_fracs, mmw = compute_equilibrium_chemistry(
...     pressures, temperatures, species,
...     metallicity_wrt_solar=0.0,
...     carbon_to_oxygen_ratio=0.55,
... )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_equilibrium_chemistry(
    pressures: np.ndarray,
    temperatures: np.ndarray,
    species: List[str],
    metallicity_wrt_solar: float = 0.0,
    carbon_to_oxygen_ratio: float = 0.55,
    adjust_co_symmetrically: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute equilibrium mass fractions using EasyChem.

    Parameters
    ----------
    pressures:
        Pressure grid in bar, shape ``(n_layers,)``.
    temperatures:
        Temperature profile in K, shape ``(n_layers,)``.
    species:
        List of species names to return (must match EasyChem output keys,
        e.g. ``"H2O_main_iso"``, ``"CO_all_iso"``).
    metallicity_wrt_solar:
        Metallicity [Fe/H] relative to solar.  EasyChem scales all
        elemental abundances (except H and He) by
        ``10 ** metallicity_wrt_solar``.
    carbon_to_oxygen_ratio:
        Carbon-to-oxygen number ratio.  Solar value is ~0.55.
    adjust_co_symmetrically:
        If True, use the retrieval-friendly path: scale C and O
        symmetrically so that both metallicity and C/O are enforced
        without disturbing other abundances.  Useful when C/O is a
        free parameter in retrieval.

    Returns
    -------
    mass_fractions:
        Dict mapping species name → mass fraction array, shape
        ``(n_layers,)``.
    mean_molecular_weight:
        Mean molecular weight array in atomic mass units, shape
        ``(n_layers,)``.

    Raises
    ------
    ImportError
        If the ``easychem`` package is not installed.
    ValueError
        If ``pressures`` and ``temperatures`` have different shapes, or
        if a requested species is not in the EasyChem output.
    """
    try:
        import easychem.easychem as ec
    except ImportError as exc:
        raise ImportError(
            "easychem is required for compute_equilibrium_chemistry().\n"
            "Install it with:  pip install easychem\n"
            "or:               pip install exoplore[prt]"
        ) from exc

    pressures = np.asarray(pressures, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)

    if pressures.shape != temperatures.shape:
        raise ValueError(
            f"pressures and temperatures must have the same shape; "
            f"got {pressures.shape} and {temperatures.shape}."
        )

    if adjust_co_symmetrically:
        mass_fracs, mmw = _solve_co_symmetric(
            ec, pressures, temperatures, species,
            metallicity_wrt_solar, carbon_to_oxygen_ratio,
        )
    else:
        mass_fracs, mmw = _solve_standard(
            ec, pressures, temperatures, species,
            metallicity_wrt_solar, carbon_to_oxygen_ratio,
        )

    return mass_fracs, mmw


def manual_mass_fractions(
    species: List[str],
    vmr: List[float],
    mean_molecular_weight: float,
    n_layers: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build constant mass fraction arrays from volume mixing ratios.

    Used when EasyChem is not desired and the user supplies explicit
    mixing ratios.

    Parameters
    ----------
    species:
        List of species names.
    vmr:
        Volume mixing ratios, same length as ``species``.
    mean_molecular_weight:
        Constant mean molecular weight in AMU.
    n_layers:
        Number of atmospheric pressure layers.

    Returns
    -------
    mass_fractions:
        Dict mapping species → constant array of shape ``(n_layers,)``.
    mean_molecular_weight_profile:
        Constant MMW array of shape ``(n_layers,)``.

    Raises
    ------
    ValueError
        If ``len(vmr) != len(species)``.
    """
    if len(vmr) != len(species):
        raise ValueError(
            f"len(vmr)={len(vmr)} must equal len(species)={len(species)}."
        )
    mass_fracs = {
        sp: float(vmr[i]) * np.ones(n_layers)
        for i, sp in enumerate(species)
    }
    mmw_profile = float(mean_molecular_weight) * np.ones(n_layers)
    return mass_fracs, mmw_profile


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _solve_standard(
    ec,
    pressures: np.ndarray,
    temperatures: np.ndarray,
    species: List[str],
    metallicity: float,
    co_ratio: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Standard EasyChem solve: set metallicity and C/O directly."""
    exo = ec.ExoAtmos()
    exo.metallicity = metallicity
    exo.co = co_ratio
    exo.solve(pressures, temperatures)
    all_mass_fracs = exo.result_mass()
    mass_fracs = _extract_species(all_mass_fracs, species)
    return mass_fracs, exo.mmw


def _solve_co_symmetric(
    ec,
    pressures: np.ndarray,
    temperatures: np.ndarray,
    species: List[str],
    metallicity: float,
    co_ratio: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Retrieval-friendly solve: symmetric C/O enforcement.

    Scales the metallicity first, then adjusts C and O by equal and
    opposite factors so that the C/O is exactly ``co_ratio`` without
    changing the total metal abundance appreciably.
    """
    exo = ec.ExoAtmos()
    atom_names = list(exo.atoms)
    abundances = exo._atomAbunds.copy()

    # Step 1: scale metallicity
    for i, name in enumerate(atom_names):
        if name not in ("H", "He"):
            abundances[i] *= 10.0 ** metallicity

    # Step 2: enforce C/O symmetrically
    i_c = atom_names.index("C")
    i_o = atom_names.index("O")
    current_co = abundances[i_c] / abundances[i_o]
    factor = np.sqrt(co_ratio / current_co)
    abundances[i_c] *= factor
    abundances[i_o] /= factor

    # Step 3: renormalise
    abundances /= abundances.sum()

    exo.updateAtomAbunds(abundances)
    exo.solve(pressures, temperatures)
    all_mass_fracs = exo.result_mass()
    mass_fracs = _extract_species(all_mass_fracs, species)
    return mass_fracs, exo.mmw


def _extract_species(
    all_mass_fracs: dict,
    species: List[str],
) -> Dict[str, np.ndarray]:
    """Extract only the requested species from EasyChem output."""
    missing = [sp for sp in species if sp not in all_mass_fracs]
    if missing:
        raise ValueError(
            f"The following species are not in the EasyChem output: {missing}.\n"
            f"Available keys: {sorted(all_mass_fracs.keys())}"
        )
    return {sp: np.asarray(all_mass_fracs[sp]) for sp in species}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mass_frac_to_vmr(
        mass_frac=None, vmr=None, mmw_species=1.,
        mmw_atmosphere=2.33, mode='direct'
        ):
    """Convert between mass fraction and volume mixing ratio (and vice versa).

    Parameters
    ----------
    mass_frac : float or None
        Mass fraction of the species.  Required when ``mode='direct'``.
    vmr : float or None
        Volume mixing ratio.  Required when ``mode='inverse'``.
    mmw_species : float
        Mean molecular weight of the species in AMU.
    mmw_atmosphere : float
        Mean molecular weight of the bulk atmosphere in AMU.
    mode : {'direct', 'inverse'}
        ``'direct'`` converts mass fraction → VMR;
        ``'inverse'`` converts VMR → mass fraction.

    Returns
    -------
    float
        Converted quantity (VMR or mass fraction).
    """
    import sys
    if mode == "direct":
        if mass_frac is None or vmr is not None:
            print("mass_frac to vmr conversion failure")
            sys.exit()
        # Conversion formula: VMR = (mass_frac × MMW_atm) / MMW_species
        # Derivation: if a fraction f by mass of the atmosphere consists of
        # species s with molar mass m_s, and the bulk atmosphere has mean molar
        # mass m_atm, then the number ratio (VMR) is
        #   VMR = (f / m_s) / (1 / m_atm) = f × m_atm / m_s
        # Mass fractions are the internal representation used by petitRADTRANS
        # when computing radiative transfer; VMR is what observers report
        # and what EasyChem returns from equilibrium chemistry solves.
        # For a solar H/He dominated atmosphere MMW_atm ≈ 2.33 amu;
        # deviations (e.g., heavy-metal-enriched or CO2-dominated atmospheres)
        # must be accounted for by passing the correct mmw_atmosphere.
        result = mass_frac * mmw_atmosphere / mmw_species
    elif mode == "inverse":
        if mass_frac is not None or vmr is None:
            print("vmr to mass_frac conversion failure")
            sys.exit()
        # Inverse: mass_frac = VMR × MMW_species / MMW_atm
        # Rearrangement of the same formula above; used when converting
        # observer-quoted VMRs back to pRT input mass fractions.
        result = vmr * mmw_species / mmw_atmosphere
    return result
