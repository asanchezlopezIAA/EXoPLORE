"""
exoplore.io.paths
=================

Output directory construction.

All output paths for one simulation are derived from the
:class:`~exoplore.config.SimulationConfig` in a single place, so there
is never path logic scattered across the codebase.

Directory layout
----------------
::

    output_root/
      <planet>/
        <instrument>/
          <event>/
            <simulation_name>/
              matrices/
              plots/
              correlations/
              inputs/
              warnings/

Examples
--------
>>> from exoplore.config import SimulationConfig
>>> from exoplore.io import build_output_tree
>>> cfg = SimulationConfig()
>>> dirs = build_output_tree(cfg, simulation_name="BL19_withsignal_1nights_SNR")
>>> dirs["matrices"]
PosixPath('outputs/HD189733b/ANDES/transit/BL19_withsignal_1nights_SNR/matrices')
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from exoplore.config.models import SimulationConfig


def simulation_output_dir(cfg: SimulationConfig, simulation_name: str) -> Path:
    """Return the top-level output directory for one simulation run.

    Parameters
    ----------
    cfg:
        Full simulation configuration.
    simulation_name:
        Human-readable name for this particular run, e.g.
        ``"BL19_withsignal_1nights_SNR_noisy_stdnoisex1.0"``.

    Returns
    -------
    pathlib.Path
        Path to ``output_root/<planet>/<instrument>/<event>/<simulation_name>/``.
    """
    return (
        Path(cfg.paths.output_root)
        / cfg.planet.name
        / cfg.instrument.name
        / cfg.observation.event_type
        / simulation_name
    )


def matrices_dir(cfg: SimulationConfig, simulation_name: str) -> Path:
    """Return the path to the spectral matrices output subdirectory.

    The subdirectory is ``<simulation_output_dir>/matrices/``.  It stores
    NPZ files produced by the simulator's Block 8 output routines, with
    typical keys ``'mat_res'``, ``'propag_noise'``, and ``'mat_star'``,
    each of shape ``(n_spectra, n_pixels)``.

    This function only constructs and returns the :class:`pathlib.Path`
    object; it does **not** create the directory on disk.  To create all
    output directories call :func:`build_output_tree` with ``create=True``.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    simulation_name : str
        Human-readable name for this run.

    Returns
    -------
    pathlib.Path
    """
    return simulation_output_dir(cfg, simulation_name) / "matrices"


def plots_dir(cfg: SimulationConfig, simulation_name: str) -> Path:
    """Return the path to the PDF figures output subdirectory.

    The subdirectory is ``<simulation_output_dir>/plots/``.  It stores
    PDF (and PNG) diagnostic figures: corner plots, Kp-Vsys maps, spectral
    matrix images, and CCF time-series plots.

    This function only constructs and returns the path; it does not create
    the directory on disk.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    simulation_name : str
        Human-readable name for this run.

    Returns
    -------
    pathlib.Path
    """
    return simulation_output_dir(cfg, simulation_name) / "plots"


def correlations_dir(cfg: SimulationConfig, simulation_name: str) -> Path:
    """Return the path to the CCF correlation products subdirectory.

    The subdirectory is ``<simulation_output_dir>/correlations/``.  It stores
    NPZ files containing the CCF matrices and Kp-Vsys significance maps
    (e.g., ``ccf_tot``, ``ccf_tot_sn``) produced during Block 7 of the
    simulator.

    This function only constructs and returns the path; it does not create
    the directory on disk.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    simulation_name : str
        Human-readable name for this run.

    Returns
    -------
    pathlib.Path
    """
    return simulation_output_dir(cfg, simulation_name) / "correlations"


def inputs_dir(cfg: SimulationConfig, simulation_name: str) -> Path:
    """Return the path to the per-simulation metadata subdirectory.

    The subdirectory is ``<simulation_output_dir>/inputs/``.  It stores
    JSON or NPZ files that snapshot the exact ``inp_dat`` parameter
    dictionary and instrument configuration used for a specific run,
    enabling full reproducibility when re-analysing archived data.

    This function only constructs and returns the path; it does not create
    the directory on disk.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    simulation_name : str
        Human-readable name for this run.

    Returns
    -------
    pathlib.Path
    """
    return simulation_output_dir(cfg, simulation_name) / "inputs"


def warnings_dir(cfg: SimulationConfig, simulation_name: str) -> Path:
    """Return the path to the flagged-spectra warnings subdirectory.

    The subdirectory is ``<simulation_output_dir>/warnings/``.  It stores
    log files listing spectra that were automatically flagged during the
    pipeline (e.g., anomalous noise levels, saturated pixels, or failed
    telluric corrections) so they can be reviewed and optionally excluded
    in post-processing.

    This function only constructs and returns the path; it does not create
    the directory on disk.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    simulation_name : str
        Human-readable name for this run.

    Returns
    -------
    pathlib.Path
    """
    return simulation_output_dir(cfg, simulation_name) / "warnings"


def build_output_tree(
    cfg: SimulationConfig,
    simulation_name: str,
    create: bool = False,
) -> Dict[str, Path]:
    """Build (and optionally create) the full output directory tree.

    Parameters
    ----------
    cfg:
        Simulation configuration.
    simulation_name:
        Name for this run.
    create:
        If True, create the directories on disk.

    Returns
    -------
    dict
        Keys: ``"root"``, ``"matrices"``, ``"plots"``, ``"correlations"``,
        ``"inputs"``, ``"warnings"``.
        Values: :class:`pathlib.Path` objects.
    """
    dirs = {
        "root": simulation_output_dir(cfg, simulation_name),
        "matrices": matrices_dir(cfg, simulation_name),
        "plots": plots_dir(cfg, simulation_name),
        "correlations": correlations_dir(cfg, simulation_name),
        "inputs": inputs_dir(cfg, simulation_name),
        "warnings": warnings_dir(cfg, simulation_name),
    }

    if create:
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    return dirs
