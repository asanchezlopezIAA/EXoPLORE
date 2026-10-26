"""
exoplore.io.paths
=================

Output directory construction.

Construct the directory tree from :class:`~exoplore.config.SimulationConfig`.
The runner and individual writers choose filenames within these directories.

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
    # Group the experiment's products under one root. build_output_tree with
    # create=True creates the directories represented by these paths.
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
    per-order archives storing their array under key ``'a'``. Residuals
    and propagated uncertainties have axes ``(night, exposure, wavelength pixel)``
    in the ordinary mode. Template products can have different axes.
    Named detection-map archives also live here.

    Return a Path object. Call :func:`build_output_tree` with
    ``create=True`` to create the output directories.

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

    Return a Path object. build_output_tree(create=True) creates directories.

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
    a location available for correlation products. The molecular runner
    currently writes its CCF and detection-map archives in ``matrices/``.

    Return a Path object. build_output_tree(create=True) creates directories.

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

    The subdirectory is ``<simulation_output_dir>/inputs/``. It provides
    a location for input metadata associated with an individual run.

    Return a Path object. build_output_tree(create=True) creates directories.

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
    the molecular runner's fully masked order report when exclusions occur.
    Individual pipelines can write further diagnostics in their own outputs.

    Return a Path object. build_output_tree(create=True) creates directories.

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
