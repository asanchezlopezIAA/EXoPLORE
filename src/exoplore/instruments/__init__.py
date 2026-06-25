"""
exoplore.instruments, instrument models and dispatcher
=========================================================

Supported instruments
---------------------

    +----------------+------------------+---------+------+------+
    | Config name    | Coverage (µm)    | Orders  | R    | Mode |
    +================+==================+=========+======+======+
    | ANDES_YJHK     | 0.4 to 1.8        |  76     | 100k | A    |
    | ANDES_YJH      | 0.4 to 1.4        |  55     | 100k | A    |
    | ANDES_K        | 1.4 to 1.8        |  21     | 100k | A    |
    | ANDES_RIZ      | 0.5 to 0.9        |  34     | 100k | A    |
    | ANDES_UBV      | 0.3 to 0.5        |  62     | 100k | A    |
    | CARMENES_NIR   | 0.96 to 1.71      |  28     |  80k | A/B/C|
    | CARMENES_VIS   | 0.514 to 0.822    |  44     |  95k | A/B/C|
    | IGRINS         | 1.43 to 2.52      |  53     |  45k | B/C  |
    | CRIRES+        | user-defined     |  varies | 100k | A    |
    +----------------+------------------+---------+------+------+

Modes: A = ETC-based (no reference night), B = reference-night synthetic,
       C = real-data analysis.  See ``instruments/base.py`` for details.

Adding a new instrument
-----------------------
1. Create ``src/exoplore/instruments/my_instrument.py`` implementing
   ``get_instrument_info(inp_dat) -> InstrumentInfo``.
2. Add one entry to ``_REGISTRY`` below, that is the only file you touch.
3. Add a test in ``tests/test_instruments.py``.
4. Document required input files in ``docs/input_files.md``.
"""

from __future__ import annotations

from .base import InstrumentInfo

# Per-instrument get_instrument_info() functions
from .andes import (
    get_instrument_info_YJHK,
    get_instrument_info_YJH,
    get_instrument_info_K,
    get_instrument_info_RIZ,
    get_instrument_info_UBV,
)
from .carmenes_nir import get_instrument_info as _carmenes_nir_info
from .carmenes_vis import get_instrument_info as _carmenes_vis_info
from .igrins       import get_instrument_info as _igrins_info
from .crires_plus  import get_instrument_info as _crires_plus_info

# Utilities still used by the simulator
from .andes import (
    ANDESInstrument,
    Load_Instrumental_Info,
    get_WaveGrid,
    From1OrderTo1Detector,
    pixel_snr_one_order,
    make_log_wave_grid,
    compute_pixel_velocity_scale,
    FromOrdersToDetectors,
    Interp_Uniform_Wvl_Grid,
    Load_CARMENES,
)

# ---------------------------------------------------------------------------
# Instrument registry, add new instruments here (one line each)
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, object] = {
    "ANDES_YJHK":   get_instrument_info_YJHK,
    "ANDES_YJH":    get_instrument_info_YJH,
    "ANDES_K":      get_instrument_info_K,
    "ANDES_RIZ":    get_instrument_info_RIZ,
    "ANDES_UBV":    get_instrument_info_UBV,
    "CARMENES_NIR": _carmenes_nir_info,
    "CARMENES_VIS": _carmenes_vis_info,
    "IGRINS":       _igrins_info,
    "CRIRES+":      _crires_plus_info,
    # Backward-compatibility alias
    "ANDES":        get_instrument_info_YJHK,
}


def load_instrument_v2(cfg: "SimulationConfig") -> InstrumentInfo:
    """Return instrument metadata directly from a config.

    Takes a :class:`~exoplore.config.models.SimulationConfig` instead of an
    ``inp_dat`` dict.  Internally builds the minimal dict the registry
    functions need, so the registry implementations themselves are unchanged.
    """
    from typing import TYPE_CHECKING
    _mini: dict = {
        "instrument":       cfg.instrument.name,
        "inputs_dir":       cfg.paths.inputs_dir,
        "Exoplanet_name":   cfg.planet.name,
        "specific_event":   cfg.observation.specific_event,
        "use_real_data":    cfg.observation.use_real_data,
        "Different_nights": cfg.observation.different_nights,
        "different_nights": cfg.observation.different_nights,
        "n_nights":         cfg.observation.n_nights,
        "fixed_snr":        cfg.noise.fixed_snr if cfg.noise.fixed_snr else 0,
        "Pix_per_resel":    cfg.instrument.pixels_per_resolution_element,
        "ETC":              False,
    }
    name = cfg.instrument.name
    if name not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown instrument '{name}'.  Supported: {supported}"
        )
    return _REGISTRY[name](_mini)


def get_WaveGrid_v2(cfg: "SimulationConfig",
                    inst: "InstrumentInfo",
                    n_orders: int):
    """Build the wavelength grid directly from a config.

    Takes SimulationConfig + InstrumentInfo instead of an inp_dat dict,
    builds the minimal dict get_WaveGrid needs, and delegates to it, so
    the underlying function is unchanged.
    """
    _mini: dict = {
        "instrument":       cfg.instrument.name,
        "Different_nights": cfg.observation.different_nights,
        "n_nights":         cfg.observation.n_nights,
        "Pix_per_resel":    cfg.instrument.pixels_per_resolution_element,
        "ETC":              False,  # CRIRES-only flag; ANDES uses snr_file=="__ETC__"
    }
    return get_WaveGrid(
        _mini,
        inst.wave_file,
        inst.sig_file,
        inst.snr_file,
        inst.JD_file,
        inst.airmass_file,
        n_orders,
    )


def load_instrument(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for the instrument named in *inp_dat*.

    This is the single entry point the simulator calls instead of the old
    monolithic ``Load_Instrumental_Info`` function.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``"instrument"``, a string
        matching one of the keys in ``_REGISTRY``.

    Returns
    -------
    InstrumentInfo
        Named tuple: observatory, file paths, n_orders_total, res, …

    Raises
    ------
    ValueError
        If the instrument name is not found in the registry.
    """
    name = inp_dat.get("instrument", "")
    if name not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown instrument '{name}'.  Supported: {supported}"
        )
    return _REGISTRY[name](inp_dat)


__all__ = [
    # Public API
    "InstrumentInfo",
    "load_instrument",
    # Per-instrument functions (for direct import)
    "get_instrument_info_YJHK",
    "get_instrument_info_YJH",
    "get_instrument_info_K",
    "get_instrument_info_RIZ",
    "get_instrument_info_UBV",
    # Utilities
    "ANDESInstrument",
    "Load_Instrumental_Info",
    "get_WaveGrid",
    "From1OrderTo1Detector",
    "pixel_snr_one_order",
    "make_log_wave_grid",
    "compute_pixel_velocity_scale",
    "FromOrdersToDetectors",
    "Interp_Uniform_Wvl_Grid",
    "Load_CARMENES",
]
