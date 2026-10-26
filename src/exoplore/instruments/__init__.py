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
   ``get_instrument_info(request) -> InstrumentInfo``.
2. Add one entry to ``_REGISTRY`` below, that is the only file you touch.
3. Add a test in ``tests/test_instruments.py``.
4. Document required input files in ``docs/input_files.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exoplore.config.models import SimulationConfig

from .base import InstrumentInfo, InstrumentInputs

# Per-instrument get_instrument_info() functions
from .wavegrid import (
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
from .wavegrid import (
    ANDESInstrument,
    get_WaveGrid,
    From1OrderTo1Detector,
    pixel_snr_one_order,
    make_log_wave_grid,
    compute_pixel_velocity_scale,
    FromOrdersToDetectors,
    Interp_Uniform_Wvl_Grid,
)

# ---------------------------------------------------------------------------
# Instrument registry, add new instruments here (one line each)
# ---------------------------------------------------------------------------
# Map instrument names to the functions that supply their resolution and input
# locations. wavegrid.py and the selected runner read the corresponding arrays.
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
    # ANDES selects the YJHK instrument setup.
    "ANDES":        get_instrument_info_YJHK,
}


def load_instrument(cfg: "SimulationConfig") -> InstrumentInfo:
    """Select the instrument inputs and read its metadata from the config."""
    inputs = InstrumentInputs(
        inputs_dir=cfg.paths.inputs_dir,
        planet_name=cfg.planet.name,
        observation=cfg.observation,
        fixed_snr=cfg.noise.fixed_snr if cfg.noise.fixed_snr else 0,
    )
    name = cfg.instrument.name
    if name not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown instrument '{name}'. Supported: {supported}")
    return _REGISTRY[name](inputs)


__all__ = [
    # Public API
    "InstrumentInfo",
    "InstrumentInputs",
    "load_instrument",
    # Per-instrument functions (for direct import)
    "get_instrument_info_YJHK",
    "get_instrument_info_YJH",
    "get_instrument_info_K",
    "get_instrument_info_RIZ",
    "get_instrument_info_UBV",
    # Utilities
    "ANDESInstrument",
    "get_WaveGrid",
    "From1OrderTo1Detector",
    "pixel_snr_one_order",
    "make_log_wave_grid",
    "compute_pixel_velocity_scale",
    "FromOrdersToDetectors",
    "Interp_Uniform_Wvl_Grid",
]
