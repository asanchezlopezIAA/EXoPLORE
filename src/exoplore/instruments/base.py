"""
instruments/base.py, Instrument interface contract for EXoPLORE
=================================================================

Every instrument module must expose a single public function::

    get_instrument_info(inp_dat: dict) -> InstrumentInfo

where ``InstrumentInfo`` is the named tuple defined below.

────────────────────────────────────────────────────────────────
Observing modes
────────────────────────────────────────────────────────────────

EXoPLORE supports three observing modes, controlled by flags in ``inp_dat``:

Mode A, ETC-based (recommended for ANDES bands and CRIRES+)
    The SNR per exposure, wavelength grid, and observing time-line are all
    derived from an Exposure Time Calculator (ETC) file.  No real observed
    spectra are needed.  The simulator builds a fully synthetic time-series
    using the planet orbital geometry.  This mode is the most self-contained
    and is ideal for feasibility studies of future instruments or targets that
    have never been observed at high resolution.

    Key flags: ``use_real_data = False``, ``fixed_snr = False``
    Required files: ETC FITS matrix with wavelength and SNR columns.

Mode B, Reference-night, synthetic spectra (recommended for CARMENES)
    A real reference night is used to extract the observing time-line (JD,
    airmass) and the per-pixel SNR from existing observations.  The stellar
    and planetary spectra are still synthesised with petitRADTRANS; no real
    science spectra are loaded.  This mode reproduces the exact cadence and
    noise properties of a real observing campaign.

    Key flags: ``use_real_data = False``, ``specific_event = True``
    Required files: JD array, airmass array, SNR cube (one per order/pixel).

Mode C, Real-data analysis
    Real observed spectra are loaded from disk and the full analysis pipeline
    is applied directly to them.

    Key flags: ``use_real_data = True``
    Required files: all Mode B files plus the observed spectral matrix.

``fixed_snr`` (any mode)
    If ``fixed_snr > 0`` the simulator broadcasts a single constant SNR value
    to every pixel in every exposure, bypassing all SNR files entirely.  This
    is useful for simulating an instrument (e.g. CARMENES) without access to
    its reference SNR cube, or for quick sensitivity studies.

────────────────────────────────────────────────────────────────
Spectral-order convention
────────────────────────────────────────────────────────────────

EXoPLORE treats all data as 3-D tensors of shape::

    (n_spectra, n_orders, n_pixels)

Every instrument module is responsible for returning the correct
``n_orders_total`` value for its configuration:

    ANDES YJHK    →  76 orders
    ANDES YJH     →  64 orders
    ANDES K       →  21 orders
    ANDES RIZ     →  (defined in andes.py)
    ANDES UBV     →  (defined in andes.py)
    CARMENES NIR  →  28 orders
    CARMENES VIS  →  44 orders
    CRIRES+       →  determined by the user-supplied ETC file

Single-order instruments (or instruments reduced to one echelle order) must
return ``n_orders_total = 1`` and provide a wavelength grid shaped
``(1, n_pixels)``.  In the config set ``"order_indices": [0]``.

If ``order_indices`` is left empty in the config, it is set to
``np.arange(n_orders_total)`` after calling ``get_instrument_info``, so all
orders are included.  To select a subset, supply the indices explicitly.

────────────────────────────────────────────────────────────────
Adding a new instrument
────────────────────────────────────────────────────────────────

1.  Create ``src/exoplore/instruments/my_instrument.py``.
2.  Implement ``get_instrument_info(inp_dat)`` returning an ``InstrumentInfo``
    named tuple (see below).  Follow the docstring conventions in
    ``carmenes_nir.py`` or ``andes.py`` as a template.
3.  Register the instrument name in ``instruments/__init__.py`` inside the
    ``_REGISTRY`` dictionary, one line.
4.  Add a test in ``tests/test_instruments.py``.
5.  Document the required input files in ``docs/input_files.md``.

No other file needs to be modified.
"""

from __future__ import annotations

from typing import NamedTuple, Optional
import numpy as np


class InstrumentInfo(NamedTuple):
    """Return type of every ``get_instrument_info`` implementation.

    Parameters
    ----------
    observatory : str
        Observatory site name passed to SkyCalc (e.g. ``"paranal"``).
    wave_file : str or None
        Path to the wavelength grid FITS file, or ``None`` if the grid is
        constructed analytically inside the module.
    sig_file : str or None
        Path to the instrumental line-spread function file, or ``None``.
    snr_file : str or None
        Path to the SNR cube FITS file (Mode B/C), or ``None`` (Mode A).
    JD_file : str or None
        Path to the Julian date array FITS file (Mode B/C), or ``None``.
    airmass_file : str or None
        Path to the airmass array FITS file (Mode B/C), or ``None``.
    gaps : list of tuple or None
        Wavelength gap intervals ``[(λ_start, λ_end), ...]`` in µm that
        should be masked, or ``None``.
    n_orders_total : int
        Total number of spectral orders for this instrument configuration.
        Used to expand an empty ``order_indices`` list into
        ``np.arange(n_orders_total)``.
    res : float
        Nominal resolving power R = λ/Δλ.
    """

    observatory: str
    wave_file: Optional[str]
    sig_file: Optional[str]
    snr_file: Optional[str]
    JD_file: Optional[str]
    airmass_file: Optional[str]
    gaps: Optional[list]
    n_orders_total: int
    res: float
