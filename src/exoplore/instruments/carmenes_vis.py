"""
exoplore.instruments.carmenes_vis
===================================

Instrument model for the CARMENES VIS channel at CAHA.

CARMENES VIS covers 0.514 to 0.822 µm at R ≈ 94,600 across 44 echelle orders.

Observing modes
---------------
The same three modes as CARMENES NIR are supported.  See ``carmenes_nir.py``
for a full description of each mode and the execution-path matrix.

Mode B (reference-night, synthetic) is recommended for most CARMENES VIS
simulations because publicly available CARMENES data can be used as the
reference night.

Required input files
--------------------
Mode A  : ``inputs_dir/CARMENES_VIS_ETC_WAVE_SNR_<planet>.fits``
           (user-supplied)
Mode B/C: ``instruments/data/wave_CARMENES_VIS.fits``  (bundled with EXoPLORE)
           and ``inputs_dir/reference_night/{julian_date,airmass,snr}.fits``

The bundled wavelength grid (``wave_CARMENES_VIS.fits``) has shape
(44 orders × 4096 pixels) in µm, averaged from 95 CARMENES VIS calibration
frames.

References
----------
Quirrenbach et al. 2014, SPIE, CARMENES overview
Reiners et al. 2018, A&A, CARMENES instrument paper
"""

from __future__ import annotations

import os as _os
from .base import InstrumentInfo

# Bundled wavelength grid (44 orders × 4096 pixels, µm)
_PKG_DATA = _os.path.join(_os.path.dirname(__file__), "data", "wave_CARMENES_VIS.fits")

_N_ORDERS = 44
_RES      = 94_600.0
_OBS      = "lasilla"   # CAHA is treated as La Silla altitude class in SkyCalc


def get_instrument_info(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for CARMENES VIS (0.514 to 0.822 µm, 44 orders).

    Selects Mode A, B, or C automatically based on flags in *inp_dat*.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Relevant keys:

        ``inputs_dir``       Base path for input files.
        ``specific_event``   True → use real JD/airmass from reference night.
        ``different_nights`` True → load per-night JD/airmass/SNR files.
        ``n_nights``         Number of nights (used when different_nights=True).
        ``Exoplanet_name``   Planet name for ETC file lookup (Mode A).
        ``fixed_snr``        If > 0, constant SNR is used (overrides all files).
        ``use_real_data``    True → Mode C (real-data analysis).

    Returns
    -------
    InstrumentInfo
        Named tuple with file paths and instrument parameters.
    """
    inputs = inp_dat.get("inputs_dir", "inputs/")
    ref    = _os.path.join(inputs, "reference_night")
    planet = inp_dat.get("Exoplanet_name", "")

    different_nights = inp_dat.get("different_nights", False) or inp_dat.get("Different_nights", False)
    n_nights       = inp_dat.get("n_nights", 1)

    # --- wavelength grid ---
    user_clean = _os.path.join(inputs, "wave_CARMENES_VIS.fits")
    if _os.path.exists(user_clean):
        wave_file = user_clean
    else:
        wave_file = _PKG_DATA

    # --- Mode A: no specific event, no real data ---
    if not inp_dat.get("specific_event", False) and not inp_dat.get("use_real_data", False):
        fixed_snr = inp_dat.get("fixed_snr", 0)
        if fixed_snr and fixed_snr > 0:
            snr_file = ""
        else:
            snr_file = _os.path.join(inputs, f"CARMENES_VIS_ETC_WAVE_SNR_{planet}.fits")
        return InstrumentInfo(
            observatory=_OBS,
            wave_file=wave_file,
            sig_file="",
            snr_file=snr_file,
            JD_file="",
            airmass_file="",
            gaps=None,
            n_orders_total=_N_ORDERS,
            res=_RES,
        )

    # --- Mode B / C: reference-night files ---
    # Always use _N suffix (single night = night 0) for consistency.
    n_ref = n_nights if different_nights else 1
    sig_file     = [_os.path.join(ref, f"sig_{i}.fits")         for i in range(n_ref)]
    snr_file     = [_os.path.join(ref, f"snr_{i}.fits")         for i in range(n_ref)]
    JD_file      = [_os.path.join(ref, f"julian_date_{i}.fits") for i in range(n_ref)]
    airmass_file = [_os.path.join(ref, f"airmass_{i}.fits")     for i in range(n_ref)]
    if not different_nights:
        sig_file, snr_file, JD_file, airmass_file = (
            sig_file[0], snr_file[0], JD_file[0], airmass_file[0])

    return InstrumentInfo(
        observatory=_OBS,
        wave_file=wave_file,
        sig_file=sig_file,
        snr_file=snr_file,
        JD_file=JD_file,
        airmass_file=airmass_file,
        gaps=None,
        n_orders_total=_N_ORDERS,
        res=_RES,
    )
