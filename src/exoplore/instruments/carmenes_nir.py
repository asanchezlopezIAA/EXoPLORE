"""
exoplore.instruments.carmenes_nir
==================================

Instrument model for the CARMENES NIR channel at CAHA.

CARMENES NIR covers 0.96 to 1.71 µm at R ≈ 80,400 across 28 echelle orders.

Observing modes
---------------
CARMENES NIR is most commonly used in **Mode B (reference-night, synthetic)**
because real CARMENES data are publicly available for many bright hot-Jupiter
hosts.  Mode A and Mode C are also fully supported.

Mode A, ETC-based (no reference night needed)
    ``use_real_data = False``, ``fixed_snr = False``
    Requires a user-supplied ETC FITS file in ``inputs_dir``.
    Use ``fixed_snr = <value>`` to bypass the ETC file and broadcast a
    constant SNR to every pixel, useful for quick sensitivity studies.

Mode B, Reference-night, synthetic spectra (recommended)
    ``use_real_data = False``, ``specific_event = True``
    Reads JD, airmass, and per-pixel SNR from a reference observation.
    Stellar and planetary spectra are still synthesised with petitRADTRANS.
    Required files in ``inputs_dir/reference_night/``:

        julian_date_0.fits, 1-D array of barycentric JDs
        airmass_0.fits, 1-D array of airmass values
        snr_0.fits, SNR cube  (n_spectra, n_orders, n_pixels)
                                   or (n_spectra, n_pixels, n_orders) for CARACAL output

    The ``_0`` suffix is always required, even for a single night.
    For multi-night simulations (``different_nights = True``) add
    ``julian_date_1.fits``, ``snr_1.fits``, etc. for each subsequent night.

Mode C, Real-data analysis
    ``use_real_data = True``
    Loads real CARMENES science spectra and runs the analysis pipeline on them
    directly.  All Mode B files are required plus the observed spectral cube.

Execution-path matrix
---------------------
``use_real_data`` × ``specific_event`` × ``different_nights`` × ``fixed_snr``
create eight paths; the most commonly useful ones are:

    (False, False, False, False)  Fully synthetic, single night, ETC SNR
    (False, False, False, True )  Fully synthetic, single night, constant SNR
    (False, True,  False, False)  Synthetic spectra on real JD/airmass grid
    (False, True,  True,  False)  Multi-night, synthetic on real grids
    (True,  True,  False, False)  Real-data injection, single night
    (True,  True,  True,  False)  Real-data injection, multi-night

Required input files
--------------------
Mode A  : ``inputs_dir/CARMENES_NIR_ETC_WAVE_SNR_<planet>.fits``
           (user-supplied; must contain wavelength and SNR arrays)
Mode B/C: ``instruments/data/wave_CARMENES_NIR.fits``  (bundled with EXoPLORE)
           and ``inputs_dir/reference_night/{julian_date,airmass,snr}.fits``

References
----------
Quirrenbach et al. 2014, SPIE, CARMENES overview
Reiners et al. 2018, A&A, CARMENES instrument paper
"""

from __future__ import annotations

import os as _os
from .base import InstrumentInfo

# Bundled wavelength grid (28 orders × 4080 pixels, µm)
_PKG_DATA = _os.path.join(_os.path.dirname(__file__), "data", "wave_CARMENES_NIR.fits")

_N_ORDERS = 28
_RES      = 80_400.0
_OBS      = "lasilla"   # CAHA is treated as La Silla altitude class in SkyCalc


def get_instrument_info(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for CARMENES NIR (0.96 to 1.71 µm, 28 orders).

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

    use_etc        = not inp_dat.get("specific_event", False) and not inp_dat.get("use_real_data", False)
    different_nights = inp_dat.get("different_nights", False) or inp_dat.get("Different_nights", False)
    n_nights       = inp_dat.get("n_nights", 1)

    # --- wavelength grid ---
    user_clean  = _os.path.join(inputs, "wave_CARMENES_NIR.fits")
    user_fallback = _os.path.join(inputs, "wave_NIR.fits")
    if _os.path.exists(user_clean):
        wave_file = user_clean
    elif _os.path.exists(_PKG_DATA):
        wave_file = _PKG_DATA
    else:
        wave_file = user_fallback  # last-resort fallback

    # --- Mode A: ETC-based ---
    if use_etc or not inp_dat.get("specific_event", False):
        fixed_snr = inp_dat.get("fixed_snr", 0)
        if fixed_snr and fixed_snr > 0:
            # Constant SNR mode, no SNR file needed at all
            snr_file = ""
        else:
            # ETC file expected by the user
            snr_file = _os.path.join(inputs, f"CARMENES_NIR_ETC_WAVE_SNR_{planet}.fits")
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
        # Unwrap single-element lists for the rest of the code
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
