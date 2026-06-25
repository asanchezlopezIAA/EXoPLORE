"""
exoplore.instruments.igrins
============================

Instrument model for IGRINS (Immersion GRating INfrared Spectrograph).

IGRINS covers 1.43 to 2.52 µm simultaneously in H and K bands at R ≈ 45,000
across 53 echelle orders (28 H-band + 25 K-band).  Orders are stored in
ascending wavelength order (H first, K second) as produced by
``scripts/prepare_igrins_night.py``.

Observing modes
---------------
Only Mode B (reference-night, synthetic) and Mode C (real-data analysis)
are supported.  Mode A (ETC-based) is not implemented because no public
IGRINS ETC is available in a standard format.

Mode B, Reference-night, synthetic spectra (recommended)
    ``use_real_data = False``, ``specific_event = True``
    Reads JD, airmass, and per-pixel SNR from the output of
    ``scripts/prepare_igrins_night.py``.
    Required files in ``inputs_dir/reference_night/``:

        julian_date_0.fits, 1-D BJD_TDB array          (n_spectra,)
        airmass_0.fits, 1-D mean airmass array      (n_spectra,)
        snr_0.fits, SNR cube                    (n_spectra, n_orders, n_pixels)

    The ``_0`` suffix is always required, even for a single night.

Mode C, Real-data analysis
    ``use_real_data = True``
    All Mode B files are required plus the observed spectral cubes written
    by ``scripts/prepare_igrins_night.py``:

        observations_berv_0.fits, BERV in km/s  (n_spectra,)
        observations_night_0_order_{h}.fits, Flux per order (n_spectra, n_pixels)

Wavelength grid
---------------
The wavelength grid is read from ``inputs_dir/reference_night/wave.fits``
(written by ``scripts/prepare_igrins_night.py``), shape (n_orders, n_pixels)
in µm.  This file is observation-specific (derived from the PLP wavelength
solution for the particular dataset) and is not bundled with EXoPLORE.

Order selection
---------------
Cheverall et al. (2026) retain 26 of the 53 orders, discarding:
    orders  0 to 5  (K-band red edge, high thermal background)
    orders 23 to 37  (H-K gap / water vapour band at 1.83 to 2.0 µm)
    orders 47 to 52  (H-band blue edge, low throughput)

Set ``instrument.order_indices`` in the config accordingly, for example::

    "order_indices": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                      38,39,40,41,42,43,44,45,46]

References
----------
Yuk et al. 2010, SPIE, IGRINS overview
Park et al. 2014, SPIE, IGRINS design
Cheverall et al. 2026, MNRAS, L 98-59 d analysis with IGRINS on Gemini South
"""

from __future__ import annotations

import os as _os
from .base import InstrumentInfo

_N_ORDERS = 53        # 28 H-band + 25 K-band, ascending wavelength
_RES      = 45_000.0  # R ≈ 45,000 across both bands
_OBS      = "cerropachon"  # Gemini South, Cerro Pachón, Chile


def get_instrument_info(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for IGRINS (1.43 to 2.52 µm, 53 orders).

    Selects Mode B or C automatically based on flags in *inp_dat*.
    Mode A (ETC-based) is not supported for IGRINS.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Relevant keys:

        ``inputs_dir``       Base path for input files.
        ``specific_event``   True → use real JD/airmass from reference night.
        ``different_nights`` True → load per-night JD/airmass/SNR files.
        ``n_nights``         Number of nights (used when different_nights=True).
        ``use_real_data``    True → Mode C (real-data analysis).

    Returns
    -------
    InstrumentInfo
        Named tuple with file paths and instrument parameters.
    """
    inputs = inp_dat.get("inputs_dir", "inputs/")
    ref    = _os.path.join(inputs, "reference_night")

    different_nights = (
        inp_dat.get("different_nights", False)
        or inp_dat.get("Different_nights", False)
    )
    n_nights = inp_dat.get("n_nights", 1)

    # Wavelength grid: written by prepare_igrins_night.py into reference_night/
    wave_file = _os.path.join(ref, "wave.fits")

    # Mode B / C: reference-night files (always _N suffix)
    n_ref        = n_nights if different_nights else 1
    sig_file     = [_os.path.join(ref, f"sig_{i}.fits")         for i in range(n_ref)]
    snr_file     = [_os.path.join(ref, f"snr_{i}.fits")         for i in range(n_ref)]
    JD_file      = [_os.path.join(ref, f"julian_date_{i}.fits") for i in range(n_ref)]
    airmass_file = [_os.path.join(ref, f"airmass_{i}.fits")     for i in range(n_ref)]

    if not different_nights:
        sig_file, snr_file, JD_file, airmass_file = (
            sig_file[0], snr_file[0], JD_file[0], airmass_file[0]
        )

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
