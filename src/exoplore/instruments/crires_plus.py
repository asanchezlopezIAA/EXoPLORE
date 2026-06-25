"""
exoplore.instruments.crires_plus
==================================

Instrument model for CRIRES+ at the VLT.

CRIRES+ is a cross-dispersed echelle spectrograph covering the near-infrared
(0.9 to 5.3 µm) at R ≈ 100,000.  Because CRIRES+ is an echelle spectrograph,
simulations are set up one wavelength setting at a time; each setting covers
a small number of spectral orders (typically 6 to 10).

Observing mode
--------------
CRIRES+ uses **Mode A (ETC-based)** exclusively in EXoPLORE.  The user must
provide an ETC FITS file containing the wavelength grid and SNR per order.

To obtain this file:

1. Run the ESO CRIRES+ ETC at https://etc.eso.org/observing/etc/ for your
   target star, integration time, and wavelength setting.
2. Export the result as a FITS file with two extensions (or columns):
       Extension 0 or column "WAVE": wavelength in µm, shape (n_orders, n_pixels)
       Extension 1 or column "SNR":  SNR per resolution element, shape (n_orders, n_pixels)
3. Place the file in ``inputs_dir`` and set the path in your config.

**Single-order note:** if your ETC file contains only one spectral order (e.g.
a single CRIRES+ detector), set ``"order_indices": [0]`` in your config.  The
wavelength grid will have shape ``(1, n_pixels)`` and the tensor throughout the
simulator will be ``(n_spectra, 1, n_pixels)``.

The total number of orders is determined automatically from the shape of the
ETC FITS file and returned as ``n_orders_total``.  An empty
``order_indices`` list is then expanded to ``np.arange(n_orders_total)``.

Required input files
--------------------
``inputs_dir/<wave_file>``, ETC wavelength grid FITS, shape (n_orders, n_pixels), µm
``inputs_dir/<snr_file>``, ETC SNR FITS, shape (n_orders, n_pixels)

Both file paths are set in the config under ``instrument.wave_file`` and
``instrument.snr_file``.  There is no bundled wavelength file for CRIRES+
because the order layout depends on the chosen wavelength setting.

References
----------
Dorn et al. 2023, CRIRES+ instrument paper, A&A
ESO CRIRES+ ETC, https://etc.eso.org
"""

from __future__ import annotations

import os as _os
from .base import InstrumentInfo

_RES = 1e5         # nominal resolving power
_OBS = "paranal"   # VLT, Paranal


def get_instrument_info(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for CRIRES+.

    The number of spectral orders is read from the ETC wavelength FITS file
    at runtime.  If the file is not available at config-load time, a sentinel
    value of -1 is returned for ``n_orders_total``; the simulator will then
    determine the true order count after loading the wavelength grid.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Relevant keys:

        ``inputs_dir``    Base path for input files.
        ``wave_file``     Filename of the ETC wavelength FITS (relative to
                          ``inputs_dir`` or absolute path).
        ``snr_file``      Filename of the ETC SNR FITS (relative to
                          ``inputs_dir`` or absolute path).
        ``specific_event``  True → use JD/airmass from reference night files.

    Returns
    -------
    InstrumentInfo
        Named tuple with file paths and instrument parameters.
        ``n_orders_total`` is set from the ETC file shape if readable, else -1.
    """
    inputs   = inp_dat.get("inputs_dir", "inputs/")
    planet   = inp_dat.get("Exoplanet_name", "")

    # Allow absolute paths or paths relative to inputs_dir
    def _resolve(key, default_name):
        val = inp_dat.get(key, "")
        if val and _os.path.isabs(val):
            return val
        return _os.path.join(inputs, val or default_name)

    wave_file = _resolve("wave_file", f"CRIRES_ETC_WAVE_{planet}.fits")
    snr_file  = _resolve("snr_file",  f"CRIRES_ETC_SNR_{planet}.fits")

    # JD / airmass for specific_event mode (_0 suffix convention)
    if inp_dat.get("specific_event", False):
        ref = _os.path.join(inputs, "reference_night")
        JD_file      = _os.path.join(ref, "julian_date_0.fits")
        airmass_file = _os.path.join(ref, "airmass_0.fits")
    else:
        JD_file = airmass_file = ""

    # Try to read n_orders from the ETC FITS file
    n_orders_total = _read_n_orders(wave_file)

    return InstrumentInfo(
        observatory=_OBS,
        wave_file=wave_file,
        sig_file="",
        snr_file=snr_file,
        JD_file=JD_file,
        airmass_file=airmass_file,
        gaps=None,
        n_orders_total=n_orders_total,
        res=_RES,
    )


def _read_n_orders(wave_file: str) -> int:
    """Return the number of orders from the ETC wavelength FITS file.

    Reads only the FITS header; no data are loaded.  Returns -1 if the
    file does not exist or cannot be parsed.
    """
    if not _os.path.exists(wave_file):
        return -1
    try:
        from astropy.io import fits
        with fits.open(wave_file) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 1:
                    shape = hdu.data.shape
                    return shape[0] if hdu.data.ndim == 2 else 1
        return -1
    except Exception:
        return -1
