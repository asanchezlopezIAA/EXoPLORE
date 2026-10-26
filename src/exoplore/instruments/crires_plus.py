"""
exoplore.instruments.crires_plus
==================================

Instrument model for CRIRES+ at the VLT.

CRIRES+ is a cross-dispersed echelle spectrograph covering the near-infrared
(0.9 to 5.3 µm) at R ≈ 100,000.  Because CRIRES+ is an echelle spectrograph,
simulations are set up one wavelength setting at a time. Each setting covers
a small number of spectral orders (typically 6 to 10).

Observing modes
---------------
For a reference-night or observed-data run (``specific_event`` or
``use_real_data``), read the files written by
``scripts/prepare_crires_night.py`` in ``inputs_dir/reference_night/``:
``wave.fits``, ``sig_0.fits``, ``snr_0.fits``, ``julian_date_0.fits`` and
``airmass_0.fits``. ``resolution_0.fits`` supplies a measured resolving power
when available. Its absence selects the nominal R = 100,000.

The ETC simulation branch reads ``CRIRES_ETC_WAVE_<planet>.fits`` and
``CRIRES_ETC_SNR_<planet>.fits`` under ``inputs_dir``. These are separate FITS
arrays with the order/pixel layout used by ``wavegrid.get_WaveGrid()``.
``InstrumentInputs`` allows direct Python callers to supply the filenames.
The JSON loader uses the reader's default names.

The wavelength file determines ``n_orders_total``. An empty
``instrument.order_indices`` selects all its orders. Select ``[0]`` for a
single segment. Observed spectral matrices use exposure and wavelength-pixel
axes within each order, with nights stored separately by the runner.

References
----------
Dorn et al. 2023, CRIRES+ instrument paper, A&A
ESO CRIRES+ ETC, https://etc.eso.org
"""

from __future__ import annotations

# Role in the calculation
# Resolve CRIRES+ input files, spectral orders and resolution for the selected
# wavelength setting. The reduction and preparation scripts produce the files read
# here.


import os as _os
from .base import InstrumentInfo, InstrumentInputs

_RES = 1e5         # nominal resolving power
_OBS = "paranal"   # VLT, Paranal


def get_instrument_info(request: InstrumentInputs) -> InstrumentInfo:
    """Return instrument metadata for CRIRES+.

    The number of spectral orders is read from the selected wavelength FITS file
    at runtime.  If the file is not available at config-load time, a sentinel
    value of -1 is returned for ``n_orders_total``. The simulator will then
    determine the true order count after loading the wavelength grid.

    Parameters
    ----------
    inputs : InstrumentInputs
        Input directory, target name, observing configuration and constant S/N.

    Returns
    -------
    InstrumentInfo
        Named tuple with file paths and instrument parameters.
        ``n_orders_total`` is set from the ETC file shape if readable, else -1.
    """
    inputs   = request.inputs_dir
    planet   = request.planet_name
    ref      = _os.path.join(inputs, "reference_night")

    # Mode C, real-data analysis (or reference-night mode): read the reduced
    # night written by scripts/prepare_crires_night.py, exactly like IGRINS.
    _real = (request.observation.use_real_data
             or request.observation.specific_event)
    if _real:
        wave_file = _os.path.join(ref, "wave.fits")
        sig_file      = _os.path.join(ref, "sig_0.fits")
        snr_file      = _os.path.join(ref, "snr_0.fits")
        JD_file       = _os.path.join(ref, "julian_date_0.fits")
        airmass_file  = _os.path.join(ref, "airmass_0.fits")
        n_orders_total = _read_n_orders(wave_file)
        # Effective resolution measured from the slit function at prepare time
        # (Nortmann A.1 super-resolution check). nominal 1e5 as a fallback.
        res = _measured_resolution(_os.path.join(ref, "resolution_0.fits"))
        return InstrumentInfo(
            observatory=_OBS, wave_file=wave_file, sig_file=sig_file,
            snr_file=snr_file, JD_file=JD_file, airmass_file=airmass_file,
            gaps=None, n_orders_total=n_orders_total, res=res,
        )

    # Mode A, ETC-based simulation (no real data).
    def _resolve(val, default_name):
        if val and _os.path.isabs(val):
            return val
        return _os.path.join(inputs, val or default_name)

    wave_file = _resolve(request.wave_file, f"CRIRES_ETC_WAVE_{planet}.fits")
    snr_file  = _resolve(request.snr_file,  f"CRIRES_ETC_SNR_{planet}.fits")
    n_orders_total = _read_n_orders(wave_file)

    return InstrumentInfo(
        observatory=_OBS,
        wave_file=wave_file,
        sig_file="",
        snr_file=snr_file,
        JD_file="",
        airmass_file="",
        gaps=None,
        n_orders_total=n_orders_total,
        res=_RES,
    )


def _measured_resolution(path: str) -> float:
    """Median effective resolution from the per segment resolution file written
    by prepare_crires_night.py (Nortmann A.1 super-resolution check).  Returns
    the nominal CRIRES+ resolution if the file is absent or unreadable."""
    if not _os.path.exists(path):
        return _RES
    try:
        from astropy.io import fits
        import numpy as np
        R = np.asarray(fits.getdata(path), float)
        R = R[np.isfinite(R) & (R > 0)]
        return float(np.median(R)) if R.size else _RES
    except Exception:
        return _RES


def _read_n_orders(wave_file: str) -> int:
    """Return the number of orders from the ETC wavelength FITS file.

    Reads only the FITS header. No data are loaded.  Returns -1 if the
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
