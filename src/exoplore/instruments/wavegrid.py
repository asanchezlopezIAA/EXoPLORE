"""
exoplore.instruments.wavegrid
=============================

Wavelength-grid and reference-night reader shared by every instrument
(CARMENES, CRIRES, CRIRES+, IGRINS, ANDES) via ``get_WaveGrid``, plus the
ANDES ETC-based instrument model (bands below).

ANDES covers five photometric bands at R ≈ 100,000:

    +--------------+----------+---------+-------------------------------+
    | Config name  | Coverage | Orders  | ETC file extension            |
    +==============+==========+=========+===============================+
    | ANDES_YJHK   | 0.4 to 1.8  |  76     | ANDES_ETC_WAVE_SNR_YJHK_*.fits|
    | ANDES_YJH    | 0.4 to 1.4  |  55     | ANDES_ETC_WAVE_SNR_YJH_*.fits |
    | ANDES_K      | 1.4 to 1.8  |  21     | ANDES_ETC_WAVE_SNR_K_*.fits   |
    | ANDES_RIZ    | 0.5 to 0.9  |  34     | ANDES_ETC_WAVE_SNR_RIZ_*.fits |
    | ANDES_UBV    | 0.3 to 0.5  |  62     | ANDES_ETC_WAVE_SNR_UBV_*.fits |
    +--------------+----------+---------+-------------------------------+

All ANDES bands use **Mode A (ETC-based)**: the SNR per exposure and
wavelength grid are derived from an ETC FITS file; no real reference night
is needed.  Place your ETC file in ``inputs/`` (see ``docs/input_files.md``).

Observing mode summary
----------------------
Mode A (ETC-based, recommended for ANDES):
    ``use_real_data = False``, ``fixed_snr = False``
    Required: ETC FITS matrix in ``inputs_dir``.

Mode A with fixed SNR:
    ``fixed_snr = <value>``, bypasses the ETC file entirely;
    every pixel in every exposure receives the same constant SNR.
    Useful for quick sensitivity studies without an ETC file.

Mode C (real-data analysis, any ANDES band):
    ``use_real_data = True``, loads real ANDES spectra from disk and
    runs the analysis pipeline on them.  Requires a reference night
    directory with JD, airmass, and spectral data.

References
----------
Marconi et al. 2022, ANDES science case
ESO ETC outputs for ANDES
"""

from __future__ import annotations

import glob as _glob
import os as _os

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .base import InstrumentInfo


# ---------------------------------------------------------------------------
# Order counts per ANDES band (derived from ESO ETC FITS files)
# ---------------------------------------------------------------------------
_N_ORDERS = {
    "ANDES_YJHK": 76,   # 55 YJH + 21 K
    "ANDES_YJH":  55,
    "ANDES_K":    21,
    "ANDES_RIZ":  34,
    "ANDES_UBV":  62,
}
_RES_ANDES = 1e5          # nominal resolving power R ≈ 100 000
_OBS_ANDES = "paranal"    # ESO Paranal observatory (ELT site)


def _andes_etc_file(inp_dat: dict, band_suffix: str) -> str:
    """Return the expected ETC FITS path for a given ANDES band suffix."""
    planet = inp_dat.get("Exoplanet_name", "HD189733b")
    inputs = inp_dat.get("inputs_dir", "inputs/")
    return _os.path.join(inputs, f"ANDES_ETC_WAVE_SNR_{band_suffix}_{planet}.fits")


def _andes_jd_airmass(inp_dat: dict):
    """Return (JD_file, airmass_file) for ANDES (only needed for specific_event=True)."""
    inputs = inp_dat.get("inputs_dir", "inputs/")
    ref = _os.path.join(inputs, "reference_night")
    if inp_dat.get("specific_event", False):
        return (
            _os.path.join(ref, "julian_date_0.fits"),
            _os.path.join(ref, "airmass_0.fits"),
        )
    return "", ""


# ---------------------------------------------------------------------------
# Per-band get_instrument_info() implementations
# ---------------------------------------------------------------------------

def get_instrument_info_YJHK(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for ANDES YJHK (0.4 to 1.8 µm, 76 orders).

    This is the full ANDES coverage combining the YJH and K arms.  The ETC
    FITS file must contain extensions ``YJH_WAVE_STARTS``, ``YJH_WAVE_MIDS``,
    ``YJH_WAVE_ENDS``, ``YJH_SNR_MID``, ``K_WAVE_STARTS``, ``K_WAVE_MIDS``,
    ``K_WAVE_ENDS``, ``K_SNR_MID``.
    """
    jd, am = _andes_jd_airmass(inp_dat)
    return InstrumentInfo(
        observatory=_OBS_ANDES,
        wave_file=_andes_etc_file(inp_dat, "YJHK"),
        sig_file="",
        snr_file="__ETC__",
        JD_file=jd,
        airmass_file=am,
        gaps=None,
        n_orders_total=_N_ORDERS["ANDES_YJHK"],
        res=_RES_ANDES,
    )


def get_instrument_info_YJH(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for ANDES YJH (0.4 to 1.4 µm, 55 orders).

    The ETC FITS file must contain extensions ``WAVE_STARTS``, ``WAVE_MIDS``,
    ``WAVE_ENDS``, ``SNR_MID``.
    """
    jd, am = _andes_jd_airmass(inp_dat)
    return InstrumentInfo(
        observatory=_OBS_ANDES,
        wave_file=_andes_etc_file(inp_dat, "YJH"),
        sig_file="",
        snr_file="__ETC__",
        JD_file=jd,
        airmass_file=am,
        gaps=None,
        n_orders_total=_N_ORDERS["ANDES_YJH"],
        res=_RES_ANDES,
    )


def get_instrument_info_K(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for ANDES K (1.4 to 1.8 µm, 21 orders).

    The ETC FITS file must contain extensions ``WAVE_STARTS``, ``WAVE_MIDS``,
    ``WAVE_ENDS``, ``SNR_MID``.
    """
    jd, am = _andes_jd_airmass(inp_dat)
    return InstrumentInfo(
        observatory=_OBS_ANDES,
        wave_file=_andes_etc_file(inp_dat, "K"),
        sig_file="",
        snr_file="__ETC__",
        JD_file=jd,
        airmass_file=am,
        gaps=None,
        n_orders_total=_N_ORDERS["ANDES_K"],
        res=_RES_ANDES,
    )


def get_instrument_info_RIZ(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for ANDES RIZ (0.5 to 0.9 µm, 34 orders).

    The ETC FITS file must contain extensions ``WAVE_STARTS``, ``WAVE_MIDS``,
    ``WAVE_ENDS``, ``SNR_MID``.
    """
    jd, am = _andes_jd_airmass(inp_dat)
    return InstrumentInfo(
        observatory=_OBS_ANDES,
        wave_file=_andes_etc_file(inp_dat, "RIZ"),
        sig_file="",
        snr_file="__ETC__",
        JD_file=jd,
        airmass_file=am,
        gaps=None,
        n_orders_total=_N_ORDERS["ANDES_RIZ"],
        res=_RES_ANDES,
    )


def get_instrument_info_UBV(inp_dat: dict) -> InstrumentInfo:
    """Return instrument metadata for ANDES UBV (0.3 to 0.5 µm, 62 orders).

    The ETC FITS file must contain extensions ``WAVE_STARTS``, ``WAVE_MIDS``,
    ``WAVE_ENDS``, ``SNR_MID``.
    """
    jd, am = _andes_jd_airmass(inp_dat)
    return InstrumentInfo(
        observatory=_OBS_ANDES,
        wave_file=_andes_etc_file(inp_dat, "UBV"),
        sig_file="",
        snr_file="__ETC__",
        JD_file=jd,
        airmass_file=am,
        gaps=None,
        n_orders_total=_N_ORDERS["ANDES_UBV"],
        res=_RES_ANDES,
    )


@dataclass
class ANDESOrder:
    """One ANDES spectral order.

    Parameters
    ----------
    index:
        Order index (0-based).
    wavelength_start_um:
        Blaze start wavelength in microns.
    wavelength_mid_um:
        Order central wavelength in microns.
    wavelength_end_um:
        Blaze end wavelength in microns.
    wavelength_grid:
        Pixel-level wavelength array in microns (if built).
    snr_per_pixel:
        SNR per pixel array (if loaded from an ETC file).
    """
    index: int
    wavelength_start_um: float
    wavelength_mid_um: float
    wavelength_end_um: float
    wavelength_grid: Optional[np.ndarray] = None
    snr_per_pixel: Optional[np.ndarray] = None


class ANDESInstrument:
    """ANDES spectrograph model.

    Parameters
    ----------
    resolving_power:
        Spectral resolving power R = λ/Δλ.  Default 100 000.
    pixels_per_resolution_element:
        Detector pixels per resolution element.  Default 2.5.
    n_pixels_per_order:
        Number of pixels per spectral order.  Default 2048.

    Examples
    --------
    >>> from exoplore.instruments import ANDESInstrument
    >>> andes = ANDESInstrument()
    >>> grids = andes.build_wavelength_grids(
    ...     lam_start=[1.0, 1.05],
    ...     lam_mid=[1.025, 1.075],
    ...     lam_end=[1.05, 1.10],
    ... )
    >>> len(grids)
    2
    """

    def __init__(
        self,
        resolving_power: float = 100_000.0,
        pixels_per_resolution_element: float = 2.5,
        n_pixels_per_order: int = 2048,
    ) -> None:
        self.resolving_power = resolving_power
        self.pixels_per_resolution_element = pixels_per_resolution_element
        self.n_pixels_per_order = n_pixels_per_order

    def build_wavelength_grids(
        self,
        lam_start: List[float],
        lam_mid: List[float],
        lam_end: List[float],
        mode: int = 2,
        scale: float = 1.0,
    ) -> List[np.ndarray]:
        """Build per-order wavelength grids.

        Two modes are available:

        - ``mode=1``: Perfect Nyquist sampling centred on ``lam_mid``.
          Does not exactly respect ``lam_start`` / ``lam_end``.
        - ``mode=2``: Fixed number of pixels spanning ``lam_start`` to
          ``lam_end`` exactly.  Effective sampling may deviate slightly
          from Nyquist.

        Parameters
        ----------
        lam_start, lam_mid, lam_end:
            ETC-provided start, central, and end wavelengths for each
            order (in microns by default).
        mode:
            Grid-building mode (1 or 2).
        scale:
            Wavelength scale factor (e.g. 1e-3 to convert µm → mm).

        Returns
        -------
        list of numpy.ndarray
            One wavelength array per order in the chosen units.
        """
        n_orders = len(lam_start)
        if len(lam_mid) != n_orders or len(lam_end) != n_orders:
            raise ValueError(
                "lam_start, lam_mid, lam_end must all have the same length."
            )

        R = self.resolving_power
        m = self.pixels_per_resolution_element
        N = self.n_pixels_per_order
        grids = []

        for k in range(n_orders):
            l0, lm, l1 = lam_start[k], lam_mid[k], lam_end[k]

            if mode == 1:
                dln = 1.0 / (m * R)
                idx_mid = N // 2
                ln_grid = np.log(lm) + (np.arange(N) - idx_mid) * dln
                grid = np.exp(ln_grid) * scale

            elif mode == 2:
                ln0, ln1 = np.log(l0), np.log(l1)
                dln = (ln1 - ln0) / (N - 1)
                grid = np.exp(ln0 + np.arange(N) * dln) * scale

            else:
                raise ValueError("mode must be 1 or 2.")

            grids.append(grid)

        return grids

    def pixel_snr(
        self,
        snr_per_resel: np.ndarray,
        dit_seconds: float,
        dit_ref_seconds: float = 900.0,
        airmass: float = 1.0,
        airmass_ref: float = 1.0,
        noise_scaling_factor: float = 1.0,
    ) -> np.ndarray:
        """Convert per-resolution-element SNR to per-pixel SNR.

        Parameters
        ----------
        snr_per_resel:
            SNR per resolution element from the ETC, shape (n_pixels,).
        dit_seconds:
            Actual exposure time in seconds.
        dit_ref_seconds:
            Reference exposure time used for the ETC SNR in seconds.
        airmass:
            Observed airmass.
        airmass_ref:
            Reference airmass used for the ETC SNR.
        noise_scaling_factor:
            Multiplicative factor applied to all noise (default 1.0).

        Returns
        -------
        numpy.ndarray
            Per-pixel SNR array.
        """
        snr_pixel = snr_per_resel / np.sqrt(self.pixels_per_resolution_element)
        snr_pixel *= np.sqrt(dit_seconds / dit_ref_seconds)
        # Telluric scaling with airmass
        snr_pixel *= np.sqrt(airmass_ref / airmass)
        snr_pixel *= noise_scaling_factor
        return snr_pixel


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def Load_Instrumental_Info(inp_dat):
    """Return file paths and parameters for the selected instrument.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``"instrument"``,
        ``"ETC"``, ``"inputs_dir"``, ``"Exoplanet_name"``, and
        ``"Different_nights"`` / ``"n_nights"`` keys.

    Returns
    -------
    observatory : str
    wave_file : str
    sig_file : str or list
    snr_file : str or list
    JD_file : str or list
    airmass_file : str or list
    gaps : None or list
    norders_fix : None or int
    res : float
    """
    if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS', 'CRIRES']:
        if not inp_dat["ETC"]:
            observatory = "lasilla"
            if inp_dat['instrument'] == 'CARMENES_NIR':
                import os as _os
                # Prefer (n_orders, n_pixels) file; fall back to (n_spectra, n_pixels, n_orders)
                _pkg_data = _os.path.join(
                    _os.path.dirname(__file__), "data", "wave_CARMENES_NIR.fits"
                )
                _user_clean = f"{inp_dat['inputs_dir']}wave_CARMENES_NIR.fits"
                _user_fallback = f"{inp_dat['inputs_dir']}wave_NIR.fits"
                if _os.path.exists(_user_clean):
                    wave_file = _user_clean
                elif _os.path.exists(_pkg_data):
                    wave_file = _pkg_data
                else:
                    wave_file = _user_fallback   # fallback
            else:
                wave_file = f"{inp_dat['inputs_dir']}wave_crires.fits"
            _n_ref = inp_dat["n_nights"] if inp_dat["Different_nights"] else 1
            sig_file = [f"{inp_dat['inputs_dir']}reference_night/sig_{i}.fits"
                        for i in range(_n_ref)]
            snr_file = [f"{inp_dat['inputs_dir']}reference_night/snr_{i}.fits"
                        for i in range(_n_ref)]
            JD_file = [f"{inp_dat['inputs_dir']}reference_night/julian_date_{i}.fits"
                       for i in range(_n_ref)]
            airmass_file = [f"{inp_dat['inputs_dir']}reference_night/airmass_{i}.fits"
                            for i in range(_n_ref)]
            if not inp_dat["Different_nights"]:
                sig_file, snr_file, JD_file, airmass_file = (
                    sig_file[0], snr_file[0], JD_file[0], airmass_file[0])
            gaps = None
            norders_fix = None
        else:
            if inp_dat['instrument'] == 'CRIRES':
                observatory = 'paranal'
            else:
                observatory = 'lasilla'
            wave_file = (
                f'/Users/alexsl/Documentos/Simulador/'
                f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/'
                f'ETC/wave_H1582_EXPT90s.fits'
            )
            sig_file = ""
            snr_file = (
                f'/Users/alexsl/Documentos/Simulador/'
                f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/'
                f'ETC/snr_H1582_EXPT90s.fits'
            )
            JD_file = ''
            airmass_file = ""
            gaps = None
            norders_fix = None

        if inp_dat['instrument'] == 'CARMENES_NIR':
            res = 80400.
        elif inp_dat['instrument'] == 'CARMENES_VIS':
            res = 94600.
        elif inp_dat['instrument'] == 'CRIRES':
            res = 1e5

    elif inp_dat['instrument'] in ('ANDES', 'ANDES_YJHK', 'ANDES_YJH',
                                    'ANDES_K', 'ANDES_RIZ', 'ANDES_UBV'):
        # Route to the per-band get_instrument_info() and unpack for
        # the tuple format expected by the rest of Load_Instrumental_Info.
        band = inp_dat['instrument']
        if band in ('ANDES', 'ANDES_YJHK'):
            info = get_instrument_info_YJHK(inp_dat)
        elif band == 'ANDES_YJH':
            info = get_instrument_info_YJH(inp_dat)
        elif band == 'ANDES_K':
            info = get_instrument_info_K(inp_dat)
        elif band == 'ANDES_RIZ':
            info = get_instrument_info_RIZ(inp_dat)
        else:
            info = get_instrument_info_UBV(inp_dat)
        observatory  = info.observatory
        wave_file    = info.wave_file
        sig_file     = info.sig_file
        snr_file     = info.snr_file
        JD_file      = info.JD_file
        airmass_file = info.airmass_file
        gaps         = info.gaps
        norders_fix  = info.n_orders_total
        res          = info.res

    elif inp_dat['instrument'] == 'IGRINS2':
        observatory = '3060m'
        wave_file = "/Users/alexsl/Documents/Simulador/IGRINS2/wave_igrins2.fits"
        sig_file = ""
        snr_file = (
            f"/Users/alexsl/Documents/Simulador/IGRINS2/"
            f"{inp_dat['Exoplanet_name']}/snr_igrins2_{inp_dat['Exoplanet_name']}.fits"
        )
        JD_file = ""
        airmass_file = ""
        gaps = None
        norders_fix = None
        res = 45000.

    return observatory, wave_file, sig_file, snr_file, JD_file, \
           airmass_file, gaps, norders_fix, res


def make_wave_grid_ANDES_modes(
        lam_start,
        lam_mid,
        lam_end,
        R: float = 100000,
        px_per_resel: float = 2.0,
        N_pixels: int = 2048,
        scale: float = 1.0,
        mode: int = 1,
) -> list:
    """Build per-order wavelength grids for ANDES from ETC start/mid/end values.

    mode=1  Perfect Nyquist at lam_mid (dln = 1/(m*R)), fixed N_pixels,
            grid centred on lam_mid; does NOT enforce lam_start/end.
    mode=2  Fixed N_pixels; grid spans lam_start → lam_end (dln variable).

    Parameters
    ----------
    lam_start, lam_mid, lam_end : array_like, shape (n_orders,)
        ETC-provided start, mid, and end wavelengths (microns).
    R : float
        Resolving power λ/Δλ.
    px_per_resel : float
        Desired pixels per resolution element (mode=1).
    N_pixels : int
        Fixed number of pixels per order (default 2048).
    scale : float
        Scale factor applied to the output wavelengths (e.g. 1e-3 converts
        microns to mm).
    mode : int
        1 or 2, see description above.

    Returns
    -------
    list of ndarray
        Wavelength grid for each spectral order.
    """
    n_orders = len(lam_start)
    if not (len(lam_mid) == n_orders and len(lam_end) == n_orders):
        raise ValueError("lam_start, lam_mid, lam_end must have the same length")

    wave_grids = []
    print(
        f"Building grids (mode={mode}) for {n_orders} orders: "
        f"R={R}, m={px_per_resel}, N_pixels={N_pixels}, scale={scale}"
    )

    for k in range(n_orders):
        lam0, lamm, lam1 = lam_start[k], lam_mid[k], lam_end[k]

        if mode == 1:
            dln = 1.0 / (px_per_resel * R)
            idx_mid = N_pixels // 2
            ln_m = np.log(lamm)
            ln_grid = ln_m + (np.arange(N_pixels) - idx_mid) * dln
            grid = np.exp(ln_grid) * scale
            d_start = (grid[0] - lam0 * scale) / (lam0 * scale)
            d_end = (grid[-1] - lam1 * scale) / (lam1 * scale)
            print(
                f"Order {k} mode1: start={grid[0]:.6f} (Δ/λ={d_start:.2e}), "
                f"end={grid[-1]:.6f} (Δ/λ={d_end:.2e})"
            )

        elif mode == 2:
            ln0, ln1 = np.log(lam0), np.log(lam1)
            dln = (ln1 - ln0) / (N_pixels - 1)
            ln_grid = ln0 + np.arange(N_pixels) * dln
            grid = np.exp(ln_grid) * scale
            m_eff = 1.0 / (R * dln)
            idx_mid = np.argmin(np.abs(grid - lamm * scale))
            d_mid = (grid[idx_mid] - lamm * scale) / (lamm * scale)
            print(
                f"Order {k} mode2: m_eff={m_eff:.2f} px/resel, "
                f"mid_pix={idx_mid}, mid_λ={grid[idx_mid]:.6f} (Δ/λ={d_mid:.1e})"
            )

        else:
            raise ValueError("mode must be 1 or 2")

        wave_grids.append(grid)

    return wave_grids


def get_WaveGrid(inp_dat, wave_file, sig_file, snr_file, JD_file,
                 airmass_file, n_orders):
    """Read or reconstruct the wavelength grid for the selected instrument.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.
    wave_file, sig_file, snr_file, JD_file, airmass_file : str
        Paths to FITS files returned by :func:`Load_Instrumental_Info`.
    n_orders : int
        Number of spectral orders (used for METIS division).

    Returns
    -------
    wave_star : ndarray  Wavelength grid, shape (n_orders, n_pixels).
    n_pixels : int       Pixels per order.
    sig_og : ndarray or None
    snr_og : ndarray or None
    JD_og : ndarray or None
    airmass_og : ndarray or None
    wave_mid_og : ndarray or None   (ANDES YJHK mode only)
    """
    from astropy.io import fits

    if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS']:
        raw = fits.open(wave_file)[0].data
        # Accept two formats:
        #   CARACAL: shape (n_spectra, n_pixels, n_orders) e.g. (46, 4080, 28)
        #   Clean  (EXoPLORE): shape (n_orders, n_pixels)           e.g. (28, 4080)
        if raw.ndim == 3:
            # CARACAL convention: (n_spectra, n_pixels, n_orders)
            # → transpose to (n_spectra, n_orders, n_pixels), take mean over spectra
            wvl_clean = np.nanmean(np.transpose(raw, (0, 2, 1)), axis=0)  # (n_orders, n_pixels)
        else:
            # Already (n_orders, n_pixels), clean format
            wvl_clean = raw  # (n_orders, n_pixels)

        n_pixels = wvl_clean.shape[1]
        # wave_star is returned as (n_pixels, n_orders) here;
        # simulator Block 2e transposes it back to (n_orders, n_pixels)
        wave_star_out = wvl_clean.T

        if not inp_dat["Different_nights"]:
            sig_og = fits.open(sig_file)[0].data if sig_file != '' else None
            snr_og = fits.open(snr_file)[0].data if snr_file != '' else None
            JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
            airmass_og = fits.open(airmass_file)[0].data if airmass_file != '' else None
            return wave_star_out, n_pixels, sig_og, snr_og, JD_og, airmass_og
        else:
            sig_og_list, snr_og_list, JD_og_list, airmass_og_list = [], [], [], []
            for i in range(inp_dat["n_nights"]):
                night_sig = f"{sig_file[i]}" if sig_file != '' else None
                night_snr = f"{snr_file[i]}" if snr_file != '' else None
                night_JD = f"{JD_file[i]}" if JD_file != '' else None
                night_am = f"{airmass_file[i]}" if airmass_file != '' else None
                sig_og_list.append(fits.open(night_sig)[0].data if night_sig else None)
                snr_og_list.append(fits.open(night_snr)[0].data if night_snr else None)
                JD_og_list.append(fits.open(night_JD)[0].data if night_JD else None)
                airmass_og_list.append(fits.open(night_am)[0].data if night_am else None)
            return (wave_star_out, n_pixels,
                    sig_og_list, snr_og_list, JD_og_list, airmass_og_list)

    elif inp_dat['instrument'] in ('ANDES_YJH', 'ANDES_K', 'ANDES_RIZ', 'ANDES_UBV'):
        # Single-arm ANDES bands: ETC file has generic extension names
        # WAVE_STARTS, WAVE_MIDS, WAVE_ENDS, SNR_MID
        with fits.open(wave_file) as hdul:
            wave_start = np.array(hdul['WAVE_STARTS'].data)
            wave_mid   = np.array(hdul['WAVE_MIDS'].data)
            wave_end   = np.array(hdul['WAVE_ENDS'].data)
            snr_mid    = np.array(hdul['SNR_MID'].data) if snr_file != '' else None
        wvl_aux = make_wave_grid_ANDES_modes(
            wave_start, wave_mid, wave_end,
            R=1e5, px_per_resel=inp_dat["Pix_per_resel"], scale=1e-3, mode=1
        )
        wvl = np.array(wvl_aux)
        sig_og = None
        snr_og = snr_mid
        JD_og      = fits.open(JD_file)[0].data      if JD_file      else None
        airmass_og = fits.open(airmass_file)[0].data if airmass_file else None
        return wvl, wvl.shape[1], sig_og, snr_og, JD_og, airmass_og, wave_mid

    elif inp_dat['instrument'] in ('ANDES_YJHK', 'ANDES'):
        with fits.open(wave_file) as hdul:
            wave_yjh_start = np.array(hdul['YJH_WAVE_STARTS'].data)
            wave_yjh_mid = np.array(hdul['YJH_WAVE_MIDS'].data)
            wave_yjh_end = np.array(hdul['YJH_WAVE_ENDS'].data)
            snr_yjh = np.array(hdul['YJH_SNR_MID'].data) if snr_file != '' else None
            wave_k_start = np.array(hdul['K_WAVE_STARTS'].data)
            wave_k_mid = np.array(hdul['K_WAVE_MIDS'].data)
            wave_k_end = np.array(hdul['K_WAVE_ENDS'].data)
            snr_k = np.array(hdul['K_SNR_MID'].data) if snr_file != '' else None

        wvl_aux_yjh = make_wave_grid_ANDES_modes(
            wave_yjh_start, wave_yjh_mid, wave_yjh_end,
            R=1e5, px_per_resel=inp_dat["Pix_per_resel"], scale=1e-3, mode=1
        )
        wvl_aux_k = make_wave_grid_ANDES_modes(
            wave_k_start, wave_k_mid, wave_k_end,
            R=1e5, px_per_resel=inp_dat["Pix_per_resel"], scale=1e-3, mode=1
        )
        wvl = np.concatenate([np.array(wvl_aux_yjh), np.array(wvl_aux_k)], axis=0)
        wave_mid_og = np.concatenate([wave_yjh_mid, wave_k_mid], axis=0)
        sig_og = None
        snr_og = np.concatenate([snr_yjh, snr_k], axis=0) if snr_file != '' else None
        JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
        airmass_og = fits.open(airmass_file)[0].data if airmass_file != '' else None
        return wvl, wvl.shape[1], sig_og, snr_og, JD_og, airmass_og, wave_mid_og

    elif inp_dat['instrument'] in ('IGRINS', 'IGRINS2', 'CRIRES+'):
        # 'IGRINS' is the clean reference-night format written by
        # scripts/prepare_igrins_night.py (wave shape (n_orders, n_pixels),
        # sig/snr cubes (n_spectra, n_orders, n_pixels)); 'IGRINS2' is the
        # single-target setup.  'CRIRES+' shares the identical layout from
        # scripts/prepare_crires_night.py.  All share the same read logic.
        wvl = fits.open(wave_file)[0].data
        sig_og = fits.open(sig_file)[0].data if sig_file != '' else None
        snr_og = fits.open(snr_file)[0].data if snr_file != '' else None
        JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
        airmass_og = fits.open(airmass_file)[0].data if airmass_file != '' else None
        return wvl, wvl.shape[1], sig_og, snr_og, JD_og, airmass_og

    elif inp_dat['instrument'] == 'CRIRES':
        if not inp_dat["ETC"]:
            wvl = fits.open(wave_file)[0].data
            sig_og = fits.open(sig_file)[0].data if sig_file != '' else None
            snr_og = fits.open(snr_file)[0].data if snr_file != '' else None
            JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
            airmass_og = fits.open(airmass_file)[0].data if airmass_file != '' else None
            return wvl[0, :, :], wvl[0, :, :].shape[1], sig_og, snr_og, JD_og, airmass_og
        else:
            hdu = fits.open(wave_file)
            wave = 1.e-3 * hdu[0].data
            hdu.close()
            hdu = fits.open(snr_file)
            snr = hdu[0].data
            hdu.close()
            return wave, wave.shape[1], None, snr, None, None

    elif inp_dat['instrument'] == 'METIS':
        with open(wave_file, 'r') as f:
            wave_star_aux, spec_star_aux, sig_star_aux = [], [], []
            n_pixels = -9
            for line in f:
                if n_pixels <= -1:
                    n_pixels += 1
                    continue
                wave_star_aux.append(float(line.split()[0]))
                spec_star_aux.append(float(line.split()[1]))
                sig_star_aux.append(float(line.split()[2]))
                n_pixels += 1
        wave_star_aux = np.asarray(wave_star_aux)
        spec_star_aux = np.asarray(spec_star_aux)
        sig_star_aux = np.asarray(sig_star_aux)

        # Mask bad pixels
        for mask_fn in [
            lambda a: ~np.isfinite(a),
            lambda a: a == 0.,
        ]:
            bad = np.where(mask_fn(sig_star_aux))[0]
            for i in bad[::-1]:
                sig_star_aux = np.delete(sig_star_aux, i)
                spec_star_aux = np.delete(spec_star_aux, i)
                wave_star_aux = np.delete(wave_star_aux, i)
        snr = spec_star_aux / sig_star_aux
        low_snr = np.where(snr < 10)[0]
        for i in low_snr[::-1]:
            sig_star_aux = np.delete(sig_star_aux, i)
            spec_star_aux = np.delete(spec_star_aux, i)
            wave_star_aux = np.delete(wave_star_aux, i)
        snr = spec_star_aux / sig_star_aux

        n_pixels = int(wave_star_aux.size / n_orders)
        wave_star = np.zeros((n_orders, n_pixels), float)
        spec_star = np.zeros((n_orders, n_pixels), float)
        sig_star = np.zeros((n_orders, n_pixels), float)
        for i in range(n_orders):
            wave_star[i, :] = wave_star_aux[i * n_pixels:(i + 1) * n_pixels]
            spec_star[i, :] = spec_star_aux[i * n_pixels:(i + 1) * n_pixels]
            sig_star[i, :] = sig_star_aux[i * n_pixels:(i + 1) * n_pixels]
        return wave_star, len(wave_star[0, :]), None, spec_star / sig_star, None, None

    else:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def From1OrderTo1Detector(variable, h):
    """Split a 2-D spectral-order array into the sub-array for one detector.

    Each ANDES order is read out by two detectors side by side.  Even-numbered
    orders (h % 2 == 0) use the left half of the pixel array; odd-numbered
    orders use the right half.

    Parameters
    ----------
    variable : ndarray, shape (n_spectra, n_pixels)
        Full-order 2-D array (spectra × pixels).
    h : int
        Order index.

    Returns
    -------
    ndarray
        Sub-array for the corresponding detector half, shape
        (n_spectra, n_pixels // 2).
    """
    n_pixels = variable.shape[1]
    if h % 2 == 0:
        new_variable = variable[:, :n_pixels // 2]
    else:
        new_variable = variable[:, n_pixels // 2:]
    return new_variable


def pixel_snr_one_order(
        wave,
        tellurics,
        snr_center,
        px_per_resel: float = 2.0,
        center_pix: int = None,
        stellar_flux=None,
):
    """Compute per-pixel S/N for a single spectral order.

    Parameters
    ----------
    wave : ndarray, shape (n_pix,)
        Wavelength grid for this order (in microns).
    tellurics : ndarray, shape (n_times, n_pix)
        Telluric transmission (0 to 1) at each time and pixel.
    snr_center : float or ndarray, shape (n_times,)
        The ETC S/N at the central resolution element (per exposure).
    px_per_resel : float, optional
        Number of pixels sampling one resolution element.  Default is 2
        (Nyquist).
    center_pix : int, optional
        Initial index in *wave* for the approximate centre.  Overridden by
        the true blaze-peak index if ``None``.
    stellar_flux : ndarray, shape (n_pix,), optional
        Full stellar spectrum including lines.  If provided, S/N scales by
        ``sqrt(flux / flux_center)``.  If ``None`` this factor is unity.

    Returns
    -------
    snr_pixel : ndarray, shape (n_times, n_pix)
        Per-pixel S/N for each exposure and pixel.
    """
    n_times, n_pix = tellurics.shape

    # 1) Convert resel S/N to pixel S/N
    snr_pix_center = np.array(snr_center, ndmin=1) / np.sqrt(px_per_resel)
    snr_pix_center = np.broadcast_to(snr_pix_center, (n_times,))

    # 2) Determine blaze peak index via analytic sinc^2 model
    if center_pix is None:
        center_pix = n_pix // 2
    lambda0_approx = wave[center_pix]
    delta_lambda = wave[-1] - wave[0]
    x = (wave - lambda0_approx) / delta_lambda
    blaze = np.sinc(x) ** 2
    i_blaze = np.argmax(blaze)
    center_pix = i_blaze
    blaze /= blaze[center_pix]
    rel_blaze = np.sqrt(blaze)[None, :]

    # 3) Telluric scaling
    T_center = tellurics[:, center_pix]
    T_center = np.where(T_center > 0, T_center, 1e-8)
    rel_thru = np.sqrt(tellurics / T_center[:, None])

    # 4) Stellar spectrum scaling
    if stellar_flux is not None:
        flux_center = stellar_flux[center_pix]
        if flux_center <= 0:
            flux_center = 1e-8
        rel_flux = np.sqrt(stellar_flux / flux_center)[None, :]
    else:
        rel_flux = 1.0

    # 5) Combine all factors
    snr_pixel = snr_pix_center[:, None] * rel_thru * rel_flux * rel_blaze
    return snr_pixel


def make_log_wave_grid(lambda_min, lambda_max, R, oversample=2):
    """Build a logarithmically-spaced wavelength grid.

    Each pixel step is ``1 / (R * oversample)`` in log-wavelength space,
    giving a constant velocity spacing of ``c / (R * oversample)`` km/s.

    Parameters
    ----------
    lambda_min : float
        Minimum wavelength (any consistent unit).
    lambda_max : float
        Maximum wavelength (same unit as *lambda_min*).
    R : float
        Spectral resolving power R = λ / Δλ.
    oversample : int or float
        Number of pixels per resolution element (default 2).

    Returns
    -------
    numpy.ndarray
        Log-spaced wavelength array covering [lambda_min, lambda_max].
    """
    dlog_lambda = 1. / (R * oversample)
    log_lambda = np.arange(np.log(lambda_min), np.log(lambda_max) + dlog_lambda, dlog_lambda)
    return np.exp(log_lambda)


def compute_pixel_velocity_scale(R, pixels_per_res):
    """Return the velocity width of one pixel in km/s.

    Computes ``v_pixel = c / (R * pixels_per_res)`` where *c* is the
    speed of light in km/s.

    Parameters
    ----------
    R : float
        Spectral resolving power.
    pixels_per_res : float
        Number of detector pixels per resolution element.

    Returns
    -------
    float
        Velocity per pixel in km/s.
    """
    from astropy.constants import c
    v_res_elem = c.to('km/s').value / R
    return v_res_elem / pixels_per_res


def FromOrdersToDetectors(variable, n_orders, n_pixels):
    """Reshape a 3-D order array into the two-detector layout.

    Each ANDES order is physically split across two detectors.  This
    function rearranges a ``(n_spectra, n_orders, n_pixels)`` array into
    ``(n_spectra, 2*n_orders, n_pixels/2)`` by placing the left and right
    detector halves of each order on consecutive rows.

    Parameters
    ----------
    variable : numpy.ndarray, shape (n_spectra, n_orders, n_pixels)
        Input spectral cube.
    n_orders : int
        Number of spectral orders.
    n_pixels : int
        Total pixels per order (must be even).

    Returns
    -------
    numpy.ndarray, shape (n_spectra, 2*n_orders, n_pixels//2)
        Detector-split spectral cube.
    """
    variable_new = np.zeros(
        (variable.shape[0], n_orders * 2, int(n_pixels/2)), float
    )
    for j in range(n_orders):
        part1 = variable[:, j, :int(n_pixels/2)]
        part2 = variable[:, j, int(n_pixels/2):]
        variable_new[:, 2 * j, :] = part1
        variable_new[:, 2 * j + 1, :] = part2
    return variable_new


def Interp_Uniform_Wvl_Grid(wave, spec, sig, new_n_pixels):
    """Interpolate spectra onto a uniform wavelength grid per order.

    For each spectral order the common wavelength range across all
    exposures is found, a uniform grid with *new_n_pixels* points is
    constructed, and both the spectrum and its uncertainty are
    interpolated onto that grid using cubic splines.

    Parameters
    ----------
    wave : numpy.ndarray, shape (n_spectra, n_orders, n_pixels)
        Per-exposure wavelength arrays.
    spec : numpy.ndarray, shape (n_spectra, n_orders, n_pixels)
        Spectral flux arrays.
    sig : numpy.ndarray, shape (n_spectra, n_orders, n_pixels)
        Spectral uncertainty arrays.
    new_n_pixels : int
        Number of pixels in the output uniform grid per order.

    Returns
    -------
    new_wave : numpy.ndarray, shape (n_spectra, n_orders, new_n_pixels)
    new_spec : numpy.ndarray, shape (n_spectra, n_orders, new_n_pixels)
    new_sig  : numpy.ndarray, shape (n_spectra, n_orders, new_n_pixels)
    """
    from scipy import interpolate
    new_wave = np.zeros((wave.shape[0], wave.shape[1], new_n_pixels))
    new_spec = np.zeros((spec.shape[0], spec.shape[1], new_n_pixels))
    new_sig = np.zeros((spec.shape[0], spec.shape[1], new_n_pixels))
    for j in range(spec.shape[1]):
        min_wavelength = np.max(wave[:,j,0])
        max_wavelength = np.min(wave[:,j,-1])
        for n in range(spec.shape[0]):
            new_wave[n, j, :] = np.linspace(min_wavelength, max_wavelength, new_n_pixels)
            interp_func = interpolate.splrep(wave[n, j, :], spec[n, j, :], k=3)
            interp_func_sig = interpolate.splrep(wave[n, j, :], sig[n, j, :], k=3)
            new_spec[n, j, :] = interpolate.splev(new_wave[n, j, :], interp_func)
            new_sig[n, j, :] = interpolate.splev(new_wave[n, j, :], interp_func_sig)
    return new_wave, new_spec, new_sig


def Load_CARMENES(inp_dat, path, keyword):
    """Read CARMENES echelle spectra from a directory of FITS files.

    Scans *path* for files matching ``*{keyword}``, loads the spectral
    data and header keywords (airmass, BERV, MJD), removes fibre-P
    calibration frames, and splits each order onto two detectors using
    :func:`FromOrdersToDetectors`.

    Parameters
    ----------
    inp_dat : dict
        Simulation input dictionary.  Must contain ``'instrument'``
        (``'CARMENES_NIR'`` or ``'CARMENES_VIS'``).
    path : str
        Directory path to search for FITS files.
    keyword : str
        Glob pattern suffix to select files (e.g. ``'_HE.fits'``).

    Returns
    -------
    datafiles : list of str
        Paths of the loaded FITS files (FP-FP frames excluded).
    wave : ndarray, shape (2*n_orders, n_pixels//2)
        Wavelength grid after detector splitting.
    spec : ndarray, shape (n_files, 2*n_orders, n_pixels//2)
        Spectral flux cube.
    sig : ndarray, same shape as *spec*
        Spectral uncertainty cube.
    mjd_utc : ndarray, shape (n_files,)
        Barycentric Julian dates.
    airmass : ndarray, shape (n_files,)
        Airmass values.
    rh : ndarray, shape (n_files,)
        Relative humidity values.
    berv : ndarray, shape (n_files,)
        Barycentric Earth radial velocity in km/s.
    """
    from astropy.io import fits
    if inp_dat['instrument'] == 'CARMENES_NIR':
        n_pixels = 4080
        n_orders = 28
    else:
        n_pixels = 4096
        n_orders = 56

    datafiles = sorted(_glob.glob(f"{path}*{keyword}"))

    wave = np.zeros((n_orders * 2, int(n_pixels / 2)), float)
    airmass = np.zeros((len(datafiles)))
    rh = np.zeros_like(airmass)
    berv = np.zeros_like(airmass)
    mjd_utc = np.zeros_like(airmass)
    spec = np.zeros((len(datafiles), n_orders * 2, int(n_pixels / 2)), float)
    sig = np.zeros_like(spec)
    fp_index_list = list()

    for n in range(len(datafiles)):
        hdu = fits.open(datafiles[n])
        fp = hdu[0].header['HIERARCH CAHA INS ICS FIB-MODE']
        if fp == 'FP,FP':
            fp_index_list.append(n)
            continue
        for j in range(n_orders):
            spec = FromOrdersToDetectors(hdu[1].data, n_orders, n_pixels)
            sig  = FromOrdersToDetectors(hdu[3].data, n_orders, n_pixels)
            if n == 0:
                wave = FromOrdersToDetectors(hdu[4].data, n_orders, n_pixels)
        airmass[n] = hdu[0].header['AIRMASS']
        rh[n]      = hdu[0].header['HIERARCH CAHA GEN AMBI RHUM']
        berv[n]    = hdu[0].header['HIERARCH CARACAL BERV']
        mjd_utc[n] = hdu[0].header['HIERARCH CARACAL BJD']
        hdu.close()

    for l in fp_index_list[::-1]:
        del spec[l, :], sig[l, :], airmass[l], rh[l], mjd_utc[l], berv[l]

    return datafiles, wave, spec, sig, mjd_utc, airmass, rh, berv
