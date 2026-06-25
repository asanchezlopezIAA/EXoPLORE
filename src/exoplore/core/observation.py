"""
exoplore.core.observation
=========================

Spectral matrix builders and related in-transit/dayside helpers.

These functions take the atmospheric spectrum computed by pRT and
Doppler-shift it to the planet's radial velocity at each exposure,
building the time-series spectral matrix that is the primary input to
the data-reduction pipeline.

Functions
---------
spec_to_mat_fraction
    Full matrix builder with optional limb-asymmetry support.
get_stellar_matrix
    Build a matrix of Doppler-shifted stellar spectra.
add_throughput
    Multiply each exposure by a random throughput factor.
block_parameter
    Compute the BATMAN light-curve blocking factor.
dayside_fraction
    Fraction of the dayside visible at each orbital phase.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


# ---------------------------------------------------------------------------
# spec_to_mat_fraction, full matrix with limb asymmetry support
# ---------------------------------------------------------------------------

def spec_to_mat_fraction(inp_dat, syn_jd, T_0, v, wave, wave_prt,
                         spec, mat_stellar, with_signal, without_signal,
                         fraction,
                         spec_morning_day=None, spec_morning_night=None,
                         spec_evening_day=None, spec_evening_night=None,
                         sf_evening_day=None, sf_evening_night=None,
                         sf_morning_day=None, sf_morning_night=None,
                         injection_setup=False, include_star=True,
                         ccf_setup=False):
    """Build the full spectral time-series matrix.

    Supports homogeneous 1-D atmospheres and limb-asymmetric modes
    ``"gradual"``, ``"asymmetric"``, and ``"simplified_step"`` via
    ``inp_dat["Limb_divisions"]``.

    Parameters
    ----------
    inp_dat : dict
        Full simulation input dictionary.
    syn_jd : array
        Julian dates of all exposures.
    T_0 : float
        Mid-transit Julian date.
    v : array
        Planet RV at each exposure (km/s), shape (n_exp,).
    wave : array
        Instrument wavelength grid (µm), shape (n_pixels,).
    wave_prt : array
        pRT wavelength grid (µm).
    spec : array
        Atmospheric spectrum (homogeneous or average limb).
    mat_stellar : array
        Normalised stellar spectrum matrix, shape (n_exp, n_pixels).
    with_signal : array
        In-transit (or out-of-eclipse) exposure indices.
    without_signal : array
        Out-of-transit (or in-eclipse) exposure indices.
    fraction : array
        Per-exposure blocking factor, shape (n_exp,).
    spec_morning_day, spec_morning_night,
    spec_evening_day, spec_evening_night : array or None
        Per-limb atmospheric spectra (required for limb-asymmetry modes).
    sf_evening_day, sf_evening_night,
    sf_morning_day, sf_morning_night : array or None
        Per-exposure limb scaling factors (from :func:`get_sflimbs`).
    injection_setup : bool
        If True, use ``inp_dat["Inject_Scale_Factor"]`` instead of
        ``inp_dat["Scale_inj"]``.
    include_star : bool
        If True, multiply the planet model onto the stellar matrix.
    ccf_setup : bool
        If True, disable limb asymmetries regardless of ``inp_dat``.

    Returns
    -------
    mat : ndarray, shape (n_exp, n_pixels)
    mat_shift : ndarray, shape (n_exp, n_pixels)
    """
    from exoplore.atmosphere.prt import convolve
    try:
        from petitRADTRANS import physical_constants as cst
        c_kms = cst.c / 1e5
    except ImportError:
        c_kms = 2.998e5  # km/s fallback

    if ccf_setup:
        Limb_asymmetries = False
    elif not ccf_setup and inp_dat["Limb_asymmetries"]:
        Limb_asymmetries = True
    else:
        Limb_asymmetries = False

    scaling_factor = (inp_dat["Inject_Scale_Factor"]
                      if injection_setup else inp_dat["Scale_inj"])

    if inp_dat["event"] == 'transit':
        mat = np.zeros((len(v), len(wave)))
        mat_shift = np.zeros((len(v), len(wave)))

        if with_signal.shape[0] == 0:
            raise Exception('No spectra in-transit!!')

        for i in range(len(v)):
            if Limb_asymmetries:
                if inp_dat["Limb_divisions"] in (
                        "gradual", "asymmetric", "simplified_step"):
                    if i in with_signal:
                        wave_pl_e = wave_prt * (1.0 + v[i] / c_kms)
                        spec_pl_e = np.interp(wave_prt, wave_pl_e, spec_evening_day)
                        wave_pl_m = wave_prt * (1.0 + v[i] / c_kms)
                        spec_pl_m = np.interp(wave_prt, wave_pl_m, spec_morning_day)
                        spec_pl_shift = (sf_evening_day[i] * spec_pl_e
                                         + sf_morning_day[i] * spec_pl_m)
                        spec_pl_shift = convolve(wave_prt, spec_pl_shift, inp_dat["res"])

                        mat_shift[i, :] = np.interp(wave, wave_prt, spec_pl_shift)
                        if include_star:
                            mat[i, :] = mat_stellar[i, :] * (
                                1. - scaling_factor * mat_shift[i, :] * fraction[i])
                        else:
                            mat[i, :] = 1. - scaling_factor * mat_shift[i, :] * fraction[i]
                    else:
                        if include_star:
                            mat[i, :] = mat_stellar[i, :]
                        else:
                            mat[i, :] = mat_stellar[i, :] * 0. + 1.
                    continue

            else:
                if i in with_signal:
                    wave_pl = wave_prt * (1.0 + v[i] / c_kms)
                    spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                    mat_shift[i, :] = np.interp(wave, wave_prt, spec_pl_shift)
                    if include_star:
                        mat[i, :] = mat_stellar[i, :] * (
                            1. - scaling_factor * mat_shift[i, :] * fraction[i])
                    else:
                        mat[i, :] = 1. - scaling_factor * mat_shift[i, :] * fraction[i]
                else:
                    if include_star:
                        mat[i, :] = mat_stellar[i, :]
                    else:
                        mat[i, :] = mat_stellar[i, :] * 0. + 1.

    elif inp_dat["event"] == 'dayside':
        mat = np.empty_like(mat_stellar)
        mat_shift = np.empty_like(mat_stellar)

        if without_signal.shape[0] == 0:
            mat[:, :] = 1.
        else:
            for i in range(len(v)):
                if i in with_signal:
                    wave_pl = wave_prt * (1.0 + v[i] / c_kms)
                    spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                    mat_shift[i, :] = np.interp(wave, wave_prt, spec_pl_shift)
                    mat[i, :] = (1. + scaling_factor * fraction[i]
                                 * mat_shift[i, :] / mat_stellar[i, :])
                else:
                    mat[i, :] = 1.

    return mat, mat_shift


# ---------------------------------------------------------------------------
# get_stellar_matrix
# ---------------------------------------------------------------------------

def get_stellar_matrix(spec_star, v_star, wave):
    """Build a matrix of Doppler-shifted stellar spectra.

    Parameters
    ----------
    spec_star : array
        Normalised stellar spectrum on ``wave``.
    v_star : array
        Stellar RV at each exposure in km/s, shape (n_exp,).
    wave : array
        Wavelength grid in µm, shape (n_pixels,).

    Returns
    -------
    ndarray, shape (n_exp, n_pixels)
    """
    try:
        from petitRADTRANS import physical_constants as cst
        c_kms = cst.c / 1e5
    except ImportError:
        c_kms = 2.998e5

    mat_star = np.zeros((len(v_star), len(wave)), dtype=float)
    for i, v in enumerate(v_star):
        wave_shift = wave * (1.0 + v / c_kms)
        mat_star[i] = np.interp(wave, wave_shift, spec_star)
    return mat_star


# ---------------------------------------------------------------------------
# add_throughput
# ---------------------------------------------------------------------------

def add_throughput(F, jitter_frac=0.02, mode="white",
                   red_smooth_sigma=2.0, seed=None):
    """Multiply each exposure by a random throughput factor.

    Parameters
    ----------
    F : ndarray, shape (n_spectra, n_pixels)
        Input flux matrix.
    jitter_frac : float
        1-sigma fractional jitter (e.g. 0.02 = 2 %).
    mode : str
        ``"white"`` for uncorrelated draws; ``"red"`` for temporally
        correlated (Gaussian-smoothed) draws.
    red_smooth_sigma : float
        Gaussian sigma in exposures for ``mode="red"``.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n_spectra, n_pixels)
        Throughput-modulated flux.
    """
    rng = np.random.default_rng(seed)
    n_spectra = F.shape[0]
    raw = rng.normal(loc=1.0, scale=jitter_frac, size=n_spectra)
    if mode == "white":
        T = raw
    elif mode == "red":
        T = scipy.ndimage.gaussian_filter1d(raw, sigma=red_smooth_sigma)
    else:
        raise ValueError("mode must be 'white' or 'red'")
    return F * T[:, None]


# ---------------------------------------------------------------------------
# block_parameter, BATMAN transit light curve blocking factor
# ---------------------------------------------------------------------------

def block_parameter(JD, T_0, P, R_P, a, R_s, i, uu,
                    e=0, omega=90, limb_dark_mode='quadratic'):
    """Compute the per-exposure transit blocking factor via BATMAN.

    Parameters
    ----------
    JD : array
        Julian dates.
    T_0 : float
        Time of inferior conjunction.
    P : float
        Orbital period (days).
    R_P : float
        Planet radius (same units as R_s).
    a : float
        Semi-major axis (same units as R_s).
    R_s : float
        Stellar radius.
    i : float
        Inclination (degrees).
    uu : tuple
        Limb-darkening coefficients.
    e : float
        Eccentricity (default 0).
    omega : float
        Longitude of periastron in degrees (default 90).
    limb_dark_mode : str
        BATMAN limb-darkening model (default ``"quadratic"``).

    Returns
    -------
    ndarray
        Normalised blocking factor per exposure (0 outside transit,
        peaks at 1 at mid-transit).
    """
    try:
        import batman
    except ImportError as exc:
        raise ImportError(
            "batman-package is required for block_parameter().\n"
            "Install with:  pip install batman-package"
        ) from exc

    params = batman.TransitParams()
    params.t0 = T_0
    params.per = P
    params.rp = R_P / R_s
    params.a = a / R_s
    params.inc = i
    params.ecc = e
    params.w = omega
    params.u = uu
    params.limb_dark = limb_dark_mode

    m = batman.TransitModel(params, JD)
    flux = m.light_curve(params)
    block = -(flux - 1)
    block /= block.max()
    return block


# ---------------------------------------------------------------------------
# dayside_fraction
# ---------------------------------------------------------------------------

def dayside_fraction(syn_jd, without_signal):
    """Fraction of the dayside visible at each orbital phase.

    Parameters
    ----------
    syn_jd : array
        Julian date array for all exposures.
    without_signal : array
        Indices of the in-eclipse (no dayside) exposures.

    Returns
    -------
    ndarray
        Fraction array: ramps up before eclipse, zero during, ramps down
        after.
    """
    fraction = np.empty_like(syn_jd)
    fraction[0:without_signal[0]] = np.linspace(0.5, 1, without_signal[0])
    fraction[without_signal] = 0
    fraction[without_signal[-1] + 1:] = np.linspace(
        1, 0.65, len(syn_jd) - without_signal[-1] - 1
    )
    return fraction
