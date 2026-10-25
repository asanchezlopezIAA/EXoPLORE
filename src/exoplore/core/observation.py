"""
exoplore.core.observation
=========================

Build planetary spectral matrices for the observing sequence.

These functions take the atmospheric spectrum computed by pRT and
Doppler-shift it to the planet's radial velocity at each exposure,
building the time-series spectral matrix that is the primary input to
the data-reduction pipeline.

Function
--------
spec_to_mat_fraction
    Build the spectral time series with optional limb asymmetries.
"""

from __future__ import annotations

# Role in the calculation
# Construct spectral time-series matrices from velocity tracks, event fractions and
# model spectra. The simulator and preparation pipeline call spec_to_mat_fraction here.


import numpy as np


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
        If True, read the injection amplitude from
        ``inp_dat["Inject_Scale_Factor"]``.
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
