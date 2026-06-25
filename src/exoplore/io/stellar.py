"""
exoplore.io.stellar
===================

Loaders for stellar model spectra (PHOENIX and related).
"""

from __future__ import annotations

import numpy as np


def LoadPhoenix(file, wave, res):
    """Load and convolve a PHOENIX stellar spectrum onto a wavelength grid.

    Loads a PHOENIX model from two FITS files (wavelength + flux),
    convolves it to the instrument resolution, and interpolates it
    onto the supplied wavelength grid.

    Parameters
    ----------
    file : list of str or None
        ``[wave_fits_path, flux_fits_path]``.  Phoenix wavelengths are
        expected in Angstrom; they are converted to microns internally.
        Pass ``None`` to raise an exception (no stellar model available).
    wave : ndarray
        Instrument wavelength grid in microns.  May be 1-D (n_pixels,)
        or 2-D (n_orders, n_pixels).  If the first axis is larger than
        the second the array is transposed automatically.
    res : float
        Instrument resolving power R = λ / Δλ.

    Returns
    -------
    ndarray
        Normalised, convolved, interpolated stellar spectrum.
        Shape matches ``wave``: (n_orders, n_pixels) for 2-D input or
        (n_pixels,) for 1-D input.

    Raises
    ------
    Exception
        If ``file`` is None.
    """
    from astropy.io import fits
    from exoplore.atmosphere.prt import convolve

    if file is None:
        raise Exception("No stellar model")

    spec_og = fits.open(file[1])[0].data
    wave_og = 1.0e-4 * fits.open(file[0])[0].data   # Å → µm

    if wave.shape[0] > wave.shape[1]:
        wave = wave.T

    if len(wave.shape) == 2:
        n_orders_local = wave.shape[0]
        spec_star_phoenix = np.zeros_like(wave)

        for j in range(n_orders_local):
            spec_star_conv = convolve(wave_og, spec_og, res)
            spec_star_phoenix[j, :] = np.interp(
                wave[j, :], wave_og, spec_star_conv
            )

            # Normalise: fit a 2nd-order polynomial to the 80 brightest
            # interval maxima and divide
            n_brightest = 80
            interval_size = spec_star_phoenix.shape[1] // n_brightest
            mean_wavelengths = np.array([
                np.mean(wave[j, i * interval_size:(i + 1) * interval_size])
                for i in range(n_brightest)
            ])
            max_values = [
                np.max(spec_star_phoenix[j, i * interval_size:(i + 1) * interval_size])
                for i in range(n_brightest)
            ]
            c1 = np.polyfit(mean_wavelengths, max_values, deg=2)
            fit = np.polyval(c1, wave[j, :])
            spec_star_phoenix[j, :] /= fit

        return spec_star_phoenix

    else:   # 1-D
        return np.interp(wave, wave_og, convolve(wave_og, spec_og, res))


def spec_to_mat_fraction(inp_dat, syn_jd, T_0, v, wave, wave_prt,
                         spec, mat_stellar, with_signal, without_signal,
                         fraction, spec_morning_day=None, spec_morning_night=None,
                         spec_evening_day=None, spec_evening_night=None,
                         sf_evening_day=None, sf_evening_night=None,
                         sf_morning_day=None, sf_morning_night=None,
                         injection_setup=False, include_star=True,
                         ccf_setup=False):
    """Thin shim, canonical implementation in :mod:`exoplore.core.observation`.

    This alias keeps ``exoplore.io.stellar.spec_to_mat_fraction`` importable
    for any code that references it here; the single source of truth is
    :func:`exoplore.core.observation.spec_to_mat_fraction`.
    """
    from exoplore.core.observation import spec_to_mat_fraction as _impl
    return _impl(
        inp_dat, syn_jd, T_0, v, wave, wave_prt,
        spec, mat_stellar, with_signal, without_signal, fraction,
        spec_morning_day=spec_morning_day,
        spec_morning_night=spec_morning_night,
        spec_evening_day=spec_evening_day,
        spec_evening_night=spec_evening_night,
        sf_evening_day=sf_evening_day,
        sf_evening_night=sf_evening_night,
        sf_morning_day=sf_morning_day,
        sf_morning_night=sf_morning_night,
        injection_setup=injection_setup,
        include_star=include_star,
        ccf_setup=ccf_setup,
    )


def get_stellar_matrix(spec_star, v_star, wave):
    """Create a matrix of Doppler-shifted stellar spectra.

    For each velocity in *v_star* the stellar spectrum is shifted by
    interpolating onto the Doppler-shifted wavelength grid, producing
    one row of the output matrix per exposure.

    Parameters
    ----------
    spec_star : numpy.ndarray, shape (n_pixels,)
        Stellar spectrum flux array.
    v_star : numpy.ndarray, shape (n_spectra,)
        Radial velocity shifts in km/s (one per exposure).
    wave : numpy.ndarray, shape (n_pixels,)
        Wavelength grid (any consistent unit).

    Returns
    -------
    mat_star : numpy.ndarray, shape (n_spectra, n_pixels)
        Matrix of Doppler-shifted stellar spectra.
    """
    c = 2.998e5  # speed of light in km/s
    mat_star = np.zeros((len(v_star), len(wave)), dtype=float)
    for i, v in enumerate(v_star):
        wave_shift = wave * (1.0 + v / c)
        mat_star[i] = np.interp(wave, wave_shift, spec_star)
    return mat_star
