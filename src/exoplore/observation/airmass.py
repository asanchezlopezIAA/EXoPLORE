"""
exoplore.observation.airmass
=============================

Synthetic airmass curves for ETC-based (non-real-data) simulations.

Parabolic airmass curves are generated when real airmass
measurements are not available.

Three observing patterns are supported:

``"up_and_down"``
    Star rises, transits the meridian, then sets.  Airmass is lowest
    at mid-observation and highest at the start/end.

``"up"``
    Star is rising throughout the observation (airmass decreasing).

``"down"``
    Star is setting throughout the observation (airmass increasing).
"""

from __future__ import annotations

import random

import numpy as np


def synthetic_airmass(
    julian_dates: np.ndarray,
    airmass_min: float,
    airmass_max: float,
    pattern: str = "up_and_down",
) -> np.ndarray:
    """Generate a synthetic parabolic airmass curve.

    Parameters
    ----------
    julian_dates:
        Array of mid-exposure Julian Dates, shape ``(n,)``.
    airmass_min:
        Minimum airmass (at the star's meridian crossing).
    airmass_max:
        Maximum airmass (at the start/end of the observation
        window).
    pattern:
        One of ``"up_and_down"``, ``"up"``, or ``"down"``.

    Returns
    -------
    np.ndarray
        Airmass at each exposure, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``pattern`` is not one of the three allowed values, or if
        ``airmass_min > airmass_max``.
    """
    if airmass_min > airmass_max:
        raise ValueError(
            f"airmass_min ({airmass_min}) must be <= airmass_max ({airmass_max})."
        )
    allowed = {"up_and_down", "up", "down"}
    if pattern not in allowed:
        raise ValueError(f"pattern must be one of {allowed}; got {pattern!r}.")

    jd = np.asarray(julian_dates, dtype=float)
    jd_min, jd_max = jd.min(), jd.max()
    span = jd_max - jd_min

    if span == 0.0:
        return np.full_like(jd, airmass_min)

    amin, amax = airmass_min, airmass_max

    if pattern == "up_and_down":
        jd_mid = 0.5 * (jd_min + jd_max)
        half_span = 0.5 * span
        a_coeff = (amax - amin) / half_span ** 2
        airmass = a_coeff * (jd - jd_mid) ** 2 + amin

    elif pattern == "up":
        # Airmass decreases from amax to amin (star rising)
        x = jd - jd_min
        a_coeff = (amin - amax) / span ** 2
        airmass = a_coeff * x ** 2 + amax

    else:  # "down"
        # Airmass increases from amin to amax (star setting)
        x = jd - jd_min
        a_coeff = (amax - amin) / span ** 2
        airmass = a_coeff * x ** 2 + amin

    return airmass


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def get_airmass(airmass_evol, julian_date, airmass_limits):
    """Calculate a synthetic airmass curve.

    Parameters
    ----------
    airmass_evol : str
        One of ``"up_and_down"``, ``"up"``, or ``"down"``.
    julian_date : numpy.ndarray
        Array of Julian Dates.
    airmass_limits : list or tuple [amin, amax]
        Minimum and maximum airmass.

    Returns
    -------
    numpy.ndarray
        Airmass at each JD.
    """
    amin, amax = airmass_limits

    jd_min = np.min(julian_date)
    jd_max = np.max(julian_date)
    span_full = jd_max - jd_min

    if span_full == 0:
        airmass = np.full_like(julian_date, amin)
    else:
        if airmass_evol == "up_and_down":
            jd_mid = 0.5 * (jd_min + jd_max)
            x = julian_date - jd_mid
            half_span = 0.5 * span_full
            a = (amax - amin) / (half_span ** 2)
            c = amin
            airmass = a * (x ** 2) + c

        elif airmass_evol == "up":
            x_half = julian_date - jd_min
            a = (amin - amax) / (span_full ** 2)
            c = amax
            airmass = a * (x_half ** 2) + c

        elif airmass_evol == "down":
            x_half = julian_date - jd_min
            a = (amax - amin) / (span_full ** 2)
            c = amin
            airmass = a * (x_half ** 2) + c

        else:
            raise ValueError(
                "`airmass_evol` must be 'up_and_down', 'up', or 'down'."
            )

    return airmass


def skycalc_model(path, julian_date, airmass_og, airmass_limits,
                  wvl_boundaries, airmass_evol, PWV, observatory):
    """Write Skycalc CLI input files and a launcher script.

    Creates one input text file per exposure plus an executable script
    ``run_skycalc_cli.txt`` that the user must run from their terminal.

    Parameters
    ----------
    path : str
        Directory where Skycalc files will be written.
    julian_date : array
        Julian dates of the exposures.
    airmass_og : array or None
        Pre-computed airmass values; if None, computed from
        ``airmass_evol`` / ``airmass_limits``.
    airmass_limits : tuple [amin, amax]
        Min/max airmass (used when ``airmass_og`` is None).
    wvl_boundaries : tuple [wmin, wmax]
        Wavelength range in nm.
    airmass_evol : str
        Airmass evolution pattern (passed to :func:`get_airmass`).
    PWV : array
        Precipitable water vapour per exposure (mm).
    observatory : str
        Observatory code understood by the Skycalc CLI.

    Returns
    -------
    None
    """
    import os

    if airmass_og is None:
        ref = airmass_limits[0] == airmass_limits[1]
        airmass = ([airmass_limits[0]] if ref
                   else get_airmass(airmass_evol, julian_date, airmass_limits, path))
        flag = '_ref' if ref else ''
    else:
        airmass, ref, flag = (airmass_og, False, '')

    # Open the exec file
    g = open(path + '/run_skycalc_cli' + flag + '.txt', 'w')

    for n in range(len(airmass)):
        if not ref:
            filename = path + '/tell_' + str(n) + '.txt'
        else:
            filename = path + '/tell_ref_airmass_' + str(airmass_limits[0]) + '.txt'

        with open(filename, 'w') as f:
            f.write('airmass         :  ' + str(np.round(airmass[n], 1)) + '\n')
            f.write('pwv_mode        :  pwv' + '\n')
            f.write('season          :  ' + str(0) + '\n')
            f.write('time            :  ' + str(0) + '\n')
            f.write('pwv             :  ' + str(PWV[n]) + '\n')
            f.write('msolflux        :  ' + str(130.0) + '\n')
            f.write('incl_moon       :  N' + '\n')
            f.write('moon_sun_sep    :  ' + str(90.0) + '\n')
            f.write('moon_target_sep :  ' + str(45.0) + '\n')
            f.write('moon_alt        :  ' + str(45.0) + '\n')
            f.write('moon_earth_dist :  ' + str(1.0) + '\n')
            f.write('incl_starlight  :  N' + '\n')
            f.write('incl_zodiacal   :  N' + '\n')
            f.write('ecl_lon         :  ' + str(135.0) + '\n')
            f.write('ecl_lat         :  ' + str(90.0) + '\n')
            f.write('incl_loweratm   :  Y' + '\n')
            f.write('incl_upperatm   :  Y' + '\n')
            f.write('incl_airglow    :  Y' + '\n')
            f.write('incl_therm      :  N' + '\n')
            f.write('therm_t1        :  ' + str(0.0) + '\n')
            f.write('therm_e1        :  ' + str(0.0) + '\n')
            f.write('therm_t2        :  ' + str(0.0) + '\n')
            f.write('therm_e2        :  ' + str(0.0) + '\n')
            f.write('therm_t3        :  ' + str(0.0) + '\n')
            f.write('therm_e3        :  ' + str(0.0) + '\n')
            f.write('vacair          :  vac' + '\n')
            f.write('wmin            :  ' + str(wvl_boundaries[0]) + '\n')
            f.write('wmax            :  ' + str(wvl_boundaries[-1]) + '\n')
            f.write('wgrid_mode      :  fixed_spectral_resolution' + '\n')
            f.write('wdelta          :  ' + str(0.01) + '\n')
            f.write('wres            :  ' + str(150000.) + '\n')
            f.write('lsf_type        :  none' + '\n')
            f.write('lsf_gauss_fwhm  :  ' + str(5.0) + '\n')
            f.write('lsf_boxcar_fwhm :  ' + str(5.0) + '\n')
            f.write('observatory     : ' + observatory + '\n')

        if not ref:
            g.write('~/.local/bin/skycalc_cli -i ' + filename + ' -o '
                    + path + '/tell_spec_' + str(n) + flag + '.fits' + '\n')
        else:
            g.write('~/.local/bin/skycalc_cli -i ' + filename + ' -o '
                    + path + '/tell_ref_airmass_'
                    + str(float(airmass[0])) + '.fits' + '\n')
    g.close()

    os.system('chmod u+x ' + path + '/run_skycalc_cli' + flag + '.txt')
    print('YOU WILL NEED TO RUN ON YOUR CONSOLE: ./run_skycalc_cli' + flag + '.txt')


def pwv_gen_skycalc(n_spectra, ref_pwv=None):
    """Generate per-exposure PWV values for Skycalc telluric simulations.

    Randomly samples precipitable water vapour (PWV) values from a
    discrete set that matches the Skycalc CLI's accepted grid.  Each
    exposure is assigned a value that is the same as, one step above, or
    one step below the reference PWV, mimicking realistic night-to-night
    PWV variation within a narrow range.

    Parameters
    ----------
    n_spectra : int
        Number of spectra (exposures) to generate PWV values for.
    ref_pwv : float or None
        Reference PWV value (mm) taken from the Skycalc grid::

            [0.05, 0.01, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5,
             10.0, 20.0, 30.0]

        If ``None``, a value is chosen randomly from the interior of
        the grid (excluding the first and last entries).

    Returns
    -------
    np.ndarray
        Array of PWV values (mm), shape ``(n_spectra,)``, dtype float64.
    """
    pwv_set = [0.05, 0.01, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5,
               10.0, 20.0, 30.0]

    if ref_pwv is None:
        ref_index = random.randint(1, len(pwv_set) - 2)
        ref_pwv = pwv_set[ref_index]
    else:
        ref_index = pwv_set.index(ref_pwv)

    pwv_values = []
    for _ in range(n_spectra):
        neighbor_index = random.randint(0, 1)
        if neighbor_index == 0:
            neighbor_index = ref_index - 1
        elif neighbor_index == 2:
            neighbor_index = ref_index + 1
        else:
            neighbor_index = ref_index
        pwv_values.append(pwv_set[neighbor_index])

    return np.asarray(pwv_values, dtype=np.float64)


def PWV_handling(constant_pwv, pwv_value, n_spectra, file_path):
    """Handle PWV values for a simulated observation and save to FITS.

    Creates a per-exposure PWV array, either constant or
    randomly varying, and writes it to a FITS file at ``file_path``
    if the file does not yet exist.

    Parameters
    ----------
    constant_pwv : bool
        If ``True``, use the same PWV value for every exposure.
        If ``False``, call :func:`pwv_gen_skycalc` to draw a
        randomly varying sequence.
    pwv_value : float or None
        PWV value (mm) to use when ``constant_pwv=True``.
        If ``None`` **and** ``constant_pwv=True``, prompts the user
        for a value interactively.
        When ``constant_pwv=False`` this is passed as ``ref_pwv`` to
        :func:`pwv_gen_skycalc` (may be ``None`` for a random choice).
    n_spectra : int
        Number of exposures.
    file_path : str
        Path where the PWV array will be written as a FITS
        ``PrimaryHDU``.  If the file already exists it is left
        unchanged and the array is still returned.

    Returns
    -------
    pwv_values : np.ndarray
        PWV per exposure (mm), shape ``(n_spectra,)``.
    """
    from astropy.io import fits
    import os

    if constant_pwv:
        if pwv_value is None:
            pwv = input("Enter the value for PWV: ")
            assert pwv, "PWV value cannot be empty."
            pwv_value = float(pwv)
        else:
            assert isinstance(pwv_value, (float, int)), (
                "PWV value should be a number within the Skycalc accepted values."
            )
        pwv_values = pwv_value * np.ones(n_spectra, float)
    else:
        pwv_values = pwv_gen_skycalc(n_spectra, pwv_value)

    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        hdu = fits.PrimaryHDU(pwv_values)
        hdu.writeto(file_path, overwrite=False)

    return pwv_values
