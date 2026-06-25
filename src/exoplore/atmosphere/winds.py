"""
exoplore.atmosphere.winds
=========================
Wind broadening, rotational broadening, and related kernels for limb-resolved
atmospheric spectra.

Functions
---------
planet_rot_vel
rotation_angle_during_transit
atmospheric_scale_height
create_velocity_grid
rotation_kernel_Maguire24
wind_broadening_triangular_kernel
convolve_spectrum_with_kernel
get_sflimbs
"""

import numpy as np
from scipy.interpolate import interp1d
from petitRADTRANS import physical_constants as cst


# ---------------------------------------------------------------------------
# Planet kinematics helpers
# ---------------------------------------------------------------------------

def planet_rot_vel(inp_dat):
    """Equatorial rotation velocity of a tidally-locked exoplanet.

    Parameters
    ----------
    inp_dat : dict  Needs ``"Period"`` (days) and ``"R_pl"`` (cm).

    Returns
    -------
    float  Equatorial rotation velocity in km/s.
    """
    seconds_per_day = 86400
    P_seconds = inp_dat["Period"] * seconds_per_day
    R_p_km = inp_dat["R_pl"] * 1e-5  # cm → km
    v_rot = (2 * np.pi * R_p_km) / P_seconds
    return v_rot


def rotation_angle_during_transit(inp_dat):
    """Rotation angle of the planet during its transit.

    Returns
    -------
    float  Rotation angle in degrees.
    """
    return 360.0 * (inp_dat["T_duration"] / inp_dat["Period"])


def atmospheric_scale_height(T, mu_amu, g):
    """Atmospheric scale height.

    Parameters
    ----------
    T : float  Temperature in Kelvin.
    mu_amu : float  Mean molecular weight in amu.
    g : float  Surface gravity in m/s².

    Returns
    -------
    float  Scale height in km.
    """
    k_B = 1.380649e-23          # Boltzmann constant, J/K
    amu_to_kg = 1.66053906660e-27  # 1 amu in kg
    mu = mu_amu * amu_to_kg
    H = k_B * T / (mu * g)
    return H * 1e-3  # m → km


# ---------------------------------------------------------------------------
# Velocity grid helper
# ---------------------------------------------------------------------------

def create_velocity_grid(wavelength, v_min, v_max, points_per_increment=3):
    """Create a velocity grid well-sampled relative to the wavelength spacing.

    Parameters
    ----------
    wavelength : array-like  Wavelength array (μm).
    v_min, v_max : float  Grid limits (km/s).
    points_per_increment : int  Points per smallest wavelength step (default 3).

    Returns
    -------
    numpy.ndarray  Symmetric velocity grid in km/s.
    """
    c = cst.c * 1e-5  # cm/s → km/s
    delta_lambda_min = np.diff(wavelength).min()
    velocity_step = (delta_lambda_min / np.mean(wavelength)) * c / points_per_increment
    num_points = int(np.ceil((v_max - v_min) / velocity_step)) + 1
    delta_v = np.linspace(v_min, v_max, num_points)
    return delta_v


# ---------------------------------------------------------------------------
# Rotation broadening kernel (Maguire et al. 2024)
# ---------------------------------------------------------------------------

def rotation_kernel_Maguire24(
    delta_v_rot, r1, d, wave, max_delta_v=100, mode='full'
):
    """Rotation broadening kernel for a rotating annulus (Maguire et al. 2024).

    Parameters
    ----------
    delta_v_rot : float  Doppler shift due to equatorial rotation (km/s).
    r1 : float  Inner radius of the annulus (km).
    d : float  Fractional thickness of the annulus (d = (r2-r1)/r1).
    wave : array-like  Wavelength array used to build the velocity grid.
    max_delta_v : float  Half-width of the velocity grid (km/s, default 100).
    mode : str  ``"morning"``, ``"evening"``, or ``"full"``.

    Returns
    -------
    For ``mode="morning"`` or ``mode="evening"``:
        ``(kernel, delta_v)``
    For ``mode="full"``:
        ``(kernel_morning, kernel_evening, kernel_total, delta_v)``
    """
    delta_v = create_velocity_grid(wave, -max_delta_v, max_delta_v, 3)

    r2 = r1 + d * r1
    x = r2 * delta_v / delta_v_rot
    a = np.sqrt((1 + d)**2 - (delta_v / delta_v_rot)**2)
    kernel = np.zeros_like(x)

    mask_outer = (np.abs(x) >= r1) & (np.abs(x) < r2)
    mask_inner = np.abs(x) < r1

    kernel[mask_outer] = a[mask_outer] / d
    kernel[mask_inner] = (
        a[mask_inner] - np.sqrt(1 - (delta_v[mask_inner] / delta_v_rot)**2)
    ) / d

    kernel_morning = np.zeros_like(x)
    kernel_evening = np.zeros_like(x)
    kernel_morning[x >= 0] = kernel[x >= 0]
    kernel_evening[x < 0] = kernel[x < 0]

    dv = np.diff(delta_v)[0]

    if mode == 'morning':
        kernel_morning /= np.sum(kernel_morning * dv)
        return kernel_morning, delta_v
    elif mode == 'evening':
        kernel_evening /= np.sum(kernel_evening * dv)
        return kernel_evening, delta_v
    elif mode == 'full':
        kernel_morning /= np.sum(kernel_morning * dv) * 2.0
        kernel_evening /= np.sum(kernel_evening * dv) * 2.0
        kernel_total = kernel_morning + kernel_evening
        return kernel_morning, kernel_evening, kernel_total, delta_v


# ---------------------------------------------------------------------------
# Wind broadening kernel
# ---------------------------------------------------------------------------

def wind_broadening_triangular_kernel(
    v_sys, v_wind, wave, max_delta_v=100, center="zero"
):
    """Normalized triangular wind broadening kernel.

    Parameters
    ----------
    v_sys : float  Systemic velocity (km/s).
    v_wind : float  FWHM of the wind velocity (km/s).
    wave : array-like  Wavelength array used to build the velocity grid.
    max_delta_v : float  Half-width of the velocity grid (km/s, default 100).
    center : str  ``"zero"`` or ``"V_sys"``.

    Returns
    -------
    kernel : numpy.ndarray
    delta_v : numpy.ndarray
    """
    delta_v = create_velocity_grid(wave, -max_delta_v, max_delta_v, 3)
    half_max_width = np.abs(v_wind) / 2

    if center == "V_sys":
        kernel = np.maximum(1 - np.abs(delta_v - v_sys) / half_max_width, 0)
    elif center == "zero":
        kernel = np.maximum(1 - np.abs(delta_v) / half_max_width, 0)

    kernel /= np.sum(kernel * np.diff(delta_v)[0])
    return kernel, delta_v


# ---------------------------------------------------------------------------
# Spectrum-kernel convolution
# ---------------------------------------------------------------------------

def convolve_spectrum_with_kernel(
    wave, spec, kernel, delta_v, mode='nearest', cval=0.0
):
    """Convolve a synthetic spectrum with an arbitrary kernel.

    Parameters
    ----------
    wave : array-like  Wavelength grid of the synthetic spectrum (μm).
    spec : array-like  Spectrum flux array.
    kernel : array-like  Kernel array (same length as *delta_v*).
    delta_v : array-like  Velocity grid for the kernel (km/s).
    mode : str  Edge-handling mode: ``"nearest"``, ``"reflect"``,
                ``"constant"``, or ``"wrap"``.
    cval : float  Fill value for ``mode="constant"``.

    Returns
    -------
    numpy.ndarray  Convolved spectrum on the original wavelength grid.
    """
    convolved_spec = np.zeros_like(spec)
    kernel = kernel[::-1]  # reverse for correct cross-correlation convention

    c_kms = cst.c * 1e-5  # cm/s → km/s

    for i in range(len(wave)):
        wave_shifted = wave[i] * np.sqrt(
            (1 + delta_v / c_kms) / (1 - delta_v / c_kms)
        )
        kernel_interpolator = interp1d(
            wave_shifted, kernel, bounds_error=False, fill_value=0
        )
        resampled_kernel = kernel_interpolator(wave)

        if mode == 'nearest':
            left_pad = np.full(i, spec[0])
            right_pad = np.full(len(wave) - i - 1, spec[-1])
        elif mode == 'reflect':
            left_pad = spec[1:i+1][::-1] if i > 0 else np.array([])
            right_pad = (
                spec[-2:len(wave)-i-1][::-1]
                if len(wave) - i - 1 > 0 else np.array([])
            )
        elif mode == 'constant':
            left_pad = np.full(i, cval)
            right_pad = np.full(len(wave) - i - 1, cval)
        elif mode == 'wrap':
            left_pad = spec[-i:] if i > 0 else np.array([])
            right_pad = spec[:len(wave) - i - 1]
        else:
            raise ValueError(
                "Invalid mode. Choose from 'nearest', 'reflect', 'constant', or 'wrap'."
            )

        extended_spec = np.concatenate((left_pad, spec, right_pad))

        resampled_kernel_sum = np.sum(resampled_kernel)
        if resampled_kernel_sum != 0:
            resampled_kernel /= resampled_kernel_sum
        else:
            resampled_kernel = np.zeros_like(resampled_kernel)

        convolved_spec[i] = np.sum(extended_spec[i:i+len(spec)] * resampled_kernel)

    return convolved_spec


# ---------------------------------------------------------------------------
# Limb scaling factors
# ---------------------------------------------------------------------------

def find_nearest(array, value):
    """Return the array element nearest to *value*.
    """
    idx = (np.abs(np.asarray(array) - value)).argmin()
    return array[idx]


def get_sflimbs(inp_dat, with_signal, without_signal, phase, syn_jd, mode="gradual"):
    """Compute per-exposure limb scaling factors for morning and evening limbs.

    Parameters
    ----------
    inp_dat : dict  Simulation input dictionary.
    with_signal : array-like  Indices of in-transit exposures.
    without_signal : array-like  Indices of out-of-transit exposures.
    phase : array-like  Orbital phase array.
    syn_jd : array-like  Julian date array.
    mode : str
        ``"gradual"`` (default): smooth cubic transition from morning-dominated
        ingress to evening-dominated egress; equal mix (0.5/0.5) at full transit.
        ``"asymmetric"``: as above but morning dominates the first quarter of
        full transit and the transition is completed by mid-transit, calibrated
        for ultra-hot Jupiters (e.g. WASP-76 b) with extreme day-night asymmetry.
        ``"simplified_step"``: hard step-function, pure morning during ingress,
        equal mix (0.5/0.5) during full transit, pure evening during egress.
        No smooth transition.

    Returns
    -------
    tuple  ``(sf_morning, sf_morning_night, sf_evening, sf_evening_night,
               ingress_idx, egress_idx)``
    The ``*_night`` arrays are always ``None`` (two-limb modes only).
    """
    b = inp_dat['a'] * np.cos(inp_dat['incl'])
    den = inp_dat['a'] * np.sin(inp_dat['incl'])
    num_full = np.sqrt((1e-5*inp_dat['R_star'] - 1e-5*inp_dat['R_pl'])**2 - b**2)
    T_full = (inp_dat['Period'] / np.pi) * np.arcsin(num_full / den)
    T_ingress = (inp_dat['T_duration'] - T_full) / 2.0

    ingress_end_idx = np.where(
        syn_jd == find_nearest(syn_jd, syn_jd[with_signal[0]] + T_ingress)
    )[0][0]
    egress_start_idx = np.where(
        syn_jd == find_nearest(syn_jd, syn_jd[with_signal[-1]] - T_ingress)
    )[0][0]

    ingress_idx = np.arange(with_signal[0], ingress_end_idx + 1, 1)
    egress_idx = np.arange(egress_start_idx, with_signal[-1] + 1, 1)
    full_transit_idx = np.arange(ingress_end_idx, egress_start_idx, 1)

    if mode == "gradual":
        sf_morning = np.zeros_like(syn_jd, dtype=float)
        sf_evening = np.zeros_like(syn_jd, dtype=float)

        half_idx = len(ingress_idx) // 2
        first_half = ingress_idx[:half_idx+1]
        second_half = ingress_idx[half_idx:]

        sf_morning[first_half] = 1.0
        sf_evening[first_half] = 0.0

        transition = np.linspace(0, 1, len(second_half))
        sf_morning[second_half] = 1.0 - 0.5 * transition
        sf_evening[second_half] = 0.5 * transition

        sf_morning[full_transit_idx] = 0.5
        sf_evening[full_transit_idx] = 0.5

        half_idx = len(egress_idx) // 2
        first_half = egress_idx[:half_idx+1]
        second_half = egress_idx[half_idx:]

        transition = np.linspace(0, 1, len(first_half))
        sf_morning[first_half] = 0.5 * (1.0 - transition)
        sf_evening[first_half] = 0.5 + 0.5 * transition

        sf_morning[second_half] = 0.0
        sf_evening[second_half] = 1.0

        sf_morning[without_signal] = 0
        sf_evening[without_signal] = 0

        norm = sf_morning + sf_evening
        for ii in with_signal:
            if norm[ii] != 0:
                sf_morning[ii] /= norm[ii]
                sf_evening[ii] /= norm[ii]

        return sf_morning, None, sf_evening, None, ingress_idx, egress_idx

    elif mode == "asymmetric":
        sf_morning = np.zeros_like(syn_jd, dtype=float)
        sf_evening = np.zeros_like(syn_jd, dtype=float)

        def smoothstep(x):
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        sf_morning[ingress_idx] = 1.0
        sf_evening[ingress_idx] = 0.0

        n_full = len(full_transit_idx)
        if n_full > 0:
            i_start = n_full // 4
            i_end = max(n_full // 2, i_start + 1)

            first_part = full_transit_idx[:i_start]
            transition_part = full_transit_idx[i_start:i_end]
            last_part = full_transit_idx[i_end:]

            sf_morning[first_part] = 1.0
            sf_evening[first_part] = 0.0

            if len(transition_part) > 0:
                x = np.linspace(0.0, 1.0, len(transition_part))
                w = smoothstep(x)
                sf_evening[transition_part] = w
                sf_morning[transition_part] = 1.0 - w

            sf_morning[last_part] = 0.0
            sf_evening[last_part] = 1.0

        sf_morning[egress_idx] = 0.0
        sf_evening[egress_idx] = 1.0

        sf_morning[without_signal] = 0.0
        sf_evening[without_signal] = 0.0

        norm = sf_morning + sf_evening
        for ii in with_signal:
            if norm[ii] != 0:
                sf_morning[ii] /= norm[ii]
                sf_evening[ii] /= norm[ii]

        return sf_morning, None, sf_evening, None, ingress_idx, egress_idx

    elif mode == "simplified_step":
        # Hard step-function: pure morning during ingress, equal mix during
        # full transit, pure evening during egress, no smooth transition.
        # This is the maximum physically possible limb asymmetry.
        sf_morning = np.zeros_like(syn_jd, dtype=float)
        sf_evening = np.zeros_like(syn_jd, dtype=float)

        sf_morning[ingress_idx] = 1.0
        sf_evening[ingress_idx] = 0.0

        sf_morning[full_transit_idx] = 0.5
        sf_evening[full_transit_idx] = 0.5

        sf_morning[egress_idx] = 0.0
        sf_evening[egress_idx] = 1.0

        sf_morning[without_signal] = 0.0
        sf_evening[without_signal] = 0.0

        norm = sf_morning + sf_evening
        for ii in with_signal:
            if norm[ii] != 0:
                sf_morning[ii] /= norm[ii]
                sf_evening[ii] /= norm[ii]

        return sf_morning, None, sf_evening, None, ingress_idx, egress_idx
