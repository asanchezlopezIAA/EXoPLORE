"""Constrain a CARMENES OH profile with an out-of-transit stellar template.

This optional extension fits the strong OH line in fibre A while keeping the
temporal sky amplitudes measured in fibre B. A free common stellar spectrum
is fitted simultaneously in the stellar frame. The weak OH correction is
retained from the existing calibrated fibre-B model. Inputs are read-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from exoplore.instruments.carmenes_oh import OHFitConfig, _sampled_profile
from exoplore.pipelines.helium_math import stellar_wavelength, window_mask


@dataclass(frozen=True)
class OOTProfileConfig:
    """Explicit fitting interval, identifiability and acceptance thresholds."""

    fit_window_nm: list[float]
    relative_svd_cutoff: float
    minimum_sky_amplitude_cv: float
    maximum_reduced_chi2: float
    maximum_profile_condition_number: float

    def __post_init__(self):
        if (len(self.fit_window_nm) != 2 or not np.isfinite(self.fit_window_nm).all()
                or not 0 < self.fit_window_nm[0] < self.fit_window_nm[1]):
            raise ValueError('Supply increasing positive topocentric vacuum nm fit bounds')
        if not 0 < self.relative_svd_cutoff < 1:
            raise ValueError('relative_svd_cutoff must lie between zero and one')
        if not np.isfinite(self.minimum_sky_amplitude_cv) or self.minimum_sky_amplitude_cv <= 0:
            raise ValueError('Supply a positive minimum relative sky-amplitude variation')
        if not np.isfinite(self.maximum_reduced_chi2) or self.maximum_reduced_chi2 <= 0:
            raise ValueError('Supply a positive OH fit acceptance chi-square')
        if (not np.isfinite(self.maximum_profile_condition_number)
                or self.maximum_profile_condition_number <= 1):
            raise ValueError('Supply an OH profile Jacobian condition-number ceiling >1')


def fit_oot_profile(wave_nm: np.ndarray, refined_wave_nm: np.ndarray,
                    raw_flux: np.ndarray, raw_error: np.ndarray,
                    telluric_transmission: np.ndarray, berv_kms: np.ndarray,
                    reference_indices: np.ndarray, science, sky: dict,
                    calibrated_transfer: dict, profile: OHFitConfig,
                    settings: OOTProfileConfig) -> dict:
    """Fit a common stellar spectrum and a changing observer-frame OH profile.

    For OOT exposure i the normalized corrected flux is
    S(lambda_star) + scale*B_amplitude_i*P(lambda_top)/(T_i*continuum_i).
    The stellar samples are linear nuisance parameters eliminated by SVD at
    each nonlinear OH fit evaluation. No transit exposure enters the fit.
    Pixel-error weights hold continuum, telluric fits and B amplitudes fixed.
    Parameter errors are diagnostic linearized conditional errors, not a full
    uncertainty budget. Acceptance gates reject insufficient sky variation,
    parameter boundaries, poor residual fits and ill-conditioned profiles.
    """
    wave, refined, flux, error, tell = [np.asarray(a, float) for a in
        (wave_nm, refined_wave_nm, raw_flux, raw_error, telluric_transmission)]
    if wave.ndim != 2 or any(a.shape != wave.shape for a in (refined, flux, error, tell)):
        raise ValueError('Supply matching native exposure-by-pixel arrays')
    refs = np.asarray(reference_indices, int)
    if len(refs) < profile.minimum_transfer_exposures or len(np.unique(refs)) != len(refs):
        raise ValueError('Insufficient distinct out-of-transit calibration exposures')
    if np.any(refs < 0) or np.any(refs >= len(wave)):
        raise ValueError('Out-of-transit indices lie outside the supplied night')
    if np.shape(berv_kms) != (len(wave),):
        raise ValueError('Supply one barycentric correction per exposure')
    strong = sky['master_parameters']['strong']
    old_centre = strong['centres_nm'][0] + calibrated_transfer['transfer_shift_nm']
    old_strong = calibrated_transfer['transfer_scale'] * sky['amplitudes'][:, None] * np.array([
        _sampled_profile(w, [old_centre], strong['sigma_nm'],
            strong['lorentz_hwhm_nm'], profile.pixel_quadrature_points) for w in wave])
    weak_model = calibrated_transfer['model_a'] - old_strong
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0) & np.isfinite(tell) & (tell >= science.telluric_min_transmission)
    corrected = np.divide(flux-weak_model, tell, out=np.full_like(flux, np.nan), where=valid)
    for w, ok in zip(wave, valid):
        if np.count_nonzero(window_mask(w, science.continuum_stellar_windows_nm) & ok) < science.minimum_continuum_pixels:
            raise ValueError('Insufficient valid continuum pixels for OOT OH fitting')
    continua = np.array([np.mean(row[window_mask(w, science.continuum_stellar_windows_nm) & ok])
        for w, row, ok in zip(wave, corrected, valid)])
    if not np.isfinite(continua).all() or np.any(continua <= 0):
        raise ValueError('Insufficient positive continuum coverage for OOT OH fitting')
    sky_strength = sky['amplitudes'][refs] / continua[refs]
    amplitude_cv = float(np.std(sky_strength)/np.mean(sky_strength))
    if amplitude_cv < settings.minimum_sky_amplitude_cv:
        raise ValueError(f'Insufficient changing sky strength to separate OH from the star: CV={amplitude_cv:.4g}')
    selected = [np.flatnonzero(valid[i] & window_mask(wave[i], [settings.fit_window_nm])) for i in refs]
    stellar = np.array([stellar_wavelength(w, b, science.gamma_kms) for w, b in zip(refined, berv_kms)])
    all_stellar = np.concatenate([stellar[i, m] for i, m in zip(refs, selected)])
    step = float(np.median(np.diff(stellar[refs[0]])))
    template_wave = np.arange(all_stellar.min()-step, all_stellar.max()+2*step, step)
    y = np.concatenate([corrected[i, m]/continua[i] for i, m in zip(refs, selected)])
    e = np.concatenate([error[i, m]/tell[i, m]/continua[i] for i, m in zip(refs, selected)])
    design = np.zeros((len(y), len(template_wave)))
    j = np.searchsorted(template_wave, all_stellar, side='right')-1
    fraction = (all_stellar-template_wave[j])/np.diff(template_wave)[j]
    rows = np.arange(len(y))
    design[rows, j] = 1-fraction
    design[rows, j+1] = fraction
    weighted_design = design/e[:, None]
    u, singular, vh = np.linalg.svd(weighted_design, full_matrices=False)
    retained = singular > settings.relative_svd_cutoff*singular.max()
    basis = u[:, retained]
    def project(v):
        return v - basis @ (basis.T @ v)
    # Scale the tiny extraction-unit multiplier to order one for optimization.
    scale_unit = calibrated_transfer['transfer_scale']
    def emission(p):
        return np.concatenate([p[0]*scale_unit*sky['amplitudes'][i]*
            _sampled_profile(wave[i, m], [old_centre+p[1]], p[2], p[3],
                profile.pixel_quadrature_points)/tell[i, m]/continua[i]
            for i, m in zip(refs, selected)])
    initial = [1., 0., strong['sigma_nm'], strong['lorentz_hwhm_nm']]
    bounds = ([0., profile.transfer_shift_bounds_nm[0], profile.sigma_bounds_nm[0], profile.lorentz_hwhm_bounds_nm[0]],
              [np.inf, profile.transfer_shift_bounds_nm[1], profile.sigma_bounds_nm[1], profile.lorentz_hwhm_bounds_nm[1]])
    fit = least_squares(lambda p: project((y-emission(p))/e), initial,
        bounds=bounds, x_scale='jac', max_nfev=1000)
    if not fit.success:
        raise RuntimeError(fit.message)
    p = fit.x
    rank = int(retained.sum())
    dof = len(y)-rank-len(p)
    if dof <= 0:
        raise ValueError('The stellar template leaves no degrees of freedom for OH fitting')
    chi = float(np.sum(fit.fun**2)/dof)
    # Dimensionless parameter scaling avoids a condition number set by units.
    scaled_jac = fit.jac * np.array([1., .01, .01, .01])[None, :]
    profile_singular = np.linalg.svd(scaled_jac, compute_uv=False)
    condition = float(profile_singular[0]/max(profile_singular[-1], np.finfo(float).tiny))
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    conditional_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    star = (vh[retained].T / singular[retained]) @ (basis.T @ ((y-emission(p))/e))
    failures = []
    # gamma=0 is the physical Gaussian limit of a Voigt profile, not an
    # invalid negative width. Retain that result explicitly in provenance.
    invalid_boundary = fit.active_mask.copy()
    if invalid_boundary[3] == -1 and profile.lorentz_hwhm_bounds_nm[0] == 0:
        invalid_boundary[3] = 0
    if np.any(invalid_boundary):
        failures.append('OH parameter reached its configured boundary')
    if chi > settings.maximum_reduced_chi2:
        failures.append('OOT residual chi-square exceeded its acceptance threshold')
    if condition > settings.maximum_profile_condition_number:
        failures.append('OH parameters are ill-conditioned after eliminating the stellar template')
    templates = np.array([_sampled_profile(w, [old_centre+p[1]], p[2], p[3],
        profile.pixel_quadrature_points) for w in wave])
    new_strong = scale_unit*p[0]*sky['amplitudes'][:, None]*templates
    model = weak_model + new_strong
    weak_template = np.divide(weak_model, sky['amplitudes'][:, None],
        out=np.zeros_like(model), where=sky['amplitudes'][:, None] != 0)
    amplitude_derivative = weak_template + scale_unit*p[0]*templates
    variance = sky['amplitude_variance'][:, None]*amplitude_derivative**2
    return dict(accepted=not failures, rejection_reasons=failures,
        model_a=model, conditional_model_variance=variance,
        corrected_flux=flux-model, conditional_error=np.sqrt(error**2+variance),
        stellar_template_wave_nm=template_wave, stellar_template_flux=star,
        stellar_template_supported_range_nm=[float(all_stellar.min()), float(all_stellar.max())],
        calibration_reference_indices=refs.tolist(), sky_amplitude_cv=amplitude_cv,
        fit_samples=len(y), stellar_template_rank=rank, degrees_of_freedom=dof,
        transfer_scale=float(scale_unit*p[0]), strong_centre_nm=float(old_centre+p[1]),
        gaussian_sigma_nm=float(p[2]), lorentz_hwhm_nm=float(p[3]),
        lorentz_lower_boundary=bool(fit.active_mask[3] == -1),
        parameter_error_scope='Conditional linearized diagnostics; a Lorentz width at zero has no symmetric confidence interval',
        conditional_parameter_errors=conditional_errors.tolist(),
        profile_condition_number=condition, transfer_reduced_chi2=chi,
        wavelength_frame='OH: native topocentric vacuum nm; template: refined stellar vacuum nm',
        weak_line_treatment='Existing calibrated fibre-B weak pair retained',
        calibration='OOT common stellar spectrum plus fibre-B-modulated strong OH profile',
        uncertainty_scope='Native A errors plus conditional B-amplitude variance; common stellar template, OH shape/scale and continuum covariance excluded')
