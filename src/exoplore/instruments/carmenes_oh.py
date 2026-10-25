"""Fibre-B OH emission models for native CARMENES NIR science spectra.

The master Voigt plus linked Gaussian pair follows Pallé et al. (2020),
doi:10.1051/0004-6361/202037719. Unequal A/B extraction units require an
transfer calibration fitted to the supplied fibre spectra.
All wavelengths are topocentric vacuum nm. Source arrays are never modified.
"""
from __future__ import annotations

# Role in the calculation
# Fit the sky spectrum in fibre B and calibrate its contribution to the science
# spectrum in fibre A. The fitted sky amplitude and inter-fibre flux factor determine
# the subtraction.


from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.special import voigt_profile


@dataclass(frozen=True)
class OHFitConfig:
    """Explicit line, fit, rejection and inter-fibre calibration choices."""

    weak_centres_nm: tuple[float, float]
    strong_centre_nm: float
    weak_fit_window_nm: tuple[float, float]
    strong_fit_window_nm: tuple[float, float]
    centre_tolerance_nm: float
    sigma_bounds_nm: tuple[float, float]
    lorentz_hwhm_bounds_nm: tuple[float, float]
    initial_sigma_nm: float
    initial_lorentz_hwhm_nm: float
    rejection_sigma: float
    pixel_quadrature_points: int
    transfer_shift_bounds_nm: tuple[float, float]
    minimum_transfer_exposures: int

    def __post_init__(self) -> None:
        """Require finite ordered wavelength intervals and physical widths."""
        for name in ('weak_fit_window_nm','strong_fit_window_nm','sigma_bounds_nm',
                     'lorentz_hwhm_bounds_nm','transfer_shift_bounds_nm'):
            pair=getattr(self,name)
            if len(pair)!=2 or not np.isfinite(pair).all() or pair[0]>=pair[1]:
                raise ValueError(f'{name} must contain two finite increasing values')
        if len(self.weak_centres_nm)!=2 or not np.isfinite(self.weak_centres_nm).all():
            raise ValueError('Specify two finite weak OH centres')
        if not np.isfinite(self.strong_centre_nm):
            raise ValueError('Specify a finite strong OH centre')
        if self.sigma_bounds_nm[0]<=0 or self.lorentz_hwhm_bounds_nm[0]<0:
            raise ValueError('Gaussian widths must be positive and Lorentz widths nonnegative')
        if not self.sigma_bounds_nm[0]<self.initial_sigma_nm<self.sigma_bounds_nm[1]:
            raise ValueError('Initial Gaussian sigma must lie inside its bounds')
        if not self.lorentz_hwhm_bounds_nm[0]<=self.initial_lorentz_hwhm_nm<self.lorentz_hwhm_bounds_nm[1]:
            raise ValueError('Initial Lorentz HWHM must lie inside its bounds')
        if not np.isfinite(self.centre_tolerance_nm) or self.centre_tolerance_nm<=0:
            raise ValueError('OH centre tolerance must be positive')
        if not np.isfinite(self.rejection_sigma) or self.rejection_sigma<=0:
            raise ValueError('OH outlier rejection threshold must be positive')
        if type(self.pixel_quadrature_points) is not int or self.pixel_quadrature_points<1:
            raise ValueError('Pixel quadrature points must be a positive integer')
        if type(self.minimum_transfer_exposures) is not int or self.minimum_transfer_exposures<2:
            raise ValueError('Transfer calibration requires at least two reference exposures')


def _sampled_profile(wave, centres, sigma, gamma, quadrature):
    """Average peak-normalised profiles over each native wavelength pixel."""
    wave = np.asarray(wave, float)
    edges = np.r_[wave[0] - (wave[1]-wave[0])/2,
                  (wave[:-1]+wave[1:])/2, wave[-1]+(wave[-1]-wave[-2])/2]
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    samples = (edges[:-1, None]+edges[1:, None])/2 + np.diff(edges)[:, None]*nodes/2
    value = np.zeros_like(wave)
    for centre in centres:
        value += (voigt_profile(samples-centre, sigma, gamma)
                  / voigt_profile(0, sigma, gamma)) @ (weights/2)
    return value


def _robust_fit(function, initial, bounds, x, y, error, rejection_sigma):
    """Fit with soft-L1 loss, then explicitly exclude gross residual outliers."""
    valid = np.isfinite(y) & np.isfinite(error) & (error > 0)
    original = valid.copy()
    fit = None
    residual_scatter = None
    for _ in range(4):
        if valid.sum() <= len(initial)+2:
            raise ValueError('Too few independent pixels for an OH fit')
        fit = least_squares(lambda p: (function(x, p)[valid]-y[valid])/error[valid],
                            initial, bounds=bounds, loss='soft_l1', f_scale=1.,
                            x_scale='jac', max_nfev=2000)
        if not fit.success:
            raise ValueError(f'OH fit failed: {fit.message}')
        residual = (function(x, fit.x)-y)/error
        centre=np.median(residual[valid])
        if residual_scatter is None:
            residual_scatter=max(1.,float(1.4826*np.median(abs(residual[valid]-centre))))
        # Keep the original residual scale: repeatedly shrinking it would
        # remove resolved profile mismatch until only a handful of pixels
        # remain. Such mismatch must stay visible in the reported chi-square.
        new = original & (np.abs(residual-centre) <= rejection_sigma*residual_scatter)
        if np.array_equal(new, valid):
            break
        valid = new
        initial = fit.x
    # Refit the final retained set so its covariance and membership agree.
    if valid.sum() <= len(initial)+2:
        raise ValueError('OH rejection left too few pixels to constrain the profile')
    fit=least_squares(lambda p:(function(x,p)[valid]-y[valid])/error[valid],
                      fit.x,bounds=bounds,loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=2000)
    if not fit.success:
        raise ValueError(f'Final OH fit failed: {fit.message}')
    # Covariance is conditional on the retained pixels and fixed model form.
    jac = fit.jac
    covariance = np.linalg.pinv(jac.T @ jac)
    chi2 = float(np.sum(((function(x, fit.x)[valid]-y[valid])/error[valid])**2))
    return fit.x, covariance, valid, chi2/max(1, valid.sum()-len(fit.x))


def fit_fibre_b(night: dict, order: int, settings: OHFitConfig) -> dict:
    """Fit a summed sky master, then one linked OH amplitude per exposure.

    A local linear background in each line group is fitted but not subtracted
    from fibre A. Rejected sky pixels are saved. science pixels are retained.
    """
    waves = night['sky_wave_nm'][:, order]
    flux, errors = night['sky_flux'][:, order], night['sky_error'][:, order]
    if not np.allclose(waves, waves[0], atol=1e-8, rtol=0):
        raise ValueError('Master OH fit requires a common native fibre-B grid')
    wave = waves[0]
    usable = np.isfinite(flux) & np.isfinite(errors) & (errors > 0)
    if not usable.all():
        raise ValueError('Handle invalid native sky pixels explicitly before summing the master')
    master = flux.sum(axis=0)
    master_error = np.sqrt((errors**2).sum(axis=0))
    s = settings
    fitted = {}
    for label, window, centres in (
        ('strong', s.strong_fit_window_nm, (s.strong_centre_nm,)),
        ('weak', s.weak_fit_window_nm, s.weak_centres_nm),
    ):
        mask = (wave >= window[0]) & (wave <= window[1])
        x, y, e = wave[mask], master[mask], master_error[mask]
        # Exclude unresolved positive spikes from the master fit only. Require
        # a peak above both neighbours as well as the extraction-error gate.
        # broad, resolved OH cores do not satisfy this isolation criterion.
        neighbour=np.maximum(np.r_[y[0],y[:-1]],np.r_[y[1:],y[-1]])
        isolated=(y>3*np.maximum(neighbour,0)) & (y-median_filter(y,size=3)>s.rejection_sigma*e)
        e=e.copy();e[isolated]=np.inf
        pivot = np.mean(window)
        n = len(centres)
        def function(x, p):
            gamma = p[n+1] if label == 'strong' else 0.
            j = n+2 if label == 'strong' else n+1
            return p[j]*_sampled_profile(x, p[:n], p[n], gamma, s.pixel_quadrature_points) + p[j+1] + p[j+2]*(x-pivot)
        widths = [s.initial_sigma_nm] + ([s.initial_lorentz_hwhm_nm] if label == 'strong' else [])
        initial = list(centres)+widths+[float(np.max(y[~isolated])-np.median(y)), float(np.median(y)), 0.]
        low = [c-s.centre_tolerance_nm for c in centres]+[s.sigma_bounds_nm[0]]
        high = [c+s.centre_tolerance_nm for c in centres]+[s.sigma_bounds_nm[1]]
        if label == 'strong':
            low += [s.lorentz_hwhm_bounds_nm[0]]; high += [s.lorentz_hwhm_bounds_nm[1]]
        p, cov, good, chi = _robust_fit(function, initial,
            (low+[0., -np.inf, -np.inf], high+[np.inf, np.inf, np.inf]), x,y,e,s.rejection_sigma)
        j = n+2 if label == 'strong' else n+1
        fitted[label] = dict(centres_nm=p[:n].tolist(), sigma_nm=float(p[n]),
            lorentz_hwhm_nm=float(p[n+1]) if label == 'strong' else 0.,
            peak_amplitude=float(p[j]), reduced_chi2=chi,
            rejected_native_indices=np.flatnonzero(mask)[~good].tolist())
    ratio = fitted['weak']['peak_amplitude']/fitted['strong']['peak_amplitude']
    def profile(x):
        a,b = fitted['strong'], fitted['weak']
        return (_sampled_profile(x,a['centres_nm'],a['sigma_nm'],a['lorentz_hwhm_nm'],s.pixel_quadrature_points)
                + ratio*_sampled_profile(x,b['centres_nm'],b['sigma_nm'],0.,s.pixel_quadrature_points))
    groups = [(wave>=window[0]) & (wave<=window[1])
              for window in (s.weak_fit_window_nm,s.strong_fit_window_nm)]
    union = groups[0] | groups[1]
    rejected_master = sorted(set(fitted['strong']['rejected_native_indices']+fitted['weak']['rejected_native_indices']))
    union[rejected_master] = False
    x = wave[union]
    template = profile(wave)
    # Independent baselines/slopes, with one common emission amplitude.
    design = np.column_stack([template]+[item for mask in groups for item in
        (mask.astype(float),mask*(wave-np.mean(wave[mask])))])
    amplitudes, variances, rejected, chi2 = [], [], [], []
    for row, error in zip(flux,errors):
        y,e = row[union], error[union]
        local = design[union]
        p,cov,good,chi = _robust_fit(lambda x,p:local@p,
            [fitted['strong']['peak_amplitude']/len(flux),0.,0.,0.,0.],
            ([0.,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf]*5),x,y,e,s.rejection_sigma)
        amplitudes.append(p[0]);variances.append(cov[0,0]);chi2.append(chi)
        rejected.append(np.flatnonzero(union)[~good].tolist())
    amplitude = np.asarray(amplitudes)
    return dict(master_wave_nm=wave,master_flux=master,master_error=master_error,
                master_parameters=fitted,weak_to_strong_peak_ratio=float(ratio),
                amplitudes=amplitude,amplitude_variance=np.asarray(variances),
                model_b=amplitude[:,None]*template, reduced_chi2=chi2,
                rejected_master_indices=rejected_master,rejected_exposure_indices=rejected)


def transfer_to_fibre_a(night: dict, order: int, sky: dict,
                        reference_indices: np.ndarray, settings: OHFitConfig,
                        *, fit_wavelength_shift: bool = True) -> dict:
    """Measure a common B-to-A flux scale and wavelength offset on OOT spectra.

    Only the strong OH interval calibrates the transfer. In-transit spectra
    never set the scale. A linear stellar background is fitted independently
    per calibration exposure. The returned uncertainties hold this global
    calibration and master profile fixed. Their systematics remain explicit.
    Matched extractions use ``fit_wavelength_shift=False`` to fit only the
    common throughput factor and per-exposure background at the fibre-B grid.
    """
    s = settings
    indices = np.asarray(reference_indices,int)
    if len(indices) < s.minimum_transfer_exposures:
        raise ValueError('Too few out-of-transit spectra for inter-fibre calibration')
    wave=night['wave_nm'][:,order]; flux=night['flux'][:,order]; errors=night['error'][:,order]
    shape=sky['master_parameters']['strong']
    def strong(w, shift):
        return _sampled_profile(w,np.asarray(shape['centres_nm'])+shift,
            shape['sigma_nm'],shape['lorentz_hwhm_nm'],s.pixel_quadrature_points)
    selected=[np.flatnonzero((wave[i]>=s.strong_fit_window_nm[0]) & (wave[i]<=s.strong_fit_window_nm[1])) for i in indices]
    # Fit all OOT exposures jointly with a positive global scale.
    y=np.concatenate([flux[i,m] for i,m in zip(indices,selected)])
    e=np.concatenate([errors[i,m] for i,m in zip(indices,selected)])
    pivot=np.mean(s.strong_fit_window_nm)
    offset=2 if fit_wavelength_shift else 1
    def function(dummy,p):
        shift=p[1] if fit_wavelength_shift else 0.
        return np.concatenate([p[0]*sky['amplitudes'][i]*strong(wave[i,m],shift)
            +p[offset+2*k]+p[offset+1+2*k]*(wave[i,m]-pivot)
            for k,(i,m) in enumerate(zip(indices,selected))])
    peak=np.median([np.ptp(flux[i,m])/sky['amplitudes'][i] for i,m in zip(indices,selected)])
    initial=[max(peak,1e-12)]+([0.] if fit_wavelength_shift else [])+[v for i,m in zip(indices,selected) for v in (np.median(flux[i,m]),0.)]
    low=[0.]+([s.transfer_shift_bounds_nm[0]] if fit_wavelength_shift else [])+[-np.inf]*(2*len(indices))
    high=[np.inf]+([s.transfer_shift_bounds_nm[1]] if fit_wavelength_shift else [])+[np.inf]*(2*len(indices))
    p,cov,good,chi=_robust_fit(function,initial,(low,high),np.arange(len(y)),y,e,s.rejection_sigma)
    if fit_wavelength_shift and not s.transfer_shift_bounds_nm[0] < p[1] < s.transfer_shift_bounds_nm[1]:
        raise ValueError('Inter-fibre wavelength offset hit its fit boundary')
    shift=float(p[1]) if fit_wavelength_shift else 0.
    weak=sky['master_parameters']['weak']
    templates=np.array([strong(w,shift)+sky['weak_to_strong_peak_ratio']*
        _sampled_profile(w,np.asarray(weak['centres_nm'])+shift,weak['sigma_nm'],0.,s.pixel_quadrature_points)
        for w in wave])
    model=p[0]*sky['amplitudes'][:,None]*templates
    variance=p[0]**2*sky['amplitude_variance'][:,None]*templates**2
    return dict(model_a=model,conditional_model_variance=variance,
                corrected_flux=flux-model,conditional_error=np.sqrt(errors**2+variance),
                transfer_scale=float(p[0]),transfer_scale_error=float(np.sqrt(max(cov[0,0],0))),
                transfer_shift_nm=shift,transfer_shift_error_nm=float(np.sqrt(max(cov[1,1],0))) if fit_wavelength_shift else 0.,
                transfer_reduced_chi2=chi,transfer_reference_indices=indices.tolist(),
                rejected_transfer_samples=np.flatnonzero(~good).tolist(),
                uncertainty_scope='Native A extraction error plus independent B amplitude variance; shared master/transfer covariance excluded',
                calibration='Empirical OOT strong-OH transfer between unequal extraction units; not published factor 0.88')


def subtract_matched_fibre_b(night: dict, order: int, sky: dict,
                            scale, settings: OHFitConfig,
                            *, reference_indices: np.ndarray | None = None) -> dict:
    """Measure the matched A/B throughput on OOT spectra and subtract the sky.

    With ``scale=None``, fit a common throughput factor using only OOT
    fibre-A spectra, holding the fibre-B profile and wavelength offset fixed.
    Existing explicit scalar/per-exposure scales remain optional overrides.
    Selecting this mode asserts compatible extraction/flux units.
    Contradictory recorded FOX extraction metadata causes an explicit error.
    The scale and master profile are held fixed in the conditional errors.
    """
    for a,b in zip(night['headers'],night['sky_headers']):
        af,bf=a.get('CARACAL FOX VERSION'),b.get('CARACAL FOX VERSION')
        if (af is None)!=(bf is None) or (af is not None and af!=bf):
            raise ValueError('fiber_b_matched_extraction conflicts with the recorded A/B FOX extraction metadata; provide compatibly extracted spectra or choose fiber_b_calibrated')
    if scale is None:
        if reference_indices is None:
            raise ValueError('Automatic matched-fibre scaling requires out-of-transit indices')
        result=transfer_to_fibre_a(night,order,sky,reference_indices,settings,
                                  fit_wavelength_shift=False)
        result.update(fiber_b_scale=[result['transfer_scale']]*len(night['flux']),
                      calibration='B-to-A throughput measured jointly on OOT spectra with matched extraction and zero inter-fibre wavelength offset')
        return result
    flux=night['flux'][:,order]; error=night['error'][:,order]
    factors=np.broadcast_to(np.asarray(scale,float),(len(flux),))
    if not np.isfinite(factors).all() or np.any(factors<=0):
        raise ValueError('fiber_b_scale must contain positive finite flux multipliers')
    a,b=sky['master_parameters']['strong'],sky['master_parameters']['weak']
    templates=np.array([_sampled_profile(w,a['centres_nm'],a['sigma_nm'],a['lorentz_hwhm_nm'],settings.pixel_quadrature_points)
        +sky['weak_to_strong_peak_ratio']*_sampled_profile(w,b['centres_nm'],b['sigma_nm'],0.,settings.pixel_quadrature_points)
        for w in night['wave_nm'][:,order]])
    model=factors[:,None]*sky['amplitudes'][:,None]*templates
    variance=factors[:,None]**2*sky['amplitude_variance'][:,None]*templates**2
    return dict(model_a=model,conditional_model_variance=variance,
                corrected_flux=flux-model,conditional_error=np.sqrt(error**2+variance),
                fiber_b_scale=factors.tolist(),transfer_shift_nm=0.,
                calibration='User-supplied B-to-A flux multiplier for matched extraction',
                uncertainty_scope='A extraction plus B amplitude variance; user scale and master shape fixed; cross-pixel sky covariance excluded')
