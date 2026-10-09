"""Fixed-band helium significance following Allart et al. (2023), Sect. 3.2.3.

Reference: https://doi.org/10.1051/0004-6361/202245832.
Absorption is measured in planetary-frame vacuum nm. Continuum scatter is
binned in wavelength and fitted in log-log space to estimate its value at the
measurement bandwidth. The finite-bin choices are explicit implementation
choices: the paper does not publish these numerical details. This module does
not perform the paper's separate bootstrap analysis or add observation noise.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Allart2023SignificanceConfig:
    """Explicit measurement band and spectral-noise diagnostic choices, in nm."""
    band_center_nm: float
    band_width_nm: float
    continuum_windows_nm: list[list[float]]
    maximum_bin_width_nm: float
    number_of_bin_sizes: int
    minimum_bins: int
    minimum_continuum_pixels: int
    standard_deviation_ddof: int
    detection_threshold_sigma: float

    def __post_init__(self):
        if not np.isfinite([self.band_center_nm,self.band_width_nm,self.maximum_bin_width_nm,self.detection_threshold_sigma]).all():
            raise ValueError('Significance settings must be finite')
        if min(self.band_center_nm,self.band_width_nm,self.detection_threshold_sigma)<=0 or self.maximum_bin_width_nm<self.band_width_nm:
            raise ValueError('Choose a positive band and noise bins extending to at least its width')
        if self.number_of_bin_sizes<3 or self.minimum_bins<3 or self.minimum_continuum_pixels<10 or self.standard_deviation_ddof not in (0,1):
            raise ValueError('Insufficient noise sampling or unsupported standard-deviation convention')
        band=[self.band_center_nm-self.band_width_nm/2,self.band_center_nm+self.band_width_nm/2]
        previous=-np.inf
        for lo,hi in self.continuum_windows_nm:
            if not np.isfinite([lo,hi]).all() or lo>=hi or lo<previous or max(lo,band[0])<=min(hi,band[1]):
                raise ValueError('Continuum windows must be ordered, disjoint and outside the helium band')
            previous=hi
        if not self.continuum_windows_nm:
            raise ValueError('Explicit continuum windows are required')


def allart2023_significance(wave_nm: np.ndarray, transmission: np.ndarray,
                           config: Allart2023SignificanceConfig) -> dict:
    """Measure band absorption and empirical, correlated-noise-adjusted error.

Use unweighted native pixels, sample standard deviation with configured ddof,
and non-overlapping bins starting at each finite continuum segment's left
edge. Bins never cross a masked pixel, configured window or detector gap.
OLS fits log(RMS) against log(bin size), without tuning to the signal value.
No floor is applied to the fitted noise scaling: the result matches the fit,
including a scale below one if the measured noise curve demands it. No
injected model is subtracted from the continuum. Report unavailable estimates
explicitly rather than substituting diagonal errors or a unity model.
    """
    w,y=np.asarray(wave_nm,float),np.asarray(transmission,float)
    if w.ndim!=1 or y.shape!=w.shape or not np.isfinite(w).all() or np.any(np.diff(w)<=0):
        raise ValueError('Expected one increasing finite wavelength grid and matching transmission')
    step=float(np.median(np.diff(w)))
    half=config.band_width_nm/2
    signal=(abs(w-config.band_center_nm)<=half)
    if signal.sum()<2 or not np.isfinite(y[signal]).all():
        raise ValueError('Incomplete or insufficient finite coverage of the significance band')
    signal_indices=np.flatnonzero(signal)
    if np.any(np.diff(w[signal_indices])>5*step):
        raise ValueError('The significance band intersects a physical detector gap')
    segments=[]
    continuum_spacings=[]
    for lo,hi in config.continuum_windows_nm:
        ids=np.flatnonzero((w>=lo)&(w<=hi)&np.isfinite(y))
        if len(ids):
            breaks=np.flatnonzero((np.diff(ids)!=1)|(np.diff(w[ids])>5*step))+1
            for part in np.split(ids,breaks):
                if len(part):segments.append(y[part]-1)
                if len(part)>1:continuum_spacings.extend(np.diff(w[part]))
    pixels=sum(map(len,segments))
    if pixels<config.minimum_continuum_pixels:
        raise ValueError('Insufficient finite continuum pixels for the spectral-noise diagnostic')
    # The full order can have a different dispersion from the local helium
    # continuum. Use local spacing to express noise bins in wavelength units.
    continuum_step=float(np.median(continuum_spacings))
    sigma_pixel=float(np.std(np.concatenate(segments),ddof=config.standard_deviation_ddof))
    maximum=int(np.floor(config.maximum_bin_width_nm/continuum_step))
    sizes=np.unique(np.rint(np.geomspace(1,maximum,config.number_of_bin_sizes)).astype(int))
    selected,rms,counts=[],[],[]
    for size in sizes:
        bins=[part[:len(part)//size*size].reshape(-1,size).mean(axis=1)
              for part in segments if len(part)>=size]
        if not bins:continue
        values=np.concatenate(bins)
        if len(values)<config.minimum_bins:continue
        deviation=float(np.std(values,ddof=config.standard_deviation_ddof))
        if deviation>0 and np.isfinite(deviation):
            selected.append(int(size));rms.append(deviation);counts.append(len(values))
    if len(selected)<3 or max(selected)<signal.sum():
        raise ValueError('Noise curve cannot constrain the measurement bandwidth without extrapolation')
    slope,intercept=np.polyfit(np.log(selected),np.log(rms),1)
    n=int(signal.sum())
    error=float(np.exp(intercept+slope*np.log(n)))
    white=float(sigma_pixel/np.sqrt(n))
    absorption=float(np.mean(1-y[signal]))
    return dict(method='allart2023_fixed_band_spectral_noise',reference='https://doi.org/10.1051/0004-6361/202245832',
        config=asdict(config),frame='planetary rest frame, vacuum nm',
        absorption_percent=100*absorption,uncertainty_percent=100*error,
        significance_sigma=absorption/error,signal_pixels=n,continuum_pixels=pixels,
        continuum_pixel_scatter_percent=100*sigma_pixel,white_noise_band_error_percent=100*white,
        fitted_noise_scale_factor=error/white,allan_fit_slope=float(slope),allan_fit_log_intercept=float(intercept),
        bin_pixels=selected,bin_widths_nm=(np.asarray(selected)*continuum_step).tolist(),
        binned_scatter_percent=(100*np.asarray(rms)).tolist(),bin_counts=counts,
        above_paper_statistical_threshold=bool(absorption/error>=config.detection_threshold_sigma),
        bootstrap_performed=False,
        scope='Empirical significance of fixed-band absorption; not a calibrated false-alarm probability or proof of planetary origin. Fit details are explicit local choices.')


def plot_allan_noise(directory, measurements: list[dict], labels: list[str]) -> None:
    """Save continuum RMS, white-noise predictions and fits used in the legends."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    directory=Path(directory)
    stem=directory/'helium_allart_noise_binning'
    if any(stem.with_suffix(ext).exists() for ext in ('.png','.pdf')):
        raise FileExistsError('Preserve existing noise-binning plots; choose a new directory')
    fig,axes=plt.subplots(len(measurements),1,figsize=(9,2.7*len(measurements)),squeeze=False,constrained_layout=True)
    for ax,m,label in zip(axes[:,0],measurements,labels):
        n=np.asarray(m['bin_pixels']);width=np.asarray(m['bin_widths_nm'])*10
        ax.loglog(width,m['binned_scatter_percent'],'ro',ms=4,label='Measured continuum scatter')
        ax.loglog(width,m['continuum_pixel_scatter_percent']/np.sqrt(n),'k:',label='Independent-pixel prediction')
        fit=np.exp(m['allan_fit_log_intercept']+m['allan_fit_slope']*np.log(n))*100
        ax.loglog(width,fit,'b--',label='Log–log fit')
        ax.axvline(m['config']['band_width_nm']*10,color='.5',ls='-')
        ax.set(title=f"{label}: {m['significance_sigma']:.1f}σ; band error {m['uncertainty_percent']:.3f}%",xlabel='Spectral bin width (Å)',ylabel='Continuum scatter (%)')
        ax.legend(fontsize=8)
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)
