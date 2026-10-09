"""
exoplore.config.czesla2024
==========================

Explicit configuration for direct He I transmission preparation following
Czesla et al. (2024), A&A 692, A230, doi:10.1051/0004-6361/202451003.
Dataset-specific exposure selections and frame conventions are mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Czesla2024Config:
    """Scientific choices for one observed, corrected CRIRES+ time series.

    ``input_path`` points to a validated per-exposure correction directory.
    ``reference_running_numbers`` and ``in_transit_running_numbers`` are
    one-based chronological selections, supplied separately for every dataset.
    Wavelengths are native topocentric vacuum nm; velocities are km/s.
    Continuum normalization is an explicit local implementation choice, since
    the paper does not specify a complete normalization recipe.
    """
    input_path: str
    wavelength_frame: str
    order_segment: str
    reference_running_numbers: list[int]
    in_transit_running_numbers: list[int]
    gamma_kms: float
    kp_kms: float
    t0_bjd_tdb: float
    period_days: float
    target_ra_deg: float
    target_dec_deg: float
    observatory_lon_lat_height: list[float]
    exposure_time_keyword: str
    continuum_stellar_windows_nm: list[list[float]]
    continuum_polynomial_degree: int
    minimum_continuum_pixels: int
    oh_mode: str
    oh_topocentric_windows_nm: list[list[float]]
    oh_topocentric_lines_nm: list[float]
    telluric_min_transmission: float
    max_wavelength_shift_kms: float
    inflate_by_telluric_rms: bool
    planet_lightcurve_window_nm: list[float]
    stellar_lightcurve_window_nm: list[float]
    helium_vacuum_lines_nm: list[float]
    helium_oscillator_strengths: list[float]
    helium_atomic_source: str
    optical_contact_phases: list[float]
    plot_stellar_window_nm: list[float]
    equivalent_width_window_nm: list[float]
    monte_carlo_draws: int
    monte_carlo_seed: int

    def __post_init__(self) -> None:
        """Reject ambiguous scientific settings before starting a run."""
        if self.wavelength_frame != 'topocentric_vacuum_nm':
            raise ValueError('czesla2024 requires topocentric_vacuum_nm; no implicit air conversion')
        if not self.input_path or not self.order_segment or not self.helium_atomic_source:
            raise ValueError('Input path, order segment and atomic source are required')
        if self.oh_mode not in ('baseline', 'mask'):
            raise ValueError('OH mode must be baseline or mask; no OH emission model is implemented')
        if self.oh_mode == 'mask' and not self.oh_topocentric_windows_nm:
            raise ValueError('OH mask mode requires explicit native topocentric windows')
        selections=[self.reference_running_numbers,self.in_transit_running_numbers]
        if any(not s or any(type(i) is not int or i<1 for i in s) or len(set(s))!=len(s) for s in selections):
            raise ValueError('Exposure selections must contain unique positive integer running numbers')
        if set(selections[0])&set(selections[1]):
            raise ValueError('Reference and in-transit selections must be disjoint')
        if len(self.observatory_lon_lat_height)!=3:
            raise ValueError('Observatory coordinates require longitude, latitude, height in metres')
        if len(self.helium_vacuum_lines_nm)!=3 or len(self.helium_oscillator_strengths)!=3:
            raise ValueError('The helium multiplet requires three explicit wavelengths and strengths')
        if not all(math.isfinite(v) and v>0 for v in self.helium_vacuum_lines_nm+self.helium_oscillator_strengths):
            raise ValueError('Atomic wavelengths and strengths must be finite and positive')
        if self.period_days<=0 or self.max_wavelength_shift_kms<=0 or not 0<self.telluric_min_transmission<=1:
            raise ValueError('Invalid period, wavelength gate or telluric floor')
        if self.continuum_polynomial_degree not in (0,1,2) or self.minimum_continuum_pixels<self.continuum_polynomial_degree+2:
            raise ValueError('Invalid continuum degree or minimum pixel count')
        if type(self.monte_carlo_draws) is not int or (self.monte_carlo_draws != 0 and self.monte_carlo_draws < 100):
            raise ValueError('Choose zero draws for a simulation without resampling, or at least 100 draws')
        windows=self.continuum_stellar_windows_nm+self.oh_topocentric_windows_nm+[
            self.planet_lightcurve_window_nm,self.stellar_lightcurve_window_nm,
            self.plot_stellar_window_nm,self.equivalent_width_window_nm]
        if not self.continuum_stellar_windows_nm or any(len(w)!=2 or not all(math.isfinite(v) for v in w) or w[0]>=w[1] for w in windows):
            raise ValueError('Wavelength windows require finite increasing bounds in nm')
