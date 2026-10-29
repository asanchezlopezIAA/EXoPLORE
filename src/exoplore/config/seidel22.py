"""Choices for stellar-reference transmission preparation, in vacuum nm."""

from dataclasses import dataclass, field
import math


@dataclass
class Seidel22Config:
    """Choose continuum intervals and stellar masks for transmission spectra.

    Continuum windows belong to the stellar frame. Core masks also belong
    to that frame and remain masked when the spectra are planet-aligned.
    Each exposure receives a weight proportional to its continuum S/N.
    """

    # Choose wavelength intervals clear of the stellar and planetary features
    # being measured. Their median flux sets each exposure's continuum level.
    continuum_windows_nm: list[list[float]]
    # These optional intervals exclude stellar residuals, such as deep cores.
    stellar_mask_windows_nm: list[list[float]] = field(default_factory=list)
    # Stop preparation if too few continuum pixels remain for normalization.
    minimum_continuum_pixels: int = 10
    # For supplied observations, record the corrections already performed.
    # The direct sodium runner performs its own telluric fit before preparation.
    input_telluric_corrected: bool = False
    input_sky_subtracted: bool = False
    # Zero uses the analytic uncertainty calculation. Two or more draws allow
    # continuum refitting and reference construction to enter the covariance.
    noise_resamples: int = 0
    # Fix the random draws used for this uncertainty calculation.
    noise_resample_seed: int | None = None

    def __post_init__(self):
        if type(self.noise_resamples) is not int or self.noise_resamples < 0 or self.noise_resamples == 1:
            raise ValueError('Seidel22 noise_resamples must be zero or at least two')
        if self.noise_resample_seed is not None and (type(self.noise_resample_seed) is not int or self.noise_resample_seed < 0):
            raise ValueError('Seidel22 noise_resample_seed must be a nonnegative integer or null')
        if (type(self.input_telluric_corrected) is not bool
                or type(self.input_sky_subtracted) is not bool):
            raise ValueError("Seidel22 input correction flags must be boolean")
        if not self.continuum_windows_nm:
            raise ValueError("Seidel22 requires explicit stellar-frame continuum windows")
        for bounds in self.continuum_windows_nm + self.stellar_mask_windows_nm:
            if (len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds)
                    or bounds[0] >= bounds[1]):
                raise ValueError("Seidel22 windows require increasing positive bounds in vacuum nm")
        if type(self.minimum_continuum_pixels) is not int or self.minimum_continuum_pixels < 2:
            raise ValueError("Seidel22 minimum_continuum_pixels must be an integer of at least 2")
