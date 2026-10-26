"""Inputs shared by spectral-preparation methods.

The simulator supplies one order and one night at a time. Configuration
objects describe the selected method and masks, while these inputs carry
values that depend on the order or observing sequence.
"""

from dataclasses import dataclass

import numpy as np

from exoplore.config.models import PipelineConfig, TelluricConfig


@dataclass(frozen=True)
class HeliumObservation:
    """Exposure information for preparation of observed helium spectra.

    Velocities are in km/s and dates are BJD. The correction flags record
    which operations have already been applied to the supplied flux arrays.
    The native-wavelength OH mask must be applied before interpolation.
    """

    berv_kms: np.ndarray
    planet_rv_kms: np.ndarray
    telluric_corrected: bool
    julian_date: np.ndarray | None = None
    phase: np.ndarray | None = None
    exposure_time_seconds: float | np.ndarray | None = None
    oh_treatment: str = "mask"
    oh_mask_applied: bool = False
    oh_treatment_applied: str | None = None


@dataclass(frozen=True)
class PreparationInputs:
    """Preparation settings for one spectral order and observing night.

    sysrem_iterations is the effective pass count for this call. It can
    differ from the configured limit after order-dependent pass selection.
    telluric_transmission carries the order's atmospheric transmission for
    methods that use an explicit telluric model.
    """

    pipeline: PipelineConfig
    tellurics: TelluricConfig
    use_real_data: bool
    sysrem_iterations: int
    telluric_transmission: np.ndarray | None = None
    helium_observation: HeliumObservation | None = None
