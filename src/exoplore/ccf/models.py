"""Configuration and reference motion used to analyse completed CCF arrays."""

from dataclasses import dataclass

import numpy as np

from exoplore.config.models import CrossCorrelationConfig, PipelineConfig


@dataclass(frozen=True)
class NightStudyInputs:
    """Settings for comparing and combining simulated observing nights.

    The CCF and pipeline configurations select the statistic and SYSREM
    optimisation. Velocities are in km/s. berv_kms is the barycentric
    correction used by the study's common observing sequence.
    """

    cross_correlation: CrossCorrelationConfig
    pipeline: PipelineConfig
    n_orders: int
    n_nights: int
    stack_group_size: int | None
    berv_kms: float | np.ndarray
    systemic_velocity_kms: float
    reference_kp_kms: float
    reference_vrest_kms: float
    sysrem_iterations: int
    exclude_around: str = "peak"


@dataclass(frozen=True)
class TimeResolvedCCFInputs:
    """Normalisation and presentation inputs for CCF transit intervals.

    resolving_power supplies the optional instrumental velocity floor,
    c/R. plots_dir includes its trailing path separator.
    """

    cross_correlation: CrossCorrelationConfig
    plots_dir: str
    resolving_power: float | None
    sysrem_iterations: int
    exclude_around: str = "peak"
