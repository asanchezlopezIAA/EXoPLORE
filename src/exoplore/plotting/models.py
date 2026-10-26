"""Labels, velocity ranges and output locations for detection figures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionPlotSettings:
    """Shared presentation settings for CCF trails and detection maps.

    Velocities and velocity spacings are in km/s. reference_kp_kms marks
    the configured orbital amplitude. plots_dir includes its trailing path
    separator because the plotting functions append their output filenames.
    sysrem_iterations selects the panel count for optimisation figures.
    """

    simulation_name: str
    plots_dir: str
    reference_kp_kms: float
    kp_max_kms: float
    velocity_half_range_kms: float
    velocity_tick_step_kms: float
    sysrem_iterations: int
    optimize_sysrem: bool
    event_type: str
