"""Completed CCFs and detection maps with named arrays, axes and units.

The simulator creates these objects from its calculated products. File readers
and plotting functions use the same array conventions.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CCFResult", "DetectionMapResult"]

# NumPy compares arrays element by element. eq=False keeps equality checks
# on these result objects separate from comparisons of their array contents.

@dataclass(frozen=True, eq=False)
class CCFResult:
    """One combined Earth-frame CCF time series.

    ``ccf`` has axes **(velocity, exposure)**. ``phase`` is in orbital cycles, zero at mid-transit.
    ``signal_exposure_indices`` identifies the exposures selected for the
    planetary-signal analysis as an integer index array.
    ``berv_kms`` is scalar or one value per exposure. All velocities are km/s.

    For nights sharing one geometry, the CCF may already be coadded across
    nights. For different geometries, the accessor returns a selected night.
    ``metadata`` records the observing settings and velocity conventions.

    Frozen fields prevent accidental reassignment. NumPy arrays themselves
    remain mutable. Simulator accessors copy by default. Their ``copy=False``
    option returns read-only views that reflect source changes.
    """

    ccf: np.ndarray
    velocity_kms: np.ndarray
    phase: np.ndarray
    signal_exposure_indices: np.ndarray
    berv_kms: np.ndarray
    order_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Require the declared (velocity, exposure) shape and matching axes.
        if self.velocity_kms.ndim != 1 or self.phase.ndim != 1:
            raise ValueError("CCF velocity and phase axes must be one-dimensional.")
        if self.ccf.shape != (self.velocity_kms.size, self.phase.size):
            raise ValueError("CCF axes must be (velocity, exposure).")
        if self.berv_kms.ndim != 0 and self.berv_kms.shape != self.phase.shape:
            raise ValueError("BERV must be scalar or have one value per exposure.")
        selected = self.signal_exposure_indices
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise ValueError("Signal exposure selection must be a one-dimensional integer index array.")
        if np.any(selected < 0) or np.any(selected >= self.phase.size):
            raise ValueError("Signal exposure index is outside the phase axis.")


@dataclass(frozen=True, eq=False)
class DetectionMapResult:
    """A saved detection statistic with axes **(Vrest, Kp)**.

    Both velocity axes are in km/s. ``metric`` distinguishes ``snr`` from
    ``welch``. Values and the Vrest convention come from the saved archive.
    A missing convention is labelled unspecified.

    Peak properties give the coordinates and value of the maximum pixel.
    Uncertainties and false-alarm probabilities require statistical inference. Ties follow NumPy's first-maximum convention.
    Arrays loaded from an archive are independent of simulator working arrays.
    """

    values: np.ndarray
    v_rest_kms: np.ndarray
    kp_kms: np.ndarray
    metric: str
    order_indices: tuple[int, ...] = ()
    v_rest_convention: str = "unspecified"

    def __post_init__(self):
        if self.metric not in ("snr", "welch"):
            raise ValueError("Detection-map metric must be 'snr' or 'welch'.")
        if self.v_rest_kms.ndim != 1 or self.kp_kms.ndim != 1:
            raise ValueError("Detection-map velocity axes must be one-dimensional.")
        if self.values.shape != (self.v_rest_kms.size, self.kp_kms.size):
            raise ValueError("Detection-map axes must be (Vrest, Kp).")

    @property
    def peak_indices(self) -> tuple[int, int]:
        """Return (Vrest index, Kp index), ignoring NaNs. Reject an all-NaN map."""
        return tuple(int(i) for i in np.unravel_index(
            np.nanargmax(self.values), self.values.shape
        ))

    @property
    def peak_value(self) -> float:
        """Maximum of the saved statistic, with its original normalization."""
        return float(self.values[self.peak_indices])

    @property
    def peak_kp_kms(self) -> float:
        """Kp coordinate of that maximum pixel, in km/s."""
        return float(self.kp_kms[self.peak_indices[1]])

    @property
    def peak_v_rest_kms(self) -> float:
        """Vrest coordinate of that maximum pixel, in km/s."""
        return float(self.v_rest_kms[self.peak_indices[0]])
