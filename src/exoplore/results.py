"""Named scientific products, independent of the simulator and plotting code.

These containers describe arrays that have already been calculated. They do
not prepare spectra, combine CCFs, transpose maps or normalize statistics.
Keeping this module independent lets IO, plotting and orchestration share the
same conventions without importing each other.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CCFResult", "DetectionMapResult"]

# NumPy arrays do not have scalar equality semantics. eq=False avoids a
# misleading automatically generated dataclass comparison of entire datasets.

@dataclass(frozen=True, eq=False)
class CCFResult:
    """One combined Earth-frame CCF time series.

    ``ccf`` has axes **(velocity, exposure)**, preserving the simulator's
    existing layout. ``phase`` is in orbital cycles, zero at mid-transit.
    ``signal_exposure_indices`` identifies the exposures selected for the
    planetary-signal analysis; it is an index array, not a boolean mask.
    ``berv_kms`` is scalar or one value per exposure. All velocities are km/s.

    For nights sharing one geometry, the CCF may already be coadded across
    nights. For different geometries, the accessor returns a selected night.
    ``metadata`` holds extensible provenance, not additional numerical axes.

    Frozen fields prevent accidental reassignment; NumPy arrays themselves
    remain mutable. Simulator accessors copy by default. Their ``copy=False``
    option returns read-only views instead, which still reflect source changes.
    """

    ccf: np.ndarray
    velocity_kms: np.ndarray
    phase: np.ndarray
    signal_exposure_indices: np.ndarray
    berv_kms: np.ndarray
    order_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Check axes at this public boundary; never guess a transpose from shape.
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
    ``welch``; values are read as saved, without a new normalization. The
    archive's Vrest convention is retained, or marked unspecified when absent.

    Peak properties locate a maximum pixel; they provide no uncertainty or
    false-alarm calibration. Ties follow NumPy's first-maximum convention.
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
        """Return (Vrest index, Kp index), ignoring NaNs; reject an all-NaN map."""
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
