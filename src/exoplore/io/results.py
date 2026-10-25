"""Read and write compact CCF and detection-map products.

The private _combined_ccf_from_state function selects arrays from a completed
molecular run and constructs a CCFResult with their axes and observing metadata.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from exoplore.results import CCFResult, DetectionMapResult

__all__ = ["load_detection_map", "save_ccf_result", "load_ccf_result"]


def _result_array(value, copy: bool) -> np.ndarray:
    """Return a copy or a read-only view of the supplied array."""
    array = np.array(value, copy=True) if copy else np.asarray(value).view()
    if not copy:
        array.setflags(write=False)
    return array


def _combined_ccf_from_state(state, order_indices, metadata, *, night_index=None, copy=True) -> CCFResult:
    """Select a completed CCF and its axes from the simulator state."""
    raw = state.get("ccf_complete")
    if raw is None:
        raise RuntimeError("This run has no combined CCF; check the configured analysis route.")
    raw = np.asarray(raw)
    if raw.ndim == 2:
        if night_index is not None:
            raise ValueError("This CCF already combines nights sharing one geometry; omit night_index.")
        phase = state["phase"]
        selected = state["with_signal"]
        berv = state["berv"]
    elif raw.ndim == 3:
        # Different nights can have different exposure counts. Remove only the
        # storage padding explicitly identified by the runner's n_spectra.
        if not metadata.get("different_nights", False):
            raise NotImplementedError("A three-dimensional CCF without different nights requires explicit axis selection.")
        if night_index is None:
            raise ValueError("Different observing sequences require an explicit night_index.")
        if isinstance(night_index, bool) or not isinstance(night_index, (int, np.integer)):
            raise ValueError("night_index must be a non-negative integer.")
        if not 0 <= night_index < raw.shape[0]:
            raise ValueError("night_index is outside the available nights.")
        count = int(state["n_spectra"][night_index])
        raw = raw[night_index, :, :count]
        phase = state["phase"][night_index]
        selected = state["with_signal"][night_index]
        berv = state["berv"][night_index]
    else:
        # Injection/iteration dimensions must be selected by their own analysis.
        # silently squeezing them could expose the wrong scientific product.
        raise NotImplementedError("CCFs with injection/iteration axes are not supported by this accessor.")

    # Convert a single phase row to a one-dimensional exposure axis.
    # A remaining second dimension identifies multiple observing sequences.
    phase = np.asarray(phase).squeeze()
    if phase.ndim > 1:
        raise ValueError("Select one observing sequence before exposing its phase axis.")
    phase = phase.reshape(-1)
    berv = np.asarray(berv).squeeze()
    details = deepcopy(metadata)
    details["night_index"] = None if night_index is None else int(night_index)
    return CCFResult(
        ccf=_result_array(raw, copy),
        velocity_kms=_result_array(state["v_ccf"], copy),
        phase=_result_array(phase, copy),
        signal_exposure_indices=_result_array(selected, copy),
        berv_kms=_result_array(berv, copy),
        order_indices=tuple(order_indices), metadata=details,
    )


def load_detection_map(path, *, metric="snr") -> DetectionMapResult:
    """Load the simulator's explicitly named S/N or Welch NPZ map.

    Array orientation and normalization come from the archive. A missing Vrest
    convention is labelled unspecified.
    """
    keys = {"snr": "ccf_tot_sn", "welch": "ccf_tot_welch"}
    if metric not in keys:
        raise ValueError("Detection-map metric must be 'snr' or 'welch'.")
    with np.load(path, allow_pickle=False) as archive:
        values = archive[keys[metric]].copy()
        if values.ndim != 2:
            raise NotImplementedError("Select injection/iteration axes before loading a two-dimensional map.")
        return DetectionMapResult(
            values=values, v_rest_kms=archive["v_rest"].copy(),
            kp_kms=archive["kp_range"].copy(), metric=metric,
            order_indices=tuple(int(i) for i in archive["order_indices"])
            if "order_indices" in archive else (),
            v_rest_convention=str(archive["v_rest_convention"].item())
            if "v_rest_convention" in archive else "unspecified",
        )


def save_ccf_result(result: CCFResult, path, *, overwrite=False) -> Path:
    """Save the CCF and its conventions. preserve existing files by default.

    The NPZ stores CCF, phase, velocity and JSON metadata arrays.
    Detection-map grids are saved as separate products.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Use a .npz filename for the compact CCF product.")
    metadata = deepcopy(result.metadata)
    metadata.update({"axis_order": ["earth_frame_velocity", "exposure"],
                     "velocity_units": "km/s",
                     "phase_units": "orbital cycles; zero at mid-transit",
                     "order_indices": list(result.order_indices)})
    if metadata.get("significant_eccentricity") is False:
        metadata["track_equation"] = "Kp*sin(2*pi*phase) + Vsys - BERV + Vrest"
    # Encode before opening the destination so invalid metadata cannot truncate it.
    encoded = json.dumps(metadata)
    with path.open("wb" if overwrite else "xb") as handle:
        np.savez_compressed(handle, ccf=result.ccf, velocity=result.velocity_kms,
                            phase=result.phase, with_signal=result.signal_exposure_indices,
                            berv=result.berv_kms, metadata_json=encoded)
    return path


def load_ccf_result(path) -> CCFResult:
    """Load CCF, phase, velocity and metadata from a compact NPZ export."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        axes = metadata.get("axis_order")
        if axes != ["earth_frame_velocity", "exposure"]:
            raise ValueError("CCF archive must declare axes (earth_frame_velocity, exposure).")
        if metadata.get("velocity_units") != "km/s":
            raise ValueError("CCF archive must declare velocity units km/s.")
        return CCFResult(
            ccf=archive["ccf"].copy(), velocity_kms=archive["velocity"].copy(),
            phase=archive["phase"].copy(), signal_exposure_indices=archive["with_signal"].copy(),
            berv_kms=archive["berv"].copy(),
            order_indices=tuple(int(i) for i in metadata.get("order_indices", [])),
            metadata=metadata,
        )
