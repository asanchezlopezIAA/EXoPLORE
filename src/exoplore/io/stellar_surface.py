"""Read specific-intensity spectra and their angular and synthesis metadata.

The RM/CLV model consumes this validated grid in exoplore.stellar. Input-file
validation is kept here with the other scientific data readers.


stellar/observation.py calls this before constructing the rotating star.
LocalIntensityGrid checks wavelength order, angular coverage and array shapes.
This reader checks the file metadata that identifies the synthesis.
"""

import numpy as np

from exoplore.stellar.surface import LocalIntensityGrid


def load_local_intensities(path, *, wavelength_medium):
    """Read explicit angular spectra and provenance from a numeric NPZ bundle.

    Input arrays are wavelength_um, mu, intensity and optional continuum.
    intensity_kind must be 'specific_intensity'. wavelength_medium is 'air'
    or 'vacuum'. intensity_unit and source are nonempty scalar strings.
    resolving_power is zero for an unconvolved synthesis or a finite positive
    resolving power for a previously convolved synthesis.
    """
    # Open numerical arrays and scalar metadata from the NPZ archive.
    with np.load(path, allow_pickle=False) as saved:
        # Require the coordinates, intensities and information needed to interpret them.
        required = {"wavelength_um", "mu", "intensity", "intensity_kind",
                    "wavelength_medium", "intensity_unit", "source", "resolving_power"}
        # List required fields absent from this file.
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"RM/CLV bundle lacks {sorted(missing)}")
        # Read one scalar value for each descriptive field.
        metadata = {}
        # Check that each field describes the whole synthesis.
        for name in ("intensity_kind", "wavelength_medium", "intensity_unit", "source"):
            if saved[name].shape != ():
                raise ValueError(f"RM/CLV metadata {name} must be scalar")
            metadata[name] = str(saved[name].item())
        # The transit calculation needs the angular dependence of specific intensity.
        if metadata["intensity_kind"] != "specific_intensity":
            raise ValueError("RM/CLV needs local specific intensities with relative angular brightness")
        # Require agreement between air or vacuum coordinates in the file and JSON.
        if metadata["wavelength_medium"] != wavelength_medium:
            raise ValueError("RM/CLV model wavelength convention differs from rm_clv.wavelength_medium")
        # Keep the synthesis source and intensity unit available for interpretation.
        if not metadata["source"].strip() or not metadata["intensity_unit"].strip():
            raise ValueError("RM/CLV model needs its synthesis source and intensity units")
        # Zero identifies an intrinsic synthesis. A positive value records prior broadening.
        resolution = float(saved["resolving_power"].item())
        if not np.isfinite(resolution) or resolution < 0:
            raise ValueError("RM/CLV resolving_power must be zero or finite and positive")
        # Construct the validated wavelength-by-angle input grid. The continuum
        # retains its physical brightness relative to the line intensities.
        local = LocalIntensityGrid(saved["wavelength_um"], saved["mu"], saved["intensity"],
                                   saved["continuum"] if "continuum" in saved.files else None)
    # Pass the spectra, resolution and metadata to the stellar observation model.
    return local, resolution, metadata
