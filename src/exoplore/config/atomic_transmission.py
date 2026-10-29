"""Inputs for a direct synthetic sodium transmission observation."""
from dataclasses import dataclass, field
import math


@dataclass
class AtomicTransmissionConfig:
    """Choose the observing sequence and reduction settings for sodium.

    Atmospheric properties remain in atmosphere.planet_model. The observing
    sequence NPZ supplies bjd and airmass. Instrument files supply wavelength
    pixels and resolving power. All wavelength intervals here use vacuum units.
    """
    # Enable the direct sodium simulation and Seidel22 transmission analysis.
    enabled: bool = False
    # Each row represents one exposure midpoint in BJD_TDB and its airmass.
    observing_sequence_file: str = ''
    # Retain this interval from the selected instrument order, in micrometres.
    wavelength_interval_um: list[float] = field(default_factory=list)
    # Calculate the pRT spectrum over a wider interval to cover Doppler shifts
    # and instrumental convolution near the observation's wavelength edges.
    model_wavelength_limits_um: list[float] = field(default_factory=list)
    # Give the Na D2 and D1 rest wavelengths in increasing vacuum nanometres.
    line_wavelengths_nm: list[float] = field(default_factory=list)
    # Longitude and latitude in degrees, then height in metres. Molecfit needs
    # the site and exposure time when calculating terrestrial absorption.
    observatory_lon_lat_height: list[float] = field(default_factory=list)
    # Average each line within this velocity distance from its planetary centre.
    band_half_width_kms: float = 20.
    # Limit simultaneous external fits to suit the available CPU and memory.
    molecfit_workers: int = 1
    # Set fitting intervals, molecules, instrumental width and acceptance limits.
    molecfit: dict = field(default_factory=dict)
    # Record the exposure time used for the ANDES ETC table. The direct runner
    # checks it against the simulation exposure time before generating spectra.
    etc_reference_exposure_seconds: float | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('atomic_transmission.enabled must be boolean')
        if not self.enabled:
            return
        if self.etc_reference_exposure_seconds is not None and (
                not math.isfinite(self.etc_reference_exposure_seconds)
                or self.etc_reference_exposure_seconds <= 0):
            raise ValueError('etc_reference_exposure_seconds must be positive')
        for name in ['wavelength_interval_um', 'model_wavelength_limits_um']:
            values = getattr(self, name)
            if (len(values) != 2 or not all(math.isfinite(v) for v in values)
                    or not 0 < values[0] < values[1]):
                raise ValueError(f'atomic_transmission.{name} requires increasing positive bounds')
        if len(self.line_wavelengths_nm) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.line_wavelengths_nm) or self.line_wavelengths_nm[0] >= self.line_wavelengths_nm[1]:
            raise ValueError('Supply the two sodium line wavelengths in increasing vacuum nm')
        if len(self.observatory_lon_lat_height) != 3 or not all(math.isfinite(v) for v in self.observatory_lon_lat_height):
            raise ValueError('Supply observatory longitude, latitude and height')
        if type(self.molecfit_workers) is not int or not 1 <= self.molecfit_workers <= 16:
            raise ValueError('Use one to sixteen Molecfit workers')
        if not math.isfinite(self.band_half_width_kms) or self.band_half_width_kms <= 0:
            raise ValueError('band_half_width_kms must be positive')
        if not self.observing_sequence_file or not self.molecfit:
            raise ValueError('Supply observing_sequence_file and Molecfit settings')
