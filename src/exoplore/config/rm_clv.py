"""JSON choices for stellar Rossiter-McLaughlin and centre-to-limb effects."""

from dataclasses import dataclass
import math


@dataclass
class RMCLVConfig:
    """Choose how the changing stellar spectrum enters the observation.

    simulate includes the stellar transit distortion before tellurics and noise.
    correct divides out an atmosphere-free stellar transit model. Seidel22
    applies this division after forming each exposure's transmission spectrum.
    Other supported preparations apply it to the flux and uncertainty first.

    effects selects "rm", "clv", or "rm_clv". The CLV control uses zero
    rotation. The RM control keeps the disk-centre line profile at every mu,
    with the continuum brightness variation supplied by the model synthesis.
    These choices let us inspect each stellar effect separately.
    coupling selects how local stellar light and planetary absorption combine.
    """

    simulate: bool = False
    correct: bool = False
    effects: str = "rm_clv"
    # factorized multiplies separately calculated stellar and planetary spectra.
    # spatial absorbs the local stellar light along rays through the planet's
    # atmosphere, then adds their contributions to the observed flux.
    coupling: str = "factorized"
    # This NPZ supplies local intensities across the stellar disk. Its arrays
    # are wavelength_um, mu, intensity and, for continuum normalization, continuum.
    # The input guide explains the required units and synthesis information.
    model_file: str = ""
    # matched uses the same star and opaque planet as the simulated observation.
    # factorized calculates a separate stellar correction. The supported choices
    # for each analysis are listed in docs/config_reference.md.
    correction_method: str = "matched"
    correction_model_file: str = ""
    wavelength_medium: str = "vacuum"
    # Supply the projected spin-orbit angle for RM. It is independent of the
    # orbital inclination already stored in the planet parameter file.
    projected_obliquity_deg: float | None = None
    cells_across_diameter: int = 200
    subpixels_per_axis: int = 1
    exposure_subsamples: int = 3
    # Spatial mode uses the atmospheric radii supplied by pRT. These counts set
    # the angular and radial sampling of the star and opaque planetary disk.
    azimuth_points: int = 128
    stellar_radial_points: int = 48
    planet_radial_points: int = 24
    # continuum divides by the integrated local continuum. band_mean divides
    # by the mean flux over the selected wavelength interval.
    normalization: str = "continuum"

    @property
    def active(self):
        """Whether the runner needs a stellar surface model."""
        return self.simulate or self.correct

    def __post_init__(self):
        if self.correction_method not in ("matched", "factorized"):
            raise ValueError("rm_clv.correction_method must be matched or factorized")
        if not isinstance(self.correction_model_file, str):
            raise ValueError("rm_clv.correction_model_file must be a path string")
        if self.correction_method == "matched" and self.correction_model_file:
            raise ValueError("Set correction_method=factorized to supply a separate correction model")
        for name in ("simulate", "correct"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"rm_clv.{name} must be true or false")
        if self.effects not in ("rm", "clv", "rm_clv"):
            raise ValueError("rm_clv.effects must be rm, clv or rm_clv")
        if self.coupling not in ("factorized", "spatial"):
            raise ValueError("rm_clv.coupling must be factorized or spatial")
        if self.normalization not in ("continuum", "band_mean"):
            raise ValueError("rm_clv.normalization must be continuum or band_mean")
        if self.wavelength_medium not in ("vacuum", "air"):
            raise ValueError("rm_clv.wavelength_medium must be vacuum or air")
        for name in ("cells_across_diameter", "subpixels_per_axis", "exposure_subsamples",
                     "azimuth_points", "stellar_radial_points", "planet_radial_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"rm_clv.{name} must be a positive integer")
        if self.projected_obliquity_deg is not None:
            value = self.projected_obliquity_deg
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("rm_clv.projected_obliquity_deg must be a finite angle")
        if self.active:
            if not isinstance(self.model_file, str) or not self.model_file:
                raise ValueError("Set rm_clv.model_file to a local specific-intensity model")
            if self.effects != "clv" and self.projected_obliquity_deg is None:
                raise ValueError("Set rm_clv.projected_obliquity_deg for the RM calculation")
