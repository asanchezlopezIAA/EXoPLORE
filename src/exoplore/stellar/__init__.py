"""Stellar surface spectra and their occultation during a planetary transit.

The surface model integrates local specific intensities over the projected
stellar disk. The observing runner supplies the planet positions and velocities.
"""

# Expose the local-intensity grid, projected surface and integrated stellar spectra
# through exoplore.stellar for callers constructing a stellar transit model.
from .surface import LocalIntensityGrid, StellarSurface, StellarTransitSpectra

# List the objects available through this package interface.
__all__ = ["LocalIntensityGrid", "StellarSurface", "StellarTransitSpectra"]
