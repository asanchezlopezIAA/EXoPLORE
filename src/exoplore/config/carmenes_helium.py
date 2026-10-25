"""Explicit CARMENES helium preparation choices following Pallé et al. (2020).

The inherited fields describe shared transmission geometry, wavelengths and
exposure selections. Normalization and averaging are implemented separately
from the Czesla et al. (2024) CRIRES+ recipe.
"""

# Reading this file
# These settings describe the CARMENES direct-helium preparation branch.
# They reuse shared geometry/selection fields while keeping that branch's
# normalization and averaging choices separate from the CRIRES recipe. The
# configuration states the choices; pipelines/carmenes_helium.py applies them.

from dataclasses import dataclass
from exoplore.config.czesla2024 import Czesla2024Config


@dataclass(frozen=True)
class CarmenesHeliumConfig(Czesla2024Config):
    """One synthetic night; OH emission is explicitly absent in this version."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.monte_carlo_draws != 0:
            raise ValueError('CARMENES simulation uses one noise realization, no resampling')
        if self.continuum_polynomial_degree != 0:
            raise ValueError('Pallé et al. (2020) normalization uses a mean continuum')

