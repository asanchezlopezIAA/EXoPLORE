"""Mutable values retained while evaluating atmospheric retrieval proposals."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class LimbRetrievalState:
    """Regional abundances retained from the latest likelihood evaluation.

    Each sequence follows its region's species order. The likelihood replaces
    the abundance sequences for each proposal. Post-sampling diagnostic plots
    read these retained values.
    """

    morning_abundances: Sequence[float]
    evening_abundances: Sequence[float]
