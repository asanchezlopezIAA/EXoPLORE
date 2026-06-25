"""
exoplore.plotting.detection
===========================

.. deprecated::
    All detection-plot functions have been consolidated into
    :mod:`exoplore.plotting.kpvsys` and :mod:`exoplore.plotting.matrices`.
    Import from those modules or from :mod:`exoplore.plotting` directly.

This module is kept as a thin re-export shim for backwards compatibility.
"""

from exoplore.plotting.kpvsys import (
    plot_Kp_Vrest,
    plot_1D_CCF,
    plot_combined_KpVrest_1DCCF,
)

__all__ = ["plot_Kp_Vrest", "plot_1D_CCF", "plot_combined_KpVrest_1DCCF"]
