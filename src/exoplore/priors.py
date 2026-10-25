"""Compatibility imports for :mod:`exoplore.retrieval.priors`.

Prior definitions belong with retrieval because they map sampler coordinates
to atmospheric/orbital parameters. Existing scripts and saved Python priors
can still resolve their original top-level names.

Maintain the implementation in ``exoplore.retrieval.priors``; this module
only preserves the older public entry points.
"""

# Import the original objects directly rather than wrapping them: signatures,
# defaults and numerical calculations then stay identical through either path.
from exoplore.retrieval.priors import (
    UniformPrior,
    GaussianPrior,
    LogUniformPrior,
    PriorSet,
    standard_1d_prior_set,
)

__all__ = ['UniformPrior', 'GaussianPrior', 'LogUniformPrior', 'PriorSet', 'standard_1d_prior_set']
