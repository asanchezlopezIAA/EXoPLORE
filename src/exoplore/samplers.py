"""Compatibility imports for :mod:`exoplore.retrieval.samplers`.

Sampler adapters belong with retrieval, alongside the likelihood and priors
they consume. The original top-level import remains valid for existing scripts.
Optional sampler dependencies are still imported only when a sampler is run.

Maintain the implementation in ``exoplore.retrieval.samplers``; this module
only preserves the older public entry points.
"""

# Import the original objects directly rather than wrapping them: signatures,
# defaults and numerical calculations then stay identical through either path.
from exoplore.retrieval.samplers import (
    run_multinest,
    run_emcee,
    posterior_summary,
)

__all__ = ['run_multinest', 'run_emcee', 'posterior_summary']
