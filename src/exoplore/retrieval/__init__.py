"""
exoplore.retrieval
==================

Bayesian atmospheric retrieval from high-resolution spectroscopic
time series.

Modules
-------
likelihood
    Log-likelihood functions: BL19, BLASP24, G22.
priors
    Prior transform functions and the PriorSet container.
samplers
    Wrappers for pymultinest (nested sampling) and emcee (MCMC).
"""

from exoplore.retrieval.likelihood import (
    log_likelihood_bl19,
    log_likelihood_blasp24,
    log_likelihood_g22,
    compute_log_likelihood,
    LIKELIHOOD_REGISTRY,
)
from exoplore.retrieval.priors import (
    UniformPrior,
    GaussianPrior,
    LogUniformPrior,
    PriorSet,
    standard_1d_prior_set,
)
from exoplore.retrieval.samplers import (
    run_multinest,
    run_emcee,
    posterior_summary,
)

__all__ = [
    # likelihood
    "log_likelihood_bl19",
    "log_likelihood_blasp24",
    "log_likelihood_g22",
    "compute_log_likelihood",
    "LIKELIHOOD_REGISTRY",
    # priors
    "UniformPrior",
    "GaussianPrior",
    "LogUniformPrior",
    "PriorSet",
    "standard_1d_prior_set",
    # samplers
    "run_multinest",
    "run_emcee",
    "posterior_summary",
]
