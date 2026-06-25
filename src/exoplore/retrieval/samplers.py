"""
exoplore.retrieval.samplers
===========================

Thin wrappers around MultiNest (pymultinest) and emcee for atmospheric
retrieval.

Both samplers are optional dependencies.  If not installed, the
functions raise ``ImportError`` with a clear message.

MultiNest (nested sampling)
---------------------------
Use :func:`run_multinest` to run nested sampling.  The function wraps
``pymultinest.run`` and saves the posterior samples in the standard
MultiNest output directory.

emcee (MCMC)
------------
Use :func:`run_emcee` to run affine-invariant MCMC.  Returns the
``EnsembleSampler`` object for direct access to the chain and flatchain.

Both functions accept a ``log_posterior`` callable:

.. code-block:: python

    def log_posterior(params: np.ndarray) -> float:
        log_p = priors.log_prior(params)
        if not np.isfinite(log_p):
            return -np.inf
        return log_p + compute_log_likelihood(...)

Examples
--------
MultiNest:

>>> from exoplore.retrieval.samplers import run_multinest
>>> run_multinest(
...     log_likelihood=my_log_L,
...     prior_transform=my_prior_transform,
...     n_params=4,
...     output_dir="output/retrieval/",
...     n_live_points=400,
... )

emcee:

>>> from exoplore.retrieval.samplers import run_emcee
>>> sampler = run_emcee(
...     log_posterior=my_log_posterior,
...     n_params=4,
...     n_walkers=32,
...     n_steps=2000,
...     initial_params=np.random.randn(32, 4),
... )
>>> flat_chain = sampler.get_chain(discard=200, flat=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# MultiNest
# ---------------------------------------------------------------------------


def run_multinest(
    log_likelihood: Callable,
    prior_transform: Callable,
    n_params: int,
    output_dir: str | Path,
    n_live_points: int = 400,
    evidence_tolerance: float = 0.5,
    sampling_efficiency: float = 0.3,
    max_iterations: int = 0,
    multimodal: bool = False,
    verbose: bool = True,
    resume: bool = False,
    run_name: str = "retrieval",
) -> dict:
    """Run MultiNest nested sampling via pymultinest.

    Parameters
    ----------
    log_likelihood:
        Callable ``f(params) → float`` returning the log-likelihood.
        Receives a 1-D NumPy array of physical parameter values.
    prior_transform:
        Callable ``f(cube) → cube`` that transforms the unit hypercube
        in-place (pymultinest convention) or returns the transformed
        array.
    n_params:
        Number of free parameters.
    output_dir:
        Directory where MultiNest output files are written.
    n_live_points:
        Number of live points.  Higher → more accurate evidence,
        slower convergence.
    evidence_tolerance:
        Stopping criterion Δ ln Z < ``evidence_tolerance``.
    sampling_efficiency:
        MultiNest sampling efficiency.  0.3 is recommended for
        parameter estimation; 0.8 for evidence.
    max_iterations:
        Maximum number of iterations (0 = unlimited).
    multimodal:
        Enable multimodal sampling.
    verbose:
        Print progress to stdout.
    resume:
        Resume from a previous run.
    run_name:
        Base name for MultiNest output files.

    Returns
    -------
    dict
        Dictionary with keys ``"log_evidence"``, ``"log_evidence_error"``,
        ``"output_dir"``, and ``"n_params"``.

    Raises
    ------
    ImportError
        If ``pymultinest`` is not installed.
    """
    try:
        import pymultinest
    except ImportError as exc:
        raise ImportError(
            "pymultinest is required for run_multinest().\n"
            "Installation instructions: https://johannesbuchner.github.io/PyMultiNest/install.html\n"
            "or:  pip install exoplore[retrieval]"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / run_name) + "_"

    # pymultinest requires the prior transform to modify cube in-place
    def _prior(cube, ndim, nparams):
        result = prior_transform(np.array([cube[i] for i in range(ndim)]))
        for i in range(ndim):
            cube[i] = result[i]

    def _loglike(cube, ndim, nparams):
        params = np.array([cube[i] for i in range(ndim)])
        return float(log_likelihood(params))

    pymultinest.run(
        LogLikelihood=_loglike,
        Prior=_prior,
        n_dims=n_params,
        outputfiles_basename=output_prefix,
        n_live_points=n_live_points,
        evidence_tolerance=evidence_tolerance,
        sampling_efficiency=sampling_efficiency,
        max_iter=max_iterations,
        multimodal=multimodal,
        verbose=verbose,
        resume=resume,
    )

    # Read evidence from summary file
    result = {
        "output_dir": str(output_dir),
        "n_params": n_params,
        "log_evidence": None,
        "log_evidence_error": None,
    }
    try:
        analyser = pymultinest.Analyser(n_params=n_params, outputfiles_basename=output_prefix)
        stats = analyser.get_stats()
        result["log_evidence"] = stats["global evidence"]
        result["log_evidence_error"] = stats["global evidence error"]
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# emcee
# ---------------------------------------------------------------------------


def run_emcee(
    log_posterior: Callable,
    n_params: int,
    n_walkers: int,
    n_steps: int,
    initial_params: np.ndarray,
    progress: bool = True,
    pool=None,
) -> "emcee.EnsembleSampler":
    """Run affine-invariant MCMC via emcee.

    Parameters
    ----------
    log_posterior:
        Callable ``f(params) → float`` returning log-posterior.
    n_params:
        Number of free parameters.
    n_walkers:
        Number of MCMC walkers.  Must be ≥ 2 × n_params.
    n_steps:
        Total number of MCMC steps per walker.
    initial_params:
        Starting positions, shape ``(n_walkers, n_params)``.
    progress:
        Show tqdm progress bar.
    pool:
        Optional multiprocessing pool for parallel log-posterior
        evaluations.

    Returns
    -------
    emcee.EnsembleSampler
        The sampler object after the run.  Use
        ``sampler.get_chain(discard=burn_in, flat=True)`` to access
        the posterior samples.

    Raises
    ------
    ImportError
        If ``emcee`` is not installed.
    ValueError
        If ``n_walkers < 2 * n_params``.
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee is required for run_emcee().\n"
            "Install with:  pip install emcee\n"
            "or:            pip install exoplore[retrieval]"
        ) from exc

    if n_walkers < 2 * n_params:
        raise ValueError(
            f"n_walkers ({n_walkers}) must be >= 2 * n_params ({2 * n_params})."
        )

    initial_params = np.asarray(initial_params, dtype=float)
    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior, pool=pool
    )
    sampler.run_mcmc(initial_params, n_steps, progress=progress)
    return sampler


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------


def posterior_summary(
    flat_chain: np.ndarray,
    param_names: Sequence[str],
    quantiles: tuple = (0.16, 0.50, 0.84),
) -> dict:
    """Compute quantile-based summary statistics from a flat MCMC chain.

    Parameters
    ----------
    flat_chain:
        Flat posterior sample array, shape ``(n_samples, n_params)``.
    param_names:
        List of parameter names, length ``n_params``.
    quantiles:
        Quantiles to compute.  Default gives 1σ credible intervals.

    Returns
    -------
    dict
        Mapping ``param_name → dict(q16, q50, q84)`` (or whatever
        quantiles were requested).
    """
    flat_chain = np.asarray(flat_chain)
    summary = {}
    for i, name in enumerate(param_names):
        vals = np.percentile(flat_chain[:, i], [100 * q for q in quantiles])
        summary[name] = {
            f"q{int(100*q):02d}": float(v)
            for q, v in zip(quantiles, vals)
        }
    return summary
