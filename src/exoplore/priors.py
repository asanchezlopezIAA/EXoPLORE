"""
exoplore.retrieval.priors
=========================

Prior transformation functions for atmospheric retrieval.

In nested sampling (MultiNest / dynesty), the prior transform maps
the unit hypercube ``[0, 1]^N`` to physical parameter space.
In emcee (MCMC), priors are evaluated as ``log_prior(params)``.

This module provides:

- Uniform prior transform functions (unit-cube → physical).
- Log-Gaussian prior functions.
- A :class:`PriorSet` container that collects per-parameter priors
  and builds the full ``prior_transform(cube)`` function expected by
  ``pymultinest``.

Typical usage (MultiNest)
-------------------------
>>> from exoplore.retrieval.priors import UniformPrior, PriorSet
>>> priors = PriorSet()
>>> priors.add("log10_H2O", UniformPrior(-8.0, -1.0))
>>> priors.add("T_equ",     UniformPrior(500.0, 3000.0))
>>> priors.add("K_p",       UniformPrior(50.0, 250.0))
>>> pt = priors.prior_transform  # pass to pymultinest.run(LogLikelihood=..., Prior=pt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


@dataclass
class UniformPrior:
    """Uniform distribution between ``low`` and ``high``.

    Maps unit-cube value *u* ∈ [0, 1] to *p* ∈ [low, high].

    Parameters
    ----------
    low, high:
        Bounds of the uniform distribution.
    """
    low: float
    high: float

    def __call__(self, u: float) -> float:
        return self.low + (self.high - self.low) * u

    def log_prior(self, value: float) -> float:
        """Log prior (0 inside bounds, -inf outside)."""
        if self.low <= value <= self.high:
            return -math.log(self.high - self.low)
        return -math.inf


@dataclass
class GaussianPrior:
    """Gaussian (normal) prior.

    Parameters
    ----------
    mean, sigma:
        Mean and standard deviation.
    """
    mean: float
    sigma: float

    def __call__(self, u: float) -> float:
        """Unit-cube → physical via the normal percent-point function."""
        from scipy.special import ndtri
        return self.mean + self.sigma * ndtri(u)

    def log_prior(self, value: float) -> float:
        """Log-Gaussian prior."""
        z = (value - self.mean) / self.sigma
        return -0.5 * z ** 2 - math.log(self.sigma * math.sqrt(2 * math.pi))


@dataclass
class LogUniformPrior:
    """Log-uniform prior: uniform in log₁₀ space.

    Maps *u* ∈ [0, 1] to *p* ∈ [10^log_low, 10^log_high].

    Parameters
    ----------
    log_low, log_high:
        Bounds in log₁₀ space.
    """
    log_low: float
    log_high: float

    def __call__(self, u: float) -> float:
        log_val = self.log_low + (self.log_high - self.log_low) * u
        return 10.0 ** log_val

    def log_prior(self, value: float) -> float:
        if value <= 0:
            return -math.inf
        log_val = math.log10(value)
        if self.log_low <= log_val <= self.log_high:
            return -math.log(
                (self.log_high - self.log_low) * value * math.log(10)
            )
        return -math.inf


# ---------------------------------------------------------------------------
# PriorSet container
# ---------------------------------------------------------------------------


class PriorSet:
    """Ordered collection of per-parameter priors.

    Parameters are added in order; the resulting prior transform
    function maps ``cube[i]`` to the i-th physical parameter.

    Examples
    --------
    >>> ps = PriorSet()
    >>> ps.add("log10_H2O", UniformPrior(-8, -1))
    >>> ps.add("T_equ",     UniformPrior(500, 3000))
    >>> ps.add("K_p",       UniformPrior(50, 250))
    >>> physical = ps.prior_transform([0.5, 0.5, 0.5])
    >>> ps.param_names
    ['log10_H2O', 'T_equ', 'K_p']
    """

    def __init__(self) -> None:
        self._names: List[str] = []
        self._priors: List[Callable] = []

    def add(self, name: str, prior: Callable) -> None:
        """Add a named parameter with its prior.

        Parameters
        ----------
        name:
            Parameter name (used for labelling outputs).
        prior:
            A callable that maps a unit-cube value *u* ∈ [0, 1] to
            the physical parameter value.
        """
        self._names.append(name)
        self._priors.append(prior)

    @property
    def n_params(self) -> int:
        """Number of parameters in this prior set."""
        return len(self._names)

    @property
    def param_names(self) -> List[str]:
        """List of parameter names in insertion order."""
        return list(self._names)

    def prior_transform(self, cube: np.ndarray) -> np.ndarray:
        """Transform unit-cube vector to physical parameters.

        Compatible with ``pymultinest.run(Prior=ps.prior_transform)``.

        Parameters
        ----------
        cube:
            Array of length ``n_params`` with values in [0, 1].

        Returns
        -------
        np.ndarray
            Physical parameter values, same length.
        """
        cube = np.asarray(cube, dtype=float)
        result = np.empty_like(cube)
        for i, prior in enumerate(self._priors):
            result[i] = prior(cube[i])
        return result

    def log_prior(self, params: np.ndarray) -> float:
        """Sum of log-priors for all parameters.

        Used with emcee.

        Parameters
        ----------
        params:
            Physical parameter values, length ``n_params``.

        Returns
        -------
        float
            Sum of log-priors, or ``-inf`` if any parameter is outside
            its prior bounds.
        """
        total = 0.0
        for prior, val in zip(self._priors, params):
            if hasattr(prior, "log_prior"):
                lp = prior.log_prior(val)
            else:
                # Fallback: evaluate transform at 0.5 (not truly correct
                # for all prior types, but safe as a dummy).
                lp = 0.0
            if not math.isfinite(lp):
                return -math.inf
            total += lp
        return total


# ---------------------------------------------------------------------------
# Convenience: standard EXoPLORE parameter sets
# ---------------------------------------------------------------------------


def standard_1d_prior_set(
    log10_vmr_range: Tuple[float, float] = (-8.0, -1.0),
    kp_range: Tuple[float, float] = (50.0, 300.0),
    t_equ_range: Tuple[float, float] = (500.0, 3000.0),
    v_wind_range: Tuple[float, float] = (-30.0, 30.0),
    beta_range: Tuple[float, float] = (0.01, 100.0),
    include_beta: bool = False,
) -> PriorSet:
    """Build a standard 1D retrieval prior set.

    Parameters are (in order): ``log10_X``, ``K_p``, ``T_equ``,
    ``v_wind``, and optionally ``beta``.

    Parameters
    ----------
    log10_vmr_range:
        (min, max) for the log₁₀ volume mixing ratio.
    kp_range:
        (min, max) for Kp in km/s.
    t_equ_range:
        (min, max) for the equilibrium temperature in K.
    v_wind_range:
        (min, max) for the wind velocity in km/s.
    beta_range:
        (min, max) for the noise scaling β.
    include_beta:
        If True, add a beta parameter (for Gibson22 log-likelihood).

    Returns
    -------
    PriorSet
    """
    ps = PriorSet()
    ps.add("log10_X", UniformPrior(*log10_vmr_range))
    ps.add("K_p", UniformPrior(*kp_range))
    ps.add("T_equ", UniformPrior(*t_equ_range))
    ps.add("v_wind", UniformPrior(*v_wind_range))
    if include_beta:
        ps.add("beta", UniformPrior(*beta_range))
    return ps
