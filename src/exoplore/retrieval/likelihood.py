"""
exoplore.retrieval.likelihood
=============================

Log-likelihood functions for atmospheric retrieval from high-resolution
spectroscopic time series.

The JSON field ``retrieval.log_likelihood`` selects the formulation:

``BL19``
    Brogi & Line (2019): scale-invariant log-likelihood based on the
    cross-covariance between data and model.
    Suitable when absolute flux calibration is unknown.

``Blain24``
    Blain, Sánchez-López & Mollière (2024): chi-squared log-likelihood
    using propagated per-pixel uncertainties.
    Requires well-calibrated uncertainties.

``Blain24_beta``
    Blain et al. (2024), Appendix G P-19: propagated uncertainties scaled
    by a fitted global β, including the Gaussian normalization ``-N ln β``.

``Gibson22``
    Gibson et al. (2022)-style: chi-squared with a global noise-scaling
    parameter β, plus a β penalty term ``-N ln β``.
    Accounts for under/over-estimated uncertainties.


All functions accept arrays of shape ``(n_in_transit, n_good_pixels)``
and return a scalar log-likelihood value.

These functions evaluate supplied data and model arrays. core/simulator.py
also defines likelihood functions inside its retrieval setup.

References
----------
Brogi, M. & Line, M. R. (2019), AJ, 157, 114.
Blain, D., Sánchez-López, A. & Mollière, P. (2024), AJ.
Gibson, N. P., et al. (2022), MNRAS, 512, 4618.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# BL19 log-likelihood
# ---------------------------------------------------------------------------


def log_likelihood_bl19(
    data_matrix: np.ndarray,
    model_matrix: np.ndarray,
) -> float:
    """Brogi & Line (2019) scale-invariant log-likelihood.

    For each frame (spectrum), computes:

    .. math::

        \\log\\mathcal{L} = -\\frac{N}{2}
            \\ln\\!\\bigl(s_f^2 - 2R + s_g^2\\bigr)

    where :math:`s_f^2` and :math:`s_g^2` are the mean squared data
    and model, and :math:`R` is their cross-covariance.
    Contributions are summed over all in-transit frames.

    Parameters
    ----------
    data_matrix:
        Residual data after pipeline cleaning, mean-subtracted per
        frame.  Shape ``(n_in_transit, n_good_pixels)``.
    model_matrix:
        Template model spectra, shape ``(n_in_transit, n_good_pixels)``.

    Returns
    -------
    float
        Sum of per-frame log-likelihoods.
    """
    data = np.asarray(data_matrix, dtype=float)
    model = np.asarray(model_matrix, dtype=float)

    # Mean-subtract per frame
    data = data - data.mean(axis=1, keepdims=True)
    model = model - model.mean(axis=1, keepdims=True)

    n = data.shape[1]
    sf2 = (1.0 / n) * np.sum(data ** 2, axis=1)
    sg2 = (1.0 / n) * np.sum(model ** 2, axis=1)
    R = (1.0 / n) * np.sum(data * model, axis=1)

    log_L = -0.5 * n * np.log(sf2 - 2.0 * R + sg2)
    return float(np.sum(log_L))


# ---------------------------------------------------------------------------
# Blain24 log-likelihood
# ---------------------------------------------------------------------------


def log_likelihood_blain24(
    data_matrix: np.ndarray,
    model_matrix: np.ndarray,
    uncertainties: np.ndarray,
) -> float:
    """Blain, Sánchez-López & Mollière (2024) chi-squared log-likelihood.

    .. math::

        \\log\\mathcal{L} = -\\frac{1}{2}
            \\sum_{n,\\lambda}
            \\left(\\frac{d_{n,\\lambda} - m_{n,\\lambda}}{\\sigma_{n,\\lambda}}\\right)^2

    Parameters
    ----------
    data_matrix:
        Residual data, shape ``(n_in_transit, n_good_pixels)``.
    model_matrix:
        Template model, shape ``(n_in_transit, n_good_pixels)``.
    uncertainties:
        Per-pixel σ, same shape as ``data_matrix``.

    Returns
    -------
    float
        Log-likelihood scalar.
    """
    data = np.asarray(data_matrix, dtype=float)
    model = np.asarray(model_matrix, dtype=float)
    sigma = np.asarray(uncertainties, dtype=float)

    chi2 = np.sum(((data - model) / sigma) ** 2)
    return float(-0.5 * chi2)


def log_likelihood_blain24_beta(
    data_matrix: np.ndarray,
    model_matrix: np.ndarray,
    uncertainties: np.ndarray,
    beta: float,
) -> float:
    """Evaluate Blain24 with one fitted uncertainty scale, as in setup P-19.

    The prepared data and model keep the Blain24 convention. Multiplying every
    uncertainty by beta gives log L = -chi2/(2 beta**2) - N log(beta), up to
    terms independent of the fitted parameters. The normalization term must
    remain when beta varies. The sampler applies the configured beta prior.
    """
    if not np.isfinite(beta) or beta <= 0:
        return -np.inf
    data = np.asarray(data_matrix, dtype=float)
    model = np.asarray(model_matrix, dtype=float)
    sigma = np.asarray(uncertainties, dtype=float)
    chi2 = np.sum(((data - model) / (beta * sigma)) ** 2)
    return float(-0.5 * chi2 - data.size * np.log(beta))


# ---------------------------------------------------------------------------
# Gibson22 log-likelihood (noise-scaling β)
# ---------------------------------------------------------------------------


def log_likelihood_gibson22(
    data_matrix: np.ndarray,
    model_matrix: np.ndarray,
    uncertainties: np.ndarray,
    beta: float,
) -> float:
    """Gibson et al. (2022) log-likelihood with noise-scaling parameter β.

    .. math::

        \\log\\mathcal{L} = -\\frac{1}{2}
            \\sum_{n,\\lambda}
            \\left(\\frac{d - m}{\\beta\\,\\sigma}\\right)^2
            - N \\ln\\beta

    where :math:`N` is the total number of data points.  The β term
    penalises over-inflation of the noise.

    A β outside ``[0.01, 100]`` returns ``-np.inf``.

    Parameters
    ----------
    data_matrix:
        Residual data, shape ``(n_in_transit, n_good_pixels)``.
    model_matrix:
        Template model, same shape.
    uncertainties:
        Per-pixel σ, same shape.
    beta:
        Noise scaling parameter (must be > 0).

    Returns
    -------
    float
        Log-likelihood scalar, or ``-np.inf`` if β is out of range.
    """
    if not (0.01 <= beta <= 100.0):
        return -np.inf

    data = np.asarray(data_matrix, dtype=float)
    model = np.asarray(model_matrix, dtype=float)
    sigma = np.asarray(uncertainties, dtype=float)

    chi2_tot = np.sum(((data - model) / (beta * sigma)) ** 2)
    N = data.size
    # Scaling sigma by beta changes both the squared residuals and the
    # Gaussian normalization. -N ln(beta) penalizes inflating the uncertainties.
    # Without it, increasing beta would keep reducing chi-squared. beta < 1
    # reduces the supplied uncertainties, while beta > 1 increases them.
    log_L = -0.5 * chi2_tot - N * np.log(beta)
    return float(log_L)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Mapping from string name to likelihood function.
LIKELIHOOD_REGISTRY: dict = {
    "BL19": log_likelihood_bl19,
    "Blain24": log_likelihood_blain24,
    "Blain24_beta": log_likelihood_blain24_beta,
    "Gibson22": log_likelihood_gibson22,
}


def compute_log_likelihood(
    choice: str,
    data_matrix: np.ndarray,
    model_matrix: np.ndarray,
    uncertainties: np.ndarray | None = None,
    beta: float = 1.0,
) -> float:
    """Dispatch to the requested log-likelihood function.

    Parameters
    ----------
    choice:
        One of ``"BL19"``, ``"Blain24"``, ``"Blain24_beta"``, ``"Gibson22"``.
    data_matrix:
        Residual data, shape ``(n_in_transit, n_good_pixels)``.
    model_matrix:
        Template model, same shape.
    Uncertainties:
        Per-pixel σ.  Required for ``"Blain24"``, ``"Blain24_beta"`` and ``"Gibson22"``.
    beta:
        Noise scaling.  Used for ``"Blain24_beta"`` and ``"Gibson22"``.

    Returns
    -------
    float
        Log-likelihood value.

    Raises
    ------
    ValueError
        If ``choice`` is not recognised or ``uncertainties`` is
        required but not supplied.
    """
    if choice not in LIKELIHOOD_REGISTRY:
        raise ValueError(
            f"Unknown log-likelihood choice: {choice!r}. "
            f"Available: {list(LIKELIHOOD_REGISTRY)}"
        )
    if choice == "BL19":
        return log_likelihood_bl19(data_matrix, model_matrix)
    if choice == "Blain24":
        if uncertainties is None:
            raise ValueError("uncertainties must be provided for Blain24.")
        return log_likelihood_blain24(data_matrix, model_matrix, uncertainties)
    if choice in ("Blain24_beta", "Gibson22"):
        if uncertainties is None:
            raise ValueError(f"uncertainties must be provided for {choice}.")
        fn = LIKELIHOOD_REGISTRY[choice]
        return fn(data_matrix, model_matrix, uncertainties, beta)
    raise ValueError(f"Unhandled choice: {choice!r}")
