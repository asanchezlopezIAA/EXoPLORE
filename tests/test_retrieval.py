"""
Tests for exoplore.retrieval, log-likelihoods and priors.
"""

import math

import numpy as np
import pytest

from exoplore.retrieval.likelihood import (
    log_likelihood_bl19,
    log_likelihood_blasp24,
    log_likelihood_g22,
    compute_log_likelihood,
)
from exoplore.retrieval.priors import (
    UniformPrior,
    GaussianPrior,
    LogUniformPrior,
    PriorSet,
    standard_1d_prior_set,
)


# ---------------------------------------------------------------------------
# Log-likelihoods
# ---------------------------------------------------------------------------

class TestLogLikelihoodBL19:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.n_frames, self.n_pix = 10, 50
        self.data = rng.normal(0, 1, (self.n_frames, self.n_pix))
        self.model = rng.normal(0, 1, (self.n_frames, self.n_pix))

    def test_returns_scalar(self):
        val = log_likelihood_bl19(self.data, self.model)
        assert isinstance(val, float)

    def test_exact_match_gives_high_logL(self):
        # When data == model, logL should be higher than a random model
        logL_exact = log_likelihood_bl19(self.data, self.data)
        logL_random = log_likelihood_bl19(self.data, self.model)
        assert logL_exact > logL_random

    def test_finite(self):
        assert math.isfinite(log_likelihood_bl19(self.data, self.model))


class TestLogLikelihoodBLASP24:
    def setup_method(self):
        rng = np.random.default_rng(1)
        self.n, self.p = 8, 40
        self.data = rng.normal(0, 1, (self.n, self.p))
        self.model = self.data + rng.normal(0, 0.1, (self.n, self.p))
        self.sigma = 0.5 * np.ones((self.n, self.p))

    def test_returns_scalar(self):
        val = log_likelihood_blasp24(self.data, self.model, self.sigma)
        assert isinstance(val, float)

    def test_closer_model_is_higher(self):
        far_model = self.data + 10.0
        logL_close = log_likelihood_blasp24(self.data, self.model, self.sigma)
        logL_far = log_likelihood_blasp24(self.data, far_model, self.sigma)
        assert logL_close > logL_far


class TestLogLikelihoodG22:
    def setup_method(self):
        rng = np.random.default_rng(2)
        self.n, self.p = 8, 40
        self.data = rng.normal(0, 1, (self.n, self.p))
        self.model = self.data.copy()
        self.sigma = np.ones((self.n, self.p))

    def test_returns_scalar(self):
        val = log_likelihood_g22(self.data, self.model, self.sigma, beta=1.0)
        assert isinstance(val, float)

    def test_out_of_range_beta_returns_neginf(self):
        val = log_likelihood_g22(self.data, self.model, self.sigma, beta=0.001)
        assert val == -np.inf

    def test_beta_above_1_penalised(self):
        logL_1 = log_likelihood_g22(self.data, self.model, self.sigma, beta=1.0)
        logL_5 = log_likelihood_g22(self.data, self.model, self.sigma, beta=5.0)
        # Large beta inflates uncertainties and adds a penalty term
        assert logL_1 > logL_5


class TestComputeLogLikelihood:
    def test_bl19_dispatch(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, (5, 30))
        # Use model != data to avoid log(0) = -inf → logL = +inf
        model = rng.normal(0, 1, (5, 30))
        val = compute_log_likelihood("BL19", data, model)
        assert math.isfinite(val)

    def test_unknown_choice_raises(self):
        rng = np.random.default_rng(4)
        data = rng.normal(0, 1, (5, 30))
        with pytest.raises(ValueError, match="Unknown log-likelihood"):
            compute_log_likelihood("XYZ", data, data)

    def test_blasp24_requires_uncertainties(self):
        rng = np.random.default_rng(5)
        data = rng.normal(0, 1, (5, 30))
        with pytest.raises(ValueError, match="uncertainties must be provided"):
            compute_log_likelihood("BLASP24", data, data)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

class TestPriors:
    def test_uniform_prior_midpoint(self):
        p = UniformPrior(-5.0, 5.0)
        assert p(0.5) == pytest.approx(0.0)

    def test_uniform_prior_bounds(self):
        p = UniformPrior(0.0, 10.0)
        assert p(0.0) == pytest.approx(0.0)
        assert p(1.0) == pytest.approx(10.0)

    def test_uniform_log_prior_inside(self):
        p = UniformPrior(0.0, 10.0)
        assert math.isfinite(p.log_prior(5.0))

    def test_uniform_log_prior_outside(self):
        p = UniformPrior(0.0, 10.0)
        assert p.log_prior(15.0) == -math.inf

    def test_log_uniform_prior(self):
        p = LogUniformPrior(-8.0, -1.0)
        val = p(0.0)
        assert val == pytest.approx(10 ** -8.0)
        val_high = p(1.0)
        assert val_high == pytest.approx(10 ** -1.0)

    def test_prior_set_transform(self):
        ps = PriorSet()
        ps.add("log10_X", UniformPrior(-8.0, -1.0))
        ps.add("K_p", UniformPrior(50.0, 250.0))
        ps.add("T_equ", UniformPrior(500.0, 3000.0))

        result = ps.prior_transform(np.array([0.5, 0.5, 0.5]))
        assert len(result) == 3
        assert result[0] == pytest.approx(-4.5)
        assert result[1] == pytest.approx(150.0)
        assert result[2] == pytest.approx(1750.0)

    def test_prior_set_n_params(self):
        ps = standard_1d_prior_set()
        assert ps.n_params == 4  # no beta

    def test_prior_set_with_beta(self):
        ps = standard_1d_prior_set(include_beta=True)
        assert ps.n_params == 5

    def test_prior_set_param_names(self):
        ps = standard_1d_prior_set()
        assert "K_p" in ps.param_names
        assert "T_equ" in ps.param_names
