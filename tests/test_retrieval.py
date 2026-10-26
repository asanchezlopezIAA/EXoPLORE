"""
Tests for exoplore.retrieval, log-likelihoods and priors.
"""

import math

import numpy as np
import pytest

from exoplore.retrieval.likelihood import (
    log_likelihood_bl19,
    log_likelihood_blain24,
    log_likelihood_gibson22,
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


class TestLogLikelihoodBlain24:
    def setup_method(self):
        rng = np.random.default_rng(1)
        self.n, self.p = 8, 40
        self.data = rng.normal(0, 1, (self.n, self.p))
        self.model = self.data + rng.normal(0, 0.1, (self.n, self.p))
        self.sigma = 0.5 * np.ones((self.n, self.p))

    def test_returns_scalar(self):
        val = log_likelihood_blain24(self.data, self.model, self.sigma)
        assert isinstance(val, float)

    def test_closer_model_is_higher(self):
        far_model = self.data + 10.0
        logL_close = log_likelihood_blain24(self.data, self.model, self.sigma)
        logL_far = log_likelihood_blain24(self.data, far_model, self.sigma)
        assert logL_close > logL_far


class TestLogLikelihoodGibson22:
    def setup_method(self):
        rng = np.random.default_rng(2)
        self.n, self.p = 8, 40
        self.data = rng.normal(0, 1, (self.n, self.p))
        self.model = self.data.copy()
        self.sigma = np.ones((self.n, self.p))

    def test_returns_scalar(self):
        val = log_likelihood_gibson22(self.data, self.model, self.sigma, beta=1.0)
        assert isinstance(val, float)

    def test_out_of_range_beta_returns_neginf(self):
        val = log_likelihood_gibson22(self.data, self.model, self.sigma, beta=0.001)
        assert val == -np.inf

    def test_beta_above_1_penalised(self):
        logL_1 = log_likelihood_gibson22(self.data, self.model, self.sigma, beta=1.0)
        logL_5 = log_likelihood_gibson22(self.data, self.model, self.sigma, beta=5.0)
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

    def test_blain24_requires_uncertainties(self):
        rng = np.random.default_rng(5)
        data = rng.normal(0, 1, (5, 30))
        with pytest.raises(ValueError, match="uncertainties must be provided"):
            compute_log_likelihood("Blain24", data, data)


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


class TestLogLikelihoodBlain24Beta:
    """Check the fitted noise scale and its Gaussian normalization."""

    @pytest.fixture
    def spectra(self):
        rng = np.random.default_rng(41)
        sigma = rng.uniform(0.1, 0.8, (5, 17))
        model = rng.normal(0.0, 0.1, sigma.shape)
        data = model + rng.normal(size=sigma.shape) * sigma
        return data, model, sigma

    def test_beta_one_recovers_fixed_blain(self, spectra):
        from exoplore.retrieval import log_likelihood_blain24_beta

        data, model, sigma = spectra
        assert log_likelihood_blain24_beta(data, model, sigma, 1.0) == (
            log_likelihood_blain24(data, model, sigma))

    def test_matches_normalized_gaussian(self, spectra):
        from scipy.stats import norm
        from exoplore.retrieval import log_likelihood_blain24_beta

        data, model, sigma = spectra
        beta = 1.7
        expected = np.sum(norm.logpdf(data, loc=model, scale=beta * sigma))
        # EXoPLORE omits only terms independent of model parameters and beta.
        fixed_terms = np.sum(np.log(sigma) + 0.5 * np.log(2 * np.pi))
        assert log_likelihood_blain24_beta(data, model, sigma, beta) == pytest.approx(
            expected + fixed_terms)

    def test_best_beta_matches_residual_rms(self, spectra):
        from exoplore.retrieval import log_likelihood_blain24_beta

        data, model, sigma = spectra
        optimum = np.sqrt(np.mean(((data - model) / sigma) ** 2))
        peak = log_likelihood_blain24_beta(data, model, sigma, optimum)
        for beta in (optimum * 0.9, optimum * 1.1, 1000.0):
            assert peak > log_likelihood_blain24_beta(data, model, sigma, beta)

    @pytest.mark.parametrize('beta', [0.0, -1.0, np.nan, np.inf])
    def test_invalid_scale_rejected(self, spectra, beta):
        from exoplore.retrieval import log_likelihood_blain24_beta

        assert log_likelihood_blain24_beta(*spectra, beta) == -np.inf

    def test_dispatch_and_missing_uncertainties(self, spectra):
        from exoplore.retrieval import log_likelihood_blain24_beta

        data, model, sigma = spectra
        assert compute_log_likelihood('Blain24_beta', data, model, sigma, 0.8) == (
            log_likelihood_blain24_beta(data, model, sigma, 0.8))
        with pytest.raises(ValueError, match='uncertainties'):
            compute_log_likelihood('Blain24_beta', data, model)

    @pytest.mark.parametrize('instrument,noise', [('andes', 'noisy'), ('carmenes', 'noiseless')])
    def test_example_matches_gibson_prior_and_preserves_blain_setup(self, instrument, noise):
        from dataclasses import asdict
        from pathlib import Path
        from exoplore.config import SimulationConfig

        root = Path(__file__).resolve().parents[1]
        stem = f'hd189733b_{instrument}_retrieval_'
        variant = SimulationConfig.from_json(root / 'configs' / f'{stem}blain24_beta_{noise}.json')
        gibson = SimulationConfig.from_json(root / 'configs' / f'{stem}gibson22_{noise}.json')
        fixed = SimulationConfig.from_json(root / 'configs' / f'{stem}blain24_{noise}.json')
        assert variant.retrieval.log_likelihood == 'Blain24_beta'
        assert variant.retrieval.prior_bounds['1D_Blain24_beta'] == (
            gibson.retrieval.prior_bounds['1D_Gibson22'])
        for section in ('planet', 'instrument', 'observation', 'noise', 'atmosphere', 'pipeline'):
            assert asdict(getattr(variant, section)) == asdict(getattr(fixed, section))
        restored = SimulationConfig.from_dict(variant.to_dict())
        assert restored.retrieval == variant.retrieval

    def test_beta_run_keeps_fixed_beta_outputs_separate(self):
        from exoplore.config import SimulationConfig
        from exoplore.core.simulator import build_simulation_name

        cfg = SimulationConfig()
        cfg.pipeline.name = 'Blain24'
        cfg.retrieval.enabled = True
        cfg.retrieval.log_likelihood = 'Blain24'
        fixed_name = build_simulation_name(cfg)
        cfg.retrieval.log_likelihood = 'Blain24_beta'
        beta_name = build_simulation_name(cfg)
        assert beta_name.startswith('Blain24_beta_')
        assert fixed_name != beta_name
        assert cfg.pipeline.name == 'Blain24'
