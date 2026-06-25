"""Likelihood emulator for fast Bayesian retrieval.

.. warning::
   **HIGHLY EXPERIMENTAL, NOT VALIDATED FOR SCIENTIFIC USE.**

   This module implements a neural likelihood emulator that maps atmospheric
   parameters θ directly to logL(θ | data), bypassing petitRADTRANS and the
   CCF/BL19 pipeline at inference time.  Both implemented variants were
   benchmarked against the true pRT retrieval on a noiseless CARMENES NIR
   order-23 1D retrieval (HD 189733 b, BLASP24 pipeline and loglike) and
   **neither performs acceptably**:

   Single-round emulator (3 000 uniform samples, 18 min training)
       Val MSE = 1.93 logL²; log₁₀(X_H₂O) biased +1.45σ from pRT;
       ln Z differs by ~10 units; total runtime 37 min vs 22 min for
       true pRT, slower AND less accurate.

   Sequential emulator (3 rounds × 1 000 to 2 000 samples)
       Not completed.  Root cause analysis showed the fundamental
       problem: the posterior occupies ~0.002% of the prior volume for
       a typical 4-parameter 1D retrieval.  With uniform round-1
       sampling, the expected number of near-posterior samples is < 0.1,
       so the round-2 Gaussian proposal has no reliable centre to focus
       on.  Sequential refinement cannot recover from this.

Root cause
----------
The BL19 log-likelihood for HRS (Brogi & Line 2019) is:

    logL = -N/2 × log(1 - CC²)

where CC is the normalised cross-correlation.  CC is extremely sensitive
to the shape and position of individual molecular lines at R ≈ 100 000.
A ±1σ posterior in T_eq spans ~150 K, which changes line depths by
several percent, changes that are fully resolvable at this resolution
but produce large shifts in CC and therefore in logL.  The likelihood
surface is narrow and highly curved.  A small MLP trained on sparse
samples in a wide prior cannot reproduce this surface accurately enough
for unbiased Bayesian inference.

This problem does not arise in low-resolution applications (R ~ 400,
e.g. Vasist et al. 2023) where the likelihood surface is smooth and
broad relative to the prior.

Possible path forward (not implemented)
----------------------------------------
For 2D retrievals where pRT is prohibitively slow (weeks), a warm-start
strategy may be viable:

1. Run a fast 1D pRT retrieval (22 min) to identify approximate Kp,
   T_eq and Vrest.
2. Use those constraints as tight prior bounds for 2D emulator training,
   so virtually all training samples fall near the posterior.
3. Train a 6-parameter likelihood emulator inside the narrow bounds.
4. Run MultiNest on the surrogate for the 2D posterior.

This has not been tested and remains an open research question.  No
literature precedent exists for validated likelihood emulation at HRS.

Configuration
-------------
    retrieval.use_likelihood_emulator : bool  (default False)
    retrieval.likelihood_emulator_n_samples : int  (default 3000)
"""

from __future__ import annotations

import os
import json
import numpy as np


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def _build_loglike_mlp(n_in: int, hidden: tuple = (256, 256, 128)):
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required.  pip install torch")

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return _MLP()


def _fit_mlp(params_norm, logL_shifted, hidden, epochs, batch, patience, seed,
             verbose):
    """Train MLP on normalised (params, logL) data. Returns (best_state, val_mse)."""
    import torch
    import torch.nn as nn

    n_params = params_norm.shape[1]
    rng      = np.random.default_rng(seed)
    idx      = rng.permutation(len(params_norm))
    n_tr     = int(0.85 * len(idx))
    i_tr, i_va = idx[:n_tr], idx[n_tr:]

    model = _build_loglike_mlp(n_params, hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                        eta_min=1e-5)
    X_tr = torch.tensor(params_norm[i_tr]); Y_tr = torch.tensor(logL_shifted[i_tr])
    X_va = torch.tensor(params_norm[i_va]); Y_va = torch.tensor(logL_shifted[i_va])
    ds   = torch.utils.data.TensorDataset(X_tr, Y_tr)
    ldr  = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    best_val, best_state, wait = np.inf, None, 0
    if verbose:
        print(f"  [loglike-emul] Training MLP "
              f"({n_tr} train / {len(i_va)} val)...")
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, Yb in ldr:
            opt.zero_grad()
            nn.functional.mse_loss(model(Xb), Yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(X_va), Y_va).item()
        if vl < best_val:
            best_val  = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  [loglike-emul]   Early stopping epoch {ep}")
                break
        if verbose and ep % 500 == 0:
            print(f"  [loglike-emul]   epoch {ep:4d}  val={vl:.3e}")
    if verbose:
        print(f"  [loglike-emul]   Best val MSE = {best_val:.3e} logL²")
    return best_state, best_val


def _eval_samples(loglike_fn, params, verbose, tag):
    """Evaluate loglike_fn on rows of params. Returns logL array."""
    import time as _t
    n = len(params)
    logL = np.empty(n, dtype=np.float32)
    t0 = _t.time()
    for i, theta in enumerate(params):
        try:
            logL[i] = float(loglike_fn(theta))
        except Exception:
            logL[i] = -1e30
        if verbose and (i + 1) % 200 == 0:
            rate = (i + 1) / (_t.time() - t0)
            eta  = (n - i - 1) / max(rate, 1e-9)
            print(f"  [loglike-emul]   {tag} {i+1}/{n}  "
                  f"{rate:.1f} calls/s  ETA {eta/60:.1f} min", flush=True)
    elapsed = _t.time() - t0
    if verbose:
        valid = logL[logL > -1e29]
        print(f"  [loglike-emul]   {tag} done in {elapsed/60:.1f} min  "
              f"logL ∈ [{valid.min():.1f}, {valid.max():.1f}]")
    return logL


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LoglikeEmulator:
    """Fast logL surrogate trained on true pRT likelihood evaluations."""

    def __init__(self, np_layers, param_bounds, logL_max):
        self._layers   = np_layers
        self._bounds   = param_bounds
        self._logL_max = float(logL_max)
        self._p_min    = param_bounds[0].astype(np.float32)
        self._p_max    = param_bounds[1].astype(np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, theta: np.ndarray) -> float:
        """Pure-numpy logL prediction. ~0.05 ms/call."""
        x = (np.asarray(theta, dtype=np.float32) - self._p_min) / \
            (self._p_max - self._p_min)
        for i, (W, b) in enumerate(self._layers):
            x = W @ x + b
            if i < len(self._layers) - 1:
                np.maximum(x, 0, out=x)
        return float(x[0]) + self._logL_max

    def posterior_region(self, delta_logL: float = 10.0,
                         n_grid: int = 50_000, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate posterior region by evaluating network on a dense random grid.

        Returns (mean, std) of parameter samples with logL > logL_max - delta_logL,
        both in PHYSICAL units (not normalised).  Used to focus round 2 sampling.
        """
        rng = np.random.default_rng(seed)
        pts = rng.random((n_grid, len(self._p_min))).astype(np.float32)
        preds = np.array([self.predict(
            self._p_min + pts[i] * (self._p_max - self._p_min))
            for i in range(n_grid)])
        threshold = preds.max() - delta_logL
        mask  = preds > threshold
        if mask.sum() < 10:
            # fallback: use full prior
            return ((self._p_min + self._p_max) / 2,
                    (self._p_max - self._p_min) / 4)
        phys = self._p_min + pts[mask] * (self._p_max - self._p_min)
        return phys.mean(axis=0), phys.std(axis=0)

    # ── Sequential training ────────────────────────────────────────────────────

    @classmethod
    def train_sequential(
        cls,
        loglike_fn,
        prior_bounds: np.ndarray,
        save_dir: str,
        n_per_round: tuple = (1000, 2000, 2000),
        delta_logL_focus: float = 10.0,
        sigma_scale: float = 2.5,
        hidden: tuple = (256, 256, 128),
        epochs: int = 2000,
        batch: int = 256,
        patience: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ) -> "LoglikeEmulator":
        """Sequential-round training: concentrates samples near the posterior.

        Parameters
        ----------
        loglike_fn:
            ``f(theta) -> float`` using the TRUE pRT + BL19 loglike.
        prior_bounds:
            shape (2, n_params), [lowers, uppers].
        save_dir:
            Directory where per-round (θ, logL) data are saved as
            ``round_{i}_params.npy`` / ``round_{i}_logL.npy``.
            Rounds already computed are SKIPPED on a second call (reuse).
        n_per_round:
            Number of new loglike evaluations per round.
        delta_logL_focus:
            Samples with logL > logL_max - delta_logL_focus define the
            posterior region used to build the next round's proposal.
        sigma_scale:
            How many posterior σ to span with the round-2+ proposal.
            2.5 keeps samples within ~2σ of the posterior peak.
        """
        from scipy.stats.qmc import LatinHypercube
        os.makedirs(save_dir, exist_ok=True)
        p_min = prior_bounds[0].astype(np.float32)
        p_max = prior_bounds[1].astype(np.float32)
        n_params = len(p_min)

        all_params = []
        all_logL   = []
        current_emu = None

        for rnd, n_samp in enumerate(n_per_round):
            tag = f"R{rnd+1}/{len(n_per_round)}"
            p_path  = os.path.join(save_dir, f"round_{rnd}_params.npy")
            ll_path = os.path.join(save_dir, f"round_{rnd}_logL.npy")

            # ── Load cached round if available ────────────────────────────────
            if os.path.exists(p_path) and os.path.exists(ll_path):
                p_rnd  = np.load(p_path).astype(np.float32)
                ll_rnd = np.load(ll_path).astype(np.float32)
                if verbose:
                    valid = ll_rnd[ll_rnd > -1e29]
                    print(f"  [loglike-emul] {tag} loaded from cache "
                          f"({len(p_rnd)} samples, "
                          f"logL ∈ [{valid.min():.1f}, {valid.max():.1f}])")
            else:
                # ── Build proposal ────────────────────────────────────────────
                if rnd == 0 or current_emu is None:
                    # Round 1: uniform LHS over full prior
                    lhs    = LatinHypercube(d=n_params, seed=seed)
                    unit   = lhs.random(n=n_samp)
                    p_rnd  = (p_min + (p_max - p_min) * unit).astype(np.float32)
                    if verbose:
                        print(f"  [loglike-emul] {tag} uniform LHS "
                              f"({n_samp} samples over full prior)...")
                else:
                    # Round 2+: Gaussian centred on posterior region
                    mu, sigma = current_emu.posterior_region(
                        delta_logL=delta_logL_focus)
                    rng = np.random.default_rng(seed + rnd)
                    p_rnd = rng.normal(
                        loc=mu,
                        scale=sigma_scale * sigma,
                        size=(n_samp, n_params),
                    ).astype(np.float32)
                    # clip to prior
                    p_rnd = np.clip(p_rnd, p_min, p_max)
                    if verbose:
                        print(f"  [loglike-emul] {tag} Gaussian proposal "
                              f"({n_samp} samples, σ_scale={sigma_scale})...")
                        for k in range(n_params):
                            print(f"  [loglike-emul]   param[{k}]: "
                                  f"μ={mu[k]:.3f}  σ={sigma[k]:.3f}")

                # ── Evaluate loglike ──────────────────────────────────────────
                ll_rnd = _eval_samples(loglike_fn, p_rnd, verbose, tag)
                np.save(p_path,  p_rnd)
                np.save(ll_path, ll_rnd)

            all_params.append(p_rnd)
            all_logL.append(ll_rnd)

            # ── Pool all rounds and train ─────────────────────────────────────
            params_all = np.vstack(all_params)
            logL_all   = np.hstack(all_logL)

            logL_max   = float(logL_all[logL_all > -1e29].max())
            shifted    = (logL_all - logL_max).astype(np.float32)
            keep       = shifted > -200.0
            if verbose:
                print(f"  [loglike-emul] Pool: {keep.sum()} / {len(params_all)} "
                      f"samples retained (logL > logL_max - 200)")

            params_norm = ((params_all[keep] - p_min) /
                           (p_max - p_min)).astype(np.float32)
            best_state, val_mse = _fit_mlp(
                params_norm, shifted[keep],
                hidden, epochs, batch, patience, seed, verbose)

            np_layers = []
            state = {k: v.numpy() for k, v in best_state.items()}
            i = 0
            while f"net.{i}.weight" in state:
                np_layers.append((state[f"net.{i}.weight"].astype(np.float32),
                                  state[f"net.{i}.bias"].astype(np.float32)))
                i += 2
            current_emu = cls(np_layers, prior_bounds.astype(np.float32), logL_max)
            if verbose:
                print(f"  [loglike-emul] {tag} complete, "
                      f"val MSE = {val_mse:.3e} logL²\n")

        return current_emu

    # ── Single-round training (kept for compatibility) ─────────────────────────

    @classmethod
    def train(
        cls,
        loglike_fn,
        prior_bounds: np.ndarray,
        n_samples: int = 3000,
        hidden: tuple = (256, 256, 128),
        epochs: int = 2000,
        batch: int = 256,
        patience: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ) -> "LoglikeEmulator":
        """Single-round uniform sampling. Kept for API compatibility.

        For better accuracy use train_sequential().
        """
        from scipy.stats.qmc import LatinHypercube
        p_min    = prior_bounds[0].astype(np.float32)
        p_max    = prior_bounds[1].astype(np.float32)
        n_params = len(p_min)
        if verbose:
            print(f"  [loglike-emul] Single-round: {n_samples} uniform samples...")
        lhs    = LatinHypercube(d=n_params, seed=seed)
        unit   = lhs.random(n=n_samples)
        params = (p_min + (p_max - p_min) * unit).astype(np.float32)
        logL   = _eval_samples(loglike_fn, params, verbose, "R1")
        logL_max = float(logL[logL > -1e29].max())
        shifted  = (logL - logL_max).astype(np.float32)
        keep     = shifted > -200.0
        norm     = ((params[keep] - p_min) / (p_max - p_min)).astype(np.float32)
        best_state, val_mse = _fit_mlp(norm, shifted[keep], hidden, epochs,
                                       batch, patience, seed, verbose)
        if verbose:
            print(f"  [loglike-emul] Val MSE = {val_mse:.3e} logL²")
        np_layers = []
        state = {k: v.numpy() for k, v in best_state.items()}
        i = 0
        while f"net.{i}.weight" in state:
            np_layers.append((state[f"net.{i}.weight"].astype(np.float32),
                              state[f"net.{i}.bias"].astype(np.float32)))
            i += 2
        return cls(np_layers, prior_bounds.astype(np.float32), logL_max)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for i, (W, b) in enumerate(self._layers):
            np.save(os.path.join(path, f"ll_W{i}.npy"), W)
            np.save(os.path.join(path, f"ll_b{i}.npy"), b)
        np.save(os.path.join(path, "ll_param_bounds.npy"), self._bounds)
        with open(os.path.join(path, "ll_meta.json"), "w") as f:
            json.dump({"n_layers": len(self._layers),
                       "logL_max": self._logL_max}, f)

    @classmethod
    def load(cls, path: str) -> "LoglikeEmulator":
        with open(os.path.join(path, "ll_meta.json")) as f:
            meta = json.load(f)
        np_layers = []
        for i in range(meta["n_layers"]):
            W = np.load(os.path.join(path, f"ll_W{i}.npy"))
            b = np.load(os.path.join(path, f"ll_b{i}.npy"))
            np_layers.append((W, b))
        bounds = np.load(os.path.join(path, "ll_param_bounds.npy"))
        return cls(np_layers, bounds, meta["logL_max"])
