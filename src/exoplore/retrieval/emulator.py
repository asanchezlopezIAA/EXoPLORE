"""petitRADTRANS spectrum emulator for fast Bayesian retrieval.

.. warning::
   **HIGHLY EXPERIMENTAL, NOT VALIDATED FOR SCIENTIFIC USE.**

   This emulator was benchmarked against the true pRT retrieval on a
   noiseless CARMENES NIR order-23 1D retrieval (HD 189733 b, BLASP24
   pipeline and loglike).  The emulator was trained on 100 000 spectra
   with a narrowed prior (log₁₀X ∈ [-5, -1], T_eq ∈ [600, 1800] K),
   achieving CCF agreement of 0.9995 (min 0.9988) on held-out test
   spectra.  Despite this spectral accuracy, **retrieval performance
   was unacceptable**:

   - Block 9 time: 11.65 min vs 22.33 min for true pRT (1.9× speedup)
   - Posterior widths: 2 to 3× broader than pRT on all parameters
   - T_eq: biased -109 K (-0.94σ from truth, -1.3σ from pRT result)
   - ln Z shifted by ~10 units relative to true pRT evidence

Root cause
----------
The BL19 log-likelihood for HRS is logL = -N/2 × log(1 - CC²).  At
R ≈ 100 000, CC is extremely sensitive to line shape and position.  The
PCA reconstruction introduces sub-pixel line profile errors that shift CC
and therefore shift the likelihood surface.  These errors are invisible
in the spectral RMSE metric (3.25 × 10⁻⁶) but large enough to broaden
and bias the posterior.  This problem does not affect low-resolution
applications (R ~ 400) where the likelihood surface is smooth.

Technical details
-----------------
PrtEmulator is a drop-in replacement for :func:`exoplore.atmosphere.prt.call_pRT`
inside Block 9 of the simulator.  It uses a PCA basis fitted on a library of
pRT spectra and a small MLP trained to predict PCA coefficients from atmospheric
parameters.  Inference takes ~1 ms instead of ~500 ms per call (the speedup
is limited by the PCA reconstruction, not the MLP).

The emulator is instrument- and order-specific.  Each emulator directory
(produced by ``scripts/emulator/train_emulator.py``) contains:

    wave_prt.npy          wavelength grid in µm (same as call_pRT returns)
    pca_mean.npy          spectral mean (n_wave,)
    pca_components.npy    PCA basis vectors (k, n_wave)
    param_bounds.npy      [[min, max], ...] per parameter (2, n_params)
    mlp_weights.pt        PyTorch MLP state dict
    training_log.json     training metadata including mlp_hidden architecture

Usage
-----
    from exoplore.retrieval.emulator import PrtEmulator
    emulator = PrtEmulator.load("emulators/carmenes_nir/order_23_v2")
    wave, spec = emulator.predict(T_eq=1200.0, log10_x_h2o=-3.0)
"""

import os
import numpy as np

# ── MLP (must match architecture in train_emulator.py) ───────────────────────
def _build_mlp(n_in, n_out, hidden=(256, 256, 128)):
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for the pRT emulator. "
                          "Install with: pip install torch")

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, n_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return _MLP()


class PrtEmulator:
    """Fast pRT spectrum emulator (PCA + MLP).

    Parameters
    ----------
    wave_prt : ndarray, shape (n_wave,)
        Wavelength grid in µm, identical to the ``wave_pRT`` return value
        of :func:`~exoplore.atmosphere.prt.call_pRT`.
    pca_mean : ndarray, shape (n_wave,)
    pca_components : ndarray, shape (k, n_wave)
    param_bounds : ndarray, shape (2, n_params)
        ``param_bounds[0]`` = minima, ``param_bounds[1]`` = maxima used
        during training.  Parameters outside this range trigger a warning.
    mlp : torch.nn.Module
    """

    def __init__(self, wave_prt, pca_mean, pca_components, param_bounds, mlp):
        self._wave     = wave_prt
        self._mu       = pca_mean
        self._W        = pca_components        # (k, n_wave)
        self._bounds   = param_bounds          # (2, n_params)
        self._mlp      = mlp
        self._mlp.eval()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path):
        """Load a trained emulator from *path* (directory produced by
        ``scripts/emulator/train_emulator.py``).
        """
        import torch

        wave = np.load(os.path.join(path, "wave_prt.npy")).astype(np.float32)
        mu   = np.load(os.path.join(path, "pca_mean.npy")).astype(np.float32)
        W    = np.load(os.path.join(path, "pca_components.npy")).astype(np.float32)
        bnd  = np.load(os.path.join(path, "param_bounds.npy")).astype(np.float32)

        n_pca = W.shape[0]
        # Read architecture from training_log.json if available
        import json as _json
        _hidden = (256, 256, 128)  # default (Phase 1 prototype)
        _log_path = os.path.join(path, "training_log.json")
        if os.path.exists(_log_path):
            with open(_log_path) as _lf:
                _log = _json.load(_lf)
            if "mlp_hidden" in _log:
                _hidden = tuple(_log["mlp_hidden"])
        mlp   = _build_mlp(n_in=bnd.shape[1], n_out=n_pca, hidden=_hidden)
        state = torch.load(os.path.join(path, "mlp_weights.pt"),
                           map_location="cpu", weights_only=True)
        mlp.load_state_dict(state)
        mlp.eval()

        return cls(wave, mu, W, bnd, mlp)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, T_eq, log10_x_h2o):
        """Return *(wave_pRT, spec_conv)* matching the call_pRT signature.

        Parameters
        ----------
        T_eq : float
            Equilibrium temperature in K.
        log10_x_h2o : float
            Log₁₀ of H₂O volume mixing ratio.

        Returns
        -------
        wave_pRT : ndarray, shape (n_wave,)
        spec_conv : ndarray, shape (n_wave,)
        """
        import torch

        params = np.array([[log10_x_h2o, T_eq]], dtype=np.float32)
        self._warn_ood(params[0])

        p_min, p_max = self._bounds[0], self._bounds[1]
        params_norm  = (params - p_min) / (p_max - p_min)

        with torch.no_grad():
            x     = torch.tensor(params_norm, dtype=torch.float32)
            coeff = self._mlp(x).numpy()[0]          # (k,)

        spec = self._mu + coeff @ self._W             # (n_wave,)
        return self._wave.copy(), spec

    def predict_batch(self, log10_x_h2o_arr, T_eq_arr):
        """Vectorised prediction for multiple parameter sets."""
        import torch

        params = np.column_stack([log10_x_h2o_arr, T_eq_arr]).astype(np.float32)
        p_min, p_max = self._bounds[0], self._bounds[1]
        params_norm  = (params - p_min) / (p_max - p_min)

        with torch.no_grad():
            x      = torch.tensor(params_norm, dtype=torch.float32)
            coeffs = self._mlp(x).numpy()            # (N, k)

        spectra = self._mu + coeffs @ self._W        # (N, n_wave)
        return self._wave.copy(), spectra

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def is_in_distribution(self, log10_x_h2o, T_eq):
        """Return True if parameters lie within the training prior bounds."""
        params = np.array([log10_x_h2o, T_eq], dtype=np.float32)
        return bool(np.all(params >= self._bounds[0]) and
                    np.all(params <= self._bounds[1]))

    def _warn_ood(self, params):
        if np.any(params < self._bounds[0]) or np.any(params > self._bounds[1]):
            import warnings
            warnings.warn(
                "PrtEmulator: parameters outside training range, "
                "predictions may be unreliable.",
                RuntimeWarning, stacklevel=3)

    def validate(self, call_prt_fn, n_samples=10, seed=42):
        """Compare emulator predictions against true pRT calls.

        Parameters
        ----------
        call_prt_fn : callable
            Function that accepts (T_eq, log10_x_h2o) and returns
            (wave, spec), a thin wrapper around call_pRT.
        n_samples : int
        seed : int

        Returns
        -------
        dict with keys: ccf_mean, ccf_min, rmse_mean
        """
        rng = np.random.default_rng(seed)
        log10_x = rng.uniform(self._bounds[0, 0], self._bounds[1, 0], n_samples)
        t_eq    = rng.uniform(self._bounds[0, 1], self._bounds[1, 1], n_samples)

        ccfs, rmses = [], []
        for lx, te in zip(log10_x, t_eq):
            _, spec_true = call_prt_fn(te, lx)
            _, spec_emul = self.predict(te, lx)
            spec_true = np.asarray(spec_true, dtype=np.float64)
            spec_emul = np.asarray(spec_emul, dtype=np.float64)
            a = spec_true - spec_true.mean()
            b = spec_emul - spec_emul.mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            ccf = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
            ccfs.append(ccf)
            rmses.append(float(np.sqrt(np.mean((spec_true - spec_emul) ** 2))))

        return {
            "ccf_mean":  float(np.mean(ccfs)),
            "ccf_min":   float(np.min(ccfs)),
            "rmse_mean": float(np.mean(rmses)),
        }
