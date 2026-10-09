"""Leaky echo state network baseline, tuned with a tree-structured Parzen estimator.

    x_{k+1} = (1 - a) x_k + a tanh(W x_k + W_in s_k + b)

The network has D nodes, where D is the quantum readout dimension, so that both models
fit the same number of readout weights (D + bias).  ESN features are noiseless.
"""
from __future__ import annotations

import numpy as np

from .readout import Splits, linear_quadratic_features, ridge_select

UNTUNED = dict(leak=1.0, radius=0.9, in_scale=1.0, bias_scale=0.0)


def esn_states(s: np.ndarray, D: int, leak: float, radius: float, in_scale: float,
               bias_scale: float, seed: int, density: float = 0.1) -> np.ndarray:
    """States (B, K, D) for inputs s (B, K); the same random network is used for the batch."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303, D]))
    W = rng.uniform(-1, 1, size=(D, D)) * (rng.uniform(size=(D, D)) < density)
    sr = np.abs(np.linalg.eigvals(W)).max()
    W *= radius / sr if sr > 0 else 0.0
    w_in = in_scale * rng.uniform(-1, 1, size=D)
    b = bias_scale * rng.uniform(-1, 1, size=D)
    B, K = s.shape
    x = np.zeros((B, D))
    out = np.empty((B, K, D))
    Wt = W.T.copy()
    for k in range(K):
        x = (1 - leak) * x + leak * np.tanh(x @ Wt + s[:, k, None] * w_in + b)
        out[:, k] = x
    return out


def esn_score(s, target, sp: Splits, D, params: dict, seeds, which="test_nrmse") -> np.ndarray:
    """NRMSE array of shape (len(seeds), n_realisations)."""
    out = np.empty((len(seeds), s.shape[0]))
    for a, seed in enumerate(seeds):
        X = esn_states(s, D, seed=seed, **params)
        for j in range(s.shape[0]):
            out[a, j] = ridge_select(X[j], target[j], sp)[which][0]
    return out


def tune_esn(s, target, sp: Splits, D: int, n_trials: int = 100, tune_seeds=(9001, 9002, 9003),
             sampler_seed: int = 0):
    """TPE search over leak rate, spectral radius, input scaling and bias scaling,
    minimising validation NRMSE averaged over input realisations and tuning seeds."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            leak=trial.suggest_float("leak", 0.05, 1.0, log=True),
            radius=trial.suggest_float("radius", 0.1, 1.5),
            in_scale=trial.suggest_float("in_scale", 1e-3, 2.0, log=True),
            bias_scale=trial.suggest_float("bias_scale", 0.0, 1.0),
        )
        v = esn_score(s, target, sp, D, params, tune_seeds, which="val_nrmse").mean()
        return float(v) if np.isfinite(v) else 1e6

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)


def baseline_row(task: str, s, target, sp: Splits, D: int, n_trials: int = 100,
                 test_seeds=tuple(range(10)), sampler_seed: int = 0) -> dict:
    """Tuned ESN, untuned ESN and linear-quadratic reference for one (task, D)."""
    params, best_val = tune_esn(s, target, sp, D, n_trials=n_trials, sampler_seed=sampler_seed)
    tuned = esn_score(s, target, sp, D, params, test_seeds)
    untuned = esn_score(s, target, sp, D, UNTUNED, test_seeds)
    lq = np.array([ridge_select(linear_quadratic_features(s[j]), target[j], sp)["test_nrmse"][0]
                   for j in range(s.shape[0])])
    return {
        "task": task, "D": D, "n_trials": n_trials,
        "esn_nrmse": float(tuned.mean()),
        "esn_nrmse_sd_seeds": float(tuned.mean(axis=1).std(ddof=1)),
        "esn_nrmse_sd_all": float(tuned.std(ddof=1)),
        "esn_val_nrmse": best_val,
        "esn_untuned_nrmse": float(untuned.mean()),
        "linquad_nrmse": float(lq.mean()),
        **{f"esn_{k}": v for k, v in params.items()},
    }
