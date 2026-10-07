"""Finite-sampling noise, ridge readout and NRMSE.  The same code scores every model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAMBDAS = np.logspace(-10, 0, 21)


@dataclass(frozen=True)
class Splits:
    washout: int = 500
    n_train: int = 2000
    n_val: int = 500
    n_test: int = 1000

    @property
    def total(self):
        return self.washout + self.n_train + self.n_val + self.n_test

    @property
    def train(self):
        return slice(self.washout, self.washout + self.n_train)

    @property
    def val(self):
        a = self.washout + self.n_train
        return slice(a, a + self.n_val)

    @property
    def test(self):
        a = self.washout + self.n_train + self.n_val
        return slice(a, a + self.n_test)


def add_shot_noise(X: np.ndarray, n_shots: float | None, rng, model: str = "binomial",
                   sigma: float = 1e-3) -> np.ndarray:
    """Replace each expectation value <O> in [-1, 1] by a finite-sample estimate.

    binomial : <O> + eps, eps ~ N(0, (1 - <O>^2)/n_shots)  (Gaussian approximation to the
               mean of n_shots +-1 outcomes), clipped to [-1, 1].
    gaussian : <O> + eps, eps ~ N(0, sigma^2)              (as in Palacios et al.).
    n_shots=None returns X unchanged (exact expectation values).
    Noise is drawn independently for every observable and time step.
    """
    if n_shots is None and model == "binomial":
        return X
    if model == "binomial":
        sd = np.sqrt(np.clip(1.0 - X**2, 0.0, None) / n_shots)
        return np.clip(X + sd * rng.standard_normal(X.shape), -1.0, 1.0)
    if model == "gaussian":
        return X + sigma * rng.standard_normal(X.shape)
    raise ValueError(model)


def nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """sqrt( mean((y - yhat)^2) / Var(y) ), variance taken over the scored segment."""
    return float(np.sqrt(np.mean((y - yhat) ** 2) / np.var(y)))


def ridge_select(X: np.ndarray, Y: np.ndarray, sp: Splits, lambdas=LAMBDAS) -> dict:
    """Ridge regression with an unpenalised bias; lambda chosen on the validation segment.

    Minimises ||y - X w - b||^2 + lambda ||w||^2 on the training segment for each lambda
    (one SVD), selects the lambda with the lowest validation NRMSE, and reports the test
    NRMSE of that (train-only) fit.  Y may hold several targets as columns; each gets its
    own lambda.  Features are centred with training means but not rescaled.
    """
    Y = Y[:, None] if Y.ndim == 1 else Y
    Xtr, Xva, Xte = X[sp.train], X[sp.val], X[sp.test]
    Ytr, Yva, Yte = Y[sp.train], Y[sp.val], Y[sp.test]
    mu, ymu = Xtr.mean(axis=0), Ytr.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    UtY = U.T @ (Ytr - ymu)                      # (r, n_targets)
    Pva, Pte = (Xva - mu) @ Vt.T, (Xte - mu) @ Vt.T
    n_t = Y.shape[1]
    val = np.empty((len(lambdas), n_t))
    for a, lam in enumerate(lambdas):
        C = (S / (S**2 + lam))[:, None] * UtY
        pred = Pva @ C + ymu
        val[a] = np.sqrt(np.mean((Yva - pred) ** 2, axis=0) / np.var(Yva, axis=0))
    best = val.argmin(axis=0)
    lam_best = np.asarray(lambdas)[best]
    C = (S[:, None] / (S[:, None] ** 2 + lam_best[None, :])) * UtY
    pred = Pte @ C + ymu
    test = np.sqrt(np.mean((Yte - pred) ** 2, axis=0) / np.var(Yte, axis=0))
    return {"test_nrmse": test, "val_nrmse": val[best, np.arange(n_t)], "lambda": lam_best,
            "at_upper": best == len(lambdas) - 1, "at_lower": best == 0}


def linear_quadratic_features(s: np.ndarray, n_lags: int = 10) -> np.ndarray:
    """Reference baseline: the last n_lags inputs and all their pairwise products."""
    K = len(s)
    lags = np.stack([np.concatenate([np.zeros(i), s[:K - i]]) for i in range(n_lags)], axis=1)
    iu = np.triu_indices(n_lags)
    quad = lags[:, iu[0]] * lags[:, iu[1]]
    return np.hstack([lags, quad])
