"""Benchmark tasks.  Every task returns (s, target): reservoir input s_k in [0, 1] and the
target to be predicted from the reservoir state after s_k has been injected."""
from __future__ import annotations

import numpy as np

NARMA_TASKS = {"narma5": 5, "narma10": 10, "narma20": 20}


def narma_target(u: np.ndarray, n: int, variant: str | None = None) -> np.ndarray:
    """NARMA-n series driven by u (1-D).  target[k] = y_{k+1}, which depends on u_0..u_k.

    standard (n = 5, 10; Atiya & Parlos 2000):
        y_{k+1} = 0.3 y_k + 0.05 y_k sum_{i=0}^{n-1} y_{k-i} + 1.5 u_{k-n+1} u_k + 0.1
    tanh (default for n = 20; Rodan & Tino 2011), bounded so that it cannot diverge:
        y_{k+1} = tanh(0.3 y_k + 0.05 y_k sum_{i=0}^{n-1} y_{k-i} + 1.5 u_{k-n+1} u_k + 0.01)
    """
    if variant is None:
        variant = "tanh" if n >= 20 else "standard"
    K = len(u)
    y = np.zeros(K + 1)
    for k in range(K):
        lo = max(0, k - n + 1)
        u_del = u[k - n + 1] if k - n + 1 >= 0 else 0.0
        core = 0.3 * y[k] + 0.05 * y[k] * y[lo:k + 1].sum() + 1.5 * u_del * u[k]
        y[k + 1] = np.tanh(core + 0.01) if variant == "tanh" else core + 0.1
        if not np.isfinite(y[k + 1]) or abs(y[k + 1]) > 1e3:
            y[k + 1:] = np.nan
            break
    return y[1:]


def narma_inputs(K: int, n_real: int, seed: int, tasks=("narma10",)):
    """Shared NARMA inputs.  u ~ U[0, 0.5] i.i.d., s = 2u.  A realisation is redrawn if
    any requested NARMA variant diverges on it.

    Returns s (n_real, K), targets {task: (n_real, K)}, n_discarded.
    """
    ss = np.random.SeedSequence([seed, 101])
    rng = np.random.default_rng(ss)
    s = np.empty((n_real, K))
    targets = {t: np.empty((n_real, K)) for t in tasks}
    discarded = 0
    j = 0
    while j < n_real:
        u = rng.uniform(0.0, 0.5, size=K)
        ys = {t: narma_target(u, NARMA_TASKS[t]) for t in tasks}
        if any(not np.all(np.isfinite(y)) for y in ys.values()):
            discarded += 1
            continue
        s[j] = 2.0 * u
        for t in tasks:
            targets[t][j] = ys[t]
        j += 1
    return s, targets, discarded


MG_LO, MG_HI = 0.2, 1.4  # fixed bounds containing the tau = 17 attractor


def mackey_glass(K: int, seed: int, tau=17.0, beta=0.2, gamma=0.1, p=10, dt_int=0.1,
                 sample=1.0, discard=1000.0, history: float | None = None) -> np.ndarray:
    """Mackey-Glass series  dx/dt = beta x(t-tau)/(1 + x(t-tau)^p) - gamma x,
    integrated with RK4 (step dt_int, linear interpolation of the delayed term at half
    steps) and sampled every `sample` time units after discarding a transient."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    lag = int(round(tau / dt_int))
    per = int(round(sample / dt_int))
    n_steps = int(round(discard / dt_int)) + per * (K + 1)
    x = np.empty(lag + 1 + n_steps)
    x[:lag + 1] = 1.2 + 0.2 * (rng.uniform(size=lag + 1) - 0.5) if history is None else history
    f = lambda xt, xd: beta * xd / (1.0 + xd**p) - gamma * xt
    for i in range(lag, lag + n_steps):
        xd0, xd1 = x[i - lag], x[i - lag + 1]
        xdh = 0.5 * (xd0 + xd1)
        k1 = f(x[i], xd0)
        k2 = f(x[i] + 0.5 * dt_int * k1, xdh)
        k3 = f(x[i] + 0.5 * dt_int * k2, xdh)
        k4 = f(x[i] + dt_int * k3, xd1)
        x[i + 1] = x[i] + dt_int * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x[lag + int(round(discard / dt_int))::per][:K + 1]   # times discard + j * sample


def mackey_glass_inputs(K: int, n_real: int, seed: int, horizon: int = 1, sample: float = 1.0):
    """s_k = (x_k - 0.2)/1.2 clipped to [0, 1]; target[k] = x_{k+horizon}."""
    s = np.empty((n_real, K))
    y = np.empty((n_real, K))
    for j in range(n_real):
        x = mackey_glass(K + horizon, seed=seed * 1000 + j, sample=sample)
        s[j] = np.clip((x[:K] - MG_LO) / (MG_HI - MG_LO), 0.0, 1.0)
        y[j] = x[horizon:K + horizon]
    return s, {"mackey_glass": y}, 0
