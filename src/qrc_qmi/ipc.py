"""Information processing capacity (Dambre et al. 2012).

Targets are products of normalised Legendre polynomials of delayed inputs u in [-1, 1].
The capacity for target z is C = 1 - min_w MSE / <z^2> = ||Q^T z||^2 / ||z||^2, where Q is
an orthonormal basis of the (centred) reservoir features.  Finite series give a positive
bias of about D/T per target, so capacities below a surrogate threshold are set to zero.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.special import eval_legendre


def _compositions(total: int, parts: int):
    """Ordered tuples of `parts` positive integers summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def target_index(max_degree: int, max_delay: int):
    """All targets with total degree 1..max_degree on delays 0..max_delay.
    Each target is a tuple of (delay, degree) pairs."""
    out = []
    for deg in range(1, max_degree + 1):
        for n_vars in range(1, deg + 1):
            for delays in combinations(range(max_delay + 1), n_vars):
                for degs in _compositions(deg, n_vars):
                    out.append(tuple(zip(delays, degs)))
    return out


def target_matrix(u: np.ndarray, index, max_degree: int, max_delay: int) -> np.ndarray:
    """Columns are the targets evaluated at times max_delay .. len(u)-1."""
    T = len(u) - max_delay
    # P[d][tau] = sqrt(2d+1) P_d(u_{t - tau})
    P = {d: [np.sqrt(2 * d + 1) * eval_legendre(d, u[max_delay - tau: max_delay - tau + T])
             for tau in range(max_delay + 1)] for d in range(1, max_degree + 1)}
    Z = np.empty((T, len(index)))
    for c, term in enumerate(index):
        col = np.ones(T)
        for tau, d in term:
            col = col * P[d][tau]
        Z[:, c] = col
    return Z


def capacities(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    Q, R = np.linalg.qr(Xc)
    keep = np.abs(np.diag(R)) > 1e-10 * np.abs(np.diag(R)).max()
    Q = Q[:, keep]
    Zc = Z - Z.mean(axis=0)
    return ((Q.T @ Zc) ** 2).sum(axis=0) / (Zc**2).sum(axis=0)


def ipc(X: np.ndarray, u: np.ndarray, max_degree: int = 3, max_delay: int = 20,
        n_surrogates: int = 1, seed: int = 0) -> dict:
    """IPC of features X (T, D) driven by u (T,) in [-1, 1] (washout already removed).

    Threshold: the largest capacity obtained when the targets are built from a time-shuffled
    copy of the input (maximum over all targets and surrogates); smaller capacities are zeroed.
    """
    index = target_index(max_degree, max_delay)
    Xs = X[max_delay:]
    C = capacities(Xs, target_matrix(u, index, max_degree, max_delay))
    rng = np.random.default_rng(seed)
    thr = 0.0
    for _ in range(n_surrogates):
        Cs = capacities(Xs, target_matrix(rng.permutation(u), index, max_degree, max_delay))
        thr = max(thr, float(Cs.max()))
    C = np.where(C > thr, C, 0.0)
    degs = np.array([sum(d for _, d in term) for term in index])
    out = {"ipc_total": float(C.sum()), "ipc_threshold": thr, "ipc_rank": int(Xs.shape[1])}
    for d in range(1, max_degree + 1):
        out[f"ipc_deg{d}"] = float(C[degs == d].sum())
    out["ipc_nonlinear"] = out["ipc_total"] - out["ipc_deg1"]
    return out
