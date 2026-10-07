"""State diagnostics.  All take a single density matrix (d, d); none sees task targets.

Entropies are in bits (base-2 logarithms).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def partial_trace(rho: np.ndarray, keep, N: int) -> np.ndarray:
    """Reduced state on the qubits in `keep` (ordered as given, must be sorted)."""
    keep = list(keep)
    drop = [i for i in range(N) if i not in keep]
    t = rho.reshape((2,) * (2 * N))
    t = np.transpose(t, keep + drop + [N + i for i in keep] + [N + i for i in drop])
    k, m = 2 ** len(keep), 2 ** len(drop)
    return np.einsum("ajbj->ab", t.reshape(k, m, k, m))


def entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy in bits."""
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())


def shannon(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 1e-14]
    return float(-(p * np.log2(p)).sum())


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.vdot(rho, rho)))


def partial_transpose(rho: np.ndarray, A, N: int) -> np.ndarray:
    t = rho.reshape((2,) * (2 * N))
    perm = list(range(2 * N))
    for i in A:
        perm[i], perm[N + i] = perm[N + i], perm[i]
    return np.transpose(t, perm).reshape(2**N, 2**N)


def negativity(rho: np.ndarray, A, N: int) -> float:
    """(||rho^{T_A}||_1 - 1)/2 = sum of |negative eigenvalues| of the partial transpose."""
    w = np.linalg.eigvalsh(partial_transpose(rho, A, N))
    return float(-w[w < 0].sum())


def bipartitions(N: int):
    """All 2^(N-1) - 1 unordered bipartitions A|B; A never contains qubit 0."""
    for r in range(1, N):
        for A in combinations(range(1, N), r):
            yield A


def mean_negativity(rho: np.ndarray, N: int, log: bool = False) -> float:
    """Negativity averaged over all bipartitions.  log=True: logarithmic negativity
    log2 ||rho^{T_A}||_1 = log2(2 N + 1)."""
    vals = [negativity(rho, A, N) for A in bipartitions(N)]
    if log:
        vals = [np.log2(2 * v + 1) for v in vals]
    return float(np.mean(vals))


def l1_coherence(rho: np.ndarray, normalised: bool = True) -> float:
    """sum_{i != j} |rho_ij| in the computational basis; normalised by its maximum d - 1."""
    d = rho.shape[0]
    c = float(np.abs(rho).sum() - np.abs(np.diag(rho)).sum())
    return c / (d - 1) if normalised else c


def total_mutual_information(rho: np.ndarray, N: int) -> float:
    """T = S(pi[rho]) - S(rho) = sum_i S(rho_i) - S(rho), pi = product of 1-qubit marginals."""
    return sum(entropy(partial_trace(rho, [i], N)) for i in range(N)) - entropy(rho)


def pairwise_mutual_information(rho: np.ndarray, N: int) -> float:
    """Mean over qubit pairs of S(rho_i) + S(rho_j) - S(rho_ij)."""
    S1 = [entropy(partial_trace(rho, [i], N)) for i in range(N)]
    vals = [S1[i] + S1[j] - entropy(partial_trace(rho, [i, j], N))
            for i in range(N) for j in range(i + 1, N)]
    return float(np.mean(vals))


def half_chain_mutual_information(rho: np.ndarray, N: int) -> float:
    A = list(range(N // 2))
    B = list(range(N // 2, N))
    return entropy(partial_trace(rho, A, N)) + entropy(partial_trace(rho, B, N)) - entropy(rho)


def classical_correlations(rho: np.ndarray, N: int) -> float:
    """K = T(Delta[rho]): total correlation of the computational-basis populations."""
    p = np.real(np.diag(rho)).clip(min=0).reshape((2,) * N)
    marg = sum(shannon(p.sum(axis=tuple(j for j in range(N) if j != i))) for i in range(N))
    return marg - shannon(p)


def hookup(rho: np.ndarray, N: int) -> float:
    """Quantum hookup M = S(rho || Delta[pi[rho]]) = sum_i H(diag rho_i) - S(rho)
    (Giorgi & Zambrini 2018).  M = T + local coherence."""
    p = np.real(np.diag(rho)).clip(min=0).reshape((2,) * N)
    marg = sum(shannon(p.sum(axis=tuple(j for j in range(N) if j != i))) for i in range(N))
    return marg - entropy(rho)


def all_diagnostics(rho: np.ndarray, N: int) -> dict:
    """Every diagnostic used in the study, with the size normalisations of Methods 2.7."""
    S = entropy(rho)
    S1 = [entropy(partial_trace(rho, [i], N)) for i in range(N)]
    T = sum(S1) - S
    pair = [S1[i] + S1[j] - entropy(partial_trace(rho, [i, j], N))
            for i in range(N) for j in range(i + 1, N)]
    negs = np.array([negativity(rho, A, N) for A in bipartitions(N)])
    return {
        "negativity": float(negs.mean()),
        "log_negativity": float(np.log2(2 * negs + 1).mean()),
        "coherence": l1_coherence(rho, normalised=True),
        "tmi": T,
        "tmi_per_qubit": T / N,
        "pair_mi": float(np.mean(pair)),
        "half_mi": half_chain_mutual_information(rho, N),
        "classical_corr": classical_correlations(rho, N),
        "hookup": hookup(rho, N),
        "purity": purity(rho),
        "entropy": S,
        "entropy_per_qubit": S / N,
    }


class DiagnosticAverager:
    """Callback for Reservoir.run that time-averages diagnostics of batch element 0."""

    def __init__(self, N: int, stride: int = 1):
        self.N, self.stride, self.rows = N, stride, []

    def __call__(self, k: int, rho: np.ndarray):
        if k % self.stride == 0:
            self.rows.append(all_diagnostics(rho[0], self.N))

    def mean(self) -> dict:
        return {k: float(np.mean([r[k] for r in self.rows])) for k in self.rows[0]}
