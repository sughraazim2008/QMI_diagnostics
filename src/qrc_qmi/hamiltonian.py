"""Transverse-field Ising reservoir Hamiltonian and spectral statistics.

Conventions used throughout the package
---------------------------------------
* Qubit 0 is the *first* tensor factor (most significant bit of the basis index).
* |0> is the +1 eigenstate of sigma^z; sigma^- = |0><1| lowers towards |0>.
* Energies are in units of J_s, times in units of 1/J_s (hbar = 1).
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2)
SX = np.array([[0.0, 1.0], [1.0, 0.0]])
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]])
SZ = np.array([[1.0, 0.0], [0.0, -1.0]])
SM = np.array([[0.0, 1.0], [0.0, 0.0]])  # sigma^- = |0><1|


def embed(op: np.ndarray, i: int, N: int) -> np.ndarray:
    """Single-qubit operator `op` acting on qubit `i` of an N-qubit register."""
    return np.kron(np.kron(np.eye(2**i), op), np.eye(2 ** (N - i - 1)))


def bit_table(N: int) -> np.ndarray:
    """bits[a, i] = value (0/1) of qubit i in computational basis state a."""
    a = np.arange(2**N)
    return np.stack([(a >> (N - 1 - i)) & 1 for i in range(N)], axis=1)


def draw_disorder(N, W, rng, Js=1.0, coupling="symmetric"):
    """Draw couplings J_ij and on-site disorder D_i.

    coupling="symmetric": J_ij ~ U[-Js/2, Js/2]  (Fujii-Nakajima / Martinez-Pena convention)
    coupling="positive":  J_ij ~ U[0, Js]        (convention reported for Palacios et al.)
    """
    iu = np.triu_indices(N, k=1)
    J = np.zeros((N, N))
    if coupling == "symmetric":
        J[iu] = rng.uniform(-Js / 2, Js / 2, size=len(iu[0]))
    elif coupling == "positive":
        J[iu] = rng.uniform(0.0, Js, size=len(iu[0]))
    else:
        raise ValueError(f"unknown coupling '{coupling}'")
    D = rng.uniform(-W, W, size=N) if W > 0 else np.zeros(N)
    return J, D


def build_hamiltonian(N, h, J, D, field_prefactor=0.5) -> np.ndarray:
    """H = sum_{i<j} J_ij X_i X_j + field_prefactor * sum_i (h + D_i) Z_i  (real symmetric)."""
    d = 2**N
    a = np.arange(d)
    H = np.zeros((d, d))
    for i in range(N):
        for j in range(i + 1, N):
            # X_i X_j flips bits i and j: a permutation matrix
            H[a, a ^ ((1 << (N - 1 - i)) | (1 << (N - 1 - j)))] += J[i, j]
    # the field term is diagonal
    z = 1.0 - 2.0 * bit_table(N)  # (d, N) eigenvalues of Z_i
    H[np.diag_indices(d)] += field_prefactor * (z @ (h + np.asarray(D)))
    return H


def parity_sectors(N: int):
    """Index sets of the two eigen-sectors of the Z2 parity P = prod_i Z_i.

    [H, P] = 0 because every X_i X_j flips two spins, so level statistics must be
    evaluated inside one sector; mixing the sectors produces spurious Poisson statistics.
    """
    par = bit_table(N).sum(axis=1) % 2
    return np.where(par == 0)[0], np.where(par == 1)[0]


def gap_ratio(H: np.ndarray, N: int) -> float:
    """Mean adjacent-gap ratio <r>, averaged over both parity sectors.

    Reference values: ~0.5307 (GOE, ergodic) and 2 ln 2 - 1 ~ 0.3863 (Poisson, localised)
    [Oganesyan & Huse 2007; Atas et al. 2013].
    """
    ratios = []
    for idx in parity_sectors(N):
        E = np.linalg.eigvalsh(H[np.ix_(idx, idx)])
        gaps = np.diff(E)
        lo = np.minimum(gaps[:-1], gaps[1:])
        hi = np.maximum(gaps[:-1], gaps[1:])
        ok = hi > 1e-14
        ratios.append(lo[ok] / hi[ok])
    r = np.concatenate(ratios)
    return float(r.mean()) if r.size else float("nan")
