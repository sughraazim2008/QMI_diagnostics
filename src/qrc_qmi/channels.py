"""Local decoherence channels.

Each channel is the *exact* solution of a single-qubit Lindblad dissipator over a time t,
so applying it to every qubit integrates the dissipative part of the master equation
without error (local dissipators on different qubits commute).

Lindblad form:  d(rho)/dt = sum_i ( L_i rho L_i^dag - 1/2 {L_i^dag L_i, rho} )

name                 jump operator L           Kraus parameter after time t
-------------------  ------------------------  ---------------------------------
dephasing            (sqrt(gamma)/2) sigma^z   p = (1 - exp(-gamma t / 2)) / 2
amplitude_damping    sqrt(gamma) sigma^-       p = 1 - exp(-gamma t)
phase_flip           sqrt(gamma) sigma^z       p = (1 - exp(-2 gamma t)) / 2
bit_flip             sqrt(gamma) sigma^x       p = (1 - exp(-2 gamma t)) / 2

With these definitions `dephasing` and `amplitude_damping` both damp single-qubit
coherences as exp(-gamma t / 2), i.e. T2 = 2/gamma.  `phase_flip`/`bit_flip` follow the
convention gamma * (sigma rho sigma - rho) used for the reproduction of Palacios et al.;
phase_flip(gamma) == dephasing(4 gamma).
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import I2, SM, SX, SZ, bit_table

CHANNELS = ("none", "dephasing", "amplitude_damping", "phase_flip", "bit_flip")
UNITAL = {"none": True, "dephasing": True, "phase_flip": True, "bit_flip": True,
          "amplitude_damping": False}


def jump_operator(name: str, gamma: float) -> np.ndarray | None:
    """Single-qubit Lindblad jump operator."""
    if name == "none" or gamma == 0:
        return None
    if name == "dephasing":
        return 0.5 * np.sqrt(gamma) * SZ
    if name == "amplitude_damping":
        return np.sqrt(gamma) * SM
    if name == "phase_flip":
        return np.sqrt(gamma) * SZ
    if name == "bit_flip":
        return np.sqrt(gamma) * SX
    raise ValueError(f"unknown channel '{name}'")


def kraus_operators(name: str, gamma: float, t: float) -> list[np.ndarray]:
    """Kraus operators of the single-qubit channel exp(D t)."""
    if name == "none" or gamma == 0 or t == 0:
        return [I2.copy()]
    if name == "dephasing":
        p = 0.5 * (1.0 - np.exp(-gamma * t / 2.0))
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * SZ]
    if name == "phase_flip":
        p = 0.5 * (1.0 - np.exp(-2.0 * gamma * t))
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * SZ]
    if name == "bit_flip":
        p = 0.5 * (1.0 - np.exp(-2.0 * gamma * t))
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * SX]
    if name == "amplitude_damping":
        p = 1.0 - np.exp(-gamma * t)
        return [np.array([[1.0, 0.0], [0.0, np.sqrt(1 - p)]]),
                np.array([[0.0, np.sqrt(p)], [0.0, 0.0]])]
    raise ValueError(f"unknown channel '{name}'")


def superoperator(kraus: list[np.ndarray]) -> np.ndarray:
    """Row-major superoperator: vec(sum_k K rho K^dag) = S vec(rho)."""
    return sum(np.kron(K, K.conj()) for K in kraus)


def choi_matrix(kraus: list[np.ndarray]) -> np.ndarray:
    """Choi matrix J = sum_ij |i><j| (x) E(|i><j|) (unnormalised, trace = 2)."""
    d = kraus[0].shape[0]
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = 1.0
            J += np.kron(E, sum(K @ E @ K.conj().T for K in kraus))
    return J


def apply_local_kraus(rho: np.ndarray, kraus: list[np.ndarray], N: int) -> np.ndarray:
    """Reference implementation: apply the same single-qubit channel to every qubit.

    rho has shape (B, d, d).  Slow but obviously correct; used by tests and bit_flip.
    """
    B = rho.shape[0]
    S = superoperator(kraus)
    t = rho.reshape((B,) + (2,) * (2 * N))
    for i in range(N):
        v = np.moveaxis(t, (1 + i, 1 + N + i), (-2, -1))
        shp = v.shape
        v = (v.reshape(-1, 4) @ S.T).reshape(shp)
        t = np.moveaxis(v, (-2, -1), (1 + i, 1 + N + i))
    return np.ascontiguousarray(t).reshape(B, 2**N, 2**N)


class LocalChannel:
    """Fast application of exp(D t) on all N qubits of a batch of density matrices.

    Calling the object may modify its argument in place; always use the return value.
    """

    def __init__(self, name: str, gamma: float, t: float, N: int):
        if name not in CHANNELS:
            raise ValueError(f"unknown channel '{name}'")
        self.name, self.gamma, self.t, self.N = name, gamma, t, N
        self.kraus = kraus_operators(name, gamma, t)
        self.trivial = name == "none" or gamma == 0 or t == 0
        if name in ("dephasing", "phase_flip") and not self.trivial:
            rate = gamma / 2.0 if name == "dephasing" else 2.0 * gamma
            bits = bit_table(N)
            hamming = (bits[:, None, :] != bits[None, :, :]).sum(axis=2)
            self._mask = np.exp(-rate * t * hamming)
        if name == "amplitude_damping" and not self.trivial:
            # Single-qubit map: rho_00 += p rho_11, then an entrywise mask
            # (rho_11 *= 1-p, rho_01 and rho_10 *= sqrt(1-p)).  Masks and transfers on
            # different qubits commute, so the N-qubit channel is N transfers followed by
            # one mask multiplication.
            self._p = p = 1.0 - np.exp(-gamma * t)
            bits = bit_table(N)
            both = (bits[:, None, :] & bits[None, :, :]).sum(axis=2)
            differ = (bits[:, None, :] != bits[None, :, :]).sum(axis=2)
            self._mask = (1.0 - p) ** both * np.sqrt(1.0 - p) ** differ

        if name == "bit_flip" and not self.trivial:
            # X_i rho X_i permutes rows and columns by flipping bit i
            a = np.arange(2**N)
            self._p = 0.5 * (1.0 - np.exp(-2.0 * gamma * t))
            self._perms = [a ^ (1 << (N - 1 - i)) for i in range(N)]

    def __call__(self, rho: np.ndarray) -> np.ndarray:
        if self.trivial:
            return rho
        if self.name == "bit_flip":
            p = self._p
            for perm in self._perms:
                rho = (1.0 - p) * rho + p * rho[:, perm][:, :, perm]
            return rho
        if self.name in ("dephasing", "phase_flip"):
            return rho * self._mask
        if self.name == "amplitude_damping":
            N, p, B = self.N, self._p, rho.shape[0]
            rho = np.ascontiguousarray(rho)
            for i in range(N):
                L, R = 2**i, 2 ** (N - i - 1)
                v = rho.reshape(B, L, 2, R, L, 2, R)           # view onto rho
                v[:, :, 0, :, :, 0, :] += p * v[:, :, 1, :, :, 1, :]
            rho *= self._mask
            return rho
        return apply_local_kraus(rho, self.kraus, N=self.N)
