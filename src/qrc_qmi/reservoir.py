"""Open-system quantum reservoir: reset-and-write input, Lindblad evolution, readout.

One input period (duration dt):
    rho  ->  rho_1(s_k) (x) Tr_1[rho]            (reset-and-write on qubit 0)
    then V sub-intervals of length tau = dt / V; after each one the z-basis observables
    <Z_i> and <Z_i Z_j> are recorded (time multiplexing).

Propagators for one sub-interval
    "exact"  : matrix exponential of the vectorised Liouvillian (practical for N <= 5)
    "strang" : n_s second-order steps  C(d/2) U(d) C(d) U(d) ... U(d) C(d/2),  d = tau/n_s,
               with C the exact local channel and U = exp(-i H d).
    "auto"   : exact for N <= 4, Strang otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .channels import LocalChannel, jump_operator
from .hamiltonian import SX, bit_table, embed

ENCODINGS = ("pure_z", "mixed_z", "pure_x")


def input_state(s: np.ndarray, encoding: str = "pure_z") -> np.ndarray:
    """Single-qubit input states, shape (B, 2, 2), for inputs s in [0, 1].

    pure_z  : |psi> = sqrt(1-s)|0> + sqrt(s)|1>                 (Fujii & Nakajima)
    mixed_z : rho = diag(s, 1-s)                                (Palacios et al., main text)
    pure_x  : |psi> = exp(-i pi s X / 2)|0> = cos(pi s/2)|0> - i sin(pi s/2)|1>
    """
    s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, 1.0)
    out = np.zeros((s.size, 2, 2), dtype=complex)
    if encoding == "pure_z":
        c = np.sqrt(s * (1 - s))
        out[:, 0, 0], out[:, 1, 1], out[:, 0, 1], out[:, 1, 0] = 1 - s, s, c, c
    elif encoding == "mixed_z":
        out[:, 0, 0], out[:, 1, 1] = s, 1 - s
    elif encoding == "pure_x":
        a, b = np.cos(np.pi * s / 2), np.sin(np.pi * s / 2)
        out[:, 0, 0], out[:, 1, 1] = a**2, b**2
        out[:, 0, 1], out[:, 1, 0] = 1j * a * b, -1j * a * b
    else:
        raise ValueError(f"unknown encoding '{encoding}'")
    return out


def observable_table(N: int, kind: str = "z+zz") -> np.ndarray:
    """Rows are the diagonals of the z-basis observables: N single-site, N(N-1)/2 pairs."""
    z = 1.0 - 2.0 * bit_table(N)  # (d, N)
    rows = []
    if "z+" in kind or kind == "z":
        rows += [z[:, i] for i in range(N)]
    if "zz" in kind:
        rows += [z[:, i] * z[:, j] for i in range(N) for j in range(i + 1, N)]
    return np.array(rows)


def sample_populations(p: np.ndarray, n_shots: float, rng) -> np.ndarray:
    """Outcome frequencies of n_shots projective z-basis measurements of states with
    populations p (rows sum to one).

    Uses the Gaussian limit of the multinomial distribution, which has the exact mean p and
    the exact covariance (diag(p) - p p^T) / n_shots:  p + (sqrt(p) z - p (sqrt(p) . z)) /
    sqrt(n_shots) with z standard normal.  Every observable computed from one sample
    therefore carries the correct variance (1 - <O>^2)/n_shots and the correct correlations
    with the other observables.
    """
    p = np.clip(p, 0.0, None)
    r = np.sqrt(p)
    z = rng.standard_normal(p.shape)
    return p + (r * z - p * (r * z).sum(axis=-1, keepdims=True)) / np.sqrt(n_shots)


def liouvillian(H: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    """Row-major Liouvillian: d vec(rho)/dt = L vec(rho), using vec(A rho B) = (A (x) B^T) vec(rho)."""
    d = H.shape[0]
    I = np.eye(d)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for A in jumps:
        AdA = A.conj().T @ A
        L = L + np.kron(A, A.conj()) - 0.5 * np.kron(AdA, I) - 0.5 * np.kron(I, AdA.T)
    return L


def lindblad_rhs(rho, H, jumps):
    out = -1j * (H @ rho - rho @ H)
    for A in jumps:
        AdA = A.conj().T @ A
        out = out + A @ rho @ A.conj().T - 0.5 * (AdA @ rho + rho @ AdA)
    return out


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.linalg.eigvalsh(a - b)).sum())


@dataclass
class Reservoir:
    H: np.ndarray
    N: int
    channel: str = "none"
    gamma: float = 0.0
    dt: float = 10.0
    V: int = 10
    n_s: int = 8
    encoding: str = "pure_z"
    propagator: str = "auto"
    observables: str = "z+zz"
    _ops: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.d = 2**self.N
        self.tau = self.dt / self.V
        if self.propagator == "auto":
            self.propagator = "exact" if self.N <= 4 else "strang"
        self.obs = observable_table(self.N, self.observables)
        self.n_features = self.V * self.obs.shape[0]
        if self.propagator == "exact":
            S = expm(liouvillian(self.H, self.jumps()) * self.tau)
            self._ops["S_T"] = np.ascontiguousarray(S.T)
        elif self.propagator == "strang":
            delta = self.tau / self.n_s
            E, P = np.linalg.eigh(self.H)
            U = (P * np.exp(-1j * E * delta)) @ P.conj().T
            self._ops["U"], self._ops["Ud"] = U, np.ascontiguousarray(U.conj().T)
            self._ops["half"] = LocalChannel(self.channel, self.gamma, delta / 2, self.N)
            self._ops["full"] = LocalChannel(self.channel, self.gamma, delta, self.N)
        else:
            raise ValueError(f"unknown propagator '{self.propagator}'")

    # ------------------------------------------------------------------ building blocks
    def jumps(self) -> list[np.ndarray]:
        A = jump_operator(self.channel, self.gamma)
        return [] if A is None else [embed(A, i, self.N) for i in range(self.N)]

    def inject(self, rho: np.ndarray, s: np.ndarray) -> np.ndarray:
        """rho -> rho_1(s) (x) Tr_1[rho] for a batch (B, d, d) and inputs s of shape (B,)."""
        B, h = rho.shape[0], self.d // 2
        rest = np.einsum("bijik->bjk", rho.reshape(B, 2, h, 2, h))
        return np.einsum("bac,bjk->bajck", input_state(s, self.encoding), rest).reshape(B, self.d, self.d)

    def step(self, rho: np.ndarray) -> np.ndarray:
        """Evolve a batch for one sub-interval tau under the master equation."""
        if self.propagator == "exact":
            B = rho.shape[0]
            return (rho.reshape(B, -1) @ self._ops["S_T"]).reshape(B, self.d, self.d)
        U, Ud, half, full = (self._ops[k] for k in ("U", "Ud", "half", "full"))
        rho = half(rho.copy())
        for m in range(self.n_s):
            rho = U @ rho @ Ud
            rho = full(rho) if m < self.n_s - 1 else half(rho)
        return rho

    def features_of(self, rho: np.ndarray) -> np.ndarray:
        return np.einsum("bii->bi", rho).real @ self.obs.T

    def initial_state(self, B: int = 1, rng=None) -> np.ndarray:
        """|0...0><0...0| by default, or Haar-random pure states if an rng is given."""
        rho = np.zeros((B, self.d, self.d), dtype=complex)
        if rng is None:
            rho[:, 0, 0] = 1.0
            return rho
        psi = rng.normal(size=(B, self.d)) + 1j * rng.normal(size=(B, self.d))
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
        return np.einsum("bi,bj->bij", psi, psi.conj())

    # ------------------------------------------------------------------ driving
    def run(self, s: np.ndarray, rho0: np.ndarray | None = None, record: bool = True,
            on_period_end=None, shots: tuple = (), rng=None):
        """Drive with inputs s of shape (B, K).

        Returns (features, rho_final); features has shape (B, K, V * n_obs) (None if
        record=False).  `on_period_end(k, rho)` is called with the state at the end of
        period k, i.e. just before the next input is injected.

        If `shots` is given, features is instead a dict {n_shots: array, None: exact array}:
        at every readout time the z-basis outcome frequencies are sampled once per shot
        count (see `sample_populations`) and all observables are computed from that one
        sample, as they would be from one batch of measurements.
        """
        s = np.atleast_2d(np.asarray(s, dtype=float))
        B, K = s.shape
        rho = self.initial_state(B) if rho0 is None else rho0
        n_obs = self.obs.shape[0]
        feats = np.empty((B, K, self.V, n_obs)) if record else None
        noisy = {n: np.empty((B, K, self.V, n_obs)) for n in shots} if record else {}
        for k in range(K):
            rho = self.inject(rho, s[:, k])
            for v in range(self.V):
                rho = self.step(rho)
                if record:
                    p = np.einsum("bii->bi", rho).real
                    feats[:, k, v, :] = p @ self.obs.T
                    for n in shots:
                        noisy[n][:, k, v, :] = sample_populations(p, n, rng) @ self.obs.T
            if on_period_end is not None:
                on_period_end(k, rho)
        if not record:
            return None, rho
        feats = feats.reshape(B, K, self.V * n_obs)
        if shots:
            out = {n: a.reshape(B, K, self.V * n_obs) for n, a in noisy.items()}
            out[None] = feats
            return out, rho
        return feats, rho

    # ------------------------------------------------------------------ validation helpers
    def echo_state_distance(self, s: np.ndarray, rng) -> float:
        """Trace distance between two Haar-random initial states after driving with s (1-D)."""
        s2 = np.tile(np.asarray(s, dtype=float)[None, :], (2, 1))
        _, rho = self.run(s2, rho0=self.initial_state(2, rng), record=False)
        return trace_distance(rho[0], rho[1])


def integrator_error(H, N, channel, gamma, n_s, dt=10.0, V=10, periods=100, seed=0,
                     encoding="pure_z") -> float:
    """Trace distance between Strang (n_s) and the exact propagator after `periods` inputs."""
    s = np.random.default_rng(seed).uniform(0, 1, size=(1, periods))
    kw = dict(H=H, N=N, channel=channel, gamma=gamma, dt=dt, V=V, encoding=encoding)
    _, a = Reservoir(propagator="exact", **kw).run(s, record=False)
    _, b = Reservoir(propagator="strang", n_s=n_s, **kw).run(s, record=False)
    return trace_distance(a[0], b[0])


def ode_reference_error(res: Reservoir, periods=5, seed=0, rtol=1e-10, atol=1e-12) -> float:
    """Independent check usable at any N: compare `res` with an adaptive Runge-Kutta
    (DOP853) solution of the Lindblad equation over a few input periods."""
    s = np.random.default_rng(seed).uniform(0, 1, size=(1, periods))
    _, a = res.run(s, record=False)
    jumps, d = res.jumps(), res.d
    rho = res.initial_state(1)
    f = lambda t, y: lindblad_rhs(y.reshape(d, d), res.H, jumps).ravel()
    for k in range(periods):
        rho = res.inject(rho, s[:, k])
        sol = solve_ivp(f, (0.0, res.dt), rho[0].ravel(), method="DOP853", rtol=rtol, atol=atol)
        rho = sol.y[:, -1].reshape(1, d, d)
    return trace_distance(a[0], rho[0])
