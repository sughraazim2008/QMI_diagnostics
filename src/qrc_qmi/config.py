"""Sweep definitions.  Values marked (dagger) are proposals to be fixed in the pre-registration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .readout import Splits

GAMMAS_MAIN = tuple(float(g) for g in np.logspace(-3, 0, 8))  # (dagger)


@dataclass(frozen=True)
class SweepConfig:
    name: str
    design_version: int = 2
    Ns: tuple = (4, 5, 6)
    channels: tuple = ("dephasing", "amplitude_damping")
    gammas: tuple = GAMMAS_MAIN
    hs: tuple = (0.1, 1.0, 10.0)            # fixed from the gap-ratio pilot
    Ws: tuple = (0.1, 5.0, 20.0)            # fixed from the gap-ratio pilot
    n_real: int = 10                        # Hamiltonian realisations per site
    narma_tasks: tuple = ("narma10",)
    mackey_glass: bool = False
    mg_horizon: int = 17                    # predict one delay time ahead
    compute_ipc: bool = True
    # reservoir
    dt: float = 10.0                        # (dagger)
    V: int = 10                             # (dagger)
    encoding: str = "pure_z"
    coupling: str = "symmetric"
    field_prefactor: float = 0.5
    propagator: str = "auto"                # exact for N <= 4, Strang otherwise
    n_s_safety: int = 2                     # multiplies the calibrated n_s when N > 5 (calibration is at N = 4, 5)
    # data
    splits: Splits = Splits(washout=2000)   # washout fixed from the echo-state pilot
    n_inputs: int = 5
    shots: tuple = (1e6, 1e5, 1e4)          # first entry is the primary setting
    noise_model: str = "same_shot"          # or "binomial" (independent noise per observable)
    # diagnostics
    diag_steps: int = 200
    diag_stride: int = 1
    esp_tol: float = 1e-6
    # IPC (dagger)
    ipc_degree: int = 3
    ipc_delay: int = 20
    base_seed: int = 20261007

    @property
    def n_configs(self) -> int:
        return len(self.Ns) * len(self.channels) * len(self.gammas) * len(self.hs) * len(self.Ws) * self.n_real

    def n_features(self, N: int) -> int:
        return self.V * (N + N * (N - 1) // 2)

    def grid(self):
        """Yield every configuration.  A *site* is one (N, gamma, h, W) point; both channels
        at a site use the same Hamiltonian realisations (paired design)."""
        for N in self.Ns:
            for ic, c in enumerate(self.channels):
                for ig, g in enumerate(self.gammas):
                    for ih, h in enumerate(self.hs):
                        for iw, W in enumerate(self.Ws):
                            for r in range(self.n_real):
                                yield dict(N=N, channel=c, gamma=g, h=h, W=W, real=r,
                                           site=f"N{N}|g{g:.6g}|h{h:g}|W{W:g}",
                                           ham_idx=(N, ig, ih, iw, r), idx=(N, ic, ig, ih, iw, r))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


MAIN = SweepConfig(name="main")

# Held-out system size for H1c.  N = 7 instead of 8, with a lighter protocol, to keep the
# sweep within reach of a workstation (see docs/PILOT_RESULTS.md, compute budget).
TRANSFER = replace(
    MAIN, name="transfer", Ns=(7,), gammas=GAMMAS_MAIN[1::2], n_real=4, n_inputs=3,
    narma_tasks=("narma10", "narma5", "narma20"), mackey_glass=True, compute_ipc=False,
    diag_stride=2,
)

# Reduced-size run of design v2 used to exercise the pipeline; NOT the pre-registered sweep.
PILOT2 = replace(
    MAIN, name="pilot2", Ns=(4,), n_real=4,
    narma_tasks=("narma10", "narma5", "narma20"), mackey_glass=True,
)

SMOKE = replace(
    MAIN, name="smoke", Ns=(3,), gammas=(0.01, 0.3), hs=(1.0,), Ws=(0.1, 20.0), n_real=2,
    splits=Splits(washout=100, n_train=300, n_val=100, n_test=150), n_inputs=2,
    narma_tasks=("narma10", "narma5"), mackey_glass=True, diag_steps=20, ipc_delay=5, ipc_degree=2,
)

PRESETS = {c.name: c for c in (MAIN, TRANSFER, PILOT2, SMOKE)}


def load_n_s(path: str | Path = "configs/n_s.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text())["n_s"] if p.exists() else {}


def n_s_for(cfg: SweepConfig, N: int, channel: str, gamma: float, table: dict | None = None,
            default: int = 16) -> int:
    """Calibrated Strang steps per sub-interval (times a safety factor for N > 5)."""
    table = load_n_s() if table is None else table
    return int(table.get(f"{channel}|{gamma:.6g}", default)) * (cfg.n_s_safety if N > 5 else 1)
