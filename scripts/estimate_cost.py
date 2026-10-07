"""Compute-budget estimate for a sweep preset (Methods 2.8: "the pilot also times one
N = 8 configuration").  Times a few Strang steps per (N, channel) on this machine and
scales by the calibrated n_s, the number of input periods and the number of drives.
Run it on an otherwise idle machine.

Usage: python scripts/estimate_cost.py main transfer
"""
import os
import sys
import time

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

from qrc_qmi.config import PRESETS, load_n_s, n_s_for  # noqa: E402
from qrc_qmi.diagnostics import all_diagnostics  # noqa: E402
from qrc_qmi.hamiltonian import build_hamiltonian, draw_disorder  # noqa: E402
from qrc_qmi.reservoir import Reservoir  # noqa: E402

table = load_n_s()
for name in sys.argv[1:] or ["main", "transfer"]:
    cfg = PRESETS[name]
    total = 0.0
    print(f"\n{name}: {cfg.n_configs} configurations, {cfg.n_inputs} input realisations, "
          f"{cfg.splits.total} periods per drive")
    for N in cfg.Ns:
        H = build_hamiltonian(N, 1.0, *draw_disorder(N, 0.1, np.random.default_rng(0)))
        s = np.random.default_rng(1).uniform(size=(cfg.n_inputs, 1))
        rho = Reservoir(H=H, N=N).initial_state(1, np.random.default_rng(2))[0]
        t = time.perf_counter(); all_diagnostics(rho, N); t_diag = time.perf_counter() - t
        for ch in cfg.channels:
            res = Reservoir(H=H, N=N, channel=ch, gamma=0.1, dt=cfg.dt, V=cfg.V, n_s=4,
                            propagator="exact" if N <= 4 else "strang")
            res.run(s)
            reps = 3 if N >= 8 else 10
            t = time.perf_counter()
            for _ in range(reps):
                res.run(s)
            per_period = (time.perf_counter() - t) / reps          # seconds, at n_s = 4
            drives = 1 + int(cfg.mackey_glass)
            n_cell = len(cfg.hs) * len(cfg.Ws) * cfg.n_real
            hours = 0.0
            for g in cfg.gammas:
                scale = 1.0 if N <= 4 else n_s_for(cfg, N, ch, g, table) / 4
                sec = per_period * scale * cfg.splits.total * drives * 1.15   # +15% washout check, diagnostics drive
                sec += drives * (cfg.diag_steps / cfg.diag_stride) * t_diag
                hours += n_cell * sec / 3600
            total += hours
            print(f"  N={N} {ch:18s} {per_period * 1e3:8.1f} ms per period at n_s=4 -> {hours:8.1f} CPU-hours")
    print(f"  total ≈ {total:.0f} CPU-hours (single thread per process)")
