"""Integrator calibration (Methods 2.10, check 2).

For every (channel, gamma) of the main grid, find the smallest number of Strang steps per
sub-interval, n_s, for which the Strang state stays within trace distance `tol` of the exact
Lindblad propagator after 100 input periods, in the worst case over the (h, W) grid and
several Hamiltonian realisations at N = 4 and 5.  Writes configs/n_s.json.

Usage: python scripts/calibrate_ns.py [--tol 1e-4] [--n-real 2] [--jobs 6]
"""
import argparse
import json
import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from qrc_qmi.config import MAIN
from qrc_qmi.hamiltonian import build_hamiltonian, draw_disorder
from qrc_qmi.reservoir import Reservoir, trace_distance

CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128)


def smallest_ns(N, channel, gamma, h, W, r, tol, dt, V):
    rng = np.random.default_rng(np.random.SeedSequence([11, N, int(h * 100), int(W * 100), r]))
    H = build_hamiltonian(N, h, *draw_disorder(N, W, rng))
    s = np.random.default_rng(r).uniform(0, 1, size=(1, 100))
    kw = dict(H=H, N=N, channel=channel, gamma=gamma, dt=dt, V=V)
    _, exact = Reservoir(propagator="exact", **kw).run(s, record=False)     # computed once
    for n_s in CANDIDATES:
        _, approx = Reservoir(propagator="strang", n_s=n_s, **kw).run(s, record=False)
        err = trace_distance(exact[0], approx[0])
        if err < tol:
            return dict(N=N, channel=channel, gamma=gamma, h=h, W=W, r=r, n_s=n_s, err=err)
    return dict(N=N, channel=channel, gamma=gamma, h=h, W=W, r=r, n_s=None, err=err)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--n-real", type=int, default=2)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--Ns", type=int, nargs="+", default=[4, 5])
    a = p.parse_args()
    jobs = [(N, c, g, h, W, r, a.tol, MAIN.dt, MAIN.V) for N in a.Ns for c in MAIN.channels
            for g in MAIN.gammas for h in MAIN.hs for W in MAIN.Ws for r in range(a.n_real)]
    res = Parallel(n_jobs=a.jobs)(delayed(smallest_ns)(*j) for j in jobs)
    table = {}
    for c in MAIN.channels:
        for g in MAIN.gammas:
            sel = [x for x in res if x["channel"] == c and x["gamma"] == g]
            if any(x["n_s"] is None for x in sel):
                raise SystemExit(f"tolerance not reached for {c}, gamma={g}")
            table[f"{c}|{g:.6g}"] = max(x["n_s"] for x in sel)
            print(f"{c:18s} gamma={g:8.4g}  n_s={table[f'{c}|{g:.6g}']:3d}  "
                  f"(per-N worst: {[max(x['n_s'] for x in sel if x['N'] == N) for N in a.Ns]})")
    out = dict(tol=a.tol, periods=100, Ns=a.Ns, n_real=a.n_real, dt=MAIN.dt, V=MAIN.V,
               hs=list(MAIN.hs), Ws=list(MAIN.Ws), n_s=table)
    Path("configs").mkdir(exist_ok=True)
    Path("configs/n_s.json").write_text(json.dumps(out, indent=2))
