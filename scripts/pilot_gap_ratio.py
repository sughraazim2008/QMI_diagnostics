"""Pilot (Methods 2.8): map the mean adjacent-gap ratio over (h, W) to fix the regime levels.

Usage: python scripts/pilot_gap_ratio.py [--N 6 8 10] [--n-real 100]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qrc_qmi.hamiltonian import build_hamiltonian, draw_disorder, gap_ratio

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, nargs="+", default=[4, 6, 8, 10])
p.add_argument("--n-real", type=int, default=100)
p.add_argument("--out", default="results/pilot/gap_ratio.csv")
a = p.parse_args()

hs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
Ws = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
rows = []
for N in a.N:
    for h in hs:
        for W in Ws:
            rng = np.random.default_rng(np.random.SeedSequence([7, N, int(h * 100), int(W * 100)]))
            r = [gap_ratio(build_hamiltonian(N, h, *draw_disorder(N, W, rng)), N)
                 for _ in range(a.n_real)]
            rows.append(dict(N=N, h=h, W=W, r_mean=np.mean(r), r_sem=np.std(r, ddof=1) / np.sqrt(len(r))))
df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
for N in a.N:
    print(f"\nN = {N}: <r> (rows h, columns W)   GOE 0.531 | Poisson 0.386")
    print(df[df.N == N].pivot(index="h", columns="W", values="r_mean").round(3).to_string())
