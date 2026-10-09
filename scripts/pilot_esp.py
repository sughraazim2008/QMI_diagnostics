"""Pilot: how long a washout does the echo state property need?

For every configuration of a preset, two Haar-random initial states are driven by the same
uniform input and their trace distance is recorded at several washout lengths.
Usage: python scripts/pilot_esp.py pilot --jobs 6
"""
import argparse
import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from qrc_qmi.config import PRESETS, load_n_s, n_s_for  # noqa: E402
from qrc_qmi.hamiltonian import build_hamiltonian, draw_disorder  # noqa: E402
from qrc_qmi.reservoir import Reservoir, trace_distance  # noqa: E402

CHECKPOINTS = (250, 500, 1000, 2000, 4000)


def one(cfg, pt, table):
    N, ch, g, h, W = (pt[k] for k in ("N", "channel", "gamma", "h", "W"))
    ss = np.random.SeedSequence([cfg.base_seed, *[int(i) for i in pt["idx"]]])
    J, D = draw_disorder(N, W, np.random.default_rng(ss.spawn(4)[0]), coupling=cfg.coupling)
    res = Reservoir(H=build_hamiltonian(N, h, J, D), N=N, channel=ch, gamma=g, dt=cfg.dt, V=cfg.V,
                    n_s=n_s_for(cfg, N, ch, g, table), encoding=cfg.encoding)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.base_seed, 505, *[int(i) for i in pt["idx"]]]))
    rho = res.initial_state(2, rng)
    out = dict(N=N, channel=ch, gamma=g, h=h, W=W, real=pt["real"])
    done = 0
    for c in CHECKPOINTS:
        s = np.tile(rng.uniform(size=(1, c - done)), (2, 1))
        _, rho = res.run(s, rho0=rho, record=False)
        out[f"d{c}"] = trace_distance(rho[0], rho[1])
        done = c
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=sorted(PRESETS))
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    cfg = PRESETS[a.preset]
    table = load_n_s()
    df = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(one)(cfg, pt, table) for pt in cfg.grid()))
    out = Path(f"results/{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "esp_vs_washout.csv", index=False)
    cols = [f"d{c}" for c in CHECKPOINTS]
    for tol in (1e-6, 1e-4):
        print(f"\nShare passing (trace distance < {tol:g}) by washout length:")
        print((df[cols] < tol).groupby([df.channel]).mean().round(2).to_string())
    print("\nShare passing at 2000 steps (tol 1e-6), channel x gamma:")
    print((df["d2000"] < 1e-6).groupby([df.channel, df.gamma]).mean().unstack().round(2).to_string())
    print("\nShare passing at 2000 steps (tol 1e-6), channel x W:")
    print((df["d2000"] < 1e-6).groupby([df.channel, df.W]).mean().unstack().round(2).to_string())
    print("\nShare passing at 2000 steps (tol 1e-6), channel x h:")
    print((df["d2000"] < 1e-6).groupby([df.channel, df.h]).mean().unstack().round(2).to_string())
