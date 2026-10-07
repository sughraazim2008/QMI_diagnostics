"""Tune and score the classical baselines for every (task, D) of a sweep preset.

The ESN sees exactly the input realisations and data splits of the quantum reservoir.
Usage: python scripts/tune_esn.py main [--trials 100] [--jobs 4]
Writes results/<preset>/esn_baseline.parquet
"""
import argparse
import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from qrc_qmi.config import PRESETS  # noqa: E402
from qrc_qmi.esn import baseline_row  # noqa: E402
from qrc_qmi.sweep import make_inputs  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=sorted(PRESETS))
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = PRESETS[a.preset]
    inputs = make_inputs(cfg)
    jobs = []
    for N in cfg.Ns:
        D = cfg.n_features(N)
        for fam in inputs.values():
            for task, target in fam["targets"].items():
                jobs.append((task, fam["s"], target, cfg.splits, D, a.trials))
    rows = Parallel(n_jobs=a.jobs)(delayed(baseline_row)(*j) for j in jobs)
    df = pd.DataFrame(rows).sort_values(["task", "D"])
    out = Path(a.out or f"results/{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "esn_baseline.parquet", index=False)
    df.to_csv(out / "esn_baseline.csv", index=False)
    print(df.round(4).to_string(index=False))
