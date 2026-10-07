"""Run a sweep preset (main | transfer | pilot | pilot6 | smoke).

Usage: python scripts/run_sweep.py main --jobs 6 [--out results/main] [--limit 48]
Interrupted runs resume from the Parquet chunks already written.
"""
import argparse
import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")          # one BLAS thread per worker process

from qrc_qmi.config import PRESETS  # noqa: E402
from qrc_qmi.sweep import run_sweep  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=sorted(PRESETS))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None, help="run only the first LIMIT configurations")
    a = p.parse_args()
    cfg = PRESETS[a.preset]
    df = run_sweep(cfg, a.out or f"results/{cfg.name}", n_jobs=a.jobs, limit=a.limit)
    print(f"{len(df)} rows; echo-state failures: {(~df.esp_ok).sum()}; "
          f"median {df.seconds.median():.1f} s per configuration")
