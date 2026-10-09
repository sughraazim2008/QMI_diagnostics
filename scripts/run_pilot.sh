#!/bin/sh
# Reduced-size pilot of design v2: exercises every stage of the pipeline.
# NOT the pre-registered sweep.  Stages run one after another; running them side by side
# oversubscribes a laptop.
set -e
PY=.venv/bin/python
[ -f results/pilot2/esn_baseline.parquet ] || $PY scripts/tune_esn.py pilot2 --jobs 4
$PY scripts/run_sweep.py pilot2 --jobs 7
$PY scripts/analyze.py --main results/pilot2/pilot2.parquet --esn results/pilot2/esn_baseline.parquet \
    --label "PILOT of design v2 (N = 4, reduced size) - not the pre-registered sweep"
