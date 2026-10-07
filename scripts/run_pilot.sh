#!/bin/sh
# Reduced-size pilot: exercises every stage of the pipeline.  NOT the pre-registered design.
# Stages run one after another; running them side by side oversubscribes a laptop.
set -e
PY=.venv/bin/python
[ -f results/pilot/esn_baseline.parquet ] || $PY scripts/tune_esn.py pilot --jobs 4
$PY scripts/run_sweep.py pilot --jobs 7
$PY scripts/analyze.py --main results/pilot/pilot.parquet --esn results/pilot/esn_baseline.parquet \
    --out results/pilot/analysis --label "PILOT (N = 4, reduced size) - not the pre-registered design"
$PY scripts/reproduce_palacios.py --jobs 7
# N = 6 pilot: several hours on a laptop (resumable)
# [ -f results/pilot6/esn_baseline.parquet ] || $PY scripts/tune_esn.py pilot6 --jobs 3
# $PY scripts/run_sweep.py pilot6 --jobs 7
