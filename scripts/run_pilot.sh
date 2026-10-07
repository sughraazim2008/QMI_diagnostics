#!/bin/sh
# Reduced-size pilot: exercises every stage of the pipeline.  NOT the pre-registered design.
set -e
PY=.venv/bin/python
$PY scripts/tune_esn.py pilot --jobs 4
$PY scripts/run_sweep.py pilot --jobs 7
$PY scripts/reproduce_palacios.py --jobs 7
$PY scripts/tune_esn.py pilot6 --jobs 3
$PY scripts/run_sweep.py pilot6 --jobs 7
