# qrc-qmi-diagnostics

Simulation and analysis code for the study *"Does quantum mutual information predict the
quantum–classical performance gap of spin-network reservoir computers?"*

Small transverse-field Ising reservoirs (N = 4 to 7) under pure dephasing or amplitude
damping are driven with NARMA and Mackey–Glass inputs, read out with finite-shot noise, and
compared with a tuned echo state network of the same readout dimension. State diagnostics
(negativity, l1 coherence, quantum mutual information) are then tested as predictors of the
performance gap.

* **Methods, as implemented:** [docs/METHODS.md](docs/METHODS.md)
* **Hypotheses, decision rules and the reasons for the design:** [docs/DESIGN_V2.md](docs/DESIGN_V2.md)
* **Pilot results and what they do and do not show:** [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md)
* **Notes for analysing results with Claude Code:** [CLAUDE.md](CLAUDE.md)

## Status

The code is complete and tested (57 tests) and implements design v2. A gap-ratio pilot, an
integrator calibration, an echo-state pilot, a qualitative reproduction of Palacios et al.
and reduced-size N = 4 pilot sweeps of both designs have been run and are in `results/`.
The main sweep (4,320 configurations at N = 4, 5, 6) and the transfer sweep (288
configurations at N = 7) have **not** been run; they should follow pre-registration.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Requires Python ≥ 3.10. `requirements-lock.txt` pins the versions used for the pilot
(Python 3.14, NumPy 2.5, SciPy 1.18).

## Quickstart

New to the codebase and just want to see it run? [`examples/quickstart.py`](examples/quickstart.py)
builds one noisy 4-qubit reservoir, scores it on NARMA-10, and prints its state diagnostics
— in under 25 lines, with no sweep configuration required:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python examples/quickstart.py
```

Expected output (exact numbers depend on the fixed seeds in the script, but the shape is
always NRMSE around 0.5 and small positive diagnostic values):

```
NARMA-10 test NRMSE: 0.4985
Negativity: 0.0000   l1 coherence: 0.1093   QMI/qubit: 0.0734
```

For the full pilot/main/transfer sweeps this example doesn't touch (sweep presets, cluster
bootstrap analysis, the matched classical baseline), see **Run** below and
[docs/METHODS.md](docs/METHODS.md).

## Run

```bash
.venv/bin/python scripts/pilot_gap_ratio.py            # regime map
.venv/bin/python scripts/calibrate_ns.py --jobs 6      # integrator calibration
.venv/bin/python scripts/reproduce_palacios.py         # validation against Palacios et al.
.venv/bin/python scripts/tune_esn.py main              # classical baselines
.venv/bin/python scripts/run_sweep.py main --jobs 6    # main sweep (resumable)
.venv/bin/python scripts/analyze.py --main results/main/main.parquet --esn results/main/esn_baseline.parquet
```

Sweep presets are defined in `src/qrc_qmi/config.py`: `main`, `transfer`, `pilot2` and `smoke`.

## Conventions

Qubit 1 of the paper is index 0 in the code and is the first tensor factor. |0⟩ is the +1
eigenstate of σᶻ and σ⁻ = |0⟩⟨1|. Energies are in units of J_s, times in units of 1/J_s,
entropies in bits.
