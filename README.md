# qrc-qmi-diagnostics

Simulation and analysis code for the study *"Does quantum mutual information predict the
quantum–classical performance gap of spin-network reservoir computers?"*

Small transverse-field Ising reservoirs (N = 4, 6, 8) under pure dephasing or amplitude
damping are driven with NARMA and Mackey–Glass inputs, read out with finite-shot noise, and
compared with a tuned echo state network of the same readout dimension. State diagnostics
(negativity, l1 coherence, quantum mutual information) are then tested as predictors of the
performance gap.

* **Methods, as implemented:** [docs/METHODS.md](docs/METHODS.md)
* **Pilot results and what they do and do not show:** [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md)
* **Notes for analysing results with Claude Code:** [CLAUDE.md](CLAUDE.md)

## Status

The code is complete and tested. The gap-ratio pilot, the integrator calibration, a
reproduction check against Palacios et al. and a reduced-size pilot sweep have been run and
are in `results/`. The pre-registered main sweep (2,880 configurations) and transfer sweep
(360 configurations at N = 8) have **not** been run; see the compute estimate in
`docs/PILOT_RESULTS.md`.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Requires Python ≥ 3.10. `requirements-lock.txt` pins the versions used for the pilot
(Python 3.14, NumPy 2.5, SciPy 1.18).

## Run

```bash
.venv/bin/python scripts/pilot_gap_ratio.py            # regime map
.venv/bin/python scripts/calibrate_ns.py --jobs 6      # integrator calibration
.venv/bin/python scripts/reproduce_palacios.py         # validation against Palacios et al.
.venv/bin/python scripts/tune_esn.py main              # classical baselines
.venv/bin/python scripts/run_sweep.py main --jobs 6    # main sweep (resumable)
.venv/bin/python scripts/analyze.py --main results/main/main.parquet --esn results/main/esn_baseline.parquet
```

Sweep presets are defined in `src/qrc_qmi/config.py`: `main`, `transfer`, `pilot`, `pilot6`
and `smoke`.

## Conventions

Qubit 1 of the paper is index 0 in the code and is the first tensor factor. |0⟩ is the +1
eigenstate of σᶻ and σ⁻ = |0⟩⟨1|. Energies are in units of J_s, times in units of 1/J_s,
entropies in bits.
