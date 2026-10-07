# Working notes for Claude Code

This repository tests whether the quantum mutual information (QMI) of a spin-network quantum
reservoir predicts its NARMA-10 performance gap to a tuned echo state network, beyond what
negativity and l1 coherence explain (H1a), whether that relation depends on the decoherence
channel (H1b), and whether it transfers across channel, size and task (H1c).

Read `docs/METHODS.md` first. It is the paper's Methods section written to match the code,
and its Section 2.12 lists the open design decisions.

## Layout

| Path | Content |
|---|---|
| `src/qrc_qmi/hamiltonian.py` | Ising Hamiltonian, parity-resolved gap ratio |
| `src/qrc_qmi/channels.py` | Exact local channels (dephasing, amplitude damping, bit/phase flip) |
| `src/qrc_qmi/reservoir.py` | Reset-and-write input, exact and Strang propagators, readout features |
| `src/qrc_qmi/diagnostics.py` | Negativity, coherence, mutual information, controls |
| `src/qrc_qmi/tasks.py`, `ipc.py` | NARMA-5/10/20, Mackey–Glass, information processing capacity |
| `src/qrc_qmi/readout.py`, `esn.py` | Shot noise, ridge readout, tuned ESN baseline |
| `src/qrc_qmi/config.py`, `sweep.py` | Sweep presets, per-configuration runner, resumable parallel sweeps |
| `src/qrc_qmi/stats.py`, `figures.py` | H1a/H1b/H1c statistics and figures |
| `scripts/` | Command-line entry points (below) |
| `results/` | Parquet tables, reports and figures |

## Commands

Always use the project environment: `.venv/bin/python`.

```bash
.venv/bin/python -m pytest -q                              # 54 tests, about 30 s
.venv/bin/python scripts/pilot_gap_ratio.py                # regime map -> results/pilot/gap_ratio.csv
.venv/bin/python scripts/calibrate_ns.py --jobs 6          # Strang steps -> configs/n_s.json
.venv/bin/python scripts/reproduce_palacios.py --jobs 6    # validation against [14]
.venv/bin/python scripts/tune_esn.py main --jobs 4         # classical baselines for a preset
.venv/bin/python scripts/run_sweep.py main --jobs 6        # resumable; presets: main transfer pilot pilot6 smoke
.venv/bin/python scripts/analyze.py --main results/main/main.parquet --esn results/main/esn_baseline.parquet \
    --transfer results/transfer/transfer.parquet --transfer-esn results/transfer/esn_baseline.parquet
```

`analyze.py` writes `analysis/report.md`, `analysis/summary.json`, `analysis/table_*.csv` and
`analysis/figures/*.png` next to the main table.

## Result table columns

One row per configuration. Identifiers: `N, channel, gamma, h, W, real, cell, D`.
Performance: `nrmse_<task>_<s6|s5|s4>` (mean test NRMSE over input realisations at 10⁶, 10⁵,
10⁴ shots), with `nrmse_sd_*`, `loglam_*`, `lam_upper_*`. Capacity: `ipc_total`, `ipc_deg1..3`,
`ipc_threshold`. Diagnostics: `diag_*` (uniform-input drive) and `diagmg_*` (Mackey–Glass
drive). Checks: `gap_ratio`, `esp_dist`, `esp_ok`, `n_s`, `propagator`, `seconds`.
`stats.prepare` adds `delta`, `rel_delta`, `regime` and the log controls.

## Rules for the results analysis

1. Report only numbers that `scripts/analyze.py` (or a script you add under `scripts/`)
   produced from the Parquet tables. Do not type results into the paper by hand, and never
   estimate a number that has not been computed.
2. The decision rules, margins (ΔR² = 0.02, interaction bound 0.1, transfer ratio 0.70) and
   bootstrap settings are fixed in `stats.py` and Methods 2.9. Do not change them after
   looking at results. Anything added afterwards is exploratory and must be labelled so.
3. `results/pilot*` is a reduced-size run made to exercise the pipeline. It is not the
   pre-registered design and its numbers must not be presented as the paper's results.
4. State sample sizes after exclusions (echo-state failures) with every result.
5. When writing the Results section, follow the order of `report.md`: data and exclusions,
   baselines, the gap, relations among diagnostics, H1a, H1b, H1c, then sensitivity analyses
   and the reproduction of [14]. Give the point estimate, the interval and the pre-specified
   decision for each hypothesis, including null and inconclusive outcomes.
6. If a physics or statistics function changes, add or update a test and rerun the whole
   test suite before rerunning any sweep. A changed `SweepConfig` gets a new hash and a new
   chunk directory, so old and new results cannot mix.
7. Citations: numbers in square brackets refer to the reference list of the paper draft.
   Entries marked "not rechecked" or "preprint" there still need checking.
