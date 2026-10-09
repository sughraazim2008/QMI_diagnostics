# Contributing

Bug reports, questions and pull requests are welcome.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

## Before opening a pull request

- Add or update a test in `tests/` for any change to `src/qrc_qmi/`.
- Run `pytest -q` and make sure it passes.
- If you change the physics (channels, diagnostics, the reservoir model) or the statistical
  analysis, say so in the PR description and note it in `docs/DESIGN_V2.md` if it's a design
  change rather than a bug fix.
- Run `python examples/quickstart.py` and check the output still looks sane.

## Reporting a bug

Please include: the command you ran, the full error or unexpected output, and your Python
and package versions (`pip show qrc-qmi`, or just paste `requirements-lock.txt`'s header).

## Scope

This package simulates small (N <= ~8) open quantum systems as spin-network reservoirs,
compares them with a tuned classical echo-state-network baseline, and computes state
diagnostics (negativity, l1 coherence, quantum mutual information) that may predict the
performance gap between them. Extensions to other reservoir substrates, other diagnostics,
or other tasks are welcome as long as they follow the same pattern: diagnostics computed
from the state alone, task performance scored the same way for the quantum and classical
side, and new statistical claims tested against a held-out set.
