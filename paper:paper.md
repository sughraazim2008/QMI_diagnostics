---
title: 'qrc-qmi: simulating noisy quantum reservoir computers and their diagnostic-vs-performance gap against a tuned classical baseline'
tags:
  - Python
  - quantum computing
  - reservoir computing
  - open quantum systems
  - machine learning
authors:
  - name: Sughra Azim
    orcid: 0000-0000-0000-0000   # TODO: replace with your real ORCID (free at orcid.org, takes 2 minutes)
    affiliation: 1
affiliations:
  - name: School of Physics and Astronomy, University of Edinburgh, United Kingdom  # TODO: confirm exact department wording
    index: 1
date: XX Month 2026   # TODO: fill in on submission
bibliography: paper.bib
---

# Summary

`qrc-qmi` simulates small (N <= ~8 qubit) quantum reservoir computers under open-system
dynamics -- transverse-field Ising spin networks driven by a time-series input and subject
to local decoherence (pure dephasing or amplitude damping) -- and scores them on standard
time-series benchmarks (NARMA, Mackey-Glass) with finite-shot measurement noise included.
Alongside each simulation it computes a set of state diagnostics (entanglement negativity,
l1-norm coherence, and total/pairwise quantum mutual information) directly from the
reservoir's quantum state, without ever looking at the task it will be scored on. The
package also builds a size- and tuning-budget-matched classical echo-state-network baseline
for every configuration, so that "quantum reservoir performance" is always reported as a
*gap* against a fairly tuned classical comparator rather than as an absolute number. A
statistical-analysis module runs the cross-validated, bootstrap- and equivalence-tested
comparisons (nested-model R^2 gain, channel-interaction tests, transfer-ratio tests) needed
to ask, rigorously, whether a cheap state diagnostic predicts that performance gap.

# Statement of need

A recurring question in the quantum-reservoir-computing literature is whether some property
of a reservoir's quantum state -- entanglement, coherence, or a correlation measure such as
quantum mutual information -- predicts when the reservoir is *useful*, ideally before the
expensive step of running it on a task at all [@palacios2024; @gotting2023; @kora2024]. Prior
studies answer this for one diagnostic, one noise channel, and against absolute task
performance rather than a fairly tuned classical baseline; a parallel line of work shows that
apparent quantum advantages in reservoir computing can disappear once the classical baseline
is tuned as carefully as the quantum model [@pandey2026; @tariq2026]. No existing open-source
tool combines (1) multiple decoherence channels simulated under a shared, validated pipeline,
(2) a battery of state diagnostics computed without access to task targets, (3) a
tuning-matched classical baseline for every configuration, and (4) the statistical machinery
(nested cross-validation, cluster bootstrap, equivalence testing, Holm-corrected interaction
tests, transfer-ratio tests) needed to test "does this diagnostic predict the *fair-baseline*
performance gap" as a falsifiable, pre-registerable hypothesis rather than an informal
correlation.

`qrc-qmi` was built to run exactly that comparison for one such study [TODO: add citation to
your paper/preprint once it has a DOI or arXiv number] and is released so other researchers
asking the same kind of question -- for a different diagnostic, channel, task, or reservoir
size -- do not need to rebuild this pipeline from scratch. The package separates physics
(`channels.py`, `hamiltonian.py`, `reservoir.py`, `diagnostics.py`), task/readout code
(`tasks.py`, `readout.py`, `esn.py`, `ipc.py`), and statistical analysis (`stats.py`) so that
each piece is independently reusable; every physical and statistical claim it makes is
covered by a unit test (`tests/`, run automatically on every change).

# Acknowledgements

[TODO: thank anyone who helped -- a supervisor, a professor who gave feedback, anyone who
reviewed the pilot design.]

# References
